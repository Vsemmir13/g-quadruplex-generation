import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DNAConvVAE(LightningModule):
    def __init__(
        self,
        seq_len=512,
        vocab_size=4,
        latent_dim=64,
        cond_dim=1,
        hidden_dim=256,
        lr=1e-3,
        beta=0.1,
        beta_warmup_steps=2000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(vocab_size, hidden_dim // 2, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.to_mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

        self.decoder_in = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, vocab_size, kernel_size=3, padding=1),
        )

        self.lr = lr
        self.beta = beta
        self.beta_warmup_steps = beta_warmup_steps
        self.test_losses = []

    def one_hot(self, x):
        return F.one_hot(x, num_classes=self.vocab_size).float()

    def encode(self, x, cond):
        x = self.one_hot(x)
        x = x.permute(0, 2, 1)
        h = self.encoder(x)
        cond_emb = self.cond_emb(cond)
        h = h + cond_emb[:, :, None]
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        h = self.decoder_in(z)
        cond_emb = self.cond_emb(cond)
        h = h + cond_emb[:, :, None]
        logits = self.decoder(h)
        logits = logits.permute(0, 2, 1)
        return logits[:, : self.seq_len, :]

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, cond)
        return logits, mu, logvar

    def loss_fn(self, logits, targets, mu, logvar):
        recon = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        warm = min(1.0, float(self.global_step) / float(self.beta_warmup_steps)) if self.beta_warmup_steps else 1.0
        beta_eff = self.beta * warm
        return recon + beta_eff * kld, recon, kld

    def training_step(self, batch, batch_idx):
        x, y, cond = batch
        logits, mu, logvar = self(x, cond)
        loss, recon, kld = self.loss_fn(logits, y, mu, logvar)
        self.log_dict(
            {
                "train_loss": loss,
                "train_recon": recon,
                "train_kld": kld,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, cond = batch
        logits, mu, logvar = self(x, cond)
        loss, recon, kld = self.loss_fn(logits, y, mu, logvar)
        self.log_dict(
            {
                "val_loss": loss,
                "val_recon": recon,
                "val_kld": kld,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        x, y, cond = batch
        logits, mu, logvar = self(x, cond)
        loss, _, _ = self.loss_fn(logits, y, mu, logvar)
        self.test_losses.append(loss.detach())
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)

    def on_test_epoch_end(self):
        avg = torch.stack(self.test_losses).mean()
        self.log("avg_test_loss", avg, logger=True)
        self.test_losses.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, cond = batch
        logits, mu, logvar = self(x, cond)
        recon = torch.argmax(logits, dim=-1)
        gen = self.generate(cond)
        return {
            "x": x.detach().cpu(),
            "levels": cond.detach().cpu(),
            "recon": recon.detach().cpu(),
            "gen": gen.detach().cpu(),
            "mu": mu.detach().cpu(),
            "logvar": logvar.detach().cpu(),
        }

    def generate(self, cond, z=None):
        if z is None:
            z = torch.randn(
                cond.size(0),
                self.hparams.latent_dim,
                self.seq_len // 4,
                device=cond.device,
            )
        elif z.dim() == 2:
            z = z[:, :, None].expand(-1, -1, self.seq_len // 4)
        logits = self.decode(z, cond)
        return torch.argmax(logits, dim=-1)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = {
            "scheduler": ReduceLROnPlateau(opt, mode="min", patience=5),
            "monitor": "val_loss",
        }
        return [opt], [sched]
