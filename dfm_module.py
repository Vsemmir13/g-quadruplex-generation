import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from dfm_flow_utils import (
    DirichletConditionalFlow,
    expand_simplex,
    sample_cond_prob_path,
    simplex_proj,
)
from dfm_model import QuadCondCNN


class QuadDFMModule(LightningModule):
    def __init__(
        self,
        *,
        seq_len: int = 512,
        vocab_size: int = 4,
        cond_dim: int = 1,
        hidden_dim: int = 256,
        num_cnn_stacks: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        alpha_max: float = 12.0,
        alpha_scale: float = 15.0,
        fix_alpha: float | None = None,
        prior_pseudocount: float = 2.0,
        num_integration_steps: int = 64,
        flow_temp: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = QuadCondCNN(
            alphabet_size=vocab_size,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            num_cnn_stacks=num_cnn_stacks,
            dropout=dropout,
            expanded_simplex=True,
        )
        self.condflow = DirichletConditionalFlow(k=vocab_size, alpha_max=alpha_max, alpha_spacing=0.001)

        self.test_losses: list[torch.Tensor] = []

    def training_step(self, batch, batch_idx):
        x, _, cond = batch
        loss, recon = self._step_loss(x, cond)
        self.log_dict({"train_loss": loss, "train_recon": recon}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, cond = batch
        loss, recon = self._step_loss(x, cond)
        self.log_dict({"val_loss": loss, "val_recon": recon}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, _, cond = batch
        loss, _ = self._step_loss(x, cond)
        self.test_losses.append(loss.detach())
        self.log("test_loss", loss)

    def on_test_epoch_end(self):
        if self.test_losses:
            avg = torch.stack(self.test_losses).mean()
            self.log("avg_test_loss", avg)
            self.test_losses.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _, cond = batch
        # For compatibility with save_examples()
        gen = self.generate(cond)
        return {
            "x": x.detach().cpu(),
            "levels": cond.detach().cpu(),
            "recon": x.detach().cpu(),
            "gen": gen.detach().cpu(),
        }

    def _step_loss(self, seq: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Sample xt along the Dirichlet path
        xt, alphas = sample_cond_prob_path(
            seq,
            self.hparams.vocab_size,
            alpha_scale=self.hparams.alpha_scale,
            alpha_max=self.hparams.alpha_max,
            fix_alpha=self.hparams.fix_alpha,
        )
        xt_inp, _ = expand_simplex(xt, alphas, self.hparams.prior_pseudocount)

        logits = self.model(xt_inp, t=alphas, cond=cond)
        recon = F.cross_entropy(logits.reshape(-1, self.hparams.vocab_size), seq.reshape(-1))
        return recon, recon

    @torch.no_grad()
    def generate(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Sample sequences conditioned on `cond`.
        """
        b = cond.size(0)
        l = int(self.hparams.seq_len)
        k = int(self.hparams.vocab_size)

        xt = torch.distributions.Dirichlet(torch.ones(b, l, k, device=self.device)).sample()
        eye = torch.eye(k, device=self.device)

        t_span = torch.linspace(1.0, float(self.hparams.alpha_max), int(self.hparams.num_integration_steps), device=self.device)
        for s, t in zip(t_span[:-1], t_span[1:]):
            s_batch = s[None].expand(b)
            xt_exp, _ = expand_simplex(xt, s_batch, float(self.hparams.prior_pseudocount))
            logits = self.model(xt_exp, t=s_batch, cond=cond)
            flow_probs = torch.softmax(logits / float(self.hparams.flow_temp), dim=-1)

            if (not torch.allclose(flow_probs.sum(-1), torch.ones_like(flow_probs[..., 0]), atol=1e-4)) or (flow_probs < 0).any():
                flow_probs = simplex_proj(flow_probs)

            c_factor = self.condflow.c_factor(xt.detach().cpu().numpy(), float(s.item()))
            c_factor = torch.from_numpy(c_factor).to(xt).float()
            if torch.isnan(c_factor).any():
                c_factor = torch.nan_to_num(c_factor)

            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)  # [B, L, K, K]
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)  # [B, L, K]
            xt = xt + flow * (t - s)

            if (not torch.allclose(xt.sum(-1), torch.ones_like(xt[..., 0]), atol=1e-4)) or (xt < 0).any():
                xt = simplex_proj(xt)

        # final decode: sample argmax (mode)
        final_logits = self.model(expand_simplex(xt, t_span[-1][None].expand(b), float(self.hparams.prior_pseudocount))[0],
                                  t=t_span[-1][None].expand(b), cond=cond)
        return torch.argmax(final_logits, dim=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.lr))

