# model.py
import torch
import torch.nn as nn
import math
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

class QuadLSTM(LightningModule):
    def __init__(self, vocab_size=5, emb_dim=128, level_dim=8, hidden_dim=256, num_layers=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.level_emb = nn.Linear(1, level_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim + level_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.test_losses = []
        self.lr = lr

    def forward(self, x, levels):
        B, T = x.shape
        token_emb = self.emb(x)
        level_emb = self.level_emb(levels)
        level_emb = level_emb.unsqueeze(1).expand(B, T, -1)
        inp = torch.cat([token_emb, level_emb], dim=-1)
        out, _ = self.lstm(inp)
        logits = self.fc_out(out)
        return logits

    def training_step(self, batch, batch_idx):
        x, y, levels = batch
        logits = self(x, levels)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, levels = batch
        logits = self(x, levels)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]
    
    def test_step(self, batch, batch_idx):
        x, y, levels = batch
        logits = self(x, levels)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_losses.append(loss.detach())
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_losses).mean()
        self.log('avg_test_loss', avg_loss, prog_bar=True, logger=True)
        perplexity = math.exp(avg_loss.item())
        self.log('test_perplexity', perplexity, prog_bar=True, logger=True)
        self.test_losses.clear()
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, levels = batch
        logits = self(x, levels)
        recon = torch.argmax(logits, dim=-1)
        gen = recon
        return {
            "x": x.detach().cpu(),
            "levels": levels.detach().cpu(),
            "recon": recon.detach().cpu(),
            "gen": gen.detach().cpu(),
        }
