from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch


def _frechet_distance(real_emb: np.ndarray, gen_emb: np.ndarray, eps: float = 1e-6) -> float:
    mu_r = np.mean(real_emb, axis=0)
    mu_g = np.mean(gen_emb, axis=0)
    cov_r = np.cov(real_emb, rowvar=False)
    cov_g = np.cov(gen_emb, rowvar=False)
    if cov_r.ndim == 0:
        cov_r = np.array([[float(cov_r)]], dtype=np.float64)
    if cov_g.ndim == 0:
        cov_g = np.array([[float(cov_g)]], dtype=np.float64)
    cov_r = cov_r + np.eye(cov_r.shape[0]) * eps
    cov_g = cov_g + np.eye(cov_g.shape[0]) * eps
    cov_prod = cov_r @ cov_g
    cov_prod = 0.5 * (cov_prod + cov_prod.T)
    eigvals = np.linalg.eigvalsh(cov_prod)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    tr_covmean = float(np.sum(np.sqrt(eigvals)))
    delta = mu_r - mu_g
    fbd = float(delta @ delta + np.trace(cov_r) + np.trace(cov_g) - 2.0 * tr_covmean)
    return max(fbd, 0.0)


@dataclass(frozen=True)
class FBDEmbedderCfg:
    checkpoint_path: str
    hidden_dim: int = 128
    num_cnn_stacks: int = 4
    p_dropout: float = 0.2
    num_classes: int = 47


class CNNCLSEmbedder:
    """
    Lightweight wrapper around the original paper's CNN classifier backbone
    (dirichlet-flow-matching `CNNModel` in clean_data mode) to produce embeddings.
    """

    def __init__(self, cfg: FBDEmbedderCfg, device: torch.device):
        import os
        import sys
        from types import SimpleNamespace

        dfm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dirichlet-flow-matching"))
        if dfm_root not in sys.path:
            sys.path.append(dfm_root)
        from model.dna_models import CNNModel  # type: ignore

        args = SimpleNamespace(
            hidden_dim=cfg.hidden_dim,
            num_cnn_stacks=cfg.num_cnn_stacks,
            dropout=cfg.p_dropout,
            clean_data=True,
            cls_expanded_simplex=False,
            mode="dirichlet",
            cls_free_guidance=False,
        )
        self.device = device
        self.model = CNNModel(args=args, alphabet_size=4, num_cls=cfg.num_classes, classifier=True).to(device)

        state = torch.load(cfg.checkpoint_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("Unsupported checkpoint format for clean classifier embedder")
        cleaned = {}
        for k, v in state.items():
            nk = k
            for prefix in ("model.", "cls_model.", "clean_cls_model."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]
            cleaned[nk] = v
        self.model.load_state_dict(cleaned, strict=False)
        self.model.eval()

    @torch.no_grad()
    def encode(self, seq_ids: torch.Tensor) -> np.ndarray:
        t = torch.zeros(seq_ids.size(0), device=self.device)
        _, emb = self.model(seq_ids.to(self.device), t=t, return_embedding=True)
        return emb.detach().cpu().numpy()


class GenerativeMetricsCallback(pl.Callback):
    def __init__(
        self,
        *,
        train_sequences: list[torch.Tensor],
        seq_len: int,
        sample_size: int = 256,
        log_prefix: str = "val_",
        mel_embedder: FBDEmbedderCfg | None = None,
        fb_embedder: FBDEmbedderCfg | None = None,
        g4hunter_window: int = 25,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.sample_size = int(sample_size)
        self.log_prefix = str(log_prefix)
        self._train_set = {tuple(s.tolist()) for s in train_sequences}
        self._mel_cfg = mel_embedder
        self._fb_cfg = fb_embedder
        self._g4hunter_window = int(g4hunter_window)

        self._last_val_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._mel: CNNCLSEmbedder | None = None
        self._fb: CNNCLSEmbedder | None = None

    @staticmethod
    def _ids_to_seq(ids: torch.Tensor) -> str:
        alphabet = "ACGT"
        return "".join(alphabet[int(t)] if 0 <= int(t) < 4 else "N" for t in ids.tolist())

    @staticmethod
    def _g4hunter_base_scores(seq: str) -> np.ndarray:
        """
        Approximation of G4Hunter nucleotide scoring:
        runs of G get +1..+4 (capped), runs of C get -1..-4 (capped), others 0.
        """
        scores = np.zeros(len(seq), dtype=np.float32)
        i = 0
        while i < len(seq):
            ch = seq[i]
            j = i + 1
            while j < len(seq) and seq[j] == ch:
                j += 1
            run_len = min(4, j - i)
            if ch == "G":
                scores[i:j] = float(run_len)
            elif ch == "C":
                scores[i:j] = float(-run_len)
            i = j
        return scores

    @classmethod
    def _g4hunter_seq_score(cls, seq: str, window: int) -> float:
        base = cls._g4hunter_base_scores(seq)
        if len(base) == 0:
            return 0.0
        if window <= 1:
            return float(np.max(np.abs(base)))
        if len(base) < window:
            # Original G4Hunter CalScore returns an empty list in this case.
            # For metric aggregation we map this to 0.0.
            return 0.0
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smooth = np.convolve(base, kernel, mode="valid")
        return float(np.max(np.abs(smooth)))

    @classmethod
    def _g4hunter_scores(cls, seqs: list[str], window: int) -> np.ndarray:
        return np.array([cls._g4hunter_seq_score(s, window=window) for s in seqs], dtype=np.float32)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self._last_val_batch is None:
            x, y, cond = batch
            self._last_val_batch = (x.detach(), y.detach(), cond.detach())

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self._last_val_batch is None:
            return
        x, y, cond = self._last_val_batch
        self._last_val_batch = None

        device = pl_module.device
        n = min(self.sample_size, int(cond.size(0)))
        if n <= 0:
            return
        cond = cond[:n].to(device)
        y = y[:n].to(device)

        # --- generation ---
        if hasattr(pl_module, "generate") and callable(getattr(pl_module, "generate")):
            try:
                gen = pl_module.generate(cond)  # type: ignore[misc]
            except TypeError:
                gen = pl_module.generate(cond, seq_len=y.size(1))  # type: ignore[misc]
        else:
            return
        gen = gen.long().detach().cpu()
        real = y.long().detach().cpu()

        # --- perplexity from val_loss (epoch) ---
        val_loss = trainer.callback_metrics.get("val_loss")
        if isinstance(val_loss, torch.Tensor):
            ppl = float(math.exp(float(val_loss.detach().cpu().item())))
            pl_module.log(self.log_prefix + "perplexity", ppl, prog_bar=True, logger=True, on_epoch=True)

        # --- novelty ---
        novelty = float(np.mean([tuple(s.tolist()) not in self._train_set for s in gen]))
        pl_module.log(self.log_prefix + "novelty", novelty, prog_bar=True, logger=True, on_epoch=True)

        # --- FBD (Melanoma / Flybrain) ---
        if self._mel_cfg is not None:
            if self._mel is None:
                self._mel = CNNCLSEmbedder(self._mel_cfg, device=device)
            mel_real = self._mel.encode(real.to(device))
            mel_gen = self._mel.encode(gen.to(device))
            pl_module.log(self.log_prefix + "melanoma_fbd", _frechet_distance(mel_real, mel_gen), prog_bar=False, logger=True, on_epoch=True)

        if self._fb_cfg is not None:
            if self._fb is None:
                self._fb = CNNCLSEmbedder(self._fb_cfg, device=device)
            fb_real = self._fb.encode(real.to(device))
            fb_gen = self._fb.encode(gen.to(device))
            pl_module.log(self.log_prefix + "flybrain_fbd", _frechet_distance(fb_real, fb_gen), prog_bar=False, logger=True, on_epoch=True)

        # --- G4Hunter similarity ---
        real_seqs = [self._ids_to_seq(s) for s in real]
        gen_seqs = [self._ids_to_seq(s) for s in gen]
        real_g4 = self._g4hunter_scores(real_seqs, window=self._g4hunter_window)
        gen_g4 = self._g4hunter_scores(gen_seqs, window=self._g4hunter_window)
        pl_module.log(self.log_prefix + "g4hunter_real_mean", float(np.mean(real_g4)), prog_bar=False, logger=True, on_epoch=True)
        pl_module.log(self.log_prefix + "g4hunter_gen_mean", float(np.mean(gen_g4)), prog_bar=False, logger=True, on_epoch=True)
        pl_module.log(
            self.log_prefix + "g4hunter_gap",
            float(np.mean(np.abs(real_g4 - gen_g4))),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

