## Discrete Flow Matching for G‑Quadruplex Generation

**Leontenkov Egor**  
Faculty of Computer Science, National Research University Higher School of Economics (HSE), Moscow, Russia  
`eeleontenkov@edu.hse.ru`

This folder contains the code for my bachelor thesis project on **conditional generation** of short DNA sequences containing **G‑quadruplex (G4)** motifs/regions, conditioned on a **G4 structural level**. The **main model** is a **Discrete Dirichlet Flow Matching** generator; **LSTM** and **VAE** are included as baselines for comparison.

The project is part of a bachelor thesis titled:

> **Discrete Flow Matching for G‑Quadruplex Generation**

### Abstract (thesis)

G4‑quadruplexes are non‑canonical DNA secondary structures with critical roles in gene regulation, genome stability, and possible therapeutic applications. Understanding and designing G4‑forming sequences are essential for synthetic genomics and bioinformatics. In this work, we propose a conditional generative framework to produce short DNA sequences containing G4, conditioned on the G4‑level. Our approach leverages a Discrete Flow Matching model trained on the EndoQuad database, which contains experimentally validated endogenous G4s. The model receives a G4 structural level annotation and generates sequences consistent with this condition. Evaluation metrics include perplexity, Fréchet Biological Distance and G4Hunter score. Our results suggest that conditional generative models can effectively generate biologically diverse G4 sequences aligned with target properties, offering a new tool for programmable DNA sequence design.

Index Terms—discrete models, flow matching, generative models, DNA sequences, G‑quadruplexes

---

## What is implemented here

- **Main model — Discrete Dirichlet Flow Matching (DFM)**: `quadruplex/dfm_module.py`, `quadruplex/dfm_model.py`, `quadruplex/dfm_flow_utils.py`  
  Conditional flow‑matching model on the simplex using a Dirichlet probability path.
- **Baseline — LSTM (autoregressive LM)**: `quadruplex/lstm.py`  
  Next‑token prediction conditioned on `level_norm`.
- **Baseline — VAE (conditional CNN, positional latent)**: `quadruplex/vae.py`  
  Conditional reconstruction/generation with a **positional latent** (no global pooling) and KL warmup.

---

## Dataset and conditioning

Dataset code: `quadruplex/data_utils.py` (`QuadDataset`).

- **Sequences**: extract windows of length `seq_len` from `hg38.fa` using coordinates from a BED file.
- **Filtering**: windows containing `N` are discarded.
- **Condition**: `level_norm = (level - 1) / 3` (float), fed to the models as `cond`.

Important fix already applied: the stored `levels` are **kept aligned** with the stored sequences (previously they could become misaligned when some windows were filtered out due to `N`).

---

## Why these design choices

- **LSTM + BOS token**: for stable AR training we use an explicit BOS token, with input `[BOS] + seq[:-1]` and target `seq`. This keeps lengths constant and typically improves optimization stability.
- **VAE without global pooling**: collapsing a length‑512 sequence into a single vector (e.g. `AdaptiveAvgPool1d(1)`) often harms reconstruction. We therefore use a **positional latent** of shape `[B, latent_dim, T/4]` and add **KL warmup** (`beta_warmup_steps`).
- **DFM (main model)**: provides a non‑autoregressive approach to discrete sequence generation with continuous conditioning.

---

## Quick start

### Installation

From the repository root:

```bash
pip install -r quadruplex/requirements.txt
```

### Smoke test (verify all 3 models run)

Runs `fast_dev_run=True` (1 train batch + 1 val batch) for LSTM/VAE/DFM:

```bash
python quadruplex/smoke_test_all.py \
  --file_path_quadruplex quadruplex/data/EQ_hg38_lifted.bed \
  --file_path_seq quadruplex/data/hg38.fa
```

---

## Training runs

### LSTM or VAE (single entrypoint)

Script: `quadruplex/main.py`

LSTM:

```bash
python quadruplex/main.py --model_type lstm \
  --file_path_quadruplex quadruplex/data/EQ_hg38_lifted.bed \
  --file_path_seq quadruplex/data/hg38.fa \
  --epochs 1 --batch_size 256
```

VAE:

```bash
python quadruplex/main.py --model_type vae \
  --file_path_quadruplex quadruplex/data/EQ_hg38_lifted.bed \
  --file_path_seq quadruplex/data/hg38.fa \
  --epochs 1 --batch_size 128
```

DFM (main model):

```bash
python quadruplex/main.py --model_type dfm \
  --file_path_quadruplex quadruplex/data/EQ_hg38_lifted.bed \
  --file_path_seq quadruplex/data/hg38.fa \
  --epochs 1 --batch_size 128 \
  --alpha_max 12 --alpha_scale 15 --num_integration_steps 64
```

After `test`, examples are saved to JSONL: `quad_<model>_examples.jsonl`.

### Dirichlet Flow Matching (separate script)

Script: `quadruplex/train_dfm.py`

```bash
python quadruplex/train_dfm.py \
  --file_path_quadruplex quadruplex/data/EQ_hg38_lifted.bed \
  --file_path_seq quadruplex/data/hg38.fa \
  --epochs 1 --batch_size 128
```

---

## Multi‑GPU and DDP (for speed)

`quadruplex/main.py` and `quadruplex/train_dfm.py` automatically enable **DDP** when:

- **CUDA** is available
- and `devices > 1`

Useful flags:

- `--devices`: number of GPUs to use (default: all available)
- `--strategy`: explicitly set a strategy (usually not needed; default: auto‑DDP on multi‑GPU)

Example:

```bash
python quadruplex/main.py --model_type vae \
  --file_path_quadruplex quadruplex/data/EQ_hg38_lifted.bed \
  --file_path_seq quadruplex/data/hg38.fa \
  --devices 4
```

---

## Likelihood metric via G4Bert (G‑DNABERT)

Reference: the [G‑DNABERT](https://github.com/mitiau/G-DNABERT) repository.

This project: `quadruplex/g4bert_metric.py`

We compute **pseudo‑log‑likelihood (PLL)** for BERT‑style masked language models:

- mask one token at a time
- score \(\log p(x_i \mid x_{\setminus i})\)
- average/sum across positions

This is a “likelihood‑like” metric for BERT (BERT does not define a standard autoregressive likelihood).

Example usage:

```python
from quadruplex.g4bert_metric import pseudo_log_likelihood_bert

seqs = ["ACGTACGTACGT", "GGGGTTTTCCCCAAAA"]
res = pseudo_log_likelihood_bert(
    seqs,
    model_name_or_path="PUT_G4BERT_CHECKPOINT_HERE",
    max_sequences=32,
)
print(res.nll_mean, res.tokens_scored)
```

Note: you must use the correct checkpoint/tokenizer (as in the G‑DNABERT notebooks), otherwise PLL is not meaningful.

---

## Metrics tables (fill later)

### Table 1 — Core metrics per model

| Model | Train loss | Val loss | Test loss | Perplexity (if applicable) | G4Bert NLL ↓ | G4Bert PLL ↑ | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| LSTM |  |  |  |  |  |  |  |
| VAE |  |  |  |  |  |  |  |
| DFM (Dirichlet) |  |  |  |  |  |  |  |

### Table 2 — Before/after comparison (show improvements)

| Model | Version | Val loss | Test loss | G4Bert NLL ↓ | Delta vs prev | What changed |
|---|---|---:|---:|---:|---:|---|
| LSTM | v0 |  |  |  |  |  |
| LSTM | v1 |  |  |  |  |  |
| VAE | v0 |  |  |  |  |  |
| VAE | v1 |  |  |  |  |  |
| DFM | v0 |  |  |  |  |  |
| DFM | v1 |  |  |  |  |  |

### Table 3 — Training speed and stability

| Model | devices | strategy | batch_size | steps/sec | epoch time | OOM? | crashed? | Notes |
|---|---:|---|---:|---:|---:|---|---|---|
| LSTM |  |  |  |  |  |  |  |  |
| VAE |  |  |  |  |  |  |  |  |
| DFM |  |  |  |  |  |  |  |  |

