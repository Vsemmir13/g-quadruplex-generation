# Discrete Flow Matching for G‑Quadruplex Generation

**Leontenkov Egor**  
Faculty of Computer Science, National Research University Higher School of Economics (HSE), Moscow, Russia  
`eeleontenkov@edu.hse.ru`

This folder contains the code for my bachelor thesis project on conditional generation of short DNA sequences containing G‑quadruplex (G4), conditioned on a structual level. The main model is a Discrete Dirichlet Flow Matching generator; LSTM and VAE are included as baselines for comparison.

The project is part of a bachelor thesis titled:

> **Discrete Flow Matching for G‑Quadruplex Generation**

### Abstract (thesis)

G4‑quadruplexes are non‑canonical DNA secondary structures with critical roles in gene regulation, genome stability, and possible therapeutic applications. Understanding and designing G4‑forming sequences are essential for synthetic genomics and bioinformatics. In this work, we propose a conditional generative framework to produce short DNA sequences containing G4, conditioned on the G4‑level. Our approach leverages a Discrete Flow Matching model trained on the EndoQuad database, which contains experimentally validated endogenous G4s. The model receives a G4 structural level annotation and generates sequences consistent with this condition. Evaluation metrics include perplexity, Fréchet Biological Distance and G4Hunter score. Our results suggest that conditional generative models can effectively generate biologically diverse G4 sequences aligned with target properties, offering a new tool for programmable DNA sequence design.

Index Terms—discrete models, flow matching, generative models, DNA sequences, G‑quadruplexes

---

## What is implemented here

- Main model — Discrete Dirichlet Flow Matching (DFM): `models/dfm_module.py`, `models/dfm_model.py`, `models/dfm_flow_utils.py`  
Conditional flow‑matching model on the simplex using a Dirichlet probability path. Two backbones are supported: residual CNN (`dfm`) and a stronger Transformer variant (`dfm_transformer`).
- Baseline — LSTM (autoregressive LM): `models/lstm.py`  
Next‑token prediction conditioned on `level_norm`.
- Baseline — VAE (conditional CNN, positional latent): `models/vae.py`  
Conditional reconstruction/generation with a positional latent and KL warmup.

---

## Dataset and conditioning

Dataset code: `utils/data_utils.py` (`QuadDataset`).

- Sequences: extract windows of length `seq_len` from `hg38.fa` using coordinates from a BED file.
- Filtering: windows containing `N` are discarded.
- Condition: `level_norm` (float), fed to the models as `cond`.

---

## Quick start

### Dry-run testing (verify all 4 models run)

Runs `fast_dev_run=True` (1 train batch + 1 val batch) for LSTM/VAE/DFM-CNN/DFM-Transformer:

```bash
python tests/test_models.py \
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa
```

---

## Training runs

### LSTM or VAE (single entrypoint)

Script: `main.py`

LSTM:

```bash
python main.py --model_type lstm \
  --experiment_name training_lstm
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --epochs 1 --batch_size 256
```

VAE:

```bash
python main.py --model_type vae \
  --experiment_name training_vae
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --epochs 1 --batch_size 128
```

DFM:

```bash
python main.py --model_type dfm \
  --experiment_name training_dfm
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --epochs 1 --batch_size 128 \
  --alpha_max 12 --alpha_scale 15 --num_integration_steps 64
```

DFM Transformer:

```bash
python main.py --model_type dfm_transformer \
  --experiment_name training_dfm_transformer \
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --epochs 1 --batch_size 64 \
  --hidden_dim 256 \
  --num_transformer_layers 6 \
  --num_attention_heads 8 \
  --transformer_ff_mult 4 \
  --alpha_max 12 --alpha_scale 15 --num_integration_steps 64
```

After `test`, examples are saved to JSONL: `examples/{model_type}/{experiment_name}.jsonl`.

---

## Multi‑GPU and DDP

`main.py` automatically enable DDP when:

- CUDA is available
- and `devices > 1`

Useful flags:

- use all available devices
- `--strategy`: explicitly set a strategy (usually not needed; default: auto‑DDP on multi‑GPU)

Example:

```bash
python main.py --model_type vae \
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
```

---

## Metrics tables

| Model | Train loss | Val loss | Test loss |  Perplexity   |  FBD  | G4Hunter | Notes |
| ---   | --         | -------- | --------- | --------------| ------| -------- | ----- |
| LSTM  |            |          |           |               |       |          |       |
| VAE   |            |          |           |               |       |          |       |
| DFM   |            |          |           |               |       |          |       |
| DFM transformer |            |          |           |               |       |          |       |
