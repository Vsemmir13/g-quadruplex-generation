# Discrete Flow Matching for G-Quadruplex Generation

**Leontenkov Egor**  
Faculty of Computer Science, National Research University Higher School of Economics (HSE), Moscow, Russia  
`eeleontenkov@edu.hse.ru`

This repository contains code for conditional generation of short DNA sequences containing G-quadruplexes (G4), conditioned on categorical G4 level. The main model is a Discrete Dirichlet Flow Matching generator. LSTM and VAE are included as baselines.

## Models

| Model | Files | What it does |
| --- | --- | --- |
| DFM CNN | `models/dfm_module.py`, `models/dfm_model.py`, `models/dfm_flow_utils.py` | Dirichlet flow matching on the simplex with a residual CNN backbone. This is the paper-like DFM setup. |
| DFM CNN large | `models/dfm_module.py`, `models/dfm_model.py`, `models/dfm_flow_utils.py` | Larger DFM CNN checkpoint used to test whether extra capacity improves generation quality. |
| LSTM | `models/lstm.py` | Autoregressive next-token language model conditioned on categorical G4 level. |
| VAE | `models/vae.py` | Conditional convolutional VAE trained by reconstruction loss plus KL warmup. |

Current trainable parameter counts:

| Model | Parameters |
| --- | ---: |
| DFM CNN, paper-like | `3,660,676` |
| DFM CNN large checkpoint | `14,595,844` |
| LSTM | `6,909,909` |
| VAE | `17,656,612` |

## Dataset

Dataset code: `utils/data_utils.py`.

- Input BED file: `data/EQ_hg38_lifted.bed`
- Reference genome: `data/hg38.fa`
- Sequence length: `512`
- Classes: raw G4 levels `4`, `5`, `6`
- Model condition ids: `0`, `1`, `2`, produced by `level - 4`
- Windows containing `N` are discarded

## Training

The main entrypoint is `main.py`. Most hyperparameters are intentionally fixed in `CFG` inside `utils/config.py`; command-line flags are kept short.

### LSTM

```bash
python main.py \
  --experiment_name g4_lstm \
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --model_type lstm \
  --max_steps 450000 \
  --max_epochs 10000 \
  --batch_size 256 \
  --num_workers 4 \
  --progress_bar
```

### VAE

```bash
python main.py \
  --experiment_name g4_vae \
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --model_type vae \
  --max_steps 450000 \
  --max_epochs 10000 \
  --batch_size 256 \
  --num_workers 4 \
  --progress_bar
```

### DFM CNN

```bash
python main.py \
  --experiment_name g4_dfm_small \
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --model_type dfm \
  --max_steps 450000 \
  --max_epochs 100000 \
  --batch_size 512 \
  --num_workers 4 \
  --guidance_mode probability_addition \
  --guidance_scale 3 \
  --progress_bar
```

### Resume From Checkpoint

```bash
python main.py \
  --experiment_name g4_dfm_small \
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --model_type dfm \
  --ckpt_path checkpoints/dfm/g4_dfm_small/last.ckpt \
  --max_steps 450000 \
  --batch_size 512 \
  --num_workers 4 \
  --progress_bar
```

### Test Only

```bash
python main.py \
  --experiment_name g4_dfm_small_test \
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --model_type dfm \
  --run_mode test \
  --ckpt_path checkpoints/dfm/g4_dfm_small/last.ckpt \
  --metric_samples 1024 \
  --batch_size 512 \
  --num_workers 4 \
  --progress_bar
```

## Class-Wise Metrics

Class-wise evaluation is preferred over one combined FBD because classes `4`, `5`, and `6` are not identical. Combined metrics can hide failures in one class.

Metric scripts live in `metrics/`:

| File | Purpose |
| --- | --- |
| `metrics/eval.py` | Generates/reuses class-wise samples and computes FBD, G4Hunter gaps, and novelty. |
| `metrics/pqsfinder.py` | Computes `pqsfinder` metrics for already generated JSONL samples. |
| `metrics/pqsfinder_metrics.R` | Small R helper used by `metrics/pqsfinder.py`. |
| `metrics/run_pqsfinder_metrics.sh` | Bash wrapper for `pqsfinder` metrics. |

Use `metrics/eval.py` for metrics-only evaluation:

```bash
python -m metrics.eval \
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --model lstm:lstm:checkpoints/lstm.ckpt \
  --model vae:vae:checkpoints/vae.ckpt \
  --model dfm:dfm:checkpoints/dfm.ckpt \
  --model dfm_large:dfm:checkpoints/dfm_large.ckpt \
  --classes 4 5 6 \
  --num_samples 2000 \
  --batch_size 512 \
  --guidance_modes probability_addition \
  --guidance_scales 0 1 2 3 \
  --embedders melanoma hyenadna \
  --output_dir generated/classwise_metrics
```

Outputs:

```text
generated/classwise_metrics/val/classwise_metrics.csv
generated/classwise_metrics/val/<model>/class_<level>/*.jsonl
```

The CSV contains:

| Metric | Meaning |
| --- | --- |
| `fbd` | Fréchet distance between real and generated embeddings for the same class |
| `g4_real_mean` | Mean G4Hunter score for real sequences |
| `g4_gen_mean` | Mean G4Hunter score for generated sequences |
| `g4_mean_gap` | Absolute gap between real and generated mean G4Hunter score |
| `g4_paired_gap` | Mean absolute pairwise G4Hunter gap |
| `g4_real_frac` | Fraction of real sequences above G4 threshold |
| `g4_gen_frac` | Fraction of generated sequences above G4 threshold |
| `g4_frac_gap` | Absolute gap between real and generated G4-positive fractions |
| `novelty_all_train` | Fraction of generated sequences absent from the full train set |
| `novelty_class_train` | Fraction of generated sequences absent from the full train set of the same class |

Existing sample files are reused by default. Add `--overwrite` only when samples should be regenerated.

### Dataset Motif Summary

The raw EndoQuad intervals contain many G-rich motifs, but not every peak matches a simple canonical motif. This supports using learned generative models and external biological metrics instead of only rule-based pattern matching.

| Subset | Valid peaks | Any G-rich motif | Percent |
| --- | ---: | ---: | ---: |
| All EndoQuad levels `1-6` | `391,355` | `193,819` | `49.53%` |
| High-confidence levels `4-6` | `140,262` | `70,532` | `50.29%` |

G-rich motif prevalence by pattern:

| Subset | Canonical `(G3+N1-7)x4` | Long-loop `(G3+N1-12)x4` | 3-tetrad `(G3+N1-7)x3` | G2-motif `(G2+N1-12)x4` |
| --- | ---: | ---: | ---: | ---: |
| All levels `1-6` | `5.63%` | `11.22%` | `20.92%` | `48.60%` |
| Levels `4-6` | `6.62%` | `12.82%` | `22.44%` | `49.40%` |

### Real Reference Metrics

These values are computed on real test sequences and provide the target biological distribution for generated samples.

| Class | Mean G4Hunter | G4Hunter-positive fraction | Mean strongest PQS score | Mean total PQS score |
| ---: | ---: | ---: | ---: | ---: |
| `4` | `2.0313` | `0.9350` | `86.5130` | `307.7075` |
| `5` | `2.0705` | `0.9505` | `89.4345` | `337.4195` |
| `6` | `2.1316` | `0.9550` | `92.5465` | `351.3410` |

### Real Class Separability

Computed with `2000` real sequences per class. Within-class rows are split-half baselines and estimate metric noise.

| Embedder | Pair | FBD |
| --- | --- | ---: |
| Melanoma CNN | `4 vs 4` | `0.6512` |
| Melanoma CNN | `5 vs 5` | `0.5532` |
| Melanoma CNN | `6 vs 6` | `0.4976` |
| Melanoma CNN | `4 vs 5` | `18.8178` |
| Melanoma CNN | `4 vs 6` | `90.6959` |
| Melanoma CNN | `5 vs 6` | `32.6544` |
| HyenaDNA | `4 vs 4` | `0.0332` |
| HyenaDNA | `5 vs 5` | `0.0282` |
| HyenaDNA | `6 vs 6` | `0.0435` |
| HyenaDNA | `4 vs 5` | `0.3347` |
| HyenaDNA | `4 vs 6` | `0.5686` |
| HyenaDNA | `5 vs 6` | `0.1018` |

### Final Model Ranking, Averaged Over Classes

Lower FBD and lower G4Hunter difference are better. For PQS metrics, generated values should be close to the real reference values above.

| Model | Params | HyenaDNA FBD | Melanoma FBD | Mean abs G4Hunter diff | Generated G4Hunter-positive fraction | Generated strongest PQS | Generated total PQS | Generated PQS-hit fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DFM large | `14.60M` | `0.1354` | `4.4116` | `0.4383` | `0.9498` | `91.1372` | `332.4100` | `0.9940` |
| DFM | `3.66M` | `0.1443` | `5.0500` | `0.4426` | `0.9445` | `90.7563` | `325.0592` | `0.9940` |
| LSTM | `6.91M` | `0.1531` | `6.3351` | `0.4597` | `0.9345` | `90.1555` | `351.3303` | `0.9918` |
| VAE | `17.66M` | `1.0950` | `75.5344` | `0.4533` | `0.8558` | `84.1975` | `279.0998` | `0.9938` |

### Final Model Metrics By Class

| Class | Model | HyenaDNA FBD | Melanoma FBD | Generated mean G4Hunter | Generated G4Hunter-positive fraction | Generated strongest PQS | Generated total PQS | Novelty vs full train |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | LSTM | `0.1366` | `1.5226` | `2.0128` | `0.9245` | `85.3315` | `302.3830` | `1.0000` |
| `4` | VAE | `1.3459` | `140.9224` | `1.7913` | `0.7755` | `76.2700` | `215.7030` | `1.0000` |
| `4` | DFM | `0.2910` | `2.0067` | `2.0408` | `0.9395` | `87.4260` | `298.1825` | `1.0000` |
| `4` | DFM large | `0.2655` | `1.2527` | `2.0536` | `0.9370` | `88.0250` | `306.4280` | `1.0000` |
| `5` | LSTM | `0.1391` | `4.0459` | `2.0940` | `0.9380` | `91.1980` | `359.6030` | `1.0000` |
| `5` | VAE | `1.0385` | `61.3385` | `1.9246` | `0.8855` | `86.3080` | `298.8590` | `1.0000` |
| `5` | DFM | `0.0617` | `1.9495` | `2.0987` | `0.9460` | `91.9370` | `338.8205` | `1.0000` |
| `5` | DFM large | `0.0632` | `1.6290` | `2.0984` | `0.9555` | `92.2470` | `344.1325` | `1.0000` |
| `6` | LSTM | `0.1835` | `13.4367` | `2.1612` | `0.9410` | `93.9370` | `392.0050` | `1.0000` |
| `6` | VAE | `0.9006` | `24.3421` | `1.9679` | `0.9065` | `90.0145` | `322.7375` | `1.0000` |
| `6` | DFM | `0.0803` | `11.1936` | `2.1231` | `0.9480` | `92.9060` | `338.1745` | `1.0000` |
| `6` | DFM large | `0.0774` | `10.3532` | `2.1436` | `0.9570` | `93.1395` | `346.6695` | `1.0000` |

### Best CFG Settings

The best DFM settings were selected by averaging over classes and considering HyenaDNA FBD together with PQS score preservation.

| Model | Guidance mode | Guidance scale | Mean HyenaDNA FBD | Mean strongest PQS score | Mean PQS hit count |
| --- | --- | ---: | ---: | ---: | ---: |
| DFM | `vectorfield_addition` | `1.0` | `0.1443` | `90.9160` | `4.9113` |
| DFM large | `score_free` | `1.0` | `0.1354` | `90.9700` | `4.9453` |

Main result: DFM large gives the strongest overall distributional match, DFM is close with fewer parameters, LSTM is a strong autoregressive baseline, and VAE is noticeably weaker for this discrete G4 generation task.

## Code Style

Style configuration lives in `pyproject.toml`.

Install dev tools:

```bash
pip install -r requirements-dev.txt
```

Run checks:

```bash
python -m ruff check .
python -m black --check .
```

Auto-fix and format:

```bash
python -m ruff check . --fix
python -m black .
```

Current rule of thumb:

- use `ruff` for import sorting and linting;
- use `black` for formatting;
- keep training defaults in `CFG` inside `utils/config.py`;
- keep metrics logic in `utils/gen_metrics_callback.py`;
- use `python -m metrics.eval` for final class-wise metrics instead of combined-set FBD;
- use `python -m metrics.pqsfinder` for PQS metrics on already generated JSONL files.
