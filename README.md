# Discrete Flow Matching for G-Quadruplex Generation

**Leontenkov Egor**  
Faculty of Computer Science, National Research University Higher School of Economics (HSE), Moscow, Russia  
`eeleontenkov@edu.hse.ru`

This repository contains code for conditional generation of short DNA sequences containing G-quadruplexes (G4), conditioned on categorical G4 level. The main model is a Discrete Dirichlet Flow Matching generator. LSTM and VAE are included as baselines.

## Models

| Model | Files | What it does |
| --- | --- | --- |
| DFM CNN | `models/dfm_module.py`, `models/dfm_model.py`, `models/dfm_flow_utils.py` | Dirichlet flow matching on the simplex with a residual CNN backbone. This is the paper-like DFM setup. |
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

### Raw Dataset Analysis

Raw BED intervals can be analyzed before `QuadDataset` creates 512bp training windows:

```bash
python analysis/dataset_analysis.py \
  --bed data/EQ_hg38_lifted.bed \
  --fasta data/hg38.fa \
  --out_dir analysis/dataset
```

The script saves tables, a Markdown summary, example raw G4 sequences, and publication-style matplotlib figures:

```text
analysis/dataset/dataset_summary.md
analysis/dataset/level_counts_comparison.png
analysis/dataset/length_distribution_comparison.png
analysis/dataset/length_hist_training_like_by_level.png
analysis/dataset/length_by_level_boxplot.png
analysis/dataset/sequence_features_by_level.png
analysis/dataset/chrom_counts_top25_raw.png
```

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

### pqsfinder Metrics

`pqsfinder` is computed after generation from existing JSONL files. It does not regenerate sequences.

This requires R and the Bioconductor packages `pqsfinder` and `Biostrings`.

Install R packages:

```r
install.packages("BiocManager")
BiocManager::install(c("pqsfinder", "Biostrings"))
```

Run on existing generated samples:

```bash
python -m metrics.pqsfinder \
  --samples_root generated/classwise_metrics/val \
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --output_csv generated/classwise_metrics/val/pqsfinder_metrics.csv \
  --split val \
  --classes 4 5 6 \
  --num_real 2000 \
  --min_score 42 \
  --strand "*"
```

To compute one specific CFG strategy only, use `--sample_glob`:

```bash
python -m metrics.pqsfinder \
  --samples_root generated/metrics_no_large/test/dfm_small \
  --sample_glob cfg_vectorfield_addition_scale_1p0.jsonl \
  --file_path_quadruplex data/EQ_hg38_lifted.bed \
  --file_path_seq data/hg38.fa \
  --output_csv generated/metrics_no_large/test/dfm_small/pqsfinder_vectorfield_addition_scale_1p0.csv \
  --split test \
  --classes 4 5 6 \
  --num_real 2000 \
  --min_score 42 \
  --strand "*"
```

Shortcut:

```bash
./metrics/run_pqsfinder_metrics.sh
```

Outputs:

```text
pqsfinder_metrics.csv              # generated-vs-real class-wise PQS gaps
pqsfinder_metrics_summary_all.csv  # real and generated PQS summaries
pqsfinder_metrics_per_sequence.csv # optional, with --keep_per_sequence
```

Main `pqsfinder` columns:

| Metric | Meaning |
| --- | --- |
| `pqs_frac` | Fraction of sequences with at least one predicted PQS hit |
| `pqs_count_mean` | Mean number of PQS hits per sequence |
| `pqs_max_score_mean` | Mean of the maximum pqsfinder score per sequence |
| `pqs_total_score_mean` | Mean sum of pqsfinder scores per sequence |
| `*_gap` | Absolute difference between generated and real sequences of the same class |

## Results

### Real-vs-Real FBD Baselines

Real-vs-real baselines estimate metric noise by comparing two real subsets.

| Embedder | Samples | Real-vs-real FBD |
| --- | ---: | ---: |
| melanoma CNN | `1024` | `0.8871` |
| HyenaDNA | `1024` | `0.0351` |

### Class Separability, Real Class-vs-Class FBD

Computed with `2000` real sequences per class.

| Embedder | Pair | FBD |
| --- | --- | ---: |
| melanoma CNN | `4 vs 4` | `0.6512` |
| melanoma CNN | `5 vs 5` | `0.5532` |
| melanoma CNN | `6 vs 6` | `0.4976` |
| melanoma CNN | `4 vs 5` | `18.8178` |
| melanoma CNN | `4 vs 6` | `90.6959` |
| melanoma CNN | `5 vs 6` | `32.6544` |
| HyenaDNA | `4 vs 4` | `0.0332` |
| HyenaDNA | `5 vs 5` | `0.0282` |
| HyenaDNA | `6 vs 6` | `0.0435` |
| HyenaDNA | `4 vs 5` | `0.3347` |
| HyenaDNA | `4 vs 6` | `0.5686` |
| HyenaDNA | `5 vs 6` | `0.1018` |

This suggests that levels `5` and `6` are close in HyenaDNA space, while level `4` is more separated.

### Combined-Set Model Metrics

Older combined-set metrics are useful for a quick overview, but class-wise metrics should be preferred for final comparison.

| Model | Params | Perplexity | Novelty | Melanoma FBD | HyenaDNA FBD |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | `6.91M` | `3.1824` | `1.0000` | `4.2649` | `0.1111` |
| DFM small | `3.66M` | `2.1920` | `1.0000` | `10.1020` | `0.3505` |
| VAE | `17.66M` | `3.5323` | `1.0000` | `81.3965` | `1.0761` |

### DFM Large Guidance Sweep

Combined-set sweep for `DFM large`.

| Guidance scale | Melanoma FBD | HyenaDNA FBD |
| ---: | ---: | ---: |
| `0` | `2.3109` | `0.1007` |
| `1` | `1.8686` | `0.0995` |
| `2` | `1.3702` | `0.0965` |
| `3` | `2.1867` | `0.1049` |

In this run, `guidance_scale=2` was best by both FBD metrics.

### G4 Metrics Snapshot

| Model | Generation | G4 real mean | G4 gen mean | G4 mean gap | G4 paired gap | G4 real frac | G4 gen frac | G4 frac gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | sample | `2.0579` | `2.0468` | `0.0111` | `0.4505` | `0.9492` | `0.9219` | `0.0273` |
| DFM small | scale 3 | `2.0666` | `2.0592` | `0.0074` | `0.4544` | `0.9551` | `0.9180` | `0.0371` |
| DFM large | scale 0 | `2.0666` | `2.0564` | `0.0102` | `0.4313` | `0.9551` | `0.9346` | `0.0205` |
| DFM large | scale 1 | `2.0666` | `2.0571` | `0.0095` | `0.4355` | `0.9551` | `0.9355` | `0.0195` |
| DFM large | scale 2 | `2.0666` | `2.0564` | `0.0102` | `0.4324` | `0.9551` | `0.9346` | `0.0205` |
| DFM large | scale 3 | `2.0666` | `2.0519` | `0.0148` | `0.4298` | `0.9551` | `0.9336` | `0.0215` |
| VAE | sample | `2.0666` | `1.8780` | `0.1886` | `0.4374` | `0.9551` | `0.8477` | `0.1074` |

### pqsfinder Snapshot

Computed on the test split with `2000` generated sequences per class and `min_score=42`.

| Model | Generation | Mean PQS frac gap | Mean PQS count gap | Mean max score gap | Mean total score gap |
| --- | --- | ---: | ---: | ---: | ---: |
| LSTM | sample | `0.0082` | `0.2668` | `1.4452` | `22.7240` |
| DFM small | vectorfield addition, scale 1 | `0.0060` | `0.1003` | `1.2583` | `8.0308` |

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
