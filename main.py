import os
import argparse
import torch
import pytorch_lightning as pl
import logging
from datetime import datetime

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from utils.data_utils import QuadDataset, load_data, save_examples
from models.dfm_module import QuadDFMModule
from utils.gen_metrics_callback import GenerativeMetricsCallback
from models.lstm import QuadLSTM
from models.vae import DNAConvVAE
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_logging(args):
    if args.log_file:
        log_path = args.log_file
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.log_dir, args.model_type, f"{args.experiment_name}_{stamp}.log")
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handlers = [logging.FileHandler(log_path, mode="a", encoding="utf-8")]
    if args.log_to_console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.info(f"Writing logs to {log_path}")
    return log_path

def _val_check_interval(value):
    if value is None:
        return None
    if value > 1 and float(value).is_integer():
        return int(value)
    return value

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate DNA sequence model")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--file_path_quadruplex", type=str, required=True)
    parser.add_argument("--file_path_seq", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=0.8, help="Ratio of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation data")
    parser.add_argument("--epochs", "--max_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=None)
    parser.add_argument("--val_check_interval", type=float, default=None)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--strategy", type=str, default=None, help="Training strategy")
    parser.add_argument("--model_type", type=str, default='lstm', choices=['lstm', 'vae', 'dfm', 'dfm_transformer'])

    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_cls", type=int, default=3)
    parser.add_argument("--level_offset", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_cnn_stacks", type=int, default=2)
    parser.add_argument("--num_transformer_layers", type=int, default=6)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--transformer_ff_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha_max", type=float, default=8.0)
    parser.add_argument("--alpha_scale", type=float, default=2.0)
    parser.add_argument("--fix_alpha", type=float, default=None)
    parser.add_argument("--prior_pseudocount", type=float, default=2.0)
    parser.add_argument("--num_integration_steps", type=int, default=64)
    parser.add_argument("--flow_temp", type=float, default=1.0)
    parser.add_argument("--classifier_free_guidance", "--cls_free_guidance", action="store_true")
    parser.add_argument("--cond_drop_prob", "--cls_free_noclass_ratio", type=float, default=0.3)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument(
        "--guidance_mode",
        type=str,
        default="score",
        choices=["score", "score_free", "probability_addition", "probability_tilt", "vectorfield_addition", "logit"],
    )
    parser.add_argument("--score_free_guidance", action="store_true")
    parser.add_argument("--probability_addition", action="store_true")
    parser.add_argument("--adaptive_prob_add", action="store_true")
    parser.add_argument("--probability_tilt", action="store_true")
    parser.add_argument("--vectorfield_addition", action="store_true")
    parser.add_argument("--allow_nan_cfactor", action="store_true")

    parser.add_argument("--val_metrics_sample_size", type=int, default=256)
    parser.add_argument("--g4hunter_window", type=int, default=25)
    parser.add_argument("--lstm_emb_dim", type=int, default=128)
    parser.add_argument("--lstm_level_dim", type=int, default=16)
    parser.add_argument("--lstm_hidden_dim", type=int, default=512)
    parser.add_argument("--lstm_num_layers", type=int, default=2)
    parser.add_argument("--lstm_mlp_layers", type=int, default=1)
    parser.add_argument("--lstm_dropout", type=float, default=0.3)
    parser.add_argument("--lstm_sample_temperature", type=float, default=1.0)
    parser.add_argument("--lstm_top_k", type=int, default=0)
    parser.add_argument("--vae_hidden_dim", type=int, default=320)
    parser.add_argument("--vae_latent_dim", type=int, default=128)
    parser.add_argument("--vae_num_res_blocks", type=int, default=2)
    parser.add_argument("--vae_dropout", type=float, default=0.1)
    parser.add_argument("--vae_sample_temperature", type=float, default=1.0)
    parser.add_argument("--vae_beta", type=float, default=0.05)
    parser.add_argument("--vae_beta_warmup_steps", type=int, default=5000)
    parser.add_argument("--log_dir", type=str, default="logs/run_logs")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log_to_console", action="store_true")
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--checkpoint_monitor", type=str, default=None)
    parser.add_argument("--checkpoint_save_top_k", type=int, default=5)

    args = parser.parse_args()
    setup_logging(args)
    if args.score_free_guidance and args.cond_drop_prob != 0:
        raise ValueError("--score_free_guidance should be used with --cls_free_noclass_ratio 0")
    if args.probability_tilt and args.score_free_guidance:
        raise ValueError("--probability_tilt and --score_free_guidance are mutually exclusive")
    if (
        args.score_free_guidance
        or args.probability_addition
        or args.adaptive_prob_add
        or args.probability_tilt
        or args.vectorfield_addition
    ) and not args.classifier_free_guidance:
        raise ValueError("Guidance modes require --cls_free_guidance")

    logging.info("Loading data...")
    df = load_data(args.file_path_quadruplex)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logging.info(f"Total data size: {len(df)}")

    logging.info("Splitting data...")
    train_size = int(len(df) * args.ratio)
    val_size = int(len(df) * args.val_ratio)
    test_size = len(df) - train_size - val_size
    train_df, remaining_df = train_test_split(df, test_size=1 - args.ratio, stratify=df['level'], random_state=42)
    test_df, val_df = train_test_split(
        remaining_df,
        test_size=args.val_ratio / (1 - args.ratio),
        stratify=remaining_df['level'],
        random_state=42,
    )
    logging.info(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    logging.info("Creating datasets and dataloaders...")
    if args.model_type == 'lstm':
        train_dataset = QuadDataset(
            train_df,
            file_path_seq=args.file_path_seq,
            typer="gen",
            seq_len=args.seq_len,
            level_offset=args.level_offset,
        )
        val_dataset = QuadDataset(
            val_df,
            file_path_seq=args.file_path_seq,
            typer="gen",
            seq_len=args.seq_len,
            level_offset=args.level_offset,
        )
        test_dataset = QuadDataset(
            test_df,
            file_path_seq=args.file_path_seq,
            typer="gen",
            seq_len=args.seq_len,
            level_offset=args.level_offset,
        )
    elif args.model_type in {'dfm', 'dfm_transformer'}:
        train_dataset = QuadDataset(
            train_df,
            file_path_seq=args.file_path_seq,
            typer="rec",
            seq_len=args.seq_len,
            level_offset=args.level_offset,
        )
        val_dataset = QuadDataset(
            val_df,
            file_path_seq=args.file_path_seq,
            typer="rec",
            seq_len=args.seq_len,
            level_offset=args.level_offset,
        )
        test_dataset = QuadDataset(
            test_df,
            file_path_seq=args.file_path_seq,
            typer="rec",
            seq_len=args.seq_len,
            level_offset=args.level_offset,
        )
    else:
        train_dataset = QuadDataset(
            train_df,
            file_path_seq=args.file_path_seq,
            typer="rec",
            seq_len=args.seq_len,
            level_offset=args.level_offset,
        )
        val_dataset = QuadDataset(
            val_df,
            file_path_seq=args.file_path_seq,
            typer="rec",
            seq_len=args.seq_len,
            level_offset=args.level_offset,
        )
        test_dataset = QuadDataset(
            test_df,
            file_path_seq=args.file_path_seq,
            typer="rec",
            seq_len=args.seq_len,
            level_offset=args.level_offset,
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logging.info("Finish creating datasets and dataloaders")

    if args.model_type == 'lstm':
        model = QuadLSTM(
            vocab_size=5,
            emb_dim=args.lstm_emb_dim,
            level_dim=args.lstm_level_dim,
            num_cls=args.num_cls,
            hidden_dim=args.lstm_hidden_dim,
            num_layers=args.lstm_num_layers,
            mlp_layers=args.lstm_mlp_layers,
            dropout=args.lstm_dropout,
            sample_temperature=args.lstm_sample_temperature,
            top_k=args.lstm_top_k,
            lr=args.lr,
        )
    elif args.model_type == 'vae':
        model = DNAConvVAE(
            seq_len=args.seq_len,
            hidden_dim=args.vae_hidden_dim,
            latent_dim=args.vae_latent_dim,
            num_cls=args.num_cls,
            num_res_blocks=args.vae_num_res_blocks,
            dropout=args.vae_dropout,
            sample_temperature=args.vae_sample_temperature,
            lr=args.lr,
            beta=args.vae_beta,
            beta_warmup_steps=args.vae_beta_warmup_steps,
        )
    elif args.model_type in {'dfm', 'dfm_transformer'}:
        model = QuadDFMModule(
            backbone="cnn" if args.model_type == "dfm" else "transformer",
            seq_len=args.seq_len,
            vocab_size=4,
            num_cls=args.num_cls,
            hidden_dim=args.hidden_dim,
            num_cnn_stacks=args.num_cnn_stacks,
            num_transformer_layers=args.num_transformer_layers,
            num_attention_heads=args.num_attention_heads,
            transformer_ff_mult=args.transformer_ff_mult,
            dropout=args.dropout,
            lr=args.lr,
            alpha_max=args.alpha_max,
            alpha_scale=args.alpha_scale,
            fix_alpha=args.fix_alpha,
            prior_pseudocount=args.prior_pseudocount,
            num_integration_steps=args.num_integration_steps,
            flow_temp=args.flow_temp,
            classifier_free_guidance=args.classifier_free_guidance,
            cond_drop_prob=args.cond_drop_prob,
            guidance_scale=args.guidance_scale,
            guidance_mode=args.guidance_mode,
            score_free_guidance=args.score_free_guidance,
            probability_addition=args.probability_addition,
            adaptive_prob_add=args.adaptive_prob_add,
            probability_tilt=args.probability_tilt,
            vectorfield_addition=args.vectorfield_addition,
            allow_nan_cfactor=args.allow_nan_cfactor,
        )
    logging.info(f"Model trainable parameters: {count_trainable_params(model):,}")

    logging.info("Init checkpoint callback...")
    checkpoint_monitor = args.checkpoint_monitor
    if checkpoint_monitor is None:
        checkpoint_monitor = "val_perplexity"
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.environ.get("MODEL_DIR", f"checkpoints/{args.model_type}/{args.experiment_name}"),
        save_top_k=args.checkpoint_save_top_k,
        save_last=True,
        monitor=checkpoint_monitor,
        mode='min'
    )

    metrics_cb = GenerativeMetricsCallback(
        train_sequences=train_dataset.encoded_seqs,
        seq_len=args.seq_len,
        sample_size=args.val_metrics_sample_size,
        g4hunter_window=args.g4hunter_window,
    )

    devices = 1
    strategy = "auto"
    if torch.cuda.is_available():
        accelerator = "gpu"
        available = torch.cuda.device_count()
        devices = available
        if devices and devices > 1:
            strategy = args.strategy or DDPStrategy(find_unused_parameters=False)
        else:
            strategy = "auto"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"

    logging.info(f"Init trainer on accelerator={accelerator} devices={devices} strategy={strategy}...")
    trainer = pl.Trainer(
        default_root_dir=os.environ.get("MODEL_DIR", f"logs/{args.model_type}/{args.experiment_name}"),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=args.progress_bar,
        callbacks=[checkpoint_callback, metrics_cb],
        logger=TensorBoardLogger(f"logs/{args.model_type}/", name=f"{args.experiment_name}"),
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=_val_check_interval(args.val_check_interval),
    )

    logging.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    logging.info("Finish training...")

    logging.info("Starting evaluation with Trainer...")
    results = trainer.test(model, dataloaders=test_loader)
    logging.info(f"Test results: {results}")
    
    logging.info("Save examples of eval...")
    predictions = trainer.predict(model, dataloaders=test_loader)
    save_examples(predictions, output_path=f"examples/{args.model_type}/{args.experiment_name}.jsonl", max_examples=30, compact=True)

    logging.info("Finish!")

if __name__ == "__main__":
    main()
