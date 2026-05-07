import os
import argparse
import inspect
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

DEFAULTS = {
    "ratio": 0.8,
    "val_ratio": 0.1,
    "check_val_every_n_epoch": 10,
    "val_check_interval": None,
    "limit_train_batches": 1.0,
    "limit_val_batches": 1.0,
    "strategy": None,
    "seq_len": 512,
    "num_cls": 3,
    "level_offset": 4,
    "hidden_dim": 256,
    "num_cnn_stacks": 4,
    "num_transformer_layers": 6,
    "num_attention_heads": 4,
    "transformer_ff_mult": 4,
    "dropout": 0.0,
    "lr": 5e-4,
    "alpha_max": 8.0,
    "alpha_scale": 2.0,
    "fix_alpha": None,
    "prior_pseudocount": 2.0,
    "num_integration_steps": 100,
    "flow_temp": 1.0,
    "classifier_free_guidance": True,
    "cond_drop_prob": 0.3,
    "guidance_scale": 3.0,
    "guidance_mode": "probability_addition",
    "score_free_guidance": False,
    "probability_addition": False,
    "adaptive_prob_add": False,
    "probability_tilt": False,
    "vectorfield_addition": False,
    "allow_nan_cfactor": True,
    "val_metrics_sample_size": 256,
    "g4hunter_window": 25,
    "lstm_emb_dim": 128,
    "lstm_level_dim": 16,
    "lstm_hidden_dim": 512,
    "lstm_num_layers": 2,
    "lstm_mlp_layers": 1,
    "lstm_dropout": 0.3,
    "lstm_sample_temperature": 1.0,
    "lstm_top_k": 0,
    "vae_hidden_dim": 320,
    "vae_latent_dim": 128,
    "vae_num_res_blocks": 2,
    "vae_dropout": 0.1,
    "vae_sample_temperature": 0.8,
    "vae_beta": 0.1,
    "vae_beta_warmup_steps": 20000,
    "log_dir": "logs/run_logs",
    "log_level": "INFO",
    "checkpoint_monitor": "val_perplexity",
    "checkpoint_save_top_k": 5,
    "devices": "auto",
}

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_checkpoint_weights(model, ckpt_path):
    if not ckpt_path:
        raise ValueError("--ckpt_path is required for --run_mode test")
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(ckpt_path, **load_kwargs)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    logging.info(f"Loaded checkpoint weights from {ckpt_path}")
    return model

def trainer_fit_kwargs(args):
    if not args.ckpt_path:
        return {}
    logging.info(f"Resuming training from checkpoint {args.ckpt_path}")
    return {"ckpt_path": args.ckpt_path}

def apply_defaults(args):
    for key, value in DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args

def apply_guidance_mode(args):
    args.score_free_guidance = args.guidance_mode == "score_free"
    args.probability_addition = args.guidance_mode == "probability_addition"
    args.probability_tilt = args.guidance_mode == "probability_tilt"
    args.vectorfield_addition = args.guidance_mode == "vectorfield_addition"
    return args

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

def log_distributed_env():
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "SLURM_JOB_ID",
        "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_GPUS",
    ]
    env = {key: os.environ.get(key) for key in keys if os.environ.get(key) is not None}
    logging.info(f"Distributed environment: {env}")
    if torch.cuda.is_available():
        logging.info(f"torch.cuda.device_count()={torch.cuda.device_count()}")

def resolve_devices(args):
    if args.devices == "auto":
        return torch.cuda.device_count()
    devices = int(args.devices)
    if devices < 1:
        raise ValueError("--devices must be a positive integer or 'auto'")
    return devices

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate DNA sequence model")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--file_path_quadruplex", type=str, required=True)
    parser.add_argument("--file_path_seq", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="dfm", choices=["lstm", "vae", "dfm", "dfm_transformer"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", "--max_epochs", type=int, default=100000)
    parser.add_argument("--max_steps", type=int, default=450000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--devices", type=str, default=DEFAULTS["devices"])
    parser.add_argument("--run_mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=DEFAULTS["guidance_scale"])
    parser.add_argument(
        "--guidance_mode",
        type=str,
        default=DEFAULTS["guidance_mode"],
        choices=["score", "score_free", "probability_addition", "probability_tilt", "vectorfield_addition", "logit"],
    )
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_to_console", action="store_true")
    parser.add_argument("--progress_bar", action="store_true")

    args = apply_guidance_mode(apply_defaults(parser.parse_args()))
    setup_logging(args)
    log_distributed_env()
    if args.run_mode == "test" and not args.ckpt_path:
        raise ValueError("--ckpt_path is required for --run_mode test")
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
    if args.run_mode == "test":
        model = load_checkpoint_weights(model, args.ckpt_path)

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
        devices = resolve_devices(args)
        if devices > torch.cuda.device_count():
            raise ValueError(f"--devices={devices} but torch sees only {torch.cuda.device_count()} CUDA device(s)")
        if devices > 1:
            strategy = args.strategy or DDPStrategy(find_unused_parameters=False)
        else:
            strategy = "auto"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"

    callbacks = [metrics_cb]
    if args.run_mode == "train":
        logging.info("Init checkpoint callback...")
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.environ.get("MODEL_DIR", f"checkpoints/{args.model_type}/{args.experiment_name}"),
            save_top_k=args.checkpoint_save_top_k,
            save_last=True,
            monitor=args.checkpoint_monitor,
            mode='min'
        )
        callbacks.insert(0, checkpoint_callback)

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
        callbacks=callbacks,
        logger=TensorBoardLogger(f"logs/{args.model_type}/", name=f"{args.experiment_name}"),
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=_val_check_interval(args.val_check_interval),
    )

    if args.run_mode == "train":
        logging.info("Starting training...")
        trainer.fit(model, train_loader, val_loader, **trainer_fit_kwargs(args))
        logging.info("Finish training...")
    else:
        logging.info("Skipping training because run_mode=test")

    logging.info("Starting evaluation with Trainer...")
    results = trainer.test(model, dataloaders=test_loader)
    logging.info(f"Test results: {results}")
    
    logging.info("Save examples of eval...")
    predictions = trainer.predict(model, dataloaders=test_loader)
    save_examples(predictions, output_path=f"examples/{args.model_type}/{args.experiment_name}.jsonl", max_examples=30, compact=True)

    logging.info("Finish!")

if __name__ == "__main__":
    main()
