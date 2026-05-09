import argparse
import inspect
import logging
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from models.dfm_module import QuadDFMModule
from models.lstm import QuadLSTM
from models.vae import DNAConvVAE
from utils.data_utils import QuadDataset, load_data, save_examples
from utils.gen_metrics_callback import GenerativeMetricsCallback

CFG = {
    "seq_len": 512,
    "num_cls": 3,
    "level_offset": 4,
    "split": 0.8,
    "val_split": 0.1,
    "lr": 5e-4,
    "check_val_every_n_epoch": 10,
    "metric_samples": 256,
    "g4hunter_window": 25,
    "checkpoint_save_top_k": 5,
    "lstm": dict(
        emb_dim=128,
        level_dim=16,
        hidden_dim=512,
        num_layers=2,
        mlp_layers=1,
        dropout=0.3,
        sample_temperature=1.0,
        top_k=0,
    ),
    "vae": dict(
        hidden_dim=320,
        latent_dim=128,
        num_res_blocks=2,
        dropout=0.1,
        sample_temperature=0.8,
        beta=0.1,
        beta_warmup_steps=20000,
    ),
    "dfm": dict(
        hidden_dim=128,
        num_cnn_stacks=4,
        num_transformer_layers=1,
        num_attention_heads=4,
        transformer_ff_mult=1,
        dropout=0.0,
        alpha_max=8.0,
        alpha_scale=2.0,
        prior_pseudocount=2.0,
        num_integration_steps=100,
        flow_temp=1.0,
        classifier_free_guidance=True,
        cond_drop_prob=0.3,
        allow_nan_cfactor=True,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Train/test G4 DNA generative models.")
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--file_path_quadruplex", required=True)
    p.add_argument("--file_path_seq", required=True)
    p.add_argument("--model_type", default="dfm", choices=["lstm", "vae", "dfm", "dfm_transformer"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_epochs", "--epochs", type=int, default=100000)
    p.add_argument("--max_steps", type=int, default=450000)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--devices", default="auto")
    p.add_argument("--run_mode", default="train", choices=["train", "test"])
    p.add_argument("--ckpt_path")
    p.add_argument(
        "--metric_samples", "--val_metrics_sample_size", type=int, default=CFG["metric_samples"]
    )
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument(
        "--guidance_mode",
        default="probability_addition",
        choices=[
            "score",
            "score_free",
            "probability_addition",
            "probability_tilt",
            "vectorfield_addition",
            "logit",
        ],
    )
    p.add_argument("--progress_bar", action="store_true")
    args = p.parse_args()
    if args.run_mode == "test" and not args.ckpt_path:
        raise ValueError("--ckpt_path is required for --run_mode test")
    return args


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def split_data(path):
    df = load_data(path).sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, rest_df = train_test_split(
        df,
        test_size=1.0 - CFG["split"],
        stratify=df["level"],
        random_state=42,
    )
    test_df, val_df = train_test_split(
        rest_df,
        test_size=CFG["val_split"] / (1.0 - CFG["split"]),
        stratify=rest_df["level"],
        random_state=42,
    )
    logging.info("Data size: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def make_loaders(args):
    typer = "gen" if args.model_type == "lstm" else "rec"
    datasets = [
        QuadDataset(
            df,
            file_path_seq=args.file_path_seq,
            typer=typer,
            seq_len=CFG["seq_len"],
            level_offset=CFG["level_offset"],
        )
        for df in split_data(args.file_path_quadruplex)
    ]
    train_ds, val_ds, test_ds = datasets
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    return train_ds, train_loader, val_loader, test_loader


def build_model(args):
    if args.model_type == "lstm":
        return QuadLSTM(vocab_size=5, num_cls=CFG["num_cls"], lr=CFG["lr"], **CFG["lstm"])

    if args.model_type == "vae":
        return DNAConvVAE(
            seq_len=CFG["seq_len"], num_cls=CFG["num_cls"], lr=CFG["lr"], **CFG["vae"]
        )

    return QuadDFMModule(
        backbone="transformer" if args.model_type == "dfm_transformer" else "cnn",
        seq_len=CFG["seq_len"],
        vocab_size=4,
        num_cls=CFG["num_cls"],
        lr=CFG["lr"],
        guidance_scale=args.guidance_scale,
        guidance_mode=args.guidance_mode,
        **CFG["dfm"],
    )


def load_weights(model, ckpt_path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    ckpt = torch.load(ckpt_path, **kwargs)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    logging.info("Loaded checkpoint weights from %s", ckpt_path)


def accelerator_and_strategy(devices_arg):
    if torch.cuda.is_available():
        devices = torch.cuda.device_count() if devices_arg == "auto" else int(devices_arg)
        if devices > torch.cuda.device_count():
            raise ValueError(
                f"--devices={devices}, but torch sees only {torch.cuda.device_count()} CUDA device(s)"
            )
        strategy = DDPStrategy(find_unused_parameters=False) if devices > 1 else "auto"
        return "gpu", devices, strategy
    if torch.backends.mps.is_available():
        return "mps", 1, "auto"
    return "cpu", 1, "auto"


def make_trainer(args, callbacks):
    accelerator, devices, strategy = accelerator_and_strategy(args.devices)
    logging.info(
        "Init trainer on accelerator=%s devices=%s strategy=%s", accelerator, devices, strategy
    )
    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        enable_progress_bar=args.progress_bar,
        callbacks=callbacks,
        logger=TensorBoardLogger(f"logs/{args.model_type}", name=args.experiment_name),
        default_root_dir=os.environ.get(
            "MODEL_DIR", f"logs/{args.model_type}/{args.experiment_name}"
        ),
        check_val_every_n_epoch=CFG["check_val_every_n_epoch"],
    )


def make_callbacks(args, train_ds):
    callbacks = [
        GenerativeMetricsCallback(
            train_sequences=train_ds.encoded_seqs,
            seq_len=CFG["seq_len"],
            sample_size=args.metric_samples,
            g4hunter_window=CFG["g4hunter_window"],
        )
    ]
    if args.run_mode == "train":
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=os.environ.get(
                    "MODEL_DIR", f"checkpoints/{args.model_type}/{args.experiment_name}"
                ),
                save_top_k=CFG["checkpoint_save_top_k"],
                save_last=True,
                monitor="val_perplexity",
                mode="min",
            ),
        )
    return callbacks


def train_or_test(args, model, trainer, train_loader, val_loader, test_loader):
    if args.run_mode == "train":
        fit_kwargs = {"ckpt_path": args.ckpt_path} if args.ckpt_path else {}
        if args.ckpt_path:
            logging.info("Resuming training from %s", args.ckpt_path)
        trainer.fit(model, train_loader, val_loader, **fit_kwargs)
    else:
        logging.info("Skipping training because run_mode=test")

    results = trainer.test(model, dataloaders=test_loader)
    logging.info("Test results: %s", results)
    predictions = trainer.predict(model, dataloaders=test_loader)
    save_examples(
        predictions,
        f"examples/{args.model_type}/{args.experiment_name}.jsonl",
        max_examples=30,
        compact=True,
    )


def main():
    args = parse_args()
    setup_logging()
    logging.info("Loading data and dataloaders")
    train_ds, train_loader, val_loader, test_loader = make_loaders(args)

    model = build_model(args)
    logging.info(
        "Model trainable parameters: %s",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )
    if args.run_mode == "test":
        load_weights(model, args.ckpt_path)

    trainer = make_trainer(args, make_callbacks(args, train_ds))
    train_or_test(args, model, trainer, train_loader, val_loader, test_loader)
    logging.info("Finish")


if __name__ == "__main__":
    main()
