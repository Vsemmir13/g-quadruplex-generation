import os
import argparse
import torch
import pytorch_lightning as pl
import logging

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from data_utils import QuadDataset, load_data, save_examples
from dfm_module import QuadDFMModule
from lstm import QuadLSTM
from vae import DNAConvVAE
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate DNA sequence model")
    parser.add_argument("--file_path_quadruplex", type=str, required=True)
    parser.add_argument("--file_path_seq", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=0.8, help="Ratio of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--devices", type=int, default=None, help="Number of devices to use (default: all available GPUs, else 1)")
    parser.add_argument("--strategy", type=str, default=None, help="Training strategy, e.g. ddp. Default: auto-enable ddp on multi-GPU CUDA.")
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--test", action="store_true", help="Run test after training")
    parser.add_argument("--model_type", type=str, default='lstm', choices=['lstm', 'vae', 'dfm'])

    # DFM-specific hyperparameters (used only when --model_type dfm)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_cnn_stacks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha_max", type=float, default=12.0)
    parser.add_argument("--alpha_scale", type=float, default=15.0)
    parser.add_argument("--fix_alpha", type=float, default=None)
    parser.add_argument("--prior_pseudocount", type=float, default=2.0)
    parser.add_argument("--num_integration_steps", type=int, default=64)
    parser.add_argument("--flow_temp", type=float, default=1.0)
    args = parser.parse_args()

    logging.info("Loading data...")
    df = load_data(args.file_path_quadruplex)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logging.info(f"Total data size: {len(df)}")

    logging.info("Splitting data...")
    train_size = int(len(df) * args.ratio)
    val_size = int(len(df) * args.val_ratio)
    test_size = len(df) - train_size - val_size
    train_df, remaining_df = train_test_split(df, test_size=1 - args.ratio, stratify=df['level'])
    val_df, test_df = train_test_split(remaining_df, test_size=args.val_ratio / (1 - args.ratio), stratify=remaining_df['level'])
    logging.info(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    logging.info("Creating datasets and dataloaders...")
    if args.model_type == 'lstm':
        train_dataset = QuadDataset(train_df, file_path_seq=args.file_path_seq, typer="gen", seq_len=args.seq_len)
        val_dataset = QuadDataset(val_df, file_path_seq=args.file_path_seq, typer="gen", seq_len=args.seq_len)
        test_dataset = QuadDataset(test_df, file_path_seq=args.file_path_seq, typer="gen", seq_len=args.seq_len)
    else:
        # VAE and DFM are trained as conditional reconstruction models (typer="rec")
        train_dataset = QuadDataset(train_df, file_path_seq=args.file_path_seq, typer="rec", seq_len=args.seq_len)
        val_dataset = QuadDataset(val_df, file_path_seq=args.file_path_seq, typer="rec", seq_len=args.seq_len)
        test_dataset = QuadDataset(test_df, file_path_seq=args.file_path_seq, typer="rec", seq_len=args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logging.info("Finish creating datasets and dataloaders")

    if args.model_type == 'lstm':
        # vocab_size=5 to include BOS token used by QuadDataset(typer="gen")
        model = QuadLSTM(vocab_size=5)
    elif args.model_type == 'vae':
        model = DNAConvVAE(seq_len=args.seq_len)
    elif args.model_type == 'dfm':
        model = QuadDFMModule(
            seq_len=args.seq_len,
            vocab_size=4,
            cond_dim=1,
            hidden_dim=args.hidden_dim,
            num_cnn_stacks=args.num_cnn_stacks,
            dropout=args.dropout,
            lr=args.lr,
            alpha_max=args.alpha_max,
            alpha_scale=args.alpha_scale,
            fix_alpha=args.fix_alpha,
            prior_pseudocount=args.prior_pseudocount,
            num_integration_steps=args.num_integration_steps,
            flow_temp=args.flow_temp,
        )

    logging.info("Init checkpoint callback...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.environ.get("MODEL_DIR", "checkpoints"),
        save_top_k=3,
        save_last=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode="min"
    )

    if torch.cuda.is_available():
        accelerator = "gpu"
        available = torch.cuda.device_count()
        devices = available if args.devices is None else args.devices
        # Auto-enable DDP when using multiple GPUs (faster than single-process DP)
        if devices and devices > 1:
            strategy = args.strategy or DDPStrategy(find_unused_parameters=False)
        else:
            strategy = args.strategy
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        strategy = None
    else:
        accelerator = "cpu"
        devices = 1
        strategy = None

    logging.info(f"Init trainer on accelerator={accelerator} devices={devices} strategy={strategy}...")
    trainer = pl.Trainer(
        default_root_dir=os.environ.get("MODEL_DIR", "logs"),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stopping],
        logger=TensorBoardLogger("logs/", name=f"quad_{args.model_type}_experiment")
    )

    logging.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    logging.info("Finish training...")

    logging.info("Starting evaluation with Trainer...")
    results = trainer.test(model, dataloaders=test_loader)
    logging.info(f"Test results: {results}")
    
    logging.info("Save examples of eval...")
    predictions = trainer.predict(model, dataloaders=test_loader)
    save_examples(predictions, output_path=f"quad_{args.model_type}_examples.jsonl", max_examples=30)

    logging.info("Finish!")

if __name__ == "__main__":
    main()
