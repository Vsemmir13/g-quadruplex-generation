import argparse
import logging

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from data_utils import QuadDataset, load_data
from dfm_module import QuadDFMModule
from lstm import QuadLSTM
from vae import DNAConvVAE


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Fast smoke test for VAE/LSTM/DFM")
    parser.add_argument("--file_path_quadruplex", type=str, required=True)
    parser.add_argument("--file_path_seq", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=128)
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)

    df = load_data(args.file_path_quadruplex).sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["level"], random_state=42)

    def small(ds):
        n = min(len(ds), args.max_items)
        return Subset(ds, list(range(n)))

    # --- LSTM (autoregressive) ---
    lstm_train = small(QuadDataset(train_df, file_path_seq=args.file_path_seq, typer="gen", seq_len=args.seq_len))
    lstm_val = small(QuadDataset(val_df, file_path_seq=args.file_path_seq, typer="gen", seq_len=args.seq_len))
    lstm_train_loader = DataLoader(lstm_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    lstm_val_loader = DataLoader(lstm_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    lstm_model = QuadLSTM(vocab_size=5)

    # --- VAE (reconstruction) ---
    vae_train = small(QuadDataset(train_df, file_path_seq=args.file_path_seq, typer="rec", seq_len=args.seq_len))
    vae_val = small(QuadDataset(val_df, file_path_seq=args.file_path_seq, typer="rec", seq_len=args.seq_len))
    vae_train_loader = DataLoader(vae_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    vae_val_loader = DataLoader(vae_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    vae_model = DNAConvVAE(seq_len=args.seq_len)

    # --- DFM (Dirichlet flow matching) ---
    dfm_train = small(QuadDataset(train_df, file_path_seq=args.file_path_seq, typer="rec", seq_len=args.seq_len))
    dfm_val = small(QuadDataset(val_df, file_path_seq=args.file_path_seq, typer="rec", seq_len=args.seq_len))
    dfm_train_loader = DataLoader(dfm_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dfm_val_loader = DataLoader(dfm_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dfm_model = QuadDFMModule(seq_len=args.seq_len)

    accelerator = "gpu" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        fast_dev_run=True,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    logging.info("Smoke test LSTM...")
    trainer.fit(lstm_model, lstm_train_loader, lstm_val_loader)

    logging.info("Smoke test VAE...")
    trainer.fit(vae_model, vae_train_loader, vae_val_loader)

    logging.info("Smoke test DFM...")
    trainer.fit(dfm_model, dfm_train_loader, dfm_val_loader)

    logging.info("All smoke tests passed (fast_dev_run).")


if __name__ == "__main__":
    main()

