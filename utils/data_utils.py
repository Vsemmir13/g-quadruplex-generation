import pandas as pd
import torch
import random
import logging
import json
from torch.utils.data import Dataset
from pyfaidx import Fasta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
VOCAB_SIZE = len(VOCAB)
ID2BASE = {v: k for k, v in VOCAB.items()}
BOS_TOKEN_ID = VOCAB_SIZE

def load_data(file_path_quadruplex):
    cols = ["chrom", "start", "end", "level_raw", "score", "strand"]
    df = pd.read_csv(file_path_quadruplex, sep="\t", names=cols)
    df["level"] = df["level_raw"].str.extract(r"(\d+)").astype(int)
    df["length"] = df["end"] - df["start"]
    max_quadruplex_length = df["length"].quantile(0.99)
    df = df[df["length"] <= max_quadruplex_length]
    df = df[df["level"] > 3]
    return df

class QuadDataset(Dataset):
    
    def __init__(self, df, file_path_seq, typer="rec", seq_len=512):
        self.file_path_seq = file_path_seq
        self.seq_len = seq_len
        self.genome = Fasta(file_path_seq)
        self.typer = typer 
        assert self.typer in ["rec", "gen"]
        self.encoded_seqs = []
        self.levels = []
        for _, row in df.iterrows():
            seq = self.generate_full_sequence(row['start'], row['end'], row["chrom"])
            if seq is not None:
                encoded_seq = self.encode_seq(seq)
                self.encoded_seqs.append(encoded_seq)
                self.levels.append(float(row["level"]))
                
    def __len__(self):
        return len(self.encoded_seqs)

    def encode_seq(self, s):
        ids = []
        for ch in s.upper():
            ids.append(VOCAB[ch])
        return torch.tensor(ids, dtype=torch.long)

    def generate_full_sequence(self, start, end, chrom):    
        chrom_sequence = self.genome[chrom]
        min_start_pos = max(0, end - self.seq_len)
        max_start_pos = min(start, len(chrom_sequence) - self.seq_len)
        if max_start_pos < min_start_pos:
            return None
        start_pos = random.randint(min_start_pos, max_start_pos)
        full_seq = chrom_sequence[start_pos:start_pos + self.seq_len].seq
        if 'N' in full_seq:
            return None
        return full_seq
    
    def __getitem__(self, idx):
        encoded_seq = self.encoded_seqs[idx]
        if self.typer == "rec":
            x = encoded_seq
            y = encoded_seq
        else:
            # Autoregressive next-token prediction with explicit BOS token.
            # Keeps sequence length constant (seq_len) for easier batching/logging.
            bos = torch.tensor([BOS_TOKEN_ID], dtype=torch.long)
            x = torch.cat([bos, encoded_seq[:-1]], dim=0)
            y = encoded_seq
        level = self.levels[idx]
        level_norm = (level - 4.0) / 2.0
        return x, y, torch.tensor([level_norm], dtype=torch.float32)

def decode_seq(ids):
    # For non-ACGT tokens (e.g. BOS), fall back to 'N'.
    return "".join(ID2BASE.get(int(i), "N") for i in ids)

def save_examples(predictions, output_path="vae_examples.jsonl", max_examples=20, *, compact: bool = False):
    saved = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for batch_out in predictions:
            x = batch_out["x"]
            cond = batch_out["levels"]
            recon = batch_out["recon"]
            gen = batch_out["gen"]
            batch_size = x.size(0)
            for i in range(batch_size):
                if compact:
                    row = {
                        "id": saved,
                        "cond": float(cond[i].view(-1)[0].item()),
                        "generation_seq": decode_seq(gen[i].tolist()),
                    }
                else:
                    row = {
                        "id": saved,
                        "cond": cond[i].tolist(),
                        "test_x": x[i].tolist(),
                        "reconstruction": recon[i].tolist(),
                        "generation": gen[i].tolist(),
                        "test_x_seq": decode_seq(x[i].tolist()),
                        "reconstruction_seq": decode_seq(recon[i].tolist()),
                        "generation_seq": decode_seq(gen[i].tolist()),
                    }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                saved += 1
                if saved >= max_examples:
                    return

