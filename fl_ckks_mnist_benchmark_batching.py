"""
FL + CKKS (TenSEAL) MNIST Benchmark — Client-side encryption batching
--------------------------------------------------------------------

What this adds (vs. your baseline script)
- Adds an **encryption-batching** mode to reduce peak memory and peak server deserialization
  costs when many clients produce large serialized ciphertexts.

Design
- Clients still compute weighted deltas `w * delta` and encrypt them locally.
- Instead of the server deserializing all client ciphertexts at once and summing them,
  the server processes clients in batches of `--enc-batch-size` clients.
  For each batch the server:
    1. Receives serialized ciphertext blobs for each client in the batch.
    2. Deserializes and sums them per-chunk, producing a *partial aggregated ciphertext* per chunk.
    3. Serializes those partial aggregated ciphertexts and stores *one* blob per chunk for the batch.
- After all batches are processed, the server deserializes the batch-level partial aggregates
  (there are far fewer of these) and sums them to obtain the final encrypted aggregate.
- This reduces peak memory and the number of live ckks_vector objects during aggregation.

Notes
- Communication bytes are still counted as the sum of client serialized ciphertexts (the network cost doesn't change),
  but the server memory and number of simultaneous deserializations is reduced.
- The protocol semantics don't change: server still only sees public-context ciphertexts and the key-owner performs decryption.

Flags added
- `--enc-batch-size` (int, default 0): if 0, original behaviour (no batching). If >0, process clients in batches of this size.

Usage examples
- No batching (original behavior):
    python fl_ckks_mnist_benchmark.py --clients 20 --rounds 3 --local-epochs 1 --iid

- Batching with groups of 10 clients:
    python fl_ckks_mnist_benchmark.py --clients 100 --rounds 3 --local-epochs 1 --iid --enc-batch-size 10

"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import psutil
import tenseal as ts


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def now_ms():
    return int(time.time() * 1000)


def sizeof_bytes(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} TB"


# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Data partitioning
# ----------------------------

def build_datasets(data_dir: str):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    return train, test


def iid_partition(train_dataset, num_clients: int):
    N = len(train_dataset)
    idxs = np.random.permutation(N)
    splits = np.array_split(idxs, num_clients)
    return [list(s) for s in splits]


def dirichlet_partition(train_dataset, num_clients: int, alpha: float = 0.3):
    labels = np.array(train_dataset.targets)
    num_classes = labels.max() + 1
    idxs = [np.where(labels == y)[0] for y in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(idxs[c])
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(idxs[c])).astype(int)[:-1]
        split = np.split(idxs[c], proportions)
        for i, s in enumerate(split):
            client_indices[i].extend(s.tolist())

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return client_indices


# ----------------------------
# FL helpers
# ----------------------------

def get_model_vector(model: nn.Module) -> np.ndarray:
    """Flatten model parameters to a single 1-D numpy array."""
    with torch.no_grad():
        return np.concatenate([
            p.detach().cpu().numpy().ravel() for p in model.parameters()
        ])


def set_model_from_vector(model: nn.Module, vec: np.ndarray):
    """Load flattened vector back into model parameters."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.copy_(torch.from_numpy(vec[offset:offset + numel]).view_as(p))
            offset += numel


def model_delta(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    return new - old


def apply_delta(old: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return old + delta


def train_local(model: nn.Module, loader: data.DataLoader, device: torch.device, epochs: int = 1, lr: float = 0.1):
    """Train the *provided* model (already seeded with global weights)."""
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    return model


def evaluate(model: nn.Module, loader: data.DataLoader, device: torch.device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item()
    return correct / total, loss_sum / total


# ----------------------------
# HE: context & ops
# ----------------------------

def make_ckks_context(poly_mod_degree=16_384, coeff_mod_bit_sizes=(60, 40, 40, 60), scale=2 ** 40):
    t0 = now_ms()
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, list(coeff_mod_bit_sizes))
    context.global_scale = scale
    # Secret (with sk) and public (without sk) split
    secret_ctx = context
    public_bytes = secret_ctx.serialize(save_public_key=True, save_secret_key=False)
    public_ctx = ts.context_from(public_bytes)
    setup_ms = now_ms() - t0
    return secret_ctx, public_ctx, setup_ms


def slot_capacity(poly_mod_degree: int) -> int:
    return poly_mod_degree // 2


def chunk_vector(vec: np.ndarray, max_len: int) -> list:
    return [vec[i:i + max_len] for i in range(0, len(vec), max_len)]


def encrypt_update_chunks(secret_ctx, vec: np.ndarray, max_slots: int):
    chunks = chunk_vector(vec, max_slots)
    return [ts.ckks_vector(secret_ctx, ch.tolist()) for ch in chunks]


def serialize_ciphertexts(ct_list) -> list:
    return [ct.serialize() for ct in ct_list]


def decrypt_concat(secret_ctx, ct_list) -> np.ndarray:
    parts = [np.array(ct.decrypt(secret_ctx.secret_key())) for ct in ct_list]
    return np.concatenate(parts)


# ----------------------------
# Benchmark loop with batching
# ----------------------------

def run_benchmark(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    os.makedirs(args.outdir, exist_ok=True)

    # Data
    train_ds, test_ds = build_datasets(args.data)
    test_loader = data.DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    if args.iid:
        parts = iid_partition(train_ds, args.clients)
    else:
        parts = dirichlet_partition(train_ds, args.clients, args.dirichlet_alpha)

    client_loaders = []
    for idxs in parts:
        subset = data.Subset(train_ds, idxs)
        loader = data.DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        client_loaders.append(loader)

    # Global model
    global_model = MLP().to(device)
    global_vec = get_model_vector(global_model)

    # HE setup
    secret_ctx, public_ctx, he_setup_ms = make_ckks_context(
        args.poly_mod_degree, tuple(args.coeff_mod_bit_sizes), 2 ** args.scale_bits
    )

    log_csv = Path(args.outdir) / "metrics.csv"
    log_jsonl = Path(args.outdir) / "metrics.jsonl"
    with open(log_csv, 'w') as f:
        f.write(
            "round,clients,train_ms,he_setup_ms,he_encrypt_ms,he_batch_aggregate_ms,he_bytes,total_bytes,he_aggregate_ms,he_decrypt_ms,apply_ms,pt_aggregate_ms,test_acc,test_loss,peak_rss_mb\n"
        )

    peak_rss_mb = 0.0

    for r in range(1, args.rounds + 1):
        round_start = now_ms()
        # Select a fraction of clients each round (simulate participation)
        m = max(1, int(args.clients * args.participation))
        selected = np.random.choice(range(args.clients), size=m, replace=False)

        # Local training per client
        local_models = []
        train_ms = 0
        for cid in selected:
            t0 = now_ms()
            # Start from current global weights
            model_copy = MLP().to(device)
            set_model_from_vector(model_copy, global_vec)
            trained = train_local(model_copy, client_loaders[cid], device, args.local_epochs, args.lr)
            train_ms += now_ms() - t0
            local_models.append(trained)

        # Compute plaintext deltas and proper FedAvg weights (by client sample count)
        deltas = []
        client_sizes = []
        for model, cid in zip(local_models, selected):
            vec_new = get_model_vector(model)
            d = model_delta(global_vec, vec_new)
            deltas.append(d)
            client_sizes.append(len(client_loaders[cid].dataset))
        client_sizes = np.asarray(client_sizes, dtype=np.float64)
        weights = client_sizes / client_sizes.sum()

        # Plaintext FedAvg baseline timing
        t0 = now_ms()
        pt_avg = np.zeros_like(global_vec)
        for w, d in zip(weights, deltas):
            pt_avg += w * d
        pt_aggregate_ms = now_ms() - t0

        # HE pipeline timing & sizes
        he_encrypt_ms = 0
        he_batch_aggregate_ms = 0
        he_aggregate_ms = 0
        he_decrypt_ms = 0
        total_bytes = 0

        max_slots = slot_capacity(args.poly_mod_degree)

        # If enc_batch_size == 0 -> original behavior: encrypt all clients and keep blobs
        if args.enc_batch_size <= 0:
            client_ct_blobs = []  # list of [blob_chunk_0, ...]
            for w, d in zip(weights, deltas):
                t0 = now_ms()
                weighted = d * w
                ct_chunks = encrypt_update_chunks(secret_ctx, weighted, max_slots)
                he_encrypt_ms += now_ms() - t0
                blobs = serialize_ciphertexts(ct_chunks)
                total_bytes += sum(len(b) for b in blobs)
                client_ct_blobs.append(blobs)

            # Server aggregates on ciphertext chunks (no secret key): just sum
            num_chunks = len(client_ct_blobs[0])
            t0 = now_ms()
            agg_chunks = []
            for ch_idx in range(num_chunks):
                ct_sum = ts.ckks_vector_from(public_ctx, client_ct_blobs[0][ch_idx])
                for client_idx in range(1, len(client_ct_blobs)):
                    ct_i = ts.ckks_vector_from(public_ctx, client_ct_blobs[client_idx][ch_idx])
                    ct_sum += ct_i
                agg_chunks.append(ct_sum)
            he_aggregate_ms = now_ms() - t0

        else:
            # Batching path: process clients in groups to keep peak memory / deserialization low
            batch_size = int(args.enc_batch_size)
            num_clients = len(weights)
            partial_batches = []  # list of batch-level serialized partial aggregates: each element is [blob_chunk_0,...]

            # Process clients in batches
            for start in range(0, num_clients, batch_size):
                end = min(start + batch_size, num_clients)
                t_batch_start = now_ms()
                # For the batch we will form a partial aggregated ciphertext per chunk
                batch_partial_cts = None
                # Iterate clients in this batch
                for idx in range(start, end):
                    w = weights[idx]
                    d = deltas[idx]
                    weighted = d * w
                    t0 = now_ms()
                    ct_chunks = encrypt_update_chunks(secret_ctx, weighted, max_slots)
                    he_encrypt_ms += now_ms() - t0
                    blobs = serialize_ciphertexts(ct_chunks)
                    total_bytes += sum(len(b) for b in blobs)

                    # sum into batch partials using public_ctx deserialization
                    if batch_partial_cts is None:
                        # Initialize with first client's chunks
                        batch_partial_cts = [ts.ckks_vector_from(public_ctx, b) for b in blobs]
                    else:
                        for ch_i, b in enumerate(blobs):
                            ct_i = ts.ckks_vector_from(public_ctx, b)
                            batch_partial_cts[ch_i] += ct_i

                    # Allow Python to drop 'blobs' and the ct_i after addition

                # Serialize batch partials and store
                partial_blobs = [ct.serialize() for ct in batch_partial_cts]
                partial_batches.append(partial_blobs)
                he_batch_aggregate_ms += now_ms() - t_batch_start

            # Final aggregation across batch-level partials
            num_chunks = len(partial_batches[0])
            t0 = now_ms()
            agg_chunks = []
            for ch_idx in range(num_chunks):
                ct_sum = ts.ckks_vector_from(public_ctx, partial_batches[0][ch_idx])
                for batch_idx in range(1, len(partial_batches)):
                    ct_i = ts.ckks_vector_from(public_ctx, partial_batches[batch_idx][ch_idx])
                    ct_sum += ct_i
                agg_chunks.append(ct_sum)
            he_aggregate_ms = now_ms() - t0

        # Decrypt aggregated update and stitch
        t0 = now_ms()
        agg_update = decrypt_concat(secret_ctx, agg_chunks)
        he_decrypt_ms = now_ms() - t0

        # Apply to global model
        t0 = now_ms()
        global_vec = apply_delta(global_vec, agg_update)
        set_model_from_vector(global_model, global_vec)
        apply_ms = now_ms() - t0

        # Eval
        acc, loss = evaluate(global_model, test_loader, device)

        # Track memory
        if psutil is not None:
            process = psutil.Process(os.getpid())
            rss_mb = process.memory_info().rss / (1024 ** 2)
            peak_rss_mb = max(peak_rss_mb, rss_mb)
        else:
            rss_mb = float('nan')

        # Persist logs
        row = {
            "round": r,
            "clients": int(m),
            "train_ms": int(train_ms),
            "he_setup_ms": int(he_setup_ms if r == 1 else 0),
            "he_encrypt_ms": int(he_encrypt_ms),
            "he_batch_aggregate_ms": int(he_batch_aggregate_ms),
            "he_bytes": int(total_bytes),
            "total_bytes": int(total_bytes),
            "he_aggregate_ms": int(he_aggregate_ms),
            "he_decrypt_ms": int(he_decrypt_ms),
            "apply_ms": int(apply_ms),
            "pt_aggregate_ms": int(pt_aggregate_ms),
            "test_acc": float(acc),
            "test_loss": float(loss),
            "peak_rss_mb": float(peak_rss_mb),
        }

        # Round summary
        with open(log_csv, 'a') as f:
            f.write(
                f"{row['round']},{row['clients']},{row['train_ms']},{row['he_setup_ms']},{row['he_encrypt_ms']},{row['he_batch_aggregate_ms']},{row['he_bytes']},{row['total_bytes']},{row['he_aggregate_ms']},{row['he_decrypt_ms']},{row['apply_ms']},{row['pt_aggregate_ms']},{row['test_acc']:.4f},{row['test_loss']:.4f},{row['peak_rss_mb']:.2f}\n"
            )
        with open(log_jsonl, 'a') as f:
            f.write(json.dumps(row) + "\n")
        print(
                f"[Round {r}/{args.rounds}] clients={m} acc={acc:.4f} loss={loss:.4f} | "
                f"train={train_ms}ms he: enc={he_encrypt_ms}ms batch_agg={he_batch_aggregate_ms}ms agg={he_aggregate_ms}ms dec={he_decrypt_ms}ms "
                f"bytes={sizeof_bytes(total_bytes)} pt_agg={pt_aggregate_ms}ms mem~{peak_rss_mb:.1f}MB"
        )

        print(f"\nLogs written to: {log_csv} and {log_jsonl}")


# ----------------------------
# Main
# ----------------------------
import sys

if __name__ == "__main__":
    #arr = [2, 5, 10, 20, 50, 100]
    #for i in arr:
        parser = argparse.ArgumentParser(description="Federated Learning + CKKS (TenSEAL) MNIST Benchmark — Batching")
        parser.add_argument('--data', type=str, default='./data')
        parser.add_argument('--outdir', type=str, default='./runs/ckks_mnist_batch')
        parser.add_argument('--clients', type=int, default=2)
        parser.add_argument('--participation', type=float, default=1.0)
        parser.add_argument('--rounds', type=int, default=10)
        parser.add_argument('--local-epochs', type=int, default=1)
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--lr', type=float, default=0.1)

        part = parser.add_mutually_exclusive_group()
        part.add_argument('--iid', action='store_true')
        part.add_argument('--dirichlet-alpha', type=float, default=0.3)

        parser.add_argument('--poly-mod-degree', type=int, default=16_384)
        parser.add_argument('--coeff-mod-bit-sizes', type=int, nargs='+', default=[60, 40, 60])
        parser.add_argument('--scale-bits', type=int, default=40)

        # Batching control
        parser.add_argument('--enc-batch-size', type=int, default=5,
                            help='0 -> no batching (original), >0 -> number of clients per encryption batch')


        parser.add_argument('--cpu', action='store_true')
        parser.add_argument('--seed', type=int, default=42)

        # This line ignores Jupyter's extra args like "-f /path/to/kernel.json"
        args, _ = parser.parse_known_args()

        run_benchmark(args)
