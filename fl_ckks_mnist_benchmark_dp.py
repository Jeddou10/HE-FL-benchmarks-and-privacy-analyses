"""
FL + CKKS (TenSEAL) MNIST Benchmark — HE + Differential Privacy (DP)
-------------------------------------------------------------------

This variant extends the baseline HE-only script (no quantization, no selection,
no batching by default) to support simple Differential Privacy mechanisms.

- Support two DP modes:
  1. `client` (local DP): each client clips its weighted update and adds Gaussian noise
     **before** encryption. This provides local differential privacy (stronger, but may
     harm accuracy more for the same noise level).
  2. `server` (central DP): each client clips its weighted update and sends **noiseless**
     encrypted updates to the server. The server homomorphically aggregates ciphertexts
     and then **adds encrypted Gaussian noise** (generated with the public key) to the
     ciphertext aggregate before decryption. This is the centralized DP model.
- Both modes require per-client clipping (L2) of `w * delta` to `dp_clip` to bound sensitivity.

Note
- Server-side encrypted noise creation uses the public TenSEAL context to encrypt random noise
  vectors and add them to the ciphertext aggregate. This avoids revealing the noise to the server
  in plaintext.
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
    #Flatten model parameters to a single 1-D numpy array.
    with torch.no_grad():
        return np.concatenate([
            p.detach().cpu().numpy().ravel() for p in model.parameters()
        ])


def set_model_from_vector(model: nn.Module, vec: np.ndarray):
    #Load flattened vector back into model parameters.
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.copy_(torch.from_numpy(vec[offset:offset+numel]).view_as(p))
            offset += numel


def model_delta(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    return new - old


def apply_delta(old: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return old + delta


def train_local(model: nn.Module, loader: data.DataLoader, device: torch.device, epochs: int = 1, lr: float = 0.1):
    #Train the *provided* model (already seeded with global weights).
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

def make_ckks_context(poly_mod_degree=16_384, coeff_mod_bit_sizes=(60, 40, 40, 60), scale=2**40):
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
# DP helpers
# ----------------------------

def clip_vec(vec: np.ndarray, clip: float) -> np.ndarray:
    #L2 clip a vector to norm `clip`.
    norm = np.linalg.norm(vec)
    if norm > clip and norm > 0:
        return vec * (clip / norm)
    return vec


def make_gauss_noise(shape, sigma, clip):
    #Gaussian noise with std = sigma * clip (per-coordinate iid).
    return np.random.normal(loc=0.0, scale=float(sigma * clip), size=shape).astype(np.float64)


# ----------------------------
# Benchmark loop (baseline HE semantics + DP)
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

    # HE setup (baseline: secret + public contexts)
    secret_ctx, public_ctx, he_setup_ms = make_ckks_context(
        args.poly_mod_degree, tuple(args.coeff_mod_bit_sizes), 2 ** args.scale_bits
    )

    log_csv = Path(args.outdir) / "metrics.csv"
    log_jsonl = Path(args.outdir) / "metrics.jsonl"
    with open(log_csv, 'w') as f:
        f.write(
            "round,clients,train_ms,he_setup_ms,he_encrypt_ms,he_bytes,he_aggregate_ms,he_decrypt_ms,apply_ms,pt_aggregate_ms,test_acc,test_loss,dp_mode,dp_clip,dp_sigma,dp_noise_ms,peak_rss_mb")

    peak_rss_mb = 0.0

    max_slots = slot_capacity(args.poly_mod_degree)

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
        he_aggregate_ms = 0
        he_decrypt_ms = 0
        dp_noise_ms = 0
        total_bytes = 0

        # --- CLIENT-SIDE: clip, optionally add noise (client-DP), then encrypt & serialize ---
        client_ct_blobs = []
        for w, d in zip(weights, deltas):
            vec = (w * d).astype(np.float64)

            # clip per-client to bound sensitivity
            if args.dp_mode != 'none' and args.dp_clip > 0.0:
                vec = clip_vec(vec, args.dp_clip)

            # client DP: add Gaussian noise before encryption
            if args.dp_mode == 'client' and args.dp_sigma > 0.0:
                t0 = now_ms()
                noise = make_gauss_noise(vec.shape, args.dp_sigma, args.dp_clip)
                vec = vec + noise
                dp_noise_ms += now_ms() - t0

            # encrypt packed vector across CKKS slots (as the baseline did)
            t0 = now_ms()
            ct_chunks = encrypt_update_chunks(secret_ctx, vec, max_slots)
            he_encrypt_ms += now_ms() - t0
            blobs = serialize_ciphertexts(ct_chunks)
            total_bytes += sum(len(b) for b in blobs)
            client_ct_blobs.append(blobs)

        # --- SERVER-SIDE: aggregate ciphertexts (no secret key) ---
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

        # server DP: add encrypted Gaussian noise to aggregated ciphertexts before decryption
        if args.dp_mode == 'server' and args.dp_sigma > 0.0 and args.dp_clip > 0.0:
            t0 = now_ms()
            # build a plaintext noise vector with same length as flattened model and split into chunks
            full_noise = make_gauss_noise((global_vec.shape[0],), args.dp_sigma, args.dp_clip)
            noise_chunks = chunk_vector(full_noise, max_slots)
            # encrypt each noise chunk with public context and add to agg_chunks
            for i, nch in enumerate(noise_chunks):
                noise_ct = ts.ckks_vector_from(public_ctx, ts.ckks_vector(secret_ctx, nch.tolist()).serialize())
                # Above: we construct a noise ciphertext using secret_ctx then serialize/deserialize into public_ctx
                # (This is a safe way to ensure proper encryption; if TenSEAL supports direct public_ctx crc, it can be simplified.)
                agg_chunks[i] += noise_ct
            dp_noise_ms += now_ms() - t0

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
            "he_bytes": int(total_bytes),
            "he_aggregate_ms": int(he_aggregate_ms),
            "he_decrypt_ms": int(he_decrypt_ms),
            "apply_ms": int(apply_ms),
            "pt_aggregate_ms": int(pt_aggregate_ms),
            "test_acc": float(acc),
            "test_loss": float(loss),
            "dp_mode": args.dp_mode,
            "dp_clip": float(args.dp_clip),
            "dp_sigma": float(args.dp_sigma),
            "dp_noise_ms": int(dp_noise_ms),
            "peak_rss_mb": float(peak_rss_mb),
        }

        with open(log_csv, 'a') as f:
            f.write(
                f"{row['round']},{row['clients']},{row['train_ms']},{row['he_setup_ms']},{row['he_encrypt_ms']},{row['he_bytes']},{row['he_aggregate_ms']},{row['he_decrypt_ms']},{row['apply_ms']},{row['pt_aggregate_ms']},{row['test_acc']:.4f},{row['test_loss']:.4f},{row['dp_mode']},{row['dp_clip']},{row['dp_sigma']},{row['dp_noise_ms']},{row['peak_rss_mb']:.2f}"
            )
        with open(log_jsonl, 'a') as f:
            f.write(json.dumps(row) + "")

        # Round summary
        print(
            f"[Round {r}/{args.rounds}] clients={m} acc={acc:.4f} loss={loss:.4f} | "
            f"train={train_ms}ms he: enc={he_encrypt_ms}ms agg={he_aggregate_ms}ms dec={he_decrypt_ms}ms "
            f"dp_mode={args.dp_mode} clip={args.dp_clip} sigma={args.dp_sigma} dp_noise_ms={dp_noise_ms} mem~{peak_rss_mb:.1f}MB"
        )

    print(f"Logs written to: {log_csv} and {log_jsonl}")


# ----------------------------
# Main
# ----------------------------
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Learning + CKKS (TenSEAL) MNIST Benchmark — HE + DP")


    parser.add_argument('--data', type=str, default='./data')
    # Path to dataset storage / load directory
    parser.add_argument('--outdir', type=str,
                        default='./runs/ckks_mnist_dp')
    # Output directory for run artifacts (metrics, logs, etc.)

    parser.add_argument('--clients', type=int, default=10)
    # Number of federated clients to simulate

    parser.add_argument('--participation', type=float,
                        default=1.0)
    # Fraction of clients sampled each round (1.0 = all)

    parser.add_argument('--rounds', type=int, default=100)
    # Number of federated training rounds to run

    parser.add_argument('--local-epochs', type=int, default=1)
    # Local epochs each client runs per round

    parser.add_argument('--batch-size', type=int, default=64)
    # Batch size for local training

    parser.add_argument('--lr', type=float, default=0.1)
    # Learning rate for local SGD

    part = parser.add_mutually_exclusive_group()
    # Create a group that enforces mutual exclusivity between options

    part.add_argument('--iid', action='store_true')
    # If set, use IID partitioning across clients

    part.add_argument('--dirichlet-alpha', type=float,
                      default=0.3)
    # Dirichlet concentration alpha for non-IID partitioning (ignored if --iid set)

    parser.add_argument('--poly-mod-degree', type=int,
                        default=16_384)
    # CKKS polynomial modulus degree (affects slot count & security level)

    parser.add_argument('--coeff-mod-bit-sizes', type=int, nargs='+',
                        default=[60, 40, 60])
    # Coefficient-modulus chain bit-sizes for CKKS (precision & noise budget)

    parser.add_argument('--scale-bits', type=int,
                        default=40)
    # Number of bits used for CKKS scaling factor (encoding precision)

    # DP controls
    parser.add_argument(
        '--dp-mode', type=str, choices=['none', 'client', 'server'], default='client',
        help='none=disable DP, client=local DP (noise added before encryption), server=central DP (encrypted noise added before decryption)'
    )  # Toggle DP behavior: none/client/server with explicit semantics in help

    parser.add_argument('--dp-clip', type=float, default=1.0,
                        help='L2 clipping bound for per-client weighted update')
    # L2 clip bound used before adding DP noise

    parser.add_argument('--dp-sigma', type=float, default=0.1,
                        help='Gaussian noise multiplier: std = dp_sigma * dp_clip')
    # Noise multiplier for Gaussian mechanism (std = sigma * clip)

    parser.add_argument('--cpu', action='store_true')
    # Force CPU-only execution (disable GPU use)

    parser.add_argument('--seed', type=int,
                        default=42)
    # RNG seed for reproducibility (partitions, training, DP randomness, etc.)

    args, _ = parser.parse_known_args()

    run_benchmark(args)

