
"""
FL + CKKS (TenSEAL) MNIST Benchmark — Quantization Strategy
-----------------------------------------------------------

What this script adds (vs. your baseline)
- **Client-side uniform quantization** of model updates before HE encryption.
  * Options: bit-width (e.g., 8, 4), percentile clipping, stochastic rounding.
  * Quantized updates are **dequantized on the client** (still before encryption)
    so the server pipeline is unchanged (sums ciphertexts) and global update
    remains compatible with FedAvg.
  * This enables a smaller CKKS scale/modulus chain (optional "HE-lite" mode)
    to reduce encryption/aggregation/decryption time and ciphertext size, while
    keeping the exact same FL loop, metrics, and logs.
- Extra metrics: quantization time and L2 error per round.

Why dequantize before encryption?
- The server sums encrypted vectors. If each client used its own per-round scale,
  the server would need to combine different scales, which is not possible without
  extra ciphertexts/metadata. By dequantizing on the client, we preserve the
  existing single-sum server path and still benefit from **reduced numeric range**
  (allowing a smaller CKKS scale/modulus chain) and potentially faster HE ops.

Tip: Use `--quant-he-lite` to switch to a tighter CKKS parameter set suitable for
quantized values (e.g., scale 2**30 and modulus chain [40, 40]).

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
            p.copy_(torch.from_numpy(vec[offset:offset+numel]).view_as(p))
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
# Quantization helpers
# ----------------------------

def quantize_uniform(vec: np.ndarray, bits: int = 8, clip_p: float = 0.999, stochastic: bool = False):
    """
    Uniform symmetric per-tensor quantization.
    - vec: float64 array
    - bits: bit-width (2..16 typical)
    - clip_p: percentile for clipping magnitude to reduce outlier impact
    - stochastic: if True, adds uniform noise in [-0.5, 0.5] before rounding

    Returns (q: int16/32 array, scale: float, Qmax: int)
    such that vec ≈ (q / Qmax) * scale
    """
    assert 2 <= bits <= 16, "bits must be in [2, 16]"
    Qmax = (1 << (bits - 1)) - 1

    # Determine clipping threshold via percentile of |vec|
    abs_vec = np.abs(vec)
    max_mag = np.percentile(abs_vec, 100 * clip_p) if vec.size > 0 else 0.0
    # Avoid degenerate scale
    scale = max(max_mag, 1e-8)

    # Normalize to [-1, 1]
    norm = np.clip(vec / scale, -1.0, 1.0)
    norm *= Qmax

    if stochastic:
        noise = np.random.uniform(-0.5, 0.5, size=norm.shape)
        norm = norm + noise

    q = np.rint(norm).astype(np.int32)

    # Choose minimal dtype that fits
    if bits <= 8:
        q = q.astype(np.int16, copy=False)
    else:
        q = q.astype(np.int32, copy=False)

    return q, float(scale), int(Qmax)


def dequantize_uniform(q: np.ndarray, scale: float, Qmax: int) -> np.ndarray:
    return (q.astype(np.float64) / float(Qmax)) * float(scale)


# ----------------------------
# HE: context & ops
# ----------------------------

def make_ckks_context(poly_mod_degree=16_384, coeff_mod_bit_sizes=(60, 40, 60), scale=2**40):
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
# Benchmark loop
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

    # CKKS params — optionally tightened for quantized data
    poly_mod_degree = args.poly_mod_degree
    coeff_mod = tuple(args.coeff_mod_bit_sizes)
    scale_bits = args.scale_bits

    if args.quant_he_lite:
        # Smaller scale and chain when using quantized values
        scale_bits = min(scale_bits, 30)
        # ensure at least two levels for additions; keep them modest
        if len(coeff_mod) >= 3:
            coeff_mod = (40, 40)
        elif len(coeff_mod) == 2:
            coeff_mod = (min(coeff_mod[0], 40), min(coeff_mod[1], 40))
        else:
            coeff_mod = (40, 40)

    # Global model
    global_model = MLP().to(device)
    global_vec = get_model_vector(global_model)

    # HE setup
    secret_ctx, public_ctx, he_setup_ms = make_ckks_context(
        poly_mod_degree, coeff_mod, 2 ** scale_bits
    )

    log_csv = Path(args.outdir) / "metrics.csv"
    log_jsonl = Path(args.outdir) / "metrics.jsonl"
    with open(log_csv, 'w') as f:
        f.write(
            "round,clients,train_ms,he_setup_ms,he_encrypt_ms,he_bytes,total_bytes,he_aggregate_ms,he_decrypt_ms,apply_ms,pt_aggregate_ms,test_acc,test_loss,quant_ms,quant_l2_err,peak_rss_mb\n"
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
        he_aggregate_ms = 0
        he_decrypt_ms = 0
        quant_ms = 0
        total_bytes = 0

        max_slots = slot_capacity(poly_mod_degree)

        # Encrypt & serialize at clients (chunked)
        client_ct_blobs = []  # list of [blob_chunk_0, ...]
        quant_l2_err_accum = 0.0
        for w, d in zip(weights, deltas):
            vec = (w * d).astype(np.float64)

            if args.quant_bits > 0:
                t0 = now_ms()
                q, scale, Qmax = quantize_uniform(
                    vec,
                    bits=args.quant_bits,
                    clip_p=args.quant_clip,
                    stochastic=args.quant_stochastic,
                )
                vec_q = dequantize_uniform(q, scale, Qmax)
                quant_ms += now_ms() - t0
                # Track quantization L2 error for information/debug
                quant_l2_err_accum += float(np.linalg.norm(vec - vec_q))
                vec_to_encrypt = vec_q
            else:
                vec_to_encrypt = vec

            t0 = now_ms()
            ct_chunks = encrypt_update_chunks(secret_ctx, vec_to_encrypt, max_slots)
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
            "total_bytes": int(total_bytes),
            "he_aggregate_ms": int(he_aggregate_ms),
            "he_decrypt_ms": int(he_decrypt_ms),
            "apply_ms": int(apply_ms),
            "pt_aggregate_ms": int(pt_aggregate_ms),
            "test_acc": float(acc),
            "test_loss": float(loss),
            "quant_ms": int(quant_ms),
            "quant_l2_err": float(quant_l2_err_accum),
            "peak_rss_mb": float(peak_rss_mb),
        }

        with open(log_csv, 'a') as f:
            f.write(
                f"{row['round']},{row['clients']},{row['train_ms']},{row['he_setup_ms']},{row['he_encrypt_ms']},{row['he_bytes']},{row['total_bytes']},{row['he_aggregate_ms']},{row['he_decrypt_ms']},{row['apply_ms']},{row['pt_aggregate_ms']},{row['test_acc']:.4f},{row['test_loss']:.4f},{row['quant_ms']},{row['quant_l2_err']:.6f},{row['peak_rss_mb']:.2f}\n"
            )
        with open(log_jsonl, 'a') as f:
            f.write(json.dumps(row) + "\n")

        # Round summary
        print(
            f"[Round {r}/{args.rounds}] clients={m} acc={acc:.4f} loss={loss:.4f} | "
            f"train={train_ms}ms he: enc={he_encrypt_ms}ms agg={he_aggregate_ms}ms dec={he_decrypt_ms}ms "
            f"bytes={sizeof_bytes(total_bytes)} pt_agg={pt_aggregate_ms}ms quant={quant_ms}ms L2err~{quant_l2_err_accum:.3e} mem~{peak_rss_mb:.1f}MB"
       )

        print(f"\nLogs written to: {log_csv} and {log_jsonl}")


# ----------------------------
# Main
# ----------------------------
import sys

if __name__ == "__main__":
#  arr = [5, 10, 20, 50, 100]

  #for i in arr:
    parser = argparse.ArgumentParser(description="Federated Learning + CKKS (TenSEAL) MNIST Benchmark — Quantization")
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='./runs/ckks_mnist_quant')

    #######################################################
    parser.add_argument('--clients', type=int, default=2)
    #######################################################

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


    # Quantization controls
    ####################################################################################################################################
    parser.add_argument('--quant-bits', type=int, default=8, help='bit-width for uniform per-tensor quantization (set 0 to disable)')###
    ####################################################################################################################################


    parser.add_argument('--quant-clip', type=float, default=0.999, help='percentile (0-1] to clip magnitudes before quantization')
    parser.add_argument('--quant-stochastic', action='store_true', help='enable stochastic rounding in quantization')
    parser.add_argument('--quant-he-lite', action='store_true', help='use smaller CKKS scale/modulus suited to quantized values')

    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    # This line ignores Jupyter's extra args like "-f /path/to/kernel.json"
    args, _ = parser.parse_args()

    run_benchmark(args)