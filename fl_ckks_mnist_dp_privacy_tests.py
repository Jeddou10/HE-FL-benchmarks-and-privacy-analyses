"""
Privacy Test Harness for FL + CKKS + Differential Privacy (DP)
----------------------------------------------------------------

This script adapts the comprehensive threat-model harness (gradient inversion,
membership inference, probing, timing leakage, collusion) to the HE+DP
benchmark. It mirrors the attack tests and logging style from
the original "threat_model_full" harness while taking into account the
DP mode (none / client / server) and the per-client clipping + Gaussian
noise described in DP benchmark.

"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image

try:
    import tenseal as ts
except ImportError:
    raise SystemExit("tenseal is required. Install via: pip install tenseal")

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

def save_image(arr: np.ndarray, path: str):
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    im = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
    Image.fromarray(im).convert('L').save(path)


def psnr(a, b, maxval=1.0):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(maxval / np.sqrt(mse))

def sizeof_bytes(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} TB"


# ----------------------------
# Model & data (same MLP + MNIST)
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
# Flatten & model vector helpers
# ----------------------------

def get_model_vector(model: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def set_model_from_vector(model: nn.Module, vec: np.ndarray):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.copy_(torch.from_numpy(vec[offset:offset+numel]).view_as(p))
            offset += numel
    for p in model.parameters():
        p.requires_grad_(True)


def model_param_slices(model: nn.Module):
    slices = []
    offset = 0
    for p in model.parameters():
        num = p.numel()
        slices.append((offset, offset + num))
        offset += num
    return slices


def model_delta(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    return new - old


def apply_delta(old: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return old + delta


# ----------------------------
# Train / eval
# ----------------------------

def train_local(model: nn.Module, loader: data.DataLoader, device: torch.device, epochs: int = 1, lr: float = 0.1):
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
    model = model.to(device)
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
# HE: context & ops (same patterns as your DP benchmark)
# ----------------------------

def make_ckks_context(poly_mod_degree=16_384, coeff_mod_bit_sizes=(60, 40, 40, 60), scale=2**40):
    t0 = now_ms()
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, list(coeff_mod_bit_sizes))
    context.global_scale = scale
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
# DP helpers (clip + gaussian noise)
# ----------------------------

def clip_vec(vec: np.ndarray, clip: float) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > clip and norm > 0:
        return vec * (clip / norm)
    return vec


def make_gauss_noise(shape, sigma, clip):
    return np.random.normal(loc=0.0, scale=float(sigma * clip), size=shape).astype(np.float64)


# ----------------------------
# Attacks & analyses adapted to DP
# ----------------------------

def compute_grad_vector(model: nn.Module, x: torch.Tensor, label: int, device: torch.device):
    model = model.to(device)
    model.eval()
    x = x.to(device)
    y = torch.tensor([label], device=device)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    flat = torch.cat([g.view(-1) for g in grads]).detach().cpu().float()
    return flat


def inversion_attack_from_aggregate(agg_update: np.ndarray, model_template: nn.Module, param_slice: tuple,
                                    img_shape=(1,28,28), steps=800, lr=0.05, restarts=3, device='cpu'):
    """Gradient inversion attempt starting from the *aggregated* update vector.

    Note: when DP noise is present (client or server) the target will be noisy and
    this attack may fail (higher loss / wrong class). The attack code itself is
    identical to the baseline harness but we expose results to the logs.
    """
    start, end = param_slice
    device = torch.device(device)
    target = torch.tensor(agg_update[start:end], dtype=torch.float32, device=device)

    best = {'label': None, 'img': None, 'loss': float('inf')}

    for restart in range(restarts):
        for lab in range(10):
            x = torch.rand((1,)+img_shape, device=device, requires_grad=True)
            opt = torch.optim.Adam([x], lr=lr)
            model = model_template.to(device)
            for p in model.parameters():
                p.requires_grad_(True)
            set_model_from_vector(model, get_model_vector(model))
            for i in range(steps):
                opt.zero_grad()
                logits = model(x)
                loss_ce = nn.CrossEntropyLoss()(logits, torch.tensor([lab], device=device))
                grads = torch.autograd.grad(loss_ce, model.parameters(), create_graph=True)
                gvec = torch.cat([g.contiguous().view(-1) for g in grads])
                gsel = gvec[start:end].to(device)
                loss = nn.MSELoss()(gsel, target)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(0.0, 1.0)
            final_loss = float(loss.detach().cpu().numpy())
            if final_loss < best['loss']:
                best['loss'] = final_loss
                best['label'] = lab
                best['img'] = x.detach().cpu().numpy().copy()
    return best


def membership_inference_loss_threshold(global_model: nn.Module, sample_x: np.ndarray, sample_y: int, threshold: float, device='cpu'):
    model = global_model.to(device)
    model.eval()
    x = torch.tensor(sample_x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        loss = float(nn.CrossEntropyLoss()(logits, torch.tensor([sample_y], device=device)).item())
    return loss < threshold, loss


def train_shadow_model(shadow_train, shadow_test, model_template: nn.Module, device, epochs=3, lr=0.1):
    model = model_template.to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    loader = data.DataLoader(shadow_train, batch_size=64, shuffle=True)
    for e in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    # collect losses for members and non-members
    model.eval()
    member_losses, nonmember_losses = [], []
    with torch.no_grad():
        for xb, yb in data.DataLoader(shadow_train, batch_size=64):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            l = nn.CrossEntropyLoss(reduction='none')(logits, yb)
            member_losses.extend(l.cpu().numpy().tolist())
        for xb, yb in data.DataLoader(shadow_test, batch_size=64):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            l = nn.CrossEntropyLoss(reduction='none')(logits, yb)
            nonmember_losses.extend(l.cpu().numpy().tolist())
    return np.array(member_losses), np.array(nonmember_losses)


def probing_marker_attack(weighted_vecs: list, secret_ctx, public_ctx, max_slots, marker_scale=1e-2, device='cpu'):
    """Inject a sparse marker into one client's weighted update and see if the
    aggregated decrypt reveals its footprint. Under DP the marker can be hidden.
    """
    vec0 = weighted_vecs[0]
    marker = np.zeros_like(vec0)
    idxs = np.random.choice(len(vec0), size=10, replace=False)
    marker[idxs] = marker_scale
    marked = [v.copy() for v in weighted_vecs]
    marked[0] = marked[0] + marker

    blobs_orig = [serialize_ciphertexts(encrypt_update_chunks(secret_ctx, v.astype(np.float64), max_slots)) for v in weighted_vecs]
    blobs_marked = [serialize_ciphertexts(encrypt_update_chunks(secret_ctx, v.astype(np.float64), max_slots)) for v in marked]

    # reconstruct ciphertext aggregates via public_ctx
    agg_orig_chunks = []
    for ch_idx in range(len(blobs_orig[0])):
        ct_sum = ts.ckks_vector_from(public_ctx, blobs_orig[0][ch_idx])
        for i in range(1, len(blobs_orig)):
            ct_sum += ts.ckks_vector_from(public_ctx, blobs_orig[i][ch_idx])
        agg_orig_chunks.append(ct_sum)

    agg_mark_chunks = []
    for ch_idx in range(len(blobs_marked[0])):
        ct_sum = ts.ckks_vector_from(public_ctx, blobs_marked[0][ch_idx])
        for i in range(1, len(blobs_marked)):
            ct_sum += ts.ckks_vector_from(public_ctx, blobs_marked[i][ch_idx])
        agg_mark_chunks.append(ct_sum)

    agg_orig = decrypt_concat(secret_ctx, agg_orig_chunks)
    agg_mark = decrypt_concat(secret_ctx, agg_mark_chunks)
    diff = agg_mark - agg_orig
    corr = np.corrcoef(diff.flatten(), marker.flatten())[0,1]
    return corr, marker, diff


def timing_and_size_leakage(weighted_vecs: list, secret_ctx, public_ctx, max_slots):
    enc_times = []
    lengths = []
    for v in weighted_vecs:
        t0 = now_ms()
        blobs = serialize_ciphertexts(encrypt_update_chunks(secret_ctx, v.astype(np.float64), max_slots))
        enc_times.append(now_ms() - t0)
        lengths.append(sum(len(b) for b in blobs))
    return enc_times, lengths


# ----------------------------
# Collusion helper (additive sharing)
# ----------------------------

def make_additive_shares(vec: np.ndarray, n: int):
    shares = [np.random.normal(0, 1e-6, size=vec.shape) for _ in range(n-1)]
    last = vec - np.sum(shares, axis=0)
    shares.append(last)
    return shares


# ----------------------------
# Orchestration: run privacy tests under DP modes
# ----------------------------

def run_privacy_tests(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    os.makedirs(args.outdir, exist_ok=True)

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

    global_model = MLP().to(device)
    global_vec = get_model_vector(global_model)

    secret_ctx, public_ctx, he_setup_ms = make_ckks_context(
        args.poly_mod_degree, tuple(args.coeff_mod_bit_sizes), 2 ** args.scale_bits
    )
    max_slots = slot_capacity(args.poly_mod_degree)
    param_slices = model_param_slices(global_model)

    logs = defaultdict(list)

    for r in range(1, args.rounds + 1):
        print(f"--- Round {r}/{args.rounds} ---")
        m = max(1, int(args.clients * args.participation))
        selected = np.random.choice(range(args.clients), size=m, replace=False)

        # Local training
        local_models = []
        for cid in selected:
            model_copy = MLP().to(device)
            set_model_from_vector(model_copy, global_vec)
            trained = train_local(model_copy, client_loaders[cid], device, args.local_epochs, args.lr)
            local_models.append(trained)

        # Deltas and weights
        deltas = []
        client_sizes = []
        for model, cid in zip(local_models, selected):
            vec_new = get_model_vector(model)
            d = model_delta(global_vec, vec_new)
            deltas.append(d)
            client_sizes.append(len(client_loaders[cid].dataset))
        client_sizes = np.asarray(client_sizes, dtype=np.float64)
        weights = client_sizes / client_sizes.sum()

        # compute weighted vectors (before DP)
        weighted = [w * d for w, d in zip(weights, deltas)]

        # PT baseline aggregate for utility logging
        pt_avg = np.zeros_like(global_vec)
        for v in weighted:
            pt_avg += v

        # Apply DP: per-client clipping and optional client-side noise
        client_ct_blobs = []
        pre_noise_weighted = [v.copy() for v in weighted]
        dp_noise_ms = 0
        total_bytes = 0

        # --- Timing & size leakage (run every round) ---
        try:
            enc_times, lengths = timing_and_size_leakage(pre_noise_weighted, secret_ctx, public_ctx, max_slots)
            norms = [float(np.linalg.norm(v)) for v in pre_noise_weighted]
            if len(norms) > 1:
                corr_time = float(np.corrcoef(norms, enc_times)[0,1])
                corr_len = float(np.corrcoef(norms, lengths)[0,1])
            else:
                corr_time = float('nan')
                corr_len = float('nan')
            logs['timing'].append({'round': r, 'corr_time_norm': corr_time, 'corr_len_norm': corr_len})
            status = 'SUCCESS' if (abs(corr_time) > 0.3 or abs(corr_len) > 0.3) else 'FAIL'
            print(f"[Round {r}] TIMING LEAKAGE -> corr_time={corr_time:.4f} corr_len={corr_len:.4f} => {status}")
        except Exception as e:
            print(f"[Round {r}] TIMING LEAKAGE -> ERROR: {e}")


        for v in weighted:
            vec = v.astype(np.float64)
            if args.dp_mode != 'none' and args.dp_clip > 0.0:
                vec = clip_vec(vec, args.dp_clip)
            if args.dp_mode == 'client' and args.dp_sigma > 0.0:
                t0 = now_ms()
                noise = make_gauss_noise(vec.shape, args.dp_sigma, args.dp_clip)
                vec = vec + noise
                dp_noise_ms += now_ms() - t0
            t0 = now_ms()
            ct_chunks = encrypt_update_chunks(secret_ctx, vec, max_slots)
            dp_noise_ms += 0  # encryption time is counted separately if desired
            he_enc_ms = now_ms() - t0
            blobs = serialize_ciphertexts(ct_chunks)
            total_bytes += sum(len(b) for b in blobs)
            client_ct_blobs.append(blobs)

        # server aggregates
        num_chunks = len(client_ct_blobs[0])
        t0 = now_ms()
        agg_chunks = []
        for ch_idx in range(num_chunks):
            ct_sum = ts.ckks_vector_from(public_ctx, client_ct_blobs[0][ch_idx])
            for client_idx in range(1, len(client_ct_blobs)):
                ct_sum += ts.ckks_vector_from(public_ctx, client_ct_blobs[client_idx][ch_idx])
            agg_chunks.append(ct_sum)
        he_agg_ms = now_ms() - t0

        # Server DP: add encrypted Gaussian noise to aggregated ciphertexts
        if args.dp_mode == 'server' and args.dp_sigma > 0.0 and args.dp_clip > 0.0:
            t0 = now_ms()
            full_noise = make_gauss_noise((global_vec.shape[0],), args.dp_sigma, args.dp_clip)
            noise_chunks = chunk_vector(full_noise, max_slots)
            for i, nch in enumerate(noise_chunks):
                # create a secret_ctx ciphertext and translate into public_ctx bytes
                noise_ct = ts.ckks_vector_from(public_ctx, ts.ckks_vector(secret_ctx, nch.tolist()).serialize())
                agg_chunks[i] += noise_ct
            dp_noise_ms += now_ms() - t0

        # Decrypt aggregate
        t0 = now_ms()
        agg_update = decrypt_concat(secret_ctx, agg_chunks)
        he_dec_ms = now_ms() - t0

        # Apply update and evaluate
        global_vec = apply_delta(global_vec, agg_update)
        set_model_from_vector(global_model, global_vec)
        acc, loss = evaluate(global_model, test_loader, device)

        # Logs for this round
        logs['rounds'].append({
            'round': r,
            'dp_mode': args.dp_mode,
            'dp_clip': args.dp_clip,
            'dp_sigma': args.dp_sigma,
            'test_acc': float(acc),
            'test_loss': float(loss),
        })

        print(f"Round {r}: acc={acc:.4f} loss={loss:.4f} dp_mode={args.dp_mode} clip={args.dp_clip} sigma={args.dp_sigma}")

        # --- ATTACKS: run when r == attack_round or if attack_all is set ---
        do_attack = (r == args.attack_round) or args.attack_all
        if not do_attack:
            continue

        print("--- Running attacks at round", r, "---")

        # Prepare data used by attacks
        # decrypt the aggregated update (we already have agg_update)

        # 1) Gradient inversion attack on a selected slice
        '''
        if args.attack_inversion:
            print("Running inversion attack...")
            slice_idx = args.attack_slice_idx if args.attack_slice_idx is not None else 0
            s = param_slices[slice_idx]
            inv = inversion_attack_from_aggregate(agg_update, MLP(), s, img_shape=(1,28,28), steps=args.attack_steps, lr=args.attack_lr, restarts=args.attack_restarts, device=device)
            # save image & stats
            outdir = args.outdir
            os.makedirs(outdir, exist_ok=True)
            if inv['img'] is not None:
                # write image as numpy .npy for inspection
                np.save(os.path.join(outdir, f"r{r}_inv_label{inv['label']}.npy"), inv['img'])
            logs['inversion'].append({'round': r, 'label_guess': int(inv['label']), 'loss': float(inv['loss'])})
            print(f"Inversion: label_guess={inv['label']} loss={inv['loss']:.6e}")
        '''
        if args.attack_inversion and r == args.attack_round:
            print("Running gradient inversion attack (multi-restarts)...")
            slice_idx = args.attack_slice_idx if args.attack_slice_idx is not None else 0
            s = param_slices[slice_idx]
            inv = inversion_attack_from_aggregate(agg_update, MLP(), s, img_shape=(1,28,28), steps=args.attack_steps, lr=args.attack_lr, restarts=args.attack_restarts, device=device)
            # save reconstructed: inversion.png
            save_image(inv['img'].squeeze(0), os.path.join(args.outdir, "inversion.png"))
            # save original that attacker is trying to reconstruct (undo normalization)
            orig_tensor = client_loaders[selected[0]].dataset[0][0].numpy()
            orig_img = (orig_tensor * 0.3081) + 0.1307
            save_image(orig_img, os.path.join(args.outdir, "original.png"))
            mse = float(np.mean((inv['img'].squeeze(0) - client_loaders[selected[0]].dataset[0][0].numpy())**2))
            p = psnr(inv['img'].squeeze(0), client_loaders[selected[0]].dataset[0][0].numpy())
            logs['inversion'].append({'round': r, 'label_guess': int(inv['label']), 'loss': float(inv['loss']), 'mse': float(mse), 'psnr': float(p)})
            print(f"inversion: label_guess={inv['label']} mse={mse:.6e} psnr={p:.2f}dB")

        # 2) Membership inference (loss-threshold)
        if args.attack_membership:
            print("Running membership inference (loss-threshold)...")
            target_client = args.attack_client if args.attack_client is not None else int(selected[0])
            ds = list(data.Subset(train_ds, parts[target_client]))
            x_true, y_true = ds[0][0].numpy(), int(ds[0][1])
            inferred, loss_val = membership_inference_loss_threshold(global_model, x_true, y_true, args.membership_threshold, device=device)
            logs['membership'].append({'round': r, 'target_client': int(target_client), 'loss': float(loss_val), 'inferred_member': bool(inferred)})
            print(f"membership: loss={loss_val:.4f} inferred={inferred}")

        # 3) Membership inference (shadow-model)
        if args.attack_membership_shadow:
            print("Running membership inference (shadow-model)...")
            shadow_idxs = np.random.choice(len(train_ds), size=min(2000, len(train_ds)), replace=False)
            half = len(shadow_idxs)//2
            shadow_train = data.Subset(train_ds, shadow_idxs[:half])
            shadow_test = data.Subset(train_ds, shadow_idxs[half:])
            mem_losses, nonmem_losses = train_shadow_model(shadow_train, shadow_test, MLP(), device, epochs=args.shadow_epochs, lr=args.shadow_lr)
            thr = float(np.percentile(mem_losses, args.shadow_percentile))
            target_client = args.attack_client if args.attack_client is not None else int(selected[0])
            ds = list(data.Subset(train_ds, parts[target_client]))
            x_true, y_true = ds[0][0].numpy(), int(ds[0][1])
            inferred, loss_val = membership_inference_loss_threshold(global_model, x_true, y_true, thr, device=device)
            logs['membership_shadow'].append({'round': r, 'threshold': thr, 'loss': float(loss_val), 'inferred_member': bool(inferred)})
            print(f"membership (shadow): thr={thr:.4f} loss={loss_val:.4f} inferred={inferred}")

        # 4) Probing marker attack
        if args.attack_probing:
            print("Running probing marker attack...")
            corr, marker, diff = probing_marker_attack(pre_noise_weighted, secret_ctx, public_ctx, max_slots, marker_scale=args.probe_scale)
            logs['probing'].append({'round': r, 'corr': float(corr)})
            print(f"probing corr={corr:.4f}")


        # 6) Collusion (additive sharing reveal)
        if args.attack_collusion:
            print("Running collusion simulation...")
            n_agg = max(2, int(args.n_aggregators))
            k = min(n_agg, args.collusion_k)
            all_shares = []
            for v in pre_noise_weighted:
                shares = make_additive_shares(v, n_agg)
                all_shares.append(shares)
            reconstructed_clients = []
            for ci in range(len(pre_noise_weighted)):
                coll_sum = sum(all_shares[ci][j] for j in range(k))
                reconstructed_clients.append(coll_sum)
            rmses = [float(np.sqrt(np.mean((reconstructed_clients[i] - pre_noise_weighted[i])**2))) for i in range(len(pre_noise_weighted))]
            avg_rmse = float(np.mean(rmses))
            logs['collusion'].append({'round': r, 'k': k, 'avg_rmse': avg_rmse})
            print(f"collusion k={k} avg_rmse={avg_rmse:.6e}")

    # Save logs
    outp = os.path.join(args.outdir, 'dp_privacy_logs.json')
    with open(outp, 'w') as f:
        json.dump({k: v for k, v in logs.items()}, f, indent=2)
    print(f"Finished. Logs saved to {outp}")


# ----------------------------
# CLI
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Privacy Test Harness for FL + CKKS + DP (MNIST)')

    parser.add_argument('--data', type=str, default='./data')
    # Path where MNIST (and other datasets) will be stored/loaded.

    parser.add_argument('--outdir', type=str, default='./runs/ckks_mnist_dp_privacy')
    # Directory to write run outputs (metrics, reconstructed artifacts, logs).

    parser.add_argument('--clients', type=int, default=8)
    # Number of federated clients to simulate.

    parser.add_argument('--participation', type=float, default=1.0)
    # Fraction of clients sampled each round (1.0 = all clients participate).

    parser.add_argument('--rounds', type=int, default=10)
    # Number of federated training rounds to execute.

    parser.add_argument('--local-epochs', type=int, default=1)
    # Number of local training epochs each client runs per round.

    parser.add_argument('--batch-size', type=int, default=64)
    # Local training batch size on each client.

    parser.add_argument('--lr', type=float, default=0.1)
    # Learning rate for clients' local optimizer (SGD).

    parser.add_argument('--attack-round', type=int, default=5)
    # Round at which to execute attacks (e.g., run privacy tests after this round).


    parser.add_argument('--attack-all', action='store_true')
    # If set, enable all attack types (convenience flag).

    parser.add_argument('--attack_inversion', action='store_true')
    # Enable gradient inversion attacks (optimization-based reconstruction).

    parser.add_argument('--attack_membership', action='store_true')
    # Enable membership inference attacks (threshold-based).

    parser.add_argument('--attack_membership_shadow', action='store_true')
    # Enable shadow-model based membership inference attacks.

    parser.add_argument('--attack_probing', action='store_true')
    # Enable probing/marker injection attacks.

    parser.add_argument('--attack_collusion', action='store_true')
    # Enable collusion/additive-share reconstruction simulations.

    parser.add_argument('--attack_timing', action='store_true')
    # Enable timing/size leakage analysis.

    parser.add_argument('--attack_steps', type=int, default=800)
    # Number of optimization steps for inversion-style attacks.

    parser.add_argument('--attack_lr', type=float, default=0.05)
    # Learning rate used by the inversion optimizer.

    parser.add_argument('--attack_restarts', type=int, default=2)
    # Number of random restarts for inversion optimization.

    parser.add_argument('--attack_slice_idx', type=int, default=0)
    # Index of model slice/parameter block to attack (if slicing is used).

    parser.add_argument('--attack_client', type=int, default=None)
    # Specific client id to target with attacks (None => use sampled/selected client).

    parser.add_argument('--membership_threshold', type=float, default=0.5)
    # Threshold used by simple loss-threshold membership inference.

    parser.add_argument('--probe_scale', type=float, default=1e-2)
    # Magnitude scaling for probing/marker injections.

    parser.add_argument('--n_aggregators', type=int, default=3)
    # Number of aggregator replicas simulated (used by some collusion scenarios).

    parser.add_argument('--collusion_k', type=int, default=1)
    # Number of colluding parties assumed in collusion experiments.

    parser.add_argument('--shadow_epochs', type=int, default=3)
    # Number of epochs used when training shadow models for membership inference.

    parser.add_argument('--shadow_lr', type=float, default=0.1)
    # Learning rate for shadow-model training.

    parser.add_argument('--shadow_percentile', type=float, default=50.0)
    # Percentile used to derive membership decision thresholds from shadow losses.

    parser.add_argument('--poly-mod-degree', type=int, default=16_384)
    # CKKS polynomial modulus degree (affects slot capacity & security).

    parser.add_argument('--coeff-mod-bit-sizes', type=int, nargs='+', default=[60, 40, 60])
    # Coefficient-modulus chain bit-sizes for CKKS (precision & noise budget).

    parser.add_argument('--scale-bits', type=int, default=40)
    # Number of bits used for CKKS scaling factor (encoding precision).

    parser.add_argument('--iid', action='store_true')
    # If set, use IID data partitioning across clients.

    parser.add_argument('--dirichlet-alpha', type=float, default=0.3)
    # Dirichlet concentration alpha for non-IID partitioning (ignored if --iid is set).

    # DP controls
    parser.add_argument('--dp-mode', type=str, choices=['none', 'client', 'server'], default='client')
    # DP mode: 'none' disables DP, 'client' adds local DP noise before encryption, 'server' adds encrypted noise at server.

    parser.add_argument('--dp-clip', type=float, default=1.0)
    # L2 clipping bound for per-client weighted updates before adding DP noise.

    parser.add_argument('--dp-sigma', type=float, default=0.1)
    # Gaussian noise multiplier for DP: std = dp_sigma * dp_clip.

    parser.add_argument('--cpu', action='store_true')
    # Force CPU-only execution (disable GPU usage).

    parser.add_argument('--seed', type=int, default=42)
    # RNG seed for reproducibility (partitions, training, DP randomness, etc.).

    args, unknown = parser.parse_known_args()
    # Parse known args and collect any unknown extras into `unknown` (useful in wrapped contexts).

    # default: enable all attacks for quick comparison (can be disabled by user flags)
    args.attack_inversion = args.attack_inversion or True
    # Ensure inversion attack is enabled by default (True unless explicitly set otherwise).

    args.attack_membership = args.attack_membership or True
    # Ensure membership attack is enabled by default.

    args.attack_membership_shadow = args.attack_membership_shadow or True
    # Ensure shadow-model membership attack is enabled by default.

    args.attack_probing = args.attack_probing or True
    # Ensure probing attack is enabled by default.

    args.attack_collusion = args.attack_collusion or True
    # Ensure collusion attack is enabled by default.

    args.attack_timing = args.attack_timing or True
    # Ensure timing/size-leakage attack is enabled by default.

    run_privacy_tests(args)
