"""
FL + CKKS (TenSEAL) MNIST — Comprehensive HE Threat Model & Attacks (Fixed)
----------------------------------------------------------------------------
- Fixes gradient inversion autograd error (no grad_fn) by keeping graph intact
  and ensuring parameters have requires_grad=True after parameter loads.
- Uses `create_graph=True` in autograd to allow gradients-of-gradients so the
  inversion attack can backpropagate into the reconstructed input.
- Fixes logging bug using loop var `r` instead of Python builtin `round()`.
- Adds missing additive sharing helper for collusion simulation.

Run (example quick):
  python fl_ckks_threat_model_full_fixed.py --clients 8 --rounds 3 --attack-round 2 --iid --outdir ./runs/he_threat



import argparse
import json
import os
import random
import time
from pathlib import Path
from collections import defaultdict

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
    for unit in ["B","KB","MB","GB"]:
        if num < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} TB"


# ----------------------------
# Model & data
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
# Flatten helpers
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
    # ensure grads are enabled after in-place copy
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
# Train / eval helpers
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
# HE helpers
# ----------------------------

def make_ckks_context(poly_mod_degree=16_384, coeff_mod_bit_sizes=(60, 40, 60), scale=2**40):
    t0 = now_ms()
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, list(coeff_mod_bit_sizes))
    ctx.global_scale = scale
    secret_ctx = ctx
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
# Attacks & analyses
# ----------------------------

def compute_grad_vector(model: nn.Module, x: torch.Tensor, label: int, device: torch.device):
    model = model.to(device)
    model.eval()
    x = x.to(device)
    y = torch.tensor([label], device=device)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False, create_graph=False)
    flat = torch.cat([g.view(-1) for g in grads]).detach().cpu().float()
    return flat


def inversion_attack_from_aggregate(agg_update: np.ndarray, model_template: nn.Module, param_slice: tuple,
                                    img_shape=(1,28,28), steps=800, lr=0.05, restarts=3, device='cpu'):
    start, end = param_slice
    device = torch.device(device)
    target = torch.tensor(agg_update[start:end], dtype=torch.float32, device=device)

    best = {'label': None, 'img': None, 'loss': float('inf')}

    for restart in range(restarts):
        for lab in range(10):
            x = torch.rand((1,)+img_shape, device=device, requires_grad=True)
            opt = torch.optim.Adam([x], lr=lr)
            model = model_template.to(device)
            # ensure model params require grad
            for p in model.parameters():
                p.requires_grad_(True)
            set_model_from_vector(model, get_model_vector(model))
            for i in range(steps):
                opt.zero_grad()
                logits = model(x)
                loss_ce = nn.CrossEntropyLoss()(logits, torch.tensor([lab], device=device))
                # IMPORTANT: keep graph; do NOT detach here
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
    # collect losses for members (shadow_train) and non-members (shadow_test)
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
    # attacker crafts sparse marker and injects into first client's update
    vec0 = weighted_vecs[0]
    marker = np.zeros_like(vec0)
    idxs = np.random.choice(len(vec0), size=10, replace=False)
    marker[idxs] = marker_scale
    marked = [v.copy() for v in weighted_vecs]
    marked[0] = marked[0] + marker
    # encrypt and decrypt both aggregates
    blobs_orig = [serialize_ciphertexts(encrypt_update_chunks(secret_ctx, v.astype(np.float64), max_slots)) for v in weighted_vecs]
    blobs_marked = [serialize_ciphertexts(encrypt_update_chunks(secret_ctx, v.astype(np.float64), max_slots)) for v in marked]
    # reconstruct decrypt
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
    # Decrypt
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
# Collusion helper (missing in original)
# ----------------------------

def make_additive_shares(vec: np.ndarray, n: int):
    #Return n shares that sum (elementwise) to vec.
    shares = [np.random.normal(0, 1e-6, size=vec.shape) for _ in range(n-1)]
    last = vec - np.sum(shares, axis=0)
    shares.append(last)
    return shares


# ----------------------------
# Orchestration
# ----------------------------


def run_full_threat_model(args):
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

        # local training
        local_models = []
        for cid in selected:
            model_copy = MLP().to(device)
            set_model_from_vector(model_copy, global_vec)
            trained = train_local(model_copy, client_loaders[cid], device, args.local_epochs, args.lr)
            local_models.append(trained)

        deltas = []
        client_sizes = []
        for model, cid in zip(local_models, selected):
            vec_new = get_model_vector(model)
            d = model_delta(global_vec, vec_new)
            deltas.append(d)
            client_sizes.append(len(client_loaders[cid].dataset))
        client_sizes = np.asarray(client_sizes, dtype=np.float64)
        weights = client_sizes / client_sizes.sum()

        weighted = [w * d for w, d in zip(weights, deltas)]

        # metadata leakage test: timings & sizes
        if args.attack_timing:
            enc_times = []
            lengths = []
            for v in weighted:
                t0 = now_ms()
                ct_chunks = encrypt_update_chunks(secret_ctx, v.astype(np.float64), max_slots)
                enc_times.append(now_ms() - t0)
                blobs = serialize_ciphertexts(ct_chunks)
                lengths.append(sum(len(b) for b in blobs))
            # correlations
            norms = [float(np.linalg.norm(v)) for v in weighted]
            if len(norms) > 1:
                corr_time = np.corrcoef(norms, enc_times)[0,1]
                corr_len = np.corrcoef(norms, lengths)[0,1]
            else:
                corr_time = float('nan')
                corr_len = float('nan')
            logs['timing'].append({'round': r, 'corr_time_norm': float(corr_time), 'corr_len_norm': float(corr_len)})
            # FIX: use r instead of built-in round
            print(f"[Round {r}] timing leakage corr(norm,enc_time)={corr_time:.4f} corr(norm,len)={corr_len:.4f}")
            success = (abs(corr_time) > 0.3 or abs(corr_len) > 0.3)
            print(f"timing leakage => {'SUCCESS' if success else 'FAIL'}")

        # HE aggregation path (simulate normal operation)
        client_blobs = []
        for v in weighted:
            ct_chunks = encrypt_update_chunks(secret_ctx, v.astype(np.float64), max_slots)
            client_blobs.append(serialize_ciphertexts(ct_chunks))

        # server aggregates ciphertexts
        num_chunks = len(client_blobs[0])
        agg_chunks = []
        for ch_idx in range(num_chunks):
            ct_sum = ts.ckks_vector_from(public_ctx, client_blobs[0][ch_idx])
            for client_idx in range(1, len(client_blobs)):
                ct_sum += ts.ckks_vector_from(public_ctx, client_blobs[client_idx][ch_idx])
            agg_chunks.append(ct_sum)

        # decrypt aggregate (simulate key compromise)
        agg_update = decrypt_concat(secret_ctx, agg_chunks)

        # apply update
        global_vec = apply_delta(global_vec, agg_update)
        set_model_from_vector(global_model, global_vec)

        acc, loss = evaluate(global_model, test_loader, device)
        print(f"utility: acc={acc:.4f} loss={loss:.4f}")

        # ATTACK: inversion
        if args.attack_inversion and r == args.attack_round:
            print("Running gradient inversion attack (multi-restarts)...")
            slice_idx = args.attack_slice_idx if args.attack_slice_idx is not None else 0
            s = param_slices[slice_idx]
            inv = inversion_attack_from_aggregate(agg_update, MLP(), s, img_shape=(1,28,28), steps=args.attack_steps, lr=args.attack_lr, restarts=args.attack_restarts, device=device)
            save_image(inv['img'].squeeze(0), os.path.join(args.outdir, f"r{r}_inv_label{inv['label']}.png"))
            mse = float(np.mean((inv['img'].squeeze(0) - client_loaders[selected[0]].dataset[0][0].numpy())**2))
            p = psnr(inv['img'].squeeze(0), client_loaders[selected[0]].dataset[0][0].numpy())
            logs['inversion'].append({'round': r, 'label_guess': int(inv['label']), 'loss': float(inv['loss']), 'mse': float(mse), 'psnr': float(p)})
            print(f"inversion: label_guess={inv['label']} mse={mse:.6e} psnr={p:.2f}dB")

        # ATTACK: membership inference (loss-threshold)
        if args.attack_membership and r == args.attack_round:
            print("Running membership inference (loss-threshold)...")
            target_client = args.attack_client if args.attack_client is not None else int(selected[0])
            ds = list(data.Subset(train_ds, parts[target_client]))
            x_true, y_true = ds[0][0].numpy(), int(ds[0][1])
            inferred, loss_val = membership_inference_loss_threshold(global_model, x_true, y_true, args.membership_threshold, device=device)
            logs['membership'].append({'round': r, 'target_client': int(target_client), 'loss': float(loss_val), 'inferred_member': bool(inferred)})
            print(f"membership (threshold) loss={loss_val:.4f} inferred={inferred}")

        # ATTACK: membership (shadow-model)
        if args.attack_membership_shadow and r == args.attack_round:
            print("Running membership inference (shadow-model)...\n")
            # build small shadow split — take small subsets for speed
            shadow_idxs = np.random.choice(len(train_ds), size=min(2000, len(train_ds)), replace=False)
            half = len(shadow_idxs)//2
            shadow_train = data.Subset(train_ds, shadow_idxs[:half])
            shadow_test = data.Subset(train_ds, shadow_idxs[half:])
            mem_losses, nonmem_losses = train_shadow_model(shadow_train, shadow_test, MLP(), device, epochs=args.shadow_epochs, lr=args.shadow_lr)
            # simple threshold from shadow: choose percentile on member losses
            thr = float(np.percentile(mem_losses, args.shadow_percentile))
            # evaluate on target sample
            target_client = args.attack_client if args.attack_client is not None else int(selected[0])
            ds = list(data.Subset(train_ds, parts[target_client]))
            x_true, y_true = ds[0][0].numpy(), int(ds[0][1])
            inferred, loss_val = membership_inference_loss_threshold(global_model, x_true, y_true, thr, device=device)
            logs['membership_shadow'].append({'round': r, 'threshold': thr, 'loss': float(loss_val), 'inferred_member': bool(inferred)})
            print(f"membership (shadow) thr={thr:.4f} loss={loss_val:.4f} inferred={inferred}")

        # ATTACK: probing client
        if args.attack_probing and r == args.attack_round:
            print("Running probing (marker) attack...")
            corr, marker, diff = probing_marker_attack(weighted, secret_ctx, public_ctx, max_slots, marker_scale=args.probe_scale)
            logs['probing'].append({'round': r, 'corr': float(corr)})
            print(f"probing correlation={corr:.4f}")

        # ATTACK: collusion (SMPC-style) — informative comparison
        if args.attack_collusion and r == args.attack_round:
            print("Running collusion simulation (additive sharing reveal)...")
            n_agg = max(2, int(args.n_aggregators))
            k = min(n_agg, args.collusion_k)
            all_shares = []
            for v in weighted:
                shares = make_additive_shares(v, n_agg)
                all_shares.append(shares)
            reconstructed_clients = []
            for ci in range(len(weighted)):
                coll_sum = sum(all_shares[ci][j] for j in range(k))
                approx = coll_sum
                reconstructed_clients.append(approx)
            rmses = [float(np.sqrt(np.mean((reconstructed_clients[i] - weighted[i])**2))) for i in range(len(weighted))]
            avg_rmse = float(np.mean(rmses))
            logs['collusion'].append({'round': r, 'k': k, 'avg_rmse': avg_rmse})
            print(f"collusion k={k} avg_rmse={avg_rmse:.6e}")
            success = avg_rmse < 0.1
            print(f"Collusion=> {'SUCCESS' if success else 'FAIL'}")

    # save logs
    outp = os.path.join(args.outdir, 'threat_model_full_logs.json')
    with open(outp, 'w') as f:
        json.dump({k: v for k, v in logs.items()}, f, indent=2)
    print(f"Finished. Logs saved to {outp}")


# ----------------------------
# CLI
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive HE threat-model harness (MNIST + CKKS) [Fixed]')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='./runs/ckks_threat_full')
    parser.add_argument('--clients', type=int, default=8)
    parser.add_argument('--participation', type=float, default=1.0)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)

    parser.add_argument('--attack-round', type=int, default=5)
    parser.add_argument('--attack-all', action='store_true')
    parser.add_argument('--attack_inversion', action='store_true')
    parser.add_argument('--attack_membership', action='store_true')
    parser.add_argument('--attack_membership_shadow', action='store_true')
    parser.add_argument('--attack_probing', action='store_true')
    parser.add_argument('--attack_collusion', action='store_true')
    parser.add_argument('--attack_timing', action='store_true')

    parser.add_argument('--attack_steps', type=int, default=800)
    parser.add_argument('--attack_lr', type=float, default=0.05)
    parser.add_argument('--attack_restarts', type=int, default=2)
    parser.add_argument('--attack_slice_idx', type=int, default=0)
    parser.add_argument('--attack_client', type=int, default=None)

    parser.add_argument('--membership_threshold', type=float, default=0.5)
    parser.add_argument('--probe_scale', type=float, default=1e-2)
    parser.add_argument('--n_aggregators', type=int, default=3)
    parser.add_argument('--collusion_k', type=int, default=1)

    # shadow model params
    parser.add_argument('--shadow_epochs', type=int, default=3)
    parser.add_argument('--shadow_lr', type=float, default=0.1)
    parser.add_argument('--shadow_percentile', type=float, default=50.0)

    parser.add_argument('--poly-mod-degree', type=int, default=16_384)
    parser.add_argument('--coeff-mod-bit-sizes', type=int, nargs='+', default=[60, 40, 60])
    parser.add_argument('--scale-bits', type=int, default=40)

    parser.add_argument('--iid', action='store_true')
    parser.add_argument('--dirichlet-alpha', type=float, default=0.3)

    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args, unknown = parser.parse_known_args()

    # enable all attacks if user asked (force-on here for demo)
    args.attack_inversion = True
    args.attack_membership = True
    args.attack_membership_shadow = True
    args.attack_probing = True
    args.attack_collusion = True
    args.attack_timing = True

    run_full_threat_model(args)

    log_file = os.path.join(args.outdir, 'threat_model_full_logs.json')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)

        print("\n=== Threat Model Attack Summary ===")
        # Gradient Inversion
        if "inversion" in logs:
            for entry in logs["inversion"]:
                success = entry["psnr"] > 20
                print(f"[Round {entry['round']}] Inversion: "
                      f"label_guess={entry['label_guess']} "
                      f"MSE={entry['mse']:.4e} PSNR={entry['psnr']:.2f}dB "
                      f"=> {'SUCCESS' if success else 'FAIL'}")

        # Membership Inference (threshold)
        if "membership" in logs:
            for entry in logs["membership"]:
                success = entry["inferred_member"]
                print(f"[Round {entry['round']}] Membership (threshold): "
                      f"loss={entry['loss']:.4f} inferred={entry['inferred_member']} "
                      f"=> {'SUCCESS' if success else 'FAIL'}")

        # Membership Inference (shadow model)
        if "membership_shadow" in logs:
            for entry in logs["membership_shadow"]:
                success = entry["inferred_member"]
                print(f"[Round {entry['round']}] Membership (shadow): "
                      f"thr={entry['threshold']:.4f} loss={entry['loss']:.4f} "
                      f"inferred={entry['inferred_member']} "
                      f"=> {'SUCCESS' if success else 'FAIL'}")

        # Probing
        if "probing" in logs:
            for entry in logs["probing"]:
                success = entry["corr"] > 0.5
                print(f"[Round {entry['round']}] Probing: "
                      f"correlation={entry['corr']:.4f} "
                      f"=> {'SUCCESS' if success else 'FAIL'}")

        # Collusion
        if "collusion" in logs:
            for entry in logs["collusion"]:
                success = entry["avg_rmse"] < 0.1
                print(f"[Round {entry['round']}] Collusion: k={entry['k']} "
                      f"avg_RMSE={entry['avg_rmse']:.4e} "
                      f"=> {'SUCCESS' if success else 'FAIL'}")

        # Timing Leakage
        if "timing" in logs:
            for entry in logs["timing"]:
                success = (abs(entry["corr_time_norm"]) > 0.3 or
                           abs(entry["corr_len_norm"]) > 0.3)
                print(f"[Round {entry['round']}] Timing leakage: "
                      f"corr_time_norm={entry['corr_time_norm']:.4f}, "
                      f"corr_len_norm={entry['corr_len_norm']:.4f} "
                      f"=> {'SUCCESS' if success else 'FAIL'}")

    else:
        print("No logs found.")
"""

"""
FL + CKKS (TenSEAL) MNIST — Comprehensive HE Threat Model & Attacks (Fixed)
----------------------------------------------------------------------------
- Fixes gradient inversion autograd error (no grad_fn) by keeping graph intact
  and ensuring parameters have requires_grad=True after parameter loads.
- Uses `create_graph=True` in autograd to allow gradients-of-gradients so the
  inversion attack can backpropagate into the reconstructed input.
- Fixes logging bug using loop var `r` instead of Python builtin `round()`.
- Adds missing additive sharing helper for collusion simulation.

Run (example quick):
  python fl_ckks_threat_model_full_fixed.py --clients 8 --rounds 3 --attack-round 2 --iid --outdir ./runs/he_threat

"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from collections import defaultdict

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
    for unit in ["B","KB","MB","GB"]:
        if num < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} TB"


# ----------------------------
# Model & data
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
# Flatten helpers
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
    # ensure grads are enabled after in-place copy
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
# Train / eval helpers
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
# HE helpers
# ----------------------------

def make_ckks_context(poly_mod_degree=16_384, coeff_mod_bit_sizes=(60, 40, 60), scale=2**40):
    t0 = now_ms()
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, list(coeff_mod_bit_sizes))
    ctx.global_scale = scale
    secret_ctx = ctx
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
# Attacks & analyses
# ----------------------------

def compute_grad_vector(model: nn.Module, x: torch.Tensor, label: int, device: torch.device):
    model = model.to(device)
    model.eval()
    x = x.to(device)
    y = torch.tensor([label], device=device)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False, create_graph=False)
    flat = torch.cat([g.view(-1) for g in grads]).detach().cpu().float()
    return flat


def inversion_attack_from_aggregate(agg_update: np.ndarray, model_template: nn.Module, param_slice: tuple,
                                    img_shape=(1,28,28), steps=800, lr=0.05, restarts=3, device='cpu'):
    start, end = param_slice
    device = torch.device(device)
    target = torch.tensor(agg_update[start:end], dtype=torch.float32, device=device)

    best = {'label': None, 'img': None, 'loss': float('inf')}

    for restart in range(restarts):
        for lab in range(10):
            x = torch.rand((1,)+img_shape, device=device, requires_grad=True)
            opt = torch.optim.Adam([x], lr=lr)
            model = model_template.to(device)
            # ensure model params require grad
            for p in model.parameters():
                p.requires_grad_(True)
            set_model_from_vector(model, get_model_vector(model))
            for i in range(steps):
                opt.zero_grad()
                logits = model(x)
                loss_ce = nn.CrossEntropyLoss()(logits, torch.tensor([lab], device=device))
                # IMPORTANT: keep graph; do NOT detach here
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
    # collect losses for members (shadow_train) and non-members (shadow_test)
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
    # attacker crafts sparse marker and injects into first client's update
    vec0 = weighted_vecs[0]
    marker = np.zeros_like(vec0)
    idxs = np.random.choice(len(vec0), size=10, replace=False)
    marker[idxs] = marker_scale
    marked = [v.copy() for v in weighted_vecs]
    marked[0] = marked[0] + marker
    # encrypt and decrypt both aggregates
    blobs_orig = [serialize_ciphertexts(encrypt_update_chunks(secret_ctx, v.astype(np.float64), max_slots)) for v in weighted_vecs]
    blobs_marked = [serialize_ciphertexts(encrypt_update_chunks(secret_ctx, v.astype(np.float64), max_slots)) for v in marked]
    # reconstruct decrypt
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
    # Decrypt
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
# Collusion helper (missing in original)
# ----------------------------

def make_additive_shares(vec: np.ndarray, n: int):
    """Return n shares that sum (elementwise) to vec."""
    shares = [np.random.normal(0, 1e-6, size=vec.shape) for _ in range(n-1)]
    last = vec - np.sum(shares, axis=0)
    shares.append(last)
    return shares


# ----------------------------
# Orchestration
# ----------------------------


def run_full_threat_model(args):
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

        # local training
        local_models = []
        for cid in selected:
            model_copy = MLP().to(device)
            set_model_from_vector(model_copy, global_vec)
            trained = train_local(model_copy, client_loaders[cid], device, args.local_epochs, args.lr)
            local_models.append(trained)

        deltas = []
        client_sizes = []
        for model, cid in zip(local_models, selected):
            vec_new = get_model_vector(model)
            d = model_delta(global_vec, vec_new)
            deltas.append(d)
            client_sizes.append(len(client_loaders[cid].dataset))
        client_sizes = np.asarray(client_sizes, dtype=np.float64)
        weights = client_sizes / client_sizes.sum()

        weighted = [w * d for w, d in zip(weights, deltas)]

        # metadata leakage test: timings & sizes
        if args.attack_timing:
            enc_times = []
            lengths = []
            for v in weighted:
                t0 = now_ms()
                ct_chunks = encrypt_update_chunks(secret_ctx, v.astype(np.float64), max_slots)
                enc_times.append(now_ms() - t0)
                blobs = serialize_ciphertexts(ct_chunks)
                lengths.append(sum(len(b) for b in blobs))
            # correlations
            norms = [float(np.linalg.norm(v)) for v in weighted]
            if len(norms) > 1:
                corr_time = np.corrcoef(norms, enc_times)[0,1]
                corr_len = np.corrcoef(norms, lengths)[0,1]
            else:
                corr_time = float('nan')
                corr_len = float('nan')
            logs['timing'].append({'round': r, 'corr_time_norm': float(corr_time), 'corr_len_norm': float(corr_len)})
            # FIX: use r instead of built-in round
            print(f"[Round {r}] timing leakage corr(norm,enc_time)={corr_time:.4f} corr(norm,len)={corr_len:.4f}")
            success = (abs(corr_time) > 0.3 or abs(corr_len) > 0.3)
            print(f"timing leakage => {'SUCCESS' if success else 'FAIL'}")

        # HE aggregation path (simulate normal operation)
        client_blobs = []
        for v in weighted:
            ct_chunks = encrypt_update_chunks(secret_ctx, v.astype(np.float64), max_slots)
            client_blobs.append(serialize_ciphertexts(ct_chunks))

        # server aggregates ciphertexts
        num_chunks = len(client_blobs[0])
        agg_chunks = []
        for ch_idx in range(num_chunks):
            ct_sum = ts.ckks_vector_from(public_ctx, client_blobs[0][ch_idx])
            for client_idx in range(1, len(client_blobs)):
                ct_sum += ts.ckks_vector_from(public_ctx, client_blobs[client_idx][ch_idx])
            agg_chunks.append(ct_sum)

        # decrypt aggregate (simulate key compromise)
        agg_update = decrypt_concat(secret_ctx, agg_chunks)

        # apply update
        global_vec = apply_delta(global_vec, agg_update)
        set_model_from_vector(global_model, global_vec)

        acc, loss = evaluate(global_model, test_loader, device)
        print(f"utility: acc={acc:.4f} loss={loss:.4f}")

        # ATTACK: inversion
        if args.attack_inversion and r == args.attack_round:
            print("Running gradient inversion attack (multi-restarts)...")
            slice_idx = args.attack_slice_idx if args.attack_slice_idx is not None else 0
            s = param_slices[slice_idx]
            inv = inversion_attack_from_aggregate(agg_update, MLP(), s, img_shape=(1,28,28), steps=args.attack_steps, lr=args.attack_lr, restarts=args.attack_restarts, device=device)
            # ALSO save reconstructed image to a fixed name: inversion.png
            save_image(inv['img'].squeeze(0), os.path.join(args.outdir, "inversion.png"))
            # save original that attacker is trying to reconstruct (undo normalization)
            orig_tensor = client_loaders[selected[0]].dataset[0][0].numpy()
            orig_img = (orig_tensor * 0.3081) + 0.1307
            save_image(orig_img, os.path.join(args.outdir, "original.png"))
            mse = float(np.mean((inv['img'].squeeze(0) - client_loaders[selected[0]].dataset[0][0].numpy())**2))
            p = psnr(inv['img'].squeeze(0), client_loaders[selected[0]].dataset[0][0].numpy())
            logs['inversion'].append({'round': r, 'label_guess': int(inv['label']), 'loss': float(inv['loss']), 'mse': float(mse), 'psnr': float(p)})
            print(f"inversion: label_guess={inv['label']} mse={mse:.6e} psnr={p:.2f}dB")

        # ATTACK: membership inference (loss-threshold)
        if args.attack_membership and r == args.attack_round:
            print("Running membership inference (loss-threshold)...")
            target_client = args.attack_client if args.attack_client is not None else int(selected[0])
            ds = list(data.Subset(train_ds, parts[target_client]))
            x_true, y_true = ds[0][0].numpy(), int(ds[0][1])
            inferred, loss_val = membership_inference_loss_threshold(global_model, x_true, y_true, args.membership_threshold, device=device)
            logs['membership'].append({'round': r, 'target_client': int(target_client), 'loss': float(loss_val), 'inferred_member': bool(inferred)})
            print(f"membership (threshold) loss={loss_val:.4f} inferred={inferred}")

        # ATTACK: membership (shadow-model)
        if args.attack_membership_shadow and r == args.attack_round:
            print("Running membership inference (shadow-model)...\n")
            # build small shadow split — take small subsets for speed
            shadow_idxs = np.random.choice(len(train_ds), size=min(2000, len(train_ds)), replace=False)
            half = len(shadow_idxs)//2
            shadow_train = data.Subset(train_ds, shadow_idxs[:half])
            shadow_test = data.Subset(train_ds, shadow_idxs[half:])
            mem_losses, nonmem_losses = train_shadow_model(shadow_train, shadow_test, MLP(), device, epochs=args.shadow_epochs, lr=args.shadow_lr)
            # simple threshold from shadow: choose percentile on member losses
            thr = float(np.percentile(mem_losses, args.shadow_percentile))
            # evaluate on target sample
            target_client = args.attack_client if args.attack_client is not None else int(selected[0])
            ds = list(data.Subset(train_ds, parts[target_client]))
            x_true, y_true = ds[0][0].numpy(), int(ds[0][1])
            inferred, loss_val = membership_inference_loss_threshold(global_model, x_true, y_true, thr, device=device)
            logs['membership_shadow'].append({'round': r, 'threshold': thr, 'loss': float(loss_val), 'inferred_member': bool(inferred)})
            print(f"membership (shadow) thr={thr:.4f} loss={loss_val:.4f} inferred={inferred}")

        # ATTACK: probing client
        if args.attack_probing and r == args.attack_round:
            print("Running probing (marker) attack...")
            corr, marker, diff = probing_marker_attack(weighted, secret_ctx, public_ctx, max_slots, marker_scale=args.probe_scale)
            logs['probing'].append({'round': r, 'corr': float(corr)})
            print(f"probing correlation={corr:.4f}")

        # ATTACK: collusion (SMPC-style) — informative comparison
        if args.attack_collusion and r == args.attack_round:
            print("Running collusion simulation (additive sharing reveal)...")
            n_agg = max(2, int(args.n_aggregators))
            k = min(n_agg, args.collusion_k)
            all_shares = []
            for v in weighted:
                shares = make_additive_shares(v, n_agg)
                all_shares.append(shares)
            reconstructed_clients = []
            for ci in range(len(weighted)):
                coll_sum = sum(all_shares[ci][j] for j in range(k))
                approx = coll_sum
                reconstructed_clients.append(approx)
            rmses = [float(np.sqrt(np.mean((reconstructed_clients[i] - weighted[i])**2))) for i in range(len(weighted))]
            avg_rmse = float(np.mean(rmses))
            logs['collusion'].append({'round': r, 'k': k, 'avg_rmse': avg_rmse})
            print(f"collusion k={k} avg_rmse={avg_rmse:.6e}")
            success = avg_rmse < 0.1
            print(f"Collusion=> {'SUCCESS' if success else 'FAIL'}")

    # save logs
    outp = os.path.join(args.outdir, 'threat_model_full_logs.json')
    with open(outp, 'w') as f:
        json.dump({k: v for k, v in logs.items()}, f, indent=2)
    print(f"Finished. Logs saved to {outp}")


# ----------------------------
# CLI
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive HE threat-model harness (MNIST + CKKS) [Fixed]')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='./runs/ckks_threat_full')
    parser.add_argument('--clients', type=int, default=8)
    parser.add_argument('--participation', type=float, default=1.0)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)

    parser.add_argument('--attack-round', type=int, default=5)
    parser.add_argument('--attack-all', action='store_true')
    parser.add_argument('--attack_inversion', action='store_true')
    parser.add_argument('--attack_membership', action='store_true')
    parser.add_argument('--attack_membership_shadow', action='store_true')
    parser.add_argument('--attack_probing', action='store_true')
    parser.add_argument('--attack_collusion', action='store_true')
    parser.add_argument('--attack_timing', action='store_true')

    parser.add_argument('--attack_steps', type=int, default=800)
    parser.add_argument('--attack_lr', type=float, default=0.05)
    parser.add_argument('--attack_restarts', type=int, default=2)
    parser.add_argument('--attack_slice_idx', type=int, default=0)
    parser.add_argument('--attack_client', type=int, default=None)

    parser.add_argument('--membership_threshold', type=float, default=0.5)
    parser.add_argument('--probe_scale', type=float, default=1e-2)
    parser.add_argument('--n_aggregators', type=int, default=3)
    parser.add_argument('--collusion_k', type=int, default=1)

    # shadow model params
    parser.add_argument('--shadow_epochs', type=int, default=3)
    parser.add_argument('--shadow_lr', type=float, default=0.1)
    parser.add_argument('--shadow_percentile', type=float, default=50.0)

    parser.add_argument('--poly-mod-degree', type=int, default=16_384)
    parser.add_argument('--coeff-mod-bit-sizes', type=int, nargs='+', default=[60, 40, 60])
    parser.add_argument('--scale-bits', type=int, default=40)

    parser.add_argument('--iid', action='store_true')
    parser.add_argument('--dirichlet-alpha', type=float, default=0.3)

    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args, unknown = parser.parse_known_args()

    # enable all attacks if user asked (force-on here for demo)
    args.attack_inversion = True
    args.attack_membership = True
    args.attack_membership_shadow = True
    args.attack_probing = True
    args.attack_collusion = True
    args.attack_timing = True

    run_full_threat_model(args)

    log_file = os.path.join(args.outdir, 'threat_model_full_logs.json')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)

        print("\n=== Threat Model Attack Summary ===")
        # Gradient Inversion
        if "inversion" in logs:
            for entry in logs["inversion"]:
                success = entry["psnr"] > 20
                print(f"[Round {entry['round']}] Inversion: "
                      f"label_guess={entry['label_guess']} "
                      f"MSE={entry['mse']:.4e} PSNR={entry['psnr']:.2f}dB "
                      f"=> {'SUCCESS' if success else 'FAIL'}")

        # Membership Inference (threshold)
        if "membership" in logs:
            for entry in logs["membership"]:
                success = entry["inferred_member"]
                print(f"[Round {entry['round']}] Membership (threshold): "
                      f"loss={entry['loss']:.4f} inferred={entry['inferred_member']} "
                      f"=> {'SUCCESS' if success else 'FAIL'}")

        # Membership Inference (shadow model)
        if "membership_shadow" in logs:
            for entry in logs["membership_shadow"]:
                success = entry["inferred_member"]
                print(f"[Round {entry['round']}] Membership (shadow): "
                      f"thr={entry['threshold']:.4f} loss={entry['loss']:.4f} "
                      f"inferred={entry['inferred_member']} "
                      f"=> {'SUCCESS' if success else 'FAIL'}")

        # Probing
        if "probing" in logs:
            for entry in logs["probing"]:
                success = entry["corr"] > 0.5
                print(f"[Round {entry['round']}] Probing: "
                      f"correlation={entry['corr']:.4f} "
                      f"=> {'SUCCESS' if success else 'FAIL'}")

        # Collusion
        if "collusion" in logs:
            for entry in logs["collusion"]:
                success = entry["avg_rmse"] < 0.1
                print(f"[Round {entry['round']}] Collusion: k={entry['k']} "
                      f"avg_RMSE={entry['avg_rmse']:.4e} "
                      f"=> {'SUCCESS' if success else 'FAIL'}")

        # Timing Leakage
        if "timing" in logs:
            for entry in logs["timing"]:
                success = (abs(entry["corr_time_norm"]) > 0.3 or
                           abs(entry["corr_len_norm"]) > 0.3)
                print(f"[Round {entry['round']}] Timing leakage: "
                      f"corr_time_norm={entry['corr_time_norm']:.4f}, "
                      f"corr_len_norm={entry['corr_len_norm']:.4f} "
                      f"=> {'SUCCESS' if success else 'FAIL'}")

    else:
        print("No logs found.")
