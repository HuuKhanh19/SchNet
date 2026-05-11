"""
Phase 1.2 v2 - Extract CLS Embeddings (using UniMolV2Feature)

Fix for v1: use UniMolV2Feature (UniMol2 format) instead of ConformerGen
(which is UniMol1 format and needs unimol_tools constants).

UniMolV2Feature is defined in your conformer.py file. It does NOT need
Dictionary, MODEL_CONFIG, WEIGHT_DIR, weight_download — only RDKit + numba.

Run:
  python 02_extract_embeddings_v2.py \
      --unimol2-path C:\\path\\to\\Uni-Mol\\unimol2 \
      --conformer-py C:\\path\\to\\conformer.py \
      --checkpoint C:\\path\\to\\unimol2_570m_pretrain.pt \
      --batch-size 8

The --conformer-py path is the conformer.py file you have locally.
"""

import argparse
import importlib.util
import json
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
def setup_paths(unimol2_path, schnet_root):
    if unimol2_path:
        sys.path.insert(0, str(Path(unimol2_path).resolve()))
    if schnet_root:
        sys.path.insert(0, str(Path(schnet_root).resolve()))


def find_conformer_py(schnet_root, explicit=None):
    """Find conformer.py — explicit path or search common locations."""
    if explicit:
        p = Path(explicit).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"--conformer-py specified but not found: {p}")

    # Auto-search
    schnet_root = Path(schnet_root) if schnet_root else Path(".")
    candidates = [
        schnet_root / "conformer.py",
        schnet_root / "unimol2" / "conformer.py",
        schnet_root / "unimol2" / "data" / "conformer.py",
        schnet_root / "unimol_tools" / "conformer.py",
        Path(__file__).parent / "conformer.py",
        Path(__file__).parent.parent / "conformer.py",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def load_conformer_module(conformer_path):
    """Load conformer.py as a module from explicit file path."""
    spec = importlib.util.spec_from_file_location("user_conformer", str(conformer_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_conformer"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed loading {conformer_path}: {e}")
    if not hasattr(module, "UniMolV2Feature"):
        raise ImportError(f"{conformer_path} does not contain UniMolV2Feature class")
    return module


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def make_args_570M():
    a = types.SimpleNamespace()
    a.encoder_layers = 32
    a.encoder_embed_dim = 1536
    a.pair_embed_dim = 512
    a.pair_hidden_dim = 64
    a.encoder_ffn_embed_dim = 1536
    a.encoder_attention_heads = 96
    a.dropout = 0.0
    a.emb_dropout = 0.0
    a.attention_dropout = 0.0
    a.activation_dropout = 0.0
    a.pooler_dropout = 0.0
    a.max_seq_len = 512
    a.activation_fn = "gelu"
    a.pooler_activation_fn = "tanh"
    a.post_ln = False
    a.masked_token_loss = -1.0
    a.masked_coord_loss = -1.0
    a.masked_dist_loss = -1.0
    a.masked_coord_dist_loss = -1.0
    a.x_norm_loss = -1.0
    a.delta_pair_repr_norm_loss = -1.0
    a.notri = False
    a.gaussian_std_width = 1.0
    a.gaussian_mean_start = 0.0
    a.gaussian_mean_stop = 9.0
    a.droppath_prob = 0.0
    a.mode = "infer"
    return a


def load_model(checkpoint_path):
    UniMol2Model = None
    for module_path in ["unimol2.models.unimol2", "unimol2.unimol2.models.unimol2", "models.unimol2"]:
        try:
            mod = __import__(module_path, fromlist=["UniMol2Model"])
            UniMol2Model = mod.UniMol2Model
            print(f"Imported UniMol2Model from {module_path}")
            break
        except ImportError:
            continue
    if UniMol2Model is None:
        raise ImportError("Cannot find UniMol2Model.")

    args = make_args_570M()
    model = UniMol2Model(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Constructed UniMol2-570M: {n_params/1e6:.1f}M params")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        state_dict = state.get("model") or state.get("state_dict") or state
    else:
        state_dict = state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {len(state_dict) - len(unexpected)} matched, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    return model


# ---------------------------------------------------------------------------
# Conformer generation via UniMolV2Feature
# ---------------------------------------------------------------------------
def generate_features_unimolv2(smiles_list, conformer_module, cache_path,
                                  max_atoms=128, force_regenerate=False):
    """
    Use UniMolV2Feature to generate features. Output: list of dicts (one per
    molecule) with keys: atom_feat, atom_mask, edge_feat, shortest_path, degree,
    pair_type, attn_bias, src_tokens, src_coord.
    """
    if cache_path.exists() and not force_regenerate:
        print(f"  Loading cached features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"  Generating features for {len(smiles_list)} molecules via UniMolV2Feature...")
    feat_gen = conformer_module.UniMolV2Feature(
        seed=42,
        max_atoms=max_atoms,
        method="rdkit_random",
        mode="heavy",  # better quality conformers
        remove_hs=True,
        multi_process=False,  # safer on Windows
    )
    inputs, mols = feat_gen.transform(smiles_list)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(inputs, f)
    print(f"  Cached to {cache_path}")
    return inputs


# ---------------------------------------------------------------------------
# Collate: UniMolV2Feature output -> UniMol2 forward batched_data
# ---------------------------------------------------------------------------
def collate_unimol2(feat_list, device, dtype=torch.bfloat16):
    """
    UniMolV2Feature output keys:
        src_tokens (list of atomic numbers, len N)
        src_coord (N, 3) np.float32
        atom_feat (N, 8) np.int32
        atom_mask (N,) np.int64
        edge_feat (N, N, 3) np.int32
        shortest_path (N, N) np.int32
        degree (N,) np.int32
        pair_type (N, N, 2) np.int32
        attn_bias (N+1, N+1) np.float32

    UniMol2 forward expects:
        src_token (B, N)             [src_tokens -> src_token]
        src_pos (B, N, 3)            [src_coord  -> src_pos]
        atom_feat (B, N, 8)
        atom_mask (B, N)
        edge_feat (B, N, N, 3)
        shortest_path (B, N, N)
        degree (B, N)
        pair_type (B, N, N, 2)
        attn_bias (B, N+1, N+1)
    """
    B = len(feat_list)
    max_atoms = max(np.asarray(f["atom_mask"]).shape[0] for f in feat_list)

    def pad_1d(arr, target_n, pad_val=0, np_dtype=None):
        arr = np.asarray(arr)
        if np_dtype is not None and arr.dtype != np_dtype:
            arr = arr.astype(np_dtype)
        n = arr.shape[0]
        if n >= target_n:
            return arr[:target_n]
        pad_shape = (target_n - n,) + arr.shape[1:]
        return np.concatenate([arr, np.full(pad_shape, pad_val, dtype=arr.dtype)])

    def pad_2d(arr, target_n, pad_val=0):
        arr = np.asarray(arr)
        n = arr.shape[0]
        if n >= target_n:
            return arr[:target_n, :target_n]
        out_shape = (target_n, target_n) + arr.shape[2:]
        out = np.full(out_shape, pad_val, dtype=arr.dtype)
        out[:n, :n] = arr
        return out

    # Pad and stack
    src_tokens = np.stack([pad_1d(f["src_tokens"], max_atoms, np_dtype=np.int64) for f in feat_list])
    src_coord = np.stack([pad_1d(f["src_coord"], max_atoms, 0.0) for f in feat_list])
    atom_feat = np.stack([pad_1d(f["atom_feat"], max_atoms) for f in feat_list])
    atom_mask = np.stack([pad_1d(f["atom_mask"], max_atoms) for f in feat_list])
    pair_type = np.stack([pad_2d(f["pair_type"], max_atoms) for f in feat_list])
    edge_feat = np.stack([pad_2d(f["edge_feat"], max_atoms) for f in feat_list])
    shortest_path = np.stack([pad_2d(f["shortest_path"], max_atoms) for f in feat_list])
    degree = np.stack([pad_1d(f["degree"], max_atoms) for f in feat_list])

    # attn_bias has shape (N+1, N+1) per sample, pad to (max_atoms+1, max_atoms+1)
    attn_bias_list = []
    for f in feat_list:
        ab = np.asarray(f["attn_bias"])  # (N+1, N+1)
        n = ab.shape[0] - 1  # actual atom count
        if max_atoms + 1 > ab.shape[0]:
            pad = max_atoms + 1 - ab.shape[0]
            ab = np.pad(ab, ((0, pad), (0, pad)), mode='constant')
        else:
            ab = ab[:max_atoms + 1, :max_atoms + 1]
        attn_bias_list.append(ab)
    attn_bias = np.stack(attn_bias_list)

    return {
        "src_token": torch.from_numpy(src_tokens).long().to(device),
        "src_pos": torch.from_numpy(src_coord).to(dtype).to(device),
        "atom_feat": torch.from_numpy(atom_feat).long().to(device),
        "atom_mask": torch.from_numpy(atom_mask).long().to(device),
        "pair_type": torch.from_numpy(pair_type).long().to(device),
        "edge_feat": torch.from_numpy(edge_feat).long().to(device),
        "shortest_path": torch.from_numpy(shortest_path).long().to(device),
        "degree": torch.from_numpy(degree).long().to(device),
        "attn_bias": torch.from_numpy(attn_bias).to(dtype).to(device),
    }


# ---------------------------------------------------------------------------
# Forward + extract CLS
# ---------------------------------------------------------------------------
def extract_embeddings(model, feat_list, batch_size, device):
    model.eval()
    all_emb = []
    n = len(feat_list)
    n_batches = (n + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n)
            batch_feats = feat_list[start:end]
            try:
                batch = collate_unimol2(batch_feats, device, dtype=torch.bfloat16)
                out = model(batch)
                if isinstance(out, tuple):
                    x, _pair = out
                else:
                    x = out
                cls = x[:, 0, :].float().cpu().numpy()
                all_emb.append(cls)
            except Exception as e:
                print(f"    [WARN] batch {i+1}/{n_batches} failed: {e}")
                # fill with zeros to keep alignment
                cls = np.zeros((end - start, 1536), dtype=np.float32)
                all_emb.append(cls)

            if i % 10 == 0 or i == n_batches - 1:
                print(f"    batch {i+1}/{n_batches} done")

    return np.concatenate(all_emb, axis=0)


# ---------------------------------------------------------------------------
# Filter failed conformers (where src_coord is all zeros)
# ---------------------------------------------------------------------------
def filter_failed(feat_list, y_list):
    """Remove molecules where conformer generation failed (src_coord all zeros)."""
    keep_idx = []
    for i, f in enumerate(feat_list):
        sc = np.asarray(f["src_coord"])
        if not np.all(sc == 0.0):
            keep_idx.append(i)
    keep = np.asarray(keep_idx, dtype=np.int64)
    n_failed = len(feat_list) - len(keep)
    if n_failed > 0:
        print(f"  Filtering {n_failed} failed conformers ({n_failed/len(feat_list)*100:.1f}%)")
    return [feat_list[i] for i in keep], y_list[keep], keep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unimol2-path", type=str, default=None)
    parser.add_argument("--schnet-root", type=str,
                        default=r"C:\Users\BKAI\ducluong\DrugOptimization\SchNet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--conformer-py", type=str, default=None,
                        help="Path to conformer.py file (auto-searched if omitted)")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--target-col", type=str, default="measured")
    parser.add_argument("--max-atoms", type=int, default=128)
    parser.add_argument("--force-regenerate", action="store_true")
    args = parser.parse_args()

    setup_paths(args.unimol2_path, args.schnet_root)

    # Find and load conformer.py
    conf_path = find_conformer_py(args.schnet_root, args.conformer_py)
    if conf_path is None:
        print("[FAIL] conformer.py not found. Common locations searched:")
        print(f"  {args.schnet_root}\\conformer.py")
        print(f"  {args.schnet_root}\\unimol2\\conformer.py")
        print(f"  ...")
        print("Pass --conformer-py /path/to/conformer.py explicitly.")
        return 1
    print(f"Using conformer.py: {conf_path}")
    conformer_module = load_conformer_module(conf_path)
    print(f"  Loaded UniMolV2Feature class")

    # Load Lipo splits
    splits_dir = Path(args.schnet_root) / "data" / "data_split" / "lipo" / f"seed_{args.split_seed}"
    if not splits_dir.exists():
        print(f"[FAIL] Splits not found at {splits_dir}")
        return 1
    print(f"\nLoading Lipo splits from {splits_dir}")
    splits = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(splits_dir / f"{split}.csv")
        splits[split] = df
        print(f"  {split}: {len(df)} molecules")

    # Load model
    print(f"\nLoading UniMol2-570M from {args.checkpoint}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint)
    model = model.to(device).bfloat16().eval()
    print(f"Model on {device}")

    # Process each split
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary = {"splits": {}, "embed_dim": None}

    for split, df in splits.items():
        print(f"\n--- Split: {split} ---")
        smiles_list = df[args.smiles_col].astype(str).tolist()
        y_list = df[args.target_col].astype(float).values

        # Generate features (cached)
        feat_cache = cache_dir / f"{split}_feats_seed{args.split_seed}.pkl"
        feats = generate_features_unimolv2(
            smiles_list, conformer_module, feat_cache,
            max_atoms=args.max_atoms, force_regenerate=args.force_regenerate
        )

        # Filter failed
        feats, y_filtered, keep_idx = filter_failed(feats, y_list)
        if len(feats) == 0:
            print(f"  [FAIL] All conformers failed for split {split}")
            continue

        # Extract embeddings
        print(f"  Extracting embeddings (batch_size={args.batch_size})...")
        t0 = time.time()
        embeddings = extract_embeddings(model, feats, args.batch_size, device)
        elapsed = time.time() - t0
        print(f"  Embeddings shape: {embeddings.shape}, time: {elapsed:.1f}s")

        # Save
        emb_path = cache_dir / f"{split}_emb_seed{args.split_seed}.npy"
        y_path = cache_dir / f"{split}_y_seed{args.split_seed}.npy"
        idx_path = cache_dir / f"{split}_keep_idx_seed{args.split_seed}.npy"
        np.save(emb_path, embeddings.astype(np.float32))
        np.save(y_path, y_filtered.astype(np.float32))
        np.save(idx_path, keep_idx)
        print(f"  Saved: {emb_path} ({embeddings.nbytes / 1e6:.1f} MB)")
        print(f"         {y_path}")

        summary["splits"][split] = {
            "n_input": len(smiles_list),
            "n_kept": len(embeddings),
            "n_failed": len(smiles_list) - len(embeddings),
            "emb_shape": list(embeddings.shape),
            "y_min": float(y_filtered.min()),
            "y_max": float(y_filtered.max()),
            "y_mean": float(y_filtered.mean()),
            "elapsed_s": round(elapsed, 1),
        }
        if summary["embed_dim"] is None and embeddings.shape[1] > 0:
            summary["embed_dim"] = embeddings.shape[1]

    summary["arch"] = "unimol2_570M"
    summary["checkpoint"] = args.checkpoint
    summary["split_seed"] = args.split_seed

    out = cache_dir / f"extract_summary_seed{args.split_seed}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {out}")
    print("\n[PASS] Embeddings extracted. Next: python 03_linear_probe.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())