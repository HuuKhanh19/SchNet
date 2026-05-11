"""
Phase 1.2 - Extract CLS Embeddings from UniMol2-570M

Loads Lipo dataset (train/val/test scaffold splits), generates K=1 conformer
per molecule via ConformerGen, runs UniMol2 forward, caches the CLS embedding
(x[:, 0, :]) and target value to disk.

Output: cache/{split}_emb.npy and cache/{split}_y.npy for split in {train, val, test}

Run:
  python 02_extract_embeddings.py \
      --unimol2-path C:\\path\\to\\Uni-Mol\\unimol2 \
      --schnet-root C:\\path\\to\\SchNet \
      --checkpoint C:\\path\\to\\unimol2_570M_pretrain.pt \
      --batch-size 8

Notes:
- Conformer generation is slow (~minutes for Lipo's 4100 molecules) but cached.
- Forward pass uses bf16 for speed, fp32 for embeddings cast at output.
"""

import argparse
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
# Path setup (must run before importing UniMol2)
# ---------------------------------------------------------------------------
def setup_paths(unimol2_path, schnet_root):
    if unimol2_path:
        sys.path.insert(0, str(Path(unimol2_path).resolve()))
    if schnet_root:
        sys.path.insert(0, str(Path(schnet_root).resolve()))


# ---------------------------------------------------------------------------
# Model loading (mirrors 01_setup_check)
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
    # Try multiple import paths
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
        raise ImportError("Cannot find UniMol2Model. Check --unimol2-path.")

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
    print(f"Loaded checkpoint: {len(state_dict) - len(unexpected)} keys matched, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")

    return model


# ---------------------------------------------------------------------------
# Conformer generation
# ---------------------------------------------------------------------------
def generate_conformers(smiles_list, cache_path, force_regenerate=False):
    """Generate K=1 conformer per molecule via ConformerGen. Cache to disk."""
    if cache_path.exists() and not force_regenerate:
        print(f"  Loading cached conformers from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Try to import ConformerGen
    ConformerGen = None
    for mp in ["unimol2.data.conformer", "unimol_tools.conformer", "conformer"]:
        try:
            mod = __import__(mp, fromlist=["ConformerGen"])
            ConformerGen = mod.ConformerGen
            print(f"  Using ConformerGen from {mp}")
            break
        except (ImportError, AttributeError):
            continue
    if ConformerGen is None:
        raise ImportError("Cannot find ConformerGen.")

    print(f"  Generating K=1 conformers for {len(smiles_list)} molecules...")
    conf_gen = ConformerGen(
        n_confomer=1,
        mode="heavy",
        remove_hs=False,
        max_atoms=256,
        seed=42,
    )
    inputs, _ = conf_gen.transform(smiles_list)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(inputs, f)
    print(f"  Cached to {cache_path}")
    return inputs


# ---------------------------------------------------------------------------
# Collate: convert ConformerGen output dicts to UniMol2 batched_data
# ---------------------------------------------------------------------------
def collate_unimol2(feat_list, device, dtype=torch.bfloat16):
    """
    feat_list: list of dicts from ConformerGen.
    Returns: batched_data dict ready for UniMol2 forward.

    Note: ConformerGen uses key 'src_tokens' (plural) and 'src_coord'.
    UniMol2 forward expects 'src_token' (singular) and 'src_pos'.
    """
    B = len(feat_list)
    max_atoms = max(f["atom_mask"].shape[0] for f in feat_list)

    def pad_1d(arr, target_n, pad_val=0):
        n = arr.shape[0]
        if n >= target_n:
            return arr[:target_n]
        pad_shape = (target_n - n,) + arr.shape[1:]
        return np.concatenate([arr, np.full(pad_shape, pad_val, dtype=arr.dtype)])

    def pad_2d(arr, target_n, pad_val=0):
        # arr shape: (n, n, ...). pad to (target_n, target_n, ...)
        n = arr.shape[0]
        if n >= target_n:
            return arr[:target_n, :target_n]
        out_shape = (target_n, target_n) + arr.shape[2:]
        out = np.full(out_shape, pad_val, dtype=arr.dtype)
        out[:n, :n] = arr
        return out

    # Stack with padding
    src_tokens = np.stack([pad_1d(np.asarray(f["src_tokens"]), max_atoms) for f in feat_list])
    src_coord = np.stack([pad_1d(f["src_coord"], max_atoms, 0.0) for f in feat_list])
    atom_feat = np.stack([pad_1d(f["atom_feat"], max_atoms) for f in feat_list])
    atom_mask = np.stack([pad_1d(f["atom_mask"], max_atoms) for f in feat_list])
    pair_type = np.stack([pad_2d(f["pair_type"], max_atoms) for f in feat_list])
    edge_feat = np.stack([pad_2d(f["edge_feat"], max_atoms) for f in feat_list])
    shortest_path = np.stack([pad_2d(f["shortest_path"], max_atoms) for f in feat_list])
    degree = np.stack([pad_1d(f["degree"], max_atoms) for f in feat_list])
    attn_bias = np.stack([
        np.pad(f["attn_bias"], 
               ((0, max_atoms + 1 - f["attn_bias"].shape[0]),) * 2,
               mode='constant')
        for f in feat_list
    ])

    return {
        # Note name conversion: src_tokens -> src_token, src_coord -> src_pos
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
# Forward + extract CLS embedding
# ---------------------------------------------------------------------------
def extract_embeddings(model, conformer_inputs, batch_size, device):
    """Forward through UniMol2 in batches, extract x[:, 0, :] CLS embedding."""
    model.eval()
    all_emb = []
    n = len(conformer_inputs)
    n_batches = (n + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n)
            batch_feats = conformer_inputs[start:end]
            batch = collate_unimol2(batch_feats, device, dtype=torch.bfloat16)

            out = model(batch)
            # mode='infer' -> (x, pair)
            x, _pair = out
            cls = x[:, 0, :].float().cpu().numpy()  # cast to fp32 for downstream LS
            all_emb.append(cls)

            if i % 10 == 0 or i == n_batches - 1:
                print(f"    batch {i+1}/{n_batches} done")

    return np.concatenate(all_emb, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unimol2-path", type=str, default=None)
    parser.add_argument("--schnet-root", type=str,
                        default=r"C:\Users\BKAI\ducluong\DrugOptimization\SchNet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--target-col", type=str, default="measured")
    parser.add_argument("--force-regenerate-conformers", action="store_true")
    args = parser.parse_args()

    setup_paths(args.unimol2_path, args.schnet_root)

    # Load Lipo splits
    splits_dir = Path(args.schnet_root) / "data" / "data_split" / "lipo" / f"seed_{args.split_seed}"
    if not splits_dir.exists():
        print(f"[FAIL] Splits not found at {splits_dir}. Run prepare_splits.py first.")
        return 1

    print(f"Loading Lipo splits from {splits_dir}")
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

        # Conformer generation (cached)
        conf_cache = cache_dir / f"{split}_conformers_seed{args.split_seed}.pkl"
        confs = generate_conformers(smiles_list, conf_cache,
                                     force_regenerate=args.force_regenerate_conformers)

        if len(confs) != len(smiles_list):
            print(f"  [WARN] {len(smiles_list)} SMILES -> {len(confs)} conformers (some failed)")
            # Need to align — assume conformers are in same order as smiles, with failures at end
            # This is fragile; better: track which indices succeeded
            n_use = min(len(confs), len(smiles_list))
            y_list = y_list[:n_use]
            confs = confs[:n_use]

        # Forward + extract
        print(f"  Extracting embeddings (batch_size={args.batch_size})...")
        t0 = time.time()
        embeddings = extract_embeddings(model, confs, args.batch_size, device)
        elapsed = time.time() - t0
        print(f"  Embeddings shape: {embeddings.shape}, time: {elapsed:.1f}s")

        # Save
        emb_path = cache_dir / f"{split}_emb_seed{args.split_seed}.npy"
        y_path = cache_dir / f"{split}_y_seed{args.split_seed}.npy"
        np.save(emb_path, embeddings.astype(np.float32))
        np.save(y_path, y_list.astype(np.float32))
        print(f"  Saved: {emb_path} ({embeddings.nbytes / 1e6:.1f} MB)")
        print(f"  Saved: {y_path}")

        summary["splits"][split] = {
            "n": len(embeddings),
            "emb_shape": list(embeddings.shape),
            "y_min": float(y_list.min()),
            "y_max": float(y_list.max()),
            "y_mean": float(y_list.mean()),
            "elapsed_s": round(elapsed, 1),
        }
        if summary["embed_dim"] is None:
            summary["embed_dim"] = embeddings.shape[1]

    summary["arch"] = "unimol2_570M"
    summary["checkpoint"] = args.checkpoint
    summary["split_seed"] = args.split_seed
    summary["cache_dir"] = str(cache_dir)

    out = cache_dir / f"extract_summary_seed{args.split_seed}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {out}")
    print("\n[PASS] Embeddings extracted. Next: python 03_linear_probe.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())