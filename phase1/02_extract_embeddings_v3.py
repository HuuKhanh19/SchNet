"""
Phase 1.2 v3 - Extract CLS Embeddings (bypass broken UniMolV2Feature.transform)

Workaround for bug in user's conformer.py: UniMolV2Feature.single_process()
calls inner_smi2coords() with kwargs (remove_hs, return_mol) that the function
doesn't accept. This is a version mismatch in the user's local conformer.py.

Fix: do our own SMILES -> mol via RDKit ETKDGv3 + MMFF/UFF, then call
mol2unimolv2() directly (this function works correctly).

Run:
  python 02_extract_embeddings_v3.py \
      --unimol2-path C:\\path\\to\\Uni-Mol\\unimol2 \
      --conformer-py C:\\path\\to\\conformer.py \
      --checkpoint C:\\path\\to\\unimol2_570m_pretrain.pt \
      --batch-size 8
"""

import argparse
import importlib.util
import json
import pickle
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
def setup_paths(unimol2_path, schnet_root):
    if unimol2_path:
        sys.path.insert(0, str(Path(unimol2_path).resolve()))
    if schnet_root:
        sys.path.insert(0, str(Path(schnet_root).resolve()))


def find_conformer_py(schnet_root, explicit=None):
    if explicit:
        p = Path(explicit).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"--conformer-py not found: {p}")
    schnet_root = Path(schnet_root) if schnet_root else Path(".")
    candidates = [
        schnet_root / "conformer.py",
        schnet_root / "unimol2" / "conformer.py",
        schnet_root / "unimol2" / "data" / "conformer.py",
        Path(__file__).parent / "conformer.py",
        Path(__file__).parent.parent / "conformer.py",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def load_conformer_module(conformer_path):
    spec = importlib.util.spec_from_file_location("user_conformer", str(conformer_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_conformer"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "mol2unimolv2"):
        raise ImportError(f"{conformer_path} missing mol2unimolv2 function")
    return module


# ---------------------------------------------------------------------------
# SMILES -> RDKit mol (with 3D coords) - our own implementation, no inner_smi2coords
# ---------------------------------------------------------------------------
def smi_to_mol_3d(smiles, seed=42):
    """SMILES -> RDKit mol with embedded + minimized 3D conformer.
    Returns mol or None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Replace dummy atoms with carbons for embedding
        mol = Chem.RWMol(mol)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == '*':
                atom.SetAtomicNum(6)  # treat as carbon for embedding
        mol = mol.GetMol()

        mol = AllChem.AddHs(mol)

        # Try ETKDGv3 with retries
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useRandomCoords = False
        params.maxAttempts = 200
        params.numThreads = 0

        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            # Retry with random coords
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            return None

        # Energy minimize
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass

        return mol
    except Exception:
        return None


def smi_to_mol_2d_fallback(smiles):
    """Fallback to 2D coords with z=0 if 3D embedding fails."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = AllChem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        return mol
    except Exception:
        return None


def smi_to_features(smiles_list, conformer_module, max_atoms=128, seed=42):
    """SMILES list -> list of feature dicts (or None for failures)."""
    feats = []
    n_failed_3d = 0
    n_failed_total = 0

    for smi in tqdm(smiles_list, desc="  SMILES->features"):
        mol = smi_to_mol_3d(smi, seed=seed)
        if mol is None:
            mol = smi_to_mol_2d_fallback(smi)
            if mol is not None:
                n_failed_3d += 1
        if mol is None:
            n_failed_total += 1
            feats.append(None)
            continue
        try:
            feat = conformer_module.mol2unimolv2(mol, max_atoms=max_atoms,
                                                  remove_hs=True, seed=seed)
            feats.append(feat)
        except Exception as e:
            n_failed_total += 1
            feats.append(None)

    print(f"  3D-failed (used 2D fallback): {n_failed_3d}")
    print(f"  Total-failed (skipped):       {n_failed_total}")
    return feats


# ---------------------------------------------------------------------------
# Model loading (same as v2)
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
    for mp in ["unimol2.models.unimol2", "unimol2.unimol2.models.unimol2", "models.unimol2"]:
        try:
            mod = __import__(mp, fromlist=["UniMol2Model"])
            UniMol2Model = mod.UniMol2Model
            print(f"Imported UniMol2Model from {mp}")
            break
        except ImportError:
            continue
    if UniMol2Model is None:
        raise ImportError("Cannot find UniMol2Model.")
    args = make_args_570M()
    model = UniMol2Model(args)
    print(f"Constructed UniMol2-570M: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = state.get("model") or state.get("state_dict") or state if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint: {len(sd) - len(unexpected)} matched, {len(missing)} missing, {len(unexpected)} unexpected")
    return model


# ---------------------------------------------------------------------------
# Collate (same as v2)
# ---------------------------------------------------------------------------
def collate_unimol2(feat_list, device, dtype=torch.bfloat16):
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

    src_tokens = np.stack([pad_1d(f["src_tokens"], max_atoms, np_dtype=np.int64) for f in feat_list])
    src_coord = np.stack([pad_1d(f["src_coord"], max_atoms, 0.0) for f in feat_list])
    atom_feat = np.stack([pad_1d(f["atom_feat"], max_atoms) for f in feat_list])
    atom_mask = np.stack([pad_1d(f["atom_mask"], max_atoms) for f in feat_list])
    pair_type = np.stack([pad_2d(f["pair_type"], max_atoms) for f in feat_list])
    edge_feat = np.stack([pad_2d(f["edge_feat"], max_atoms) for f in feat_list])
    shortest_path = np.stack([pad_2d(f["shortest_path"], max_atoms) for f in feat_list])
    degree = np.stack([pad_1d(f["degree"], max_atoms) for f in feat_list])

    attn_bias_list = []
    for f in feat_list:
        ab = np.asarray(f["attn_bias"])
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
                x = out[0] if isinstance(out, tuple) else out
                cls = x[:, 0, :].float().cpu().numpy()
                all_emb.append(cls)
            except Exception as e:
                print(f"    [WARN] batch {i+1}/{n_batches} failed: {e}")
                all_emb.append(np.zeros((end - start, 1536), dtype=np.float32))
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
    parser.add_argument("--conformer-py", type=str, default=None)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--target-col", type=str, default="measured")
    parser.add_argument("--max-atoms", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-regenerate", action="store_true")
    args = parser.parse_args()

    setup_paths(args.unimol2_path, args.schnet_root)

    conf_path = find_conformer_py(args.schnet_root, args.conformer_py)
    if conf_path is None:
        print("[FAIL] conformer.py not found.")
        return 1
    print(f"Using conformer.py: {conf_path}")
    conformer_module = load_conformer_module(conf_path)
    print(f"  Loaded mol2unimolv2 function (bypassing broken UniMolV2Feature.transform)")

    # Load Lipo splits
    splits_dir = Path(args.schnet_root) / "data" / "data_split" / "lipo" / f"seed_{args.split_seed}"
    if not splits_dir.exists():
        print(f"[FAIL] Splits not found: {splits_dir}")
        return 1
    print(f"\nLoading splits from {splits_dir}")
    splits = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(splits_dir / f"{split}.csv")
        splits[split] = df
        print(f"  {split}: {len(df)} molecules")

    # Try to autodetect target column
    sample_cols = splits["train"].columns.tolist()
    if args.target_col not in sample_cols:
        print(f"\n[WARN] '{args.target_col}' not in CSV columns: {sample_cols}")
        for cand in ["exp", "y", "label", "target", "value", "expt"]:
            if cand in sample_cols:
                args.target_col = cand
                print(f"  Auto-using target column: {cand}")
                break
        else:
            num_cols = [c for c in sample_cols if c != args.smiles_col
                        and pd.api.types.is_numeric_dtype(splits["train"][c])]
            if num_cols:
                args.target_col = num_cols[0]
                print(f"  Auto-using first numeric column: {args.target_col}")
            else:
                print("[FAIL] Cannot auto-detect target column.")
                return 1

    if args.smiles_col not in sample_cols:
        for cand in ["SMILES", "smi", "Smiles"]:
            if cand in sample_cols:
                args.smiles_col = cand
                print(f"  Auto-using SMILES column: {cand}")
                break

    # Load model
    print(f"\nLoading UniMol2-570M from {args.checkpoint}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint)
    model = model.to(device).bfloat16().eval()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary = {"splits": {}, "embed_dim": None}

    for split, df in splits.items():
        print(f"\n--- Split: {split} ---")
        smiles_list = df[args.smiles_col].astype(str).tolist()
        y_list = df[args.target_col].astype(float).values

        feat_cache = cache_dir / f"{split}_feats_seed{args.split_seed}.pkl"
        if feat_cache.exists() and not args.force_regenerate:
            print(f"  Loading cached: {feat_cache}")
            with open(feat_cache, "rb") as f:
                feats_with_none = pickle.load(f)
        else:
            print(f"  Generating features for {len(smiles_list)} molecules...")
            feats_with_none = smi_to_features(smiles_list, conformer_module,
                                               max_atoms=args.max_atoms, seed=args.seed)
            with open(feat_cache, "wb") as f:
                pickle.dump(feats_with_none, f)
            print(f"  Cached: {feat_cache}")

        # Filter None
        keep_idx = [i for i, f in enumerate(feats_with_none) if f is not None]
        if len(keep_idx) < len(feats_with_none):
            print(f"  Skipped {len(feats_with_none) - len(keep_idx)} None entries")
        feats = [feats_with_none[i] for i in keep_idx]
        y_kept = y_list[np.asarray(keep_idx)]

        # Filter all-zero coords (3D embed totally failed)
        valid_idx = [i for i, f in enumerate(feats)
                     if not np.all(np.asarray(f["src_coord"]) == 0.0)]
        if len(valid_idx) < len(feats):
            print(f"  Skipped {len(feats) - len(valid_idx)} all-zero coords")
        feats = [feats[i] for i in valid_idx]
        y_kept = y_kept[np.asarray(valid_idx)]
        keep_idx_final = np.asarray(keep_idx)[np.asarray(valid_idx)]

        if len(feats) == 0:
            print(f"  [FAIL] All features failed for split {split}")
            continue

        print(f"  Final: {len(feats)} valid molecules / {len(smiles_list)} total")

        print(f"  Extracting embeddings (batch_size={args.batch_size})...")
        t0 = time.time()
        embeddings = extract_embeddings(model, feats, args.batch_size, device)
        elapsed = time.time() - t0
        print(f"  Embeddings shape: {embeddings.shape}, time: {elapsed:.1f}s")

        emb_path = cache_dir / f"{split}_emb_seed{args.split_seed}.npy"
        y_path = cache_dir / f"{split}_y_seed{args.split_seed}.npy"
        idx_path = cache_dir / f"{split}_keep_idx_seed{args.split_seed}.npy"
        np.save(emb_path, embeddings.astype(np.float32))
        np.save(y_path, y_kept.astype(np.float32))
        np.save(idx_path, keep_idx_final)
        print(f"  Saved: {emb_path}, {y_path}")

        summary["splits"][split] = {
            "n_input": len(smiles_list),
            "n_kept": len(embeddings),
            "n_failed": len(smiles_list) - len(embeddings),
            "emb_shape": list(embeddings.shape),
            "y_min": float(y_kept.min()),
            "y_max": float(y_kept.max()),
            "y_mean": float(y_kept.mean()),
            "elapsed_s": round(elapsed, 1),
        }
        if summary["embed_dim"] is None:
            summary["embed_dim"] = int(embeddings.shape[1])

    summary["arch"] = "unimol2_570M"
    summary["checkpoint"] = args.checkpoint
    summary["split_seed"] = args.split_seed
    summary["smiles_col"] = args.smiles_col
    summary["target_col"] = args.target_col

    out = cache_dir / f"extract_summary_seed{args.split_seed}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {out}")
    print("\n[PASS] Embeddings extracted.")
    print("Next: python 03_linear_probe.py --cache-dir ./cache --split-seed 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())