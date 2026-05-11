"""
Phase 1.1 - Setup Verification

Checks before running embedding extraction:
1. unicore importable
2. UniMol2Model importable from your repo
3. ConformerGen importable
4. Lipo data exists
5. Pretrained checkpoint exists (or print download instructions)
6. UniMol2-570M model can be constructed
7. Checkpoint loads with reasonable shape match
8. Sample forward pass produces expected output shape

Run:  python 01_setup_check.py [--unimol2-path PATH] [--checkpoint PATH]
Output: results_01_setup.json

Pass criteria: all 8 checks ✓
"""

import argparse
import json
import sys
import types
from pathlib import Path


def header(s):
    print(f"\n{'=' * 64}\n {s}\n{'=' * 64}")


def setup_paths(unimol2_path, schnet_root):
    """Add user's repo to sys.path so imports work."""
    if unimol2_path:
        p = Path(unimol2_path).resolve()
        if not p.exists():
            return f"--unimol2-path does not exist: {p}"
        sys.path.insert(0, str(p))
        # If user's path is parent of unimol2/ package, also add the parent
        if (p / "unimol2").exists():
            sys.path.insert(0, str(p))
    if schnet_root:
        sys.path.insert(0, str(Path(schnet_root).resolve()))
    return None


def check_unicore():
    try:
        import unicore  # noqa
        from unicore import utils  # noqa
        from unicore.models import BaseUnicoreModel, register_model  # noqa
        from unicore.modules import LayerNorm  # noqa
        return True, f"unicore version: {getattr(unicore, '__version__', 'unknown')}"
    except ImportError as e:
        return False, f"ImportError: {e}\n  Install: pip install unicore (or build from https://github.com/dptech-corp/Uni-Core)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_unimol2():
    """Try multiple common import paths."""
    attempts = [
        ("unimol2.models.unimol2", "UniMol2Model"),
        ("unimol2.unimol2.models.unimol2", "UniMol2Model"),
        ("models.unimol2", "UniMol2Model"),
    ]
    for module_path, cls_name in attempts:
        try:
            module = __import__(module_path, fromlist=[cls_name])
            cls = getattr(module, cls_name)
            return True, cls, f"Imported from: {module_path}"
        except ImportError:
            continue
        except Exception as e:
            return False, None, f"{type(e).__name__}: {e}"
    return False, None, "Could not find UniMol2Model. Pass --unimol2-path /path/to/Uni-Mol/unimol2"


def check_conformer():
    attempts = [
        "unimol2.data.conformer",
        "unimol_tools.conformer",
        "conformer",
    ]
    for module_path in attempts:
        try:
            module = __import__(module_path, fromlist=["ConformerGen"])
            cls = getattr(module, "ConformerGen")
            return True, cls, f"Imported from: {module_path}"
        except (ImportError, AttributeError):
            continue
        except Exception as e:
            return False, None, f"{type(e).__name__}: {e}"
    return False, None, "ConformerGen not found. Check unimol2.data path or unimol_tools install."


def check_lipo_data(schnet_root):
    """Look for existing Lipo splits from SchNet baseline."""
    paths_to_try = [
        Path(schnet_root) / "data" / "data_split" / "lipo" / "seed_0",
        Path(schnet_root) / "data" / "lipo.csv",
        Path(".") / "data" / "data_split" / "lipo" / "seed_0",
        Path(".") / "data" / "lipo.csv",
    ]
    for p in paths_to_try:
        if p.exists():
            return True, str(p), f"Lipo data found at: {p}"
    return False, None, f"Lipo data not found in: {[str(p) for p in paths_to_try]}"


def check_checkpoint(ckpt_path):
    if not ckpt_path:
        return False, None, ("No --checkpoint provided.\n"
                              "  Download UniMol2-570M from:\n"
                              "  https://github.com/deepmodeling/Uni-Mol/tree/main/unimol2\n"
                              "  Look for 'unimol2_570m_pretrain.pt' or similar in the README")
    p = Path(ckpt_path).resolve()
    if not p.exists():
        return False, None, f"Checkpoint not found: {p}"
    size_mb = p.stat().st_size / 1e6
    return True, str(p), f"{p} ({size_mb:.1f} MB)"


def make_args_570M():
    """Build argparse Namespace for UniMol2-570M."""
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


def construct_model(UniMol2Model):
    try:
        args = make_args_570M()
        model = UniMol2Model(args)
        n_params = sum(p.numel() for p in model.parameters())
        return True, model, f"Model constructed. Total params: {n_params/1e6:.1f}M"
    except Exception as e:
        import traceback
        return False, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def load_checkpoint(model, ckpt_path):
    import torch
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Common checkpoint formats: {"model": state_dict} or just state_dict
        if isinstance(state, dict):
            if "model" in state:
                state_dict = state["model"]
            elif "state_dict" in state:
                state_dict = state["state_dict"]
            else:
                state_dict = state
        else:
            return False, f"Unexpected checkpoint type: {type(state)}"
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        n_loaded = len(state_dict) - len(unexpected)
        msg = f"Loaded {n_loaded} keys"
        if missing:
            msg += f", {len(missing)} missing (showing first 3): {missing[:3]}"
        if unexpected:
            msg += f", {len(unexpected)} unexpected (showing first 3): {unexpected[:3]}"
        return True, msg
    except Exception as e:
        import traceback
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def sample_forward(model):
    """Run forward pass with synthetic batch. Verify output shape."""
    import torch
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device).bfloat16().eval()
        
        B, N = 2, 16  # tiny batch, 16 atoms
        batch = {
            "src_token": torch.randint(1, 100, (B, N), device=device, dtype=torch.long),
            "atom_feat": torch.randint(1, 100, (B, N, 8), device=device, dtype=torch.long),
            "atom_mask": torch.ones(B, N, device=device, dtype=torch.long),
            "edge_feat": torch.randint(1, 50, (B, N, N, 3), device=device, dtype=torch.long),
            "shortest_path": torch.randint(0, 10, (B, N, N), device=device, dtype=torch.long),
            "degree": torch.randint(2, 8, (B, N), device=device, dtype=torch.long),
            "pair_type": torch.randint(0, 256, (B, N, N, 2), device=device, dtype=torch.long),
            "attn_bias": torch.zeros(B, N + 1, N + 1, device=device, dtype=torch.bfloat16),
            "src_pos": torch.randn(B, N, 3, device=device, dtype=torch.bfloat16) * 5.0,
        }
        
        with torch.no_grad():
            out = model(batch)
        
        # mode='infer' returns (x, pair)
        if isinstance(out, tuple) and len(out) == 2:
            x, pair = out
            cls_embed = x[:, 0, :]  # CLS at position 0
            return True, (
                f"Output x shape: {tuple(x.shape)}\n"
                f"   pair shape: {tuple(pair.shape)}\n"
                f"   CLS embed (x[:, 0, :]) shape: {tuple(cls_embed.shape)}\n"
                f"   CLS embed dtype: {cls_embed.dtype}\n"
                f"   Embed values (first 3): {cls_embed[0, :3].float().tolist()}"
            )
        else:
            return False, f"Unexpected output type: {type(out)}"
    except Exception as e:
        import traceback
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unimol2-path", type=str, default=None,
                        help="Path to Uni-Mol/unimol2 package directory")
    parser.add_argument("--schnet-root", type=str,
                        default=r"C:\Users\BKAI\ducluong\DrugOptimization\SchNet",
                        help="Path to your SchNet repo (where data/ lives)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to UniMol2-570M pretrained checkpoint .pt file")
    args = parser.parse_args()

    # Setup sys.path
    err = setup_paths(args.unimol2_path, args.schnet_root)
    if err:
        print(f"[FAIL] {err}")
        return 1

    results = {}

    header("1. Check unicore import")
    ok, msg = check_unicore()
    print(f"  {'[OK]' if ok else '[FAIL]'} {msg}")
    results["unicore"] = {"ok": ok, "msg": msg}
    if not ok:
        save(results); return 1

    header("2. Check UniMol2Model import")
    ok, UniMol2Model, msg = check_unimol2()
    print(f"  {'[OK]' if ok else '[FAIL]'} {msg}")
    results["unimol2"] = {"ok": ok, "msg": msg}
    if not ok:
        save(results); return 1

    header("3. Check ConformerGen import")
    ok, ConformerGen, msg = check_conformer()
    print(f"  {'[OK]' if ok else '[WARN]'} {msg}")
    results["conformer"] = {"ok": ok, "msg": msg}
    # Not blocker for setup check, only needed for embedding extraction

    header("4. Check Lipo data")
    ok, lipo_path, msg = check_lipo_data(args.schnet_root)
    print(f"  {'[OK]' if ok else '[WARN]'} {msg}")
    results["lipo"] = {"ok": ok, "path": lipo_path, "msg": msg}
    if not ok:
        print("  Run prepare_splits.py first to generate scaffold splits.")

    header("5. Check checkpoint")
    ok, ckpt_path, msg = check_checkpoint(args.checkpoint)
    print(f"  {'[OK]' if ok else '[WARN]'} {msg}")
    results["checkpoint"] = {"ok": ok, "path": ckpt_path, "msg": msg}

    header("6. Construct UniMol2-570M model")
    ok, model, msg = construct_model(UniMol2Model)
    print(f"  {'[OK]' if ok else '[FAIL]'} {msg}")
    results["construct"] = {"ok": ok, "msg": msg}
    if not ok:
        save(results); return 1

    if ckpt_path:
        header("7. Load checkpoint")
        ok, msg = load_checkpoint(model, ckpt_path)
        print(f"  {'[OK]' if ok else '[FAIL]'} {msg}")
        results["load_checkpoint"] = {"ok": ok, "msg": msg}
    else:
        print("\n[SKIP] No checkpoint, skipping load step.")
        results["load_checkpoint"] = {"ok": False, "msg": "skipped"}

    header("8. Sample forward pass")
    ok, msg = sample_forward(model)
    print(f"  {'[OK]' if ok else '[FAIL]'} {msg}")
    results["forward"] = {"ok": ok, "msg": msg}

    header("VERDICT")
    critical_checks = ["unicore", "unimol2", "construct", "forward"]
    all_critical_ok = all(results[k]["ok"] for k in critical_checks)
    has_lipo = results["lipo"]["ok"]
    has_ckpt = results["checkpoint"]["ok"]
    has_conformer = results["conformer"]["ok"]

    if all_critical_ok and has_lipo and has_ckpt and has_conformer:
        print("[PASS] Ready for Phase 1.2 (embedding extraction).")
        print("Next: python 02_extract_embeddings.py")
    elif all_critical_ok:
        print("[PARTIAL] Critical checks pass. Missing:")
        if not has_ckpt:
            print("  - Pretrained checkpoint (download from UniMol2 repo)")
        if not has_lipo:
            print("  - Lipo data splits (run prepare_splits.py)")
        if not has_conformer:
            print("  - ConformerGen (check unimol_tools or unimol2.data path)")
    else:
        print("[FAIL] Critical issues, see above.")

    save(results)
    return 0 if all_critical_ok else 1


def save(results):
    out = Path(__file__).parent / "results_01_setup.json"
    with open(out, "w", encoding="utf-8") as f:
        # Strip non-JSON-serializable values
        clean = {}
        for k, v in results.items():
            if isinstance(v, dict):
                clean[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (str, bool, int, float, type(None)))}
            else:
                clean[k] = v
        json.dump(clean, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    sys.exit(main())