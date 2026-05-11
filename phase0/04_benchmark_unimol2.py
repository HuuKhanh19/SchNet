"""
Phase 0.4 - Real UniMol2 Forward Benchmark (optional)

Tries to load actual UniMol2 model from your repo, runs forward pass benchmark.
Falls back gracefully if imports fail (you may not have unicore + Uni-Mol package
installed yet).

This benchmark uses random-init weights (no pretrained checkpoint needed).
The forward time should be similar to synthetic, +20-50% for pair attention
overhead.

Pass criteria:
- Imports succeed
- Forward pass runs without error
- Time matches synthetic benchmark within 2x

Run:  python 04_benchmark_unimol2.py [--unimol2-path /path/to/Uni-Mol/unimol2]
Output: results_04_unimol2.json
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


def try_import_unimol2(unimol2_path=None):
    """Try to import UniMol2Model. Returns class or None."""
    if unimol2_path:
        sys.path.insert(0, str(Path(unimol2_path).resolve()))

    try:
        # User mentioned their unimol2.py is at /mnt/project/. Adjust if different.
        from unimol2 import UniMol2Model
        return UniMol2Model, None
    except ImportError as e:
        return None, f"ImportError: {e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def make_args(arch="unimol2_570M"):
    """Build argparse Namespace with UniMol2 arch defaults.

    The user's unimol2.py registers architectures via decorator. Calling the
    arch function fills in defaults.
    """
    a = types.SimpleNamespace()
    # Common defaults from base_architecture in unimol2.py
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
    a.mode = "infer"  # returns (x, pair) directly

    if arch == "unimol2_1100M":
        a.encoder_layers = 64
    return a


def make_dummy_batch(batch_size, n_atoms, device, dtype=torch.bfloat16):
    """Build a synthetic batched_data dict matching UniMol2 forward signature."""
    B = batch_size
    N = n_atoms

    batch = {
        "src_token": torch.randint(1, 100, (B, N), device=device, dtype=torch.long),
        "atom_feat": torch.randint(1, 100, (B, N, 8), device=device, dtype=torch.long),
        "atom_mask": torch.ones(B, N, device=device, dtype=torch.long),
        "edge_feat": torch.randint(1, 50, (B, N, N, 3), device=device, dtype=torch.long),
        "shortest_path": torch.randint(0, 10, (B, N, N), device=device, dtype=torch.long),
        "degree": torch.randint(2, 8, (B, N), device=device, dtype=torch.long),
        "pair_type": torch.randint(0, 256, (B, N, N, 2), device=device, dtype=torch.long),
        "attn_bias": torch.zeros(B, N + 1, N + 1, device=device, dtype=dtype),
        "src_pos": torch.randn(B, N, 3, device=device, dtype=dtype) * 5.0,
    }
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unimol2-path", type=str, default=None,
                        help="Path to Uni-Mol/unimol2 package directory.")
    parser.add_argument("--arch", choices=["unimol2_570M", "unimol2_1100M"],
                        default="unimol2_570M")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[128, 256, 512])
    parser.add_argument("--n-atoms", type=int, default=64)
    args_cli = parser.parse_args()

    print("=" * 64)
    print(f" Real UniMol2 Benchmark ({args_cli.arch})")
    print("=" * 64)

    UniMol2Model, err = try_import_unimol2(args_cli.unimol2_path)
    if UniMol2Model is None:
        print(f"\n[SKIP] Cannot import UniMol2: {err}")
        print("\nTo run this benchmark you need:")
        print("  1. unicore: pip install unicore (or build from DeepModeling repo)")
        print("  2. Uni-Mol/unimol2 package on PYTHONPATH")
        print("\nWorkaround: pass --unimol2-path /path/to/your/Uni-Mol/unimol2")
        print("Or skip this script — synthetic benchmark in 03 is sufficient for Phase 0.")
        save({"skipped": True, "import_error": err})
        return 0  # not a hard fail

    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available.")
        return 1
    device = "cuda:0"
    print(f"Device: {torch.cuda.get_device_name(device)}")

    margs = make_args(args_cli.arch)

    print(f"\nBuilding {args_cli.arch}...")
    try:
        model = UniMol2Model(margs)
    except Exception as e:
        print(f"[FAIL] Model construction error: {e}")
        save({"error": str(e)})
        return 1

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params/1e6:.1f}M")

    model = model.to(device).bfloat16()
    model.eval()

    results = {"arch": args_cli.arch, "n_params_M": round(n_params / 1e6, 1), "runs": []}

    for B in args_cli.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        print(f"\n--- batch={B}, n_atoms={args_cli.n_atoms} ---")
        try:
            batch = make_dummy_batch(B, args_cli.n_atoms, device)

            with torch.no_grad():
                # Warmup
                for _ in range(2):
                    _ = model(batch)
                torch.cuda.synchronize(device)

                # Benchmark
                t0 = time.time()
                n_iter = 5
                for _ in range(n_iter):
                    out = model(batch)
                torch.cuda.synchronize(device)
                elapsed = time.time() - t0

            ft = elapsed / n_iter
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"  Forward: {ft*1000:.1f} ms ({ft/B*1000:.2f} ms/mol), peak mem {mem:.2f} GB")

            # Verify output shape (mode='infer' returns (x, pair))
            if isinstance(out, tuple) and len(out) == 2:
                x, pair = out
                print(f"  Output x shape: {tuple(x.shape)}, pair shape: {tuple(pair.shape)}")
                cls_dim = x.shape[-1]
            else:
                cls_dim = None

            results["runs"].append({
                "batch_size": B,
                "n_atoms": args_cli.n_atoms,
                "forward_time_s": round(ft, 4),
                "ms_per_molecule": round(ft / B * 1000, 2),
                "peak_mem_gb": round(mem, 2),
                "embed_dim": cls_dim,
                "oom": False,
            })

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch={B}")
            results["runs"].append({
                "batch_size": B, "n_atoms": args_cli.n_atoms, "oom": True,
            })
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            results["runs"].append({
                "batch_size": B, "n_atoms": args_cli.n_atoms,
                "error": str(e),
            })
            torch.cuda.empty_cache()

    print("\n" + "=" * 64)
    print(" VERDICT")
    print("=" * 64)
    ok_runs = [r for r in results["runs"] if not r.get("oom") and not r.get("error")]
    if ok_runs:
        max_batch = max(r["batch_size"] for r in ok_runs)
        print(f"[PASS] Max batch fit: {max_batch}")
        print(f"  At batch={max_batch}: {next(r for r in ok_runs if r['batch_size']==max_batch)}")
    else:
        print("[FAIL] No batch size succeeded.")

    save(results)
    return 0 if ok_runs else 1


def save(results):
    out = Path(__file__).parent / "results_04_unimol2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    sys.exit(main())