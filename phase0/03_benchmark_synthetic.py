"""
Phase 0.3 - Synthetic Transformer Forward Benchmark

Builds a transformer with same shape parameters as UniMol2-570M and 1.1B,
benchmarks forward pass time + peak memory across batch sizes.

This script does NOT depend on UniMol2 codebase. It's a hardware-only test
to find OOM threshold and forward throughput. Real UniMol2 in 04 may be
slightly slower due to pair attention overhead.

Pass criteria:
- 570M forward succeeds at batch >= 256 (= chunk 2 with m=128)
- 1.1B forward succeeds at batch >= 128 (= chunk 1 with m=128)

Run:  python 03_benchmark_synthetic.py
Output: results_03_synthetic.json
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIGS = {
    "unimol2_570M": {"layers": 32, "embed": 1536, "ffn": 1536, "heads": 96},
    "unimol2_1100M": {"layers": 64, "embed": 1536, "ffn": 1536, "heads": 96},
}


class Block(nn.Module):
    """Approximation of UniMol2 transformer layer.
    Skips pair update (which is significant cost in UniMol2 — real benchmark
    in script 04 measures it directly). This is a lower bound on forward time."""

    def __init__(self, d, ffn, h):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.head_dim = d // h
        self.norm1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.norm2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)

    def forward(self, x):
        b, n, d = x.shape
        h = self.h
        hd = self.head_dim

        residual = x
        x = self.norm1(x)
        q = self.q(x).view(b, n, h, hd).transpose(1, 2)
        k = self.k(x).view(b, n, h, hd).transpose(1, 2)
        v = self.v(x).view(b, n, h, hd).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, n, d)
        x = residual + self.o(attn)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class Synth(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Linear(8, cfg["embed"])
        self.blocks = nn.ModuleList([
            Block(cfg["embed"], cfg["ffn"], cfg["heads"])
            for _ in range(cfg["layers"])
        ])
        self.norm = nn.LayerNorm(cfg["embed"])

    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def bench_one(name, cfg, batch_size, n_atoms, device, n_iter=10):
    """Run one benchmark config. Returns dict or None on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        model = Synth(cfg).to(device).bfloat16()
        n_params = count_params(model)
        weight_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9

        x = torch.randn(batch_size, n_atoms, 8, dtype=torch.bfloat16, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
            torch.cuda.synchronize(device)

        # Benchmark
        with torch.no_grad():
            t0 = time.time()
            for _ in range(n_iter):
                y = model(x)
            torch.cuda.synchronize(device)
            elapsed = time.time() - t0

        forward_time = elapsed / n_iter
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

        result = {
            "config": name,
            "batch_size": batch_size,
            "n_atoms": n_atoms,
            "n_params_M": round(n_params / 1e6, 1),
            "weight_gb": round(weight_gb, 2),
            "forward_time_s": round(forward_time, 4),
            "ms_per_molecule": round(forward_time / batch_size * 1000, 2),
            "peak_mem_gb": round(peak_mem, 2),
            "oom": False,
        }

        del model, x, y
        torch.cuda.empty_cache()
        return result

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "config": name,
            "batch_size": batch_size,
            "n_atoms": n_atoms,
            "oom": True,
            "peak_mem_gb": None,
        }


def main():
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available")
        return 1

    device = "cuda:0"
    print("=" * 64)
    print(" Synthetic Transformer Benchmark")
    print("=" * 64)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    free, total = torch.cuda.mem_get_info(device)
    print(f"GPU memory: {free/1e9:.2f} / {total/1e9:.2f} GB free")

    n_atoms = 64  # typical for small drug-like molecules
    m = 128  # base batch (molecules per perturbation)

    # We test "effective batch" = chunk_size * m
    # chunk_size is the number of perturbations processed in one forward pass
    # OOM at batch X means chunk_size = X / m max

    results = {"configs": {}, "device_memory_gb": round(total / 1e9, 2)}

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n--- {cfg_name} ---")
        cfg_results = []
        for chunk in [1, 2, 4, 8, 16]:
            batch = chunk * m
            print(f"  chunk={chunk:2d} (batch={batch:4d}): ", end="", flush=True)
            r = bench_one(cfg_name, cfg, batch, n_atoms, device)
            if r["oom"]:
                print("OOM")
                cfg_results.append({**r, "chunk_size": chunk})
                break  # higher chunks will also OOM
            else:
                print(
                    f"{r['forward_time_s']*1000:7.1f} ms  "
                    f"({r['ms_per_molecule']:5.2f} ms/mol, "
                    f"peak {r['peak_mem_gb']:5.2f} GB)"
                )
                cfg_results.append({**r, "chunk_size": chunk})
        results["configs"][cfg_name] = cfg_results

    # Verdict
    print("\n" + "=" * 64)
    print(" VERDICT")
    print("=" * 64)

    for cfg_name, runs in results["configs"].items():
        max_ok = max((r["chunk_size"] for r in runs if not r["oom"]), default=0)
        print(f"{cfg_name}: max chunk size = {max_ok}")
        if max_ok == 0:
            print(f"  [FAIL] Even chunk=1 OOMs. {cfg_name} not viable on this GPU.")
        elif max_ok < 2:
            print(f"  [WARN] Only chunk=1 fits. Lose EGGROLL arithmetic intensity advantage.")
        else:
            best = next(r for r in runs if r["chunk_size"] == max_ok)
            print(f"  ms/molecule at chunk={max_ok}: {best['ms_per_molecule']}")

    out = Path(__file__).parent / "results_03_synthetic.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    print("\nNext: python 04_benchmark_unimol2.py (real model, optional)")
    print("   or python 05_estimate_pipeline.py (extrapolate from synthetic)")

    return 0


if __name__ == "__main__":
    sys.exit(main())