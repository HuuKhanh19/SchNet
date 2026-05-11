"""
Phase 0.2 - Multi-GPU Distributed Test

Tests gloo (Windows-compatible) and NCCL (Linux/WSL2 only) backends.
For ES population-parallel, we send tiny tensors per generation, so even slow
gloo bandwidth is acceptable. Main goal: confirm 2 GPUs can sync at all.

Pass criteria:
- gloo all-reduce + broadcast work
- bandwidth > 0.1 GB/s for 100 MB tensor (gloo is slow, this is fine)

Run:  python 02_test_multigpu.py
Output: results_02_multigpu.json

Note: must be run with `if __name__ == '__main__'` guard for mp.spawn on Windows.
"""

import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

OUTPUT_DIR = Path(__file__).parent
TMP_PREFIX = "_tmp_mgpu"


def worker(rank, world_size, backend, port):
    """Each rank writes its own JSON; main process collects."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    result = {
        "backend": backend,
        "rank": rank,
        "all_reduce_ok": False,
        "broadcast_ok": False,
        "bandwidth_gbs": 0.0,
        "error": None,
    }

    try:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        torch.cuda.set_device(rank)

        # Test 1: all-reduce correctness
        device = f"cuda:{rank}"
        t = torch.ones(10, device=device) * (rank + 1)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = sum(r + 1 for r in range(world_size))
        result["all_reduce_ok"] = bool(abs(t[0].item() - expected) < 1e-3)

        # Test 2: broadcast
        if rank == 0:
            t = torch.tensor([42.0], device=device)
        else:
            t = torch.tensor([0.0], device=device)
        dist.broadcast(t, src=0)
        result["broadcast_ok"] = bool(abs(t.item() - 42.0) < 1e-3)

        # Test 3: bandwidth measurement
        n = 25_000_000  # 100 MB at fp32
        t = torch.randn(n, device=device)
        for _ in range(3):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        t0 = time.time()
        n_iter = 10
        for _ in range(n_iter):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        result["bandwidth_gbs"] = n * 4 * n_iter / elapsed / 1e9

        dist.destroy_process_group()
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    out = OUTPUT_DIR / f"{TMP_PREFIX}_{backend}_rank{rank}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f)


def test_backend(backend, port):
    print(f"\n--- Backend: {backend} ---")

    # Pre-check NCCL availability
    if backend == "nccl" and not dist.is_nccl_available():
        print("  NCCL not available in PyTorch build (expected on native Windows).")
        print("  To get NCCL: switch to WSL2 or Linux.")
        return {"backend": "nccl", "available_in_build": False}

    # Clean tmp files from previous run
    for r in range(2):
        f = OUTPUT_DIR / f"{TMP_PREFIX}_{backend}_rank{r}.json"
        if f.exists():
            f.unlink()

    try:
        mp.spawn(worker, args=(2, backend, port), nprocs=2, join=True, daemon=False)
    except Exception as e:
        print(f"  [FAIL] mp.spawn error: {e}")
        return {"backend": backend, "available_in_build": True, "error": str(e)}

    # Collect results
    results_per_rank = []
    for r in range(2):
        f = OUTPUT_DIR / f"{TMP_PREFIX}_{backend}_rank{r}.json"
        if f.exists():
            with open(f, "r", encoding="utf-8") as fh:
                results_per_rank.append(json.load(fh))
            f.unlink()
        else:
            print(f"  [WARN] No result file from rank {r}")

    if not results_per_rank:
        return {"backend": backend, "error": "no_results"}

    rank0 = results_per_rank[0]
    if rank0.get("error"):
        print(f"  [FAIL] {rank0['error']}")
        return {"backend": backend, "available_in_build": True, "error": rank0["error"]}

    print(f"  all_reduce: {'OK' if rank0['all_reduce_ok'] else 'FAIL'}")
    print(f"  broadcast:  {'OK' if rank0['broadcast_ok'] else 'FAIL'}")
    print(f"  bandwidth:  {rank0['bandwidth_gbs']:.2f} GB/s (100 MB all-reduce)")

    return {
        "backend": backend,
        "available_in_build": True,
        "all_reduce_ok": rank0["all_reduce_ok"],
        "broadcast_ok": rank0["broadcast_ok"],
        "bandwidth_gbs": rank0["bandwidth_gbs"],
        "error": None,
    }


def main():
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available.")
        return 1
    if torch.cuda.device_count() < 2:
        print(f"[FAIL] Need 2 GPUs, found {torch.cuda.device_count()}")
        return 1

    print("=" * 64)
    print(" Multi-GPU Distributed Tests")
    print("=" * 64)
    print(f"PyTorch:        {torch.__version__}")
    print(f"NCCL in build:  {dist.is_nccl_available()}")
    print(f"Gloo in build:  {dist.is_gloo_available()}")

    results = {
        "nccl_in_build": dist.is_nccl_available(),
        "gloo_in_build": dist.is_gloo_available(),
        "gloo": test_backend("gloo", 29500),
        "nccl": test_backend("nccl", 29501),
    }

    # Verdict
    print("\n" + "=" * 64)
    print(" VERDICT")
    print("=" * 64)

    gloo_ok = results["gloo"].get("error") is None and results["gloo"].get("bandwidth_gbs", 0) > 0
    nccl_ok = results["nccl"].get("error") is None and results["nccl"].get("bandwidth_gbs", 0) > 0

    if nccl_ok and gloo_ok:
        speedup = results["nccl"]["bandwidth_gbs"] / max(results["gloo"]["bandwidth_gbs"], 1e-3)
        print("[PASS] Both backends work.")
        print(f"  Gloo: {results['gloo']['bandwidth_gbs']:.2f} GB/s")
        print(f"  NCCL: {results['nccl']['bandwidth_gbs']:.2f} GB/s ({speedup:.1f}x)")
        print("Recommend: NCCL for production (you are likely on Linux/WSL2).")
        results["recommend_backend"] = "nccl"
    elif gloo_ok:
        bw = results["gloo"]["bandwidth_gbs"]
        print(f"[PARTIAL PASS] Only gloo works ({bw:.2f} GB/s). NCCL unavailable.")
        print("This is EXPECTED on native Windows.")
        if bw >= 0.1:
            print("Bandwidth fine for ES (we send <100 KB per generation).")
            print("Recommend: gloo on Windows OR move to WSL2 for NCCL speedup.")
            results["recommend_backend"] = "gloo"
        else:
            print("[WARN] Bandwidth very low. Consider WSL2 for better perf.")
            results["recommend_backend"] = "gloo_or_wsl2"
    elif nccl_ok:
        print("[STRANGE] NCCL works but gloo doesn't. Check installation.")
        results["recommend_backend"] = "nccl"
    else:
        print("[FAIL] No working backend.")
        results["recommend_backend"] = None

    out = OUTPUT_DIR / "results_02_multigpu.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    if gloo_ok or nccl_ok:
        print("\nNext: python 03_benchmark_synthetic.py")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())