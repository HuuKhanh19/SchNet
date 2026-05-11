"""
Phase 0.2 v3 - Multi-GPU via CUDA Streams (no torch.distributed)

Bypasses gloo entirely. Uses CUDA streams to run work on both GPUs in parallel
within a single Python process. This is the right architecture for population-
parallel ES on a single node anyway, and avoids all Windows/Docker gloo issues.

Tests:
1. Parallel matmul on 2 GPUs via streams (should be ~2x speedup vs sequential)
2. Gather small tensors (fitnesses) from both GPUs to CPU
3. Broadcast tensor (weight update) from CPU to both GPUs

Pass criteria:
- Parallel speedup >= 1.5x (ideal 2x)
- Gather/broadcast overhead negligible

Run:  python 02_test_streams.py
Output: results_02_streams.json
"""

import json
import sys
import time
from pathlib import Path

import torch


def header(s):
    print(f"\n{'=' * 64}\n {s}\n{'=' * 64}")


def main():
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available")
        return 1
    if torch.cuda.device_count() < 2:
        print(f"[FAIL] Need 2 GPUs, found {torch.cuda.device_count()}")
        return 1

    header("Multi-GPU via CUDA Streams")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"GPU 1: {torch.cuda.get_device_name(1)}")
    print("Approach: single process, no torch.distributed")

    results = {}

    # =========================================================================
    # Test 1: Parallel matmul - measure speedup
    # =========================================================================
    header("Test 1: Parallel matmul (per-GPU streams)")

    n = 4096
    n_iter = 100

    # Allocate tensors on each GPU
    x0 = torch.randn(n, n, dtype=torch.bfloat16, device="cuda:0")
    y0 = torch.randn(n, n, dtype=torch.bfloat16, device="cuda:0")
    x1 = torch.randn(n, n, dtype=torch.bfloat16, device="cuda:1")
    y1 = torch.randn(n, n, dtype=torch.bfloat16, device="cuda:1")

    # Warmup
    for _ in range(10):
        _ = x0 @ y0
        _ = x1 @ y1
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)

    # Sequential: GPU 0 then GPU 1 (Python issues kernels sequentially)
    t0 = time.time()
    for _ in range(n_iter):
        z0 = x0 @ y0
        torch.cuda.synchronize(0)  # force sync after each iter
        z1 = x1 @ y1
        torch.cuda.synchronize(1)
    seq_time = time.time() - t0
    print(f"Sequential (with sync): {seq_time:.3f}s")

    # Async without explicit streams (PyTorch auto-overlaps cross-device ops)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        z0 = x0 @ y0
        z1 = x1 @ y1
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    auto_time = time.time() - t0
    print(f"Async (auto-overlap):   {auto_time:.3f}s  ({seq_time/auto_time:.2f}x)")

    # Explicit streams (most aggressive parallelism)
    s0 = torch.cuda.Stream(device=0)
    s1 = torch.cuda.Stream(device=1)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        with torch.cuda.stream(s0):
            z0 = x0 @ y0
        with torch.cuda.stream(s1):
            z1 = x1 @ y1
    s0.synchronize()
    s1.synchronize()
    stream_time = time.time() - t0
    print(f"Explicit streams:       {stream_time:.3f}s  ({seq_time/stream_time:.2f}x)")

    speedup = seq_time / min(auto_time, stream_time)
    parallel_works = speedup >= 1.5

    results["test1_parallel"] = {
        "sequential_s": round(seq_time, 4),
        "auto_overlap_s": round(auto_time, 4),
        "streams_s": round(stream_time, 4),
        "best_speedup": round(speedup, 2),
        "pass": parallel_works,
    }

    del x0, y0, x1, y1, z0, z1
    torch.cuda.empty_cache()

    # =========================================================================
    # Test 2: Gather small tensors (fitness scores) from both GPUs to CPU
    # =========================================================================
    header("Test 2: Gather fitness scores GPU -> CPU")

    f0 = torch.randn(128, device="cuda:0")  # 128 fitness scores from GPU 0
    f1 = torch.randn(128, device="cuda:1")  # 128 from GPU 1

    # Warmup
    for _ in range(10):
        all_f = torch.cat([f0.cpu(), f1.cpu()])

    n_iter = 1000
    t0 = time.time()
    for _ in range(n_iter):
        all_f = torch.cat([f0.cpu(), f1.cpu()])
    elapsed = time.time() - t0
    gather_us = elapsed / n_iter * 1e6
    print(f"Gather 256 floats: {gather_us:.1f} us avg ({elapsed*1000/n_iter:.3f} ms)")

    results["test2_gather"] = {
        "n_floats": 256,
        "avg_microseconds": round(gather_us, 1),
        "pass": gather_us < 1000,  # under 1ms
    }

    # =========================================================================
    # Test 3: Broadcast tensor (weight update) from CPU to both GPUs
    # =========================================================================
    header("Test 3: Broadcast weight update CPU -> 2 GPUs")

    # Simulate a typical update tensor: a single matmul matrix update
    # For UniMol2-1.1B Q projection: 1536 x 1536 = 2.36M params, 4.7 MB at bf16
    update = torch.randn(1536, 1536, dtype=torch.bfloat16, device="cpu")

    # Warmup
    for _ in range(5):
        u0 = update.cuda(0)
        u1 = update.cuda(1)
    torch.cuda.synchronize()

    n_iter = 100
    t0 = time.time()
    for _ in range(n_iter):
        u0 = update.cuda(0)
        u1 = update.cuda(1)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    bcast_ms = elapsed / n_iter * 1000
    print(f"Broadcast 4.7MB to 2 GPUs: {bcast_ms:.2f} ms avg")

    results["test3_broadcast"] = {
        "tensor_mb": 4.7,
        "avg_ms": round(bcast_ms, 2),
        "pass": bcast_ms < 100,
    }

    # =========================================================================
    # Test 4: Realistic ES generation simulation
    # =========================================================================
    header("Test 4: Simulated ES generation pattern")

    # Simulate: each GPU does a forward (heavy compute), then we gather
    # This mimics one generation of population-parallel ES.
    # Use chunk_size=4 forward as proxy.

    work_n = 1536  # mimic embed_dim
    x0 = torch.randn(512, work_n, dtype=torch.bfloat16, device="cuda:0")
    w0 = torch.randn(work_n, work_n, dtype=torch.bfloat16, device="cuda:0")
    x1 = torch.randn(512, work_n, dtype=torch.bfloat16, device="cuda:1")
    w1 = torch.randn(work_n, work_n, dtype=torch.bfloat16, device="cuda:1")

    def one_gen():
        # GPU 0: forward + compute fitness
        z0 = x0 @ w0
        for _ in range(20):  # simulate multi-layer
            z0 = z0 @ w0
        f0 = z0.sum(dim=1)[:128]
        # GPU 1: forward + compute fitness
        z1 = x1 @ w1
        for _ in range(20):
            z1 = z1 @ w1
        f1 = z1.sum(dim=1)[:128]
        # Gather to CPU
        all_f = torch.cat([f0.cpu(), f1.cpu()])
        return all_f

    # Warmup
    for _ in range(3):
        _ = one_gen()
    torch.cuda.synchronize()

    n_iter = 20
    t0 = time.time()
    for _ in range(n_iter):
        _ = one_gen()
    torch.cuda.synchronize()
    gen_time = (time.time() - t0) / n_iter
    print(f"Sim. generation time: {gen_time*1000:.1f} ms")
    print(f"  (20 layers x 1536x1536 matmul on each GPU + gather)")

    results["test4_simulated_gen"] = {
        "gen_time_ms": round(gen_time * 1000, 1),
        "pass": True,
    }

    # =========================================================================
    # VERDICT
    # =========================================================================
    header("VERDICT")
    all_pass = all(r["pass"] for r in results.values())

    if all_pass:
        print("[PASS] Multi-GPU via CUDA streams works.")
        print(f"  Parallel speedup: {speedup:.2f}x (ideal 2x)")
        print(f"  Gather overhead:  {gather_us:.0f} us per gen")
        print(f"  Broadcast overhead: {bcast_ms:.1f} ms per matrix update")
        print()
        print(">>> APPROACH FOR FULL PIPELINE: <<<")
        print("  Single Python process. Use torch.cuda.set_device() and")
        print("  torch.cuda.Stream() to run work on both GPUs in parallel.")
        print("  Gather fitnesses via .cpu() (fast).")
        print("  Compute EGGROLL update on CPU or GPU 0.")
        print("  Broadcast updated weights to both GPUs each generation.")
    elif results["test1_parallel"]["pass"]:
        print("[PARTIAL] Parallelism OK, but some overhead high.")
    else:
        print(f"[FAIL] Parallelism speedup only {speedup:.2f}x.")
        print("  Possible causes:")
        print("  - WDDM scheduler limiting cross-GPU concurrency")
        print("  - Memory bandwidth bottleneck")
        print("  - Process priority issues")
        print("  Recommend: try WSL2 for better scheduler.")

    out = Path(__file__).parent / "results_02_streams.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())