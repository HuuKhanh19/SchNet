"""
Phase 0.2 v2 - Multi-GPU with Windows fixes

Fix for gloo "unsupported gloo device" error on Windows with Docker Desktop:
- USE_LIBUV=0 disables libuv-based TCPStore (avoids hostname resolution issues)
- File-based init as fallback if TCP still fails

CRITICAL: USE_LIBUV must be set BEFORE any torch.distributed import.

Run:  python 02_test_multigpu_v2.py
Output: results_02_multigpu_v2.json
"""

import os
# MUST be set before importing torch.distributed
os.environ["USE_LIBUV"] = "0"

import json
import sys
import time
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


HERE = Path(__file__).parent


def common_tests(rank, world_size, device):
    """Run all-reduce, broadcast, bandwidth tests. Returns dict."""
    out = {"all_reduce_ok": False, "broadcast_ok": False, "bandwidth_gbs": 0.0}

    # All-reduce
    t = torch.ones(10, device=device) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(r + 1 for r in range(world_size))
    out["all_reduce_ok"] = bool(abs(t[0].item() - expected) < 1e-3)

    # Broadcast
    if rank == 0:
        t = torch.tensor([42.0], device=device)
    else:
        t = torch.tensor([0.0], device=device)
    dist.broadcast(t, src=0)
    out["broadcast_ok"] = bool(abs(t.item() - 42.0) < 1e-3)

    # Bandwidth
    n = 25_000_000  # 100 MB at fp32
    t = torch.randn(n, device=device)
    for _ in range(2):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    out["bandwidth_gbs"] = n * 4 * 10 / (time.time() - t0) / 1e9

    return out


def worker_tcp(rank, world_size, port):
    os.environ["USE_LIBUV"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    result = {"method": "tcp", "rank": rank, "error": None}
    try:
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=world_size,
            timeout=timedelta(seconds=30),
        )
        torch.cuda.set_device(rank)
        result.update(common_tests(rank, world_size, f"cuda:{rank}"))
        dist.destroy_process_group()
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        if dist.is_initialized():
            try: dist.destroy_process_group()
            except Exception: pass

    with open(HERE / f"_tmp_tcp_rank{rank}.json", "w") as f:
        json.dump(result, f)


def worker_file(rank, world_size, init_file):
    os.environ["USE_LIBUV"] = "0"

    result = {"method": "file", "rank": rank, "error": None}
    try:
        # File path needs proper URI format on Windows
        init_method = f"file:///{init_file.replace(chr(92), '/')}"
        dist.init_process_group(
            backend="gloo", init_method=init_method,
            rank=rank, world_size=world_size,
            timeout=timedelta(seconds=30),
        )
        torch.cuda.set_device(rank)
        result.update(common_tests(rank, world_size, f"cuda:{rank}"))
        dist.destroy_process_group()
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        if dist.is_initialized():
            try: dist.destroy_process_group()
            except Exception: pass

    with open(HERE / f"_tmp_file_rank{rank}.json", "w") as f:
        json.dump(result, f)


def cleanup_tmp(prefix):
    for r in range(2):
        f = HERE / f"_tmp_{prefix}_rank{r}.json"
        if f.exists():
            f.unlink()


def collect_results(prefix):
    results = []
    for r in range(2):
        f = HERE / f"_tmp_{prefix}_rank{r}.json"
        if f.exists():
            with open(f) as fh:
                results.append(json.load(fh))
            f.unlink()
    return results


def try_method(method_name, worker_fn, *args):
    print(f"\n--- gloo via {method_name} ---")
    cleanup_tmp(method_name)

    try:
        mp.spawn(worker_fn, args=args, nprocs=2, join=True)
    except Exception as e:
        print(f"  [FAIL] mp.spawn: {e}")
        return None

    results = collect_results(method_name)
    if not results:
        print("  [FAIL] No results collected")
        return None
    if results[0].get("error"):
        print(f"  [FAIL] {results[0]['error']}")
        return None

    r = results[0]
    print(f"  all_reduce: {'OK' if r['all_reduce_ok'] else 'FAIL'}")
    print(f"  broadcast:  {'OK' if r['broadcast_ok'] else 'FAIL'}")
    print(f"  bandwidth:  {r['bandwidth_gbs']:.2f} GB/s")
    return r


def main():
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available")
        return 1
    if torch.cuda.device_count() < 2:
        print(f"[FAIL] Need 2 GPUs, found {torch.cuda.device_count()}")
        return 1

    print("=" * 64)
    print(" Multi-GPU Tests (Windows fixes)")
    print("=" * 64)
    print(f"PyTorch: {torch.__version__}")
    print(f"USE_LIBUV: {os.environ.get('USE_LIBUV')}")
    print(f"NCCL avail: {dist.is_nccl_available()}")
    print(f"Gloo avail: {dist.is_gloo_available()}")

    if not dist.is_gloo_available():
        print("[FAIL] Gloo not in build")
        return 1

    # Try TCP method first
    r_tcp = try_method("tcp", worker_tcp, 2, 29500)

    # Fall back to file-based init
    init_file = tempfile.NamedTemporaryFile(delete=False, suffix=".init").name
    if Path(init_file).exists():
        Path(init_file).unlink()  # init expects non-existent file initially

    r_file = try_method("file", worker_file, 2, init_file)

    # Cleanup init file if remaining
    if Path(init_file).exists():
        try: Path(init_file).unlink()
        except Exception: pass

    print("\n" + "=" * 64)
    print(" VERDICT")
    print("=" * 64)

    final = r_tcp or r_file
    method_used = "tcp" if r_tcp else ("file" if r_file else None)

    if final and final["all_reduce_ok"]:
        print(f"[PASS] gloo works via {method_used}")
        print(f"  Bandwidth: {final['bandwidth_gbs']:.2f} GB/s")
        if final["bandwidth_gbs"] >= 0.5:
            print("  Bandwidth fine for ES (we send tiny tensors per generation).")
            print("\n>>> Stay on Windows native is OK. <<<")
        else:
            print("  [WARN] Low bandwidth. Consider WSL2 if perf matters.")
    else:
        print("[FAIL] Gloo not working via TCP or file.")
        print("\nNext steps:")
        print("  1. Check Docker Desktop is OFF (it modifies hosts file)")
        print("  2. Try in PowerShell as Administrator")
        print("  3. If still fails: switch to WSL2")

    out = HERE / "results_02_multigpu_v2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "use_libuv": False,
            "tcp_result": r_tcp,
            "file_result": r_file,
            "method_used": method_used,
            "recommend_backend": "gloo" if final else None,
        }, f, indent=2)
    print(f"\nSaved: {out}")

    return 0 if final else 1


if __name__ == "__main__":
    sys.exit(main())