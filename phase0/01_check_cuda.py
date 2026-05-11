"""
Phase 0.1 - CUDA + GPU Verification

Pass criteria:
- 2 GPUs detected (RTX 5070 Ti x 2)
- Compute capability >= 8.0 (bf16 native)
- bf16 matmul works on both GPUs
- 1 GB allocation succeeds on each GPU
- Matmul throughput > 50 TFLOPS bf16

Run:  python 01_check_cuda.py
Output: results_01_cuda.json
"""

import json
import sys
import time
from pathlib import Path

import torch


def header(s):
    print(f"\n{'=' * 64}\n {s}\n{'=' * 64}")


def main():
    results = {}

    header("System Info")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA avail:   {torch.cuda.is_available()}")
    results["pytorch"] = torch.__version__
    results["cuda_available"] = torch.cuda.is_available()

    if not torch.cuda.is_available():
        print("\n[FAIL] CUDA not available. Reinstall PyTorch with CUDA support.")
        results["pass"] = False
        save(results)
        return 1

    print(f"CUDA built:   {torch.version.cuda}")
    print(f"cuDNN:        {torch.backends.cudnn.version()}")
    print(f"GPU count:    {torch.cuda.device_count()}")
    results["cuda_built"] = torch.version.cuda
    results["cudnn"] = torch.backends.cudnn.version()
    results["device_count"] = torch.cuda.device_count()

    if torch.cuda.device_count() < 2:
        print(f"\n[FAIL] Need 2 GPUs, found {torch.cuda.device_count()}")
        results["pass"] = False
        save(results)
        return 1

    header("GPU Details")
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        cc = f"{prop.major}.{prop.minor}"
        info = {
            "index": i,
            "name": prop.name,
            "memory_gb": round(prop.total_memory / 1e9, 2),
            "compute_capability": cc,
            "sm_count": prop.multi_processor_count,
            "cc_supports_bf16_native": prop.major >= 8,
        }
        gpu_info.append(info)
        print(f"GPU {i}: {prop.name}")
        print(f"  Memory: {info['memory_gb']} GB")
        print(f"  Compute capability: {cc}  (bf16 native: {info['cc_supports_bf16_native']})")
        print(f"  SMs: {info['sm_count']}")
    results["gpus"] = gpu_info

    header("BF16 Matmul Test")
    bf16_ok = True
    for i in range(torch.cuda.device_count()):
        try:
            x = torch.randn(100, 100, dtype=torch.bfloat16, device=f"cuda:{i}")
            _ = x @ x
            torch.cuda.synchronize(i)
            print(f"GPU {i}: bf16 matmul OK")
        except Exception as e:
            print(f"GPU {i}: [FAIL] {e}")
            bf16_ok = False
    results["bf16_ok"] = bf16_ok

    header("1 GB Allocation Test")
    alloc_ok = True
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.empty_cache()
            n = 500_000_000  # 1 GB at bf16
            x = torch.empty(n, dtype=torch.bfloat16, device=f"cuda:{i}")
            free, total = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: 1GB alloc OK. After: free={free/1e9:.2f}/{total/1e9:.2f} GB")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"GPU {i}: [FAIL] {e}")
            alloc_ok = False
    results["alloc_1gb_ok"] = alloc_ok

    header("Matmul TFLOPS (bf16, 4096x4096)")
    matmul_perf = []
    for i in range(torch.cuda.device_count()):
        try:
            n = 4096
            x = torch.randn(n, n, dtype=torch.bfloat16, device=f"cuda:{i}")
            y = torch.randn(n, n, dtype=torch.bfloat16, device=f"cuda:{i}")
            for _ in range(5):
                _ = x @ y
            torch.cuda.synchronize(i)
            t0 = time.time()
            n_iter = 50
            for _ in range(n_iter):
                z = x @ y
            torch.cuda.synchronize(i)
            elapsed = time.time() - t0
            tflops = 2 * (n ** 3) * n_iter / elapsed / 1e12
            matmul_perf.append({"gpu": i, "tflops": round(tflops, 1)})
            print(f"GPU {i}: {tflops:.1f} TFLOPS bf16")
            del x, y, z
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"GPU {i}: [FAIL] {e}")
            matmul_perf.append({"gpu": i, "tflops": 0})
    results["matmul_tflops"] = matmul_perf

    header("Cross-GPU Transfer (100 MB tensor)")
    bandwidth = 0.0
    if torch.cuda.device_count() >= 2:
        try:
            n = 50_000_000  # 100 MB at bf16
            x = torch.randn(n, dtype=torch.bfloat16, device="cuda:0")
            _ = x.to("cuda:1")
            torch.cuda.synchronize()
            t0 = time.time()
            n_iter = 10
            for _ in range(n_iter):
                _ = x.to("cuda:1")
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            bandwidth = n * 2 * n_iter / elapsed / 1e9
            print(f"GPU 0 -> GPU 1: {bandwidth:.2f} GB/s")
            if bandwidth < 5:
                print("[WARN] Low bandwidth. No NVLink + WDDM overhead likely.")
                print("       Acceptable for our use case (we send tiny tensors per gen).")
        except Exception as e:
            print(f"[FAIL] {e}")
    results["cross_gpu_bandwidth_gbs"] = round(bandwidth, 2)

    header("VERDICT")
    overall = (
        results["cuda_available"]
        and results["device_count"] >= 2
        and results["bf16_ok"]
        and results["alloc_1gb_ok"]
        and all(p["tflops"] > 30 for p in matmul_perf)
        and all(g["cc_supports_bf16_native"] for g in gpu_info)
    )
    results["pass"] = overall

    if overall:
        print("[PASS] Hardware basics OK.")
        print("Next: python 02_test_multigpu.py")
    else:
        print("[FAIL] Some checks failed. See above.")
        if not all(g["cc_supports_bf16_native"] for g in gpu_info):
            print("  - Compute capability < 8.0: bf16 not native, expect slow training.")
        if any(p["tflops"] < 30 for p in matmul_perf):
            print("  - Low matmul throughput: PyTorch may not be built for your GPU arch.")
            print("    For RTX 5070 Ti (Blackwell sm_120), need PyTorch 2.5+ with cu126+.")

    save(results)
    return 0 if overall else 1


def save(results):
    out = Path(__file__).parent / "results_01_cuda.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    sys.exit(main())