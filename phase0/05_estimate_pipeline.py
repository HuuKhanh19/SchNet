"""
Phase 0.5 - Pipeline Time Estimate

Reads JSONs from 01-04 and projects:
- Time per generation
- Total training time for various configs
- Recommended config based on hardware

Run:  python 05_estimate_pipeline.py
Output: PHASE0_REPORT.md
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def load_json(name):
    p = HERE / name
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Cannot load {name}: {e}")
        return None


def estimate(forward_time_per_chunk, chunk_size, pop_size, n_gpus, n_generations,
             ls_time_per_perturbation, batch_m=128, d_emb=1537):
    """Project pipeline time.

    chunk_size: # perturbations per forward call
    pop_size: total perturbations per generation (across all GPUs)
    n_gpus: 2
    forward_time_per_chunk: from synthetic/real benchmark
    ls_time_per_perturbation: rough estimate from d^3 scaling
    """
    pop_per_gpu = pop_size // n_gpus
    chunks_per_gpu = pop_per_gpu // chunk_size

    forward_per_gen = chunks_per_gpu * forward_time_per_chunk
    ls_per_gen = pop_per_gpu * ls_time_per_perturbation
    comm_per_gen = 0.05  # small overhead for fitness gather (tiny tensor)
    total_per_gen = forward_per_gen + ls_per_gen + comm_per_gen
    total = total_per_gen * n_generations

    return {
        "pop_size": pop_size,
        "chunk_size": chunk_size,
        "chunks_per_gpu": chunks_per_gpu,
        "forward_per_gen_s": round(forward_per_gen, 2),
        "ls_per_gen_s": round(ls_per_gen, 2),
        "total_per_gen_s": round(total_per_gen, 2),
        "total_hours": round(total / 3600, 1),
        "total_days": round(total / 86400, 2),
    }


def estimate_ls_time(d=1537, gpu_tflops_fp32=44):
    """Cholesky O(d^3) dominates. fp32 throughput on RTX 5070 Ti ~44 TFLOPS."""
    ops = d ** 3 * 1.0  # rough op count for chol + triangular solve
    return ops / (gpu_tflops_fp32 * 1e12)


def main():
    print("=" * 64)
    print(" Pipeline Time Estimate")
    print("=" * 64)

    cuda_r = load_json("results_01_cuda.json")
    mgpu_r = load_json("results_02_multigpu.json")
    synth_r = load_json("results_03_synthetic.json")
    unimol_r = load_json("results_04_unimol2.json")

    if cuda_r is None or synth_r is None:
        print("[FAIL] Missing results from 01 or 03. Run those first.")
        return 1

    # Pull forward time data
    print("\n--- Forward time data ---")
    forward_data = {}  # config_name -> {chunk_size: forward_time_s}
    for cfg, runs in synth_r.get("configs", {}).items():
        forward_data[cfg] = {}
        for r in runs:
            if not r.get("oom"):
                forward_data[cfg][r["chunk_size"]] = r["forward_time_s"]

    print("(synthetic, lower bound)")
    for cfg, ct in forward_data.items():
        print(f"  {cfg}: {ct}")

    # Adjust if real UniMol2 is available
    real_overhead = 1.0
    if unimol_r and unimol_r.get("runs"):
        real_runs = [r for r in unimol_r["runs"] if not r.get("oom") and not r.get("error")]
        if real_runs:
            # Compare batch=128 forward time
            arch = unimol_r.get("arch", "unimol2_570M")
            real_t128 = next((r["forward_time_s"] for r in real_runs if r["batch_size"] == 128), None)
            synth_t128 = forward_data.get(arch, {}).get(1, None)
            if real_t128 and synth_t128:
                real_overhead = real_t128 / synth_t128
                print(f"\n(real UniMol2 / synthetic ratio: {real_overhead:.2f}x)")

    # LS estimate
    avg_tflops = sum(p["tflops"] for p in cuda_r.get("matmul_tflops", [])) / max(1, len(cuda_r.get("matmul_tflops", [])))
    # bf16 TFLOPS, but LS needs fp32, assume 4x lower
    ls_tflops_fp32 = avg_tflops / 4
    ls_t = estimate_ls_time(d=1537, gpu_tflops_fp32=max(ls_tflops_fp32, 10))
    print(f"\n--- LS time estimate ---")
    print(f"FP32 throughput estimate: {ls_tflops_fp32:.1f} TFLOPS")
    print(f"LS solve per perturbation (d=1537): {ls_t*1000:.1f} ms")

    # Estimates for various configs
    print("\n--- Pipeline projections ---")
    print(f"(real_overhead applied: {real_overhead:.2f}x)\n")

    table_rows = []
    n_generations = 5000

    for cfg in ["unimol2_570M", "unimol2_1100M"]:
        if cfg not in forward_data or not forward_data[cfg]:
            continue
        # Pick max viable chunk size
        max_chunk = max(forward_data[cfg].keys())
        ft = forward_data[cfg][max_chunk] * real_overhead

        for pop in [128, 256, 512]:
            r = estimate(
                forward_time_per_chunk=ft,
                chunk_size=max_chunk,
                pop_size=pop,
                n_gpus=2,
                n_generations=n_generations,
                ls_time_per_perturbation=ls_t,
            )
            row = {"arch": cfg, **r}
            table_rows.append(row)
            print(f"{cfg:15s}  pop={pop:4d}  chunk={max_chunk:2d}  "
                  f"per-gen={r['total_per_gen_s']:5.2f}s  "
                  f"5000 gens = {r['total_hours']:5.1f}h ({r['total_days']:.2f} days)")

    # Recommendation
    print("\n" + "=" * 64)
    print(" RECOMMENDATION")
    print("=" * 64)

    # Pick configs that fit in <2 days
    viable = [r for r in table_rows if r["total_days"] < 2]
    if not viable:
        print("[WARN] No config fits in 2 days. Consider:")
        print("  - Smaller model (unimol2_310M not in this benchmark)")
        print("  - Fewer generations (T=2000 instead of 5000)")
        print("  - Smaller population")
    else:
        print("Viable configs (training < 2 days):")
        for r in viable[:5]:
            print(f"  {r['arch']:15s}  pop={r['pop_size']:4d}  "
                  f"chunk={r['chunk_size']:2d}  -> {r['total_days']:.2f} days")
        print()

        # Best for debug iteration
        debug = min((r for r in viable if r["arch"] == "unimol2_570M"),
                    key=lambda x: x["total_days"], default=None)
        if debug:
            print(f"For Phase 1-5 (debug iteration): {debug['arch']} pop={debug['pop_size']} "
                  f"chunk={debug['chunk_size']} (~{debug['total_hours']:.1f}h per full run)")

        # Best for final
        final = min((r for r in viable if r["arch"] == "unimol2_1100M"),
                    key=lambda x: x["total_days"], default=None)
        if final:
            print(f"For Phase 8 (final scale): {final['arch']} pop={final['pop_size']} "
                  f"chunk={final['chunk_size']} (~{final['total_hours']:.1f}h per full run)")
        else:
            print("Phase 8 (1.1B): not viable on this hardware. Stick with 570M.")

    # Decision: WSL2 vs native Windows
    print("\n--- WSL2 vs Native Windows ---")
    backend = mgpu_r.get("recommend_backend") if mgpu_r else None
    nccl_in_build = mgpu_r.get("nccl_in_build", False) if mgpu_r else False

    if backend == "nccl":
        print("NCCL is available -> already on Linux/WSL2. No move needed.")
    elif backend == "gloo":
        bw_gloo = mgpu_r["gloo"].get("bandwidth_gbs", 0) if mgpu_r else 0
        if bw_gloo > 1.0:
            print(f"Gloo bandwidth {bw_gloo:.2f} GB/s is acceptable for ES.")
            print("Decision: native Windows is FINE for this workload.")
            print("  WSL2 would give ~2x speedup on cross-GPU ops, but those are tiny in our pipeline.")
        else:
            print(f"Gloo bandwidth {bw_gloo:.2f} GB/s is low.")
            print("Decision: consider WSL2 for better perf, OR proceed with Windows native and accept overhead.")
    else:
        print("[WARN] No working backend. Move to WSL2 mandatory.")

    # Write report
    report = HERE / "PHASE0_REPORT.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Phase 0 Report\n\n")
        f.write("## CUDA / GPU\n")
        if cuda_r:
            for g in cuda_r.get("gpus", []):
                f.write(f"- GPU {g['index']}: {g['name']}, {g['memory_gb']} GB, "
                        f"compute {g['compute_capability']}\n")
            f.write(f"- bf16 OK: {cuda_r.get('bf16_ok')}\n")
            f.write(f"- Cross-GPU bandwidth: {cuda_r.get('cross_gpu_bandwidth_gbs')} GB/s\n")
            for p in cuda_r.get("matmul_tflops", []):
                f.write(f"- GPU {p['gpu']} matmul: {p['tflops']} TFLOPS bf16\n")

        f.write("\n## Multi-GPU\n")
        if mgpu_r:
            f.write(f"- Recommended backend: {mgpu_r.get('recommend_backend')}\n")
            if mgpu_r.get("gloo", {}).get("bandwidth_gbs"):
                f.write(f"- Gloo bandwidth: {mgpu_r['gloo']['bandwidth_gbs']:.2f} GB/s\n")
            if mgpu_r.get("nccl", {}).get("bandwidth_gbs"):
                f.write(f"- NCCL bandwidth: {mgpu_r['nccl']['bandwidth_gbs']:.2f} GB/s\n")

        f.write("\n## Forward Benchmark\n")
        for cfg, runs in synth_r.get("configs", {}).items():
            f.write(f"### {cfg}\n")
            for r in runs:
                if r.get("oom"):
                    f.write(f"- chunk={r['chunk_size']}: OOM\n")
                else:
                    f.write(f"- chunk={r['chunk_size']}: {r['forward_time_s']*1000:.1f} ms, "
                            f"peak {r['peak_mem_gb']} GB\n")

        f.write("\n## Pipeline Projections (5000 generations)\n\n")
        f.write("| Arch | Pop | Chunk | Per-gen | Total |\n")
        f.write("|------|-----|-------|---------|-------|\n")
        for r in table_rows:
            f.write(f"| {r['arch']} | {r['pop_size']} | {r['chunk_size']} | "
                    f"{r['total_per_gen_s']}s | {r['total_hours']}h ({r['total_days']}d) |\n")

    print(f"\nReport written: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())