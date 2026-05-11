# Phase 0 Report

## CUDA / GPU
- GPU 0: NVIDIA GeForce RTX 5070 Ti, 17.09 GB, compute 12.0
- GPU 1: NVIDIA GeForce RTX 5070 Ti, 17.09 GB, compute 12.0
- bf16 OK: True
- Cross-GPU bandwidth: 16.82 GB/s
- GPU 0 matmul: 98.9 TFLOPS bf16
- GPU 1 matmul: 83.6 TFLOPS bf16

## Multi-GPU
- Recommended backend: None

## Forward Benchmark
### unimol2_570M
- chunk=1: 101.6 ms, peak 1.82 GB
- chunk=2: 197.0 ms, peak 1.83 GB
- chunk=4: 393.6 ms, peak 1.83 GB
- chunk=8: 784.7 ms, peak 2.73 GB
- chunk=16: 1563.7 ms, peak 4.54 GB
### unimol2_1100M
- chunk=1: 203.5 ms, peak 3.64 GB
- chunk=2: 393.2 ms, peak 3.64 GB
- chunk=4: 785.4 ms, peak 3.64 GB
- chunk=8: 1567.8 ms, peak 3.64 GB
- chunk=16: 3131.5 ms, peak 5.45 GB

## Pipeline Projections (5000 generations)

| Arch | Pop | Chunk | Per-gen | Total |
|------|-----|-------|---------|-------|
| unimol2_570M | 128 | 16 | 6.31s | 8.8h (0.37d) |
| unimol2_570M | 256 | 16 | 12.58s | 17.5h (0.73d) |
| unimol2_570M | 512 | 16 | 25.11s | 34.9h (1.45d) |
| unimol2_1100M | 128 | 16 | 12.59s | 17.5h (0.73d) |
| unimol2_1100M | 256 | 16 | 25.12s | 34.9h (1.45d) |
| unimol2_1100M | 512 | 16 | 50.19s | 69.7h (2.9d) |
