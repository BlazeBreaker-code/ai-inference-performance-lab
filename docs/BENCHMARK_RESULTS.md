# Benchmark Results

This document captures benchmark results for the AI Inference Performance Lab.

The current experiment benchmarks a synthetic PyTorch inference workload across CPU and CUDA execution paths, then compares how batch size affects throughput, p50 latency, p95 latency, p99 latency, model execution time, and host to device transfer overhead.

## Experiment Setup

Model configuration:

```text
Input dimension: 1024
Hidden dimension: 2048
Output dimension: 512
Depth: 4
Activation: GELU
Warmup iterations: 25
Measurement iterations: 200
Batch sizes: 1, 2, 4, 8, 16, 32, 64
```

Metrics collected:

```text
Throughput: requests per second
p50 latency: median end to end latency
p95 latency: tail latency signal
p99 latency: worst tail latency signal
Model p95: model execution p95 latency
Transfer p95: host to device transfer p95 latency
```

## Experiment 001: CPU Batch Sweep

Goal:

Measure how inference latency and throughput change as batch size increases on CPU.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_cpu_sweep.ps1
```

Environment:

```text
Device: cpu
Warmup iterations: 25
Measurement iterations: 200
Input dimension: 1024
Hidden dimension: 2048
Output dimension: 512
Depth: 4
```

Results:

| Batch | Throughput     | p50 latency | p95 latency | p99 latency | Model p95 | Transfer p95 |
| ----- | -------------- | ----------- | ----------- | ----------- | --------- | ------------ |
| 1     | 653.93 req/s   | 1.4892 ms   | 1.7620 ms   | 2.2474 ms   | 1.7620 ms | 0.0000 ms    |
| 2     | 2276.32 req/s  | 0.8692 ms   | 0.9328 ms   | 1.0587 ms   | 0.9328 ms | 0.0000 ms    |
| 4     | 3036.62 req/s  | 1.4202 ms   | 1.7259 ms   | 1.8952 ms   | 1.7259 ms | 0.0000 ms    |
| 8     | 3918.51 req/s  | 2.0319 ms   | 2.7475 ms   | 3.2479 ms   | 2.7475 ms | 0.0000 ms    |
| 16    | 6861.52 req/s  | 2.2989 ms   | 3.2259 ms   | 3.7300 ms   | 3.2259 ms | 0.0000 ms    |
| 32    | 9724.32 req/s  | 3.2740 ms   | 3.9512 ms   | 4.2780 ms   | 3.9512 ms | 0.0000 ms    |
| 64    | 12676.54 req/s | 5.0428 ms   | 5.7289 ms   | 5.9324 ms   | 5.7289 ms | 0.0000 ms    |

Observations:

Throughput increased from 653.93 req/s at batch size 1 to 12676.54 req/s at batch size 64.

p95 latency increased from 1.7620 ms at batch size 1 to 5.7289 ms at batch size 64.

Batch size 2 produced the lowest CPU p95 latency in this run at 0.9328 ms.

On CPU, host to device transfer latency is 0.0000 ms because no GPU transfer occurs.

Takeaway:

The CPU benchmark shows the core batching tradeoff. Larger batches significantly improve throughput, but they also increase p95 and p99 latency as more work is grouped into each inference call.

## Experiment 002: GPU Batch Sweep

Goal:

Measure how CUDA inference latency and throughput change as batch size increases, including host to device transfer overhead.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_gpu_sweep.ps1
```

Environment:

```text
Device: cuda
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
PyTorch: 2.11.0+cu128
CUDA build: 12.8
Warmup iterations: 25
Measurement iterations: 200
Input dimension: 1024
Hidden dimension: 2048
Output dimension: 512
Depth: 4
```

Results:

| Batch | Throughput     | p50 latency | p95 latency | p99 latency | Model p95 | Transfer p95 |
| ----- | -------------- | ----------- | ----------- | ----------- | --------- | ------------ |
| 1     | 3367.69 req/s  | 0.2958 ms   | 0.3067 ms   | 0.3144 ms   | 0.2798 ms | 0.0277 ms    |
| 2     | 7244.33 req/s  | 0.2743 ms   | 0.2859 ms   | 0.3480 ms   | 0.2593 ms | 0.0281 ms    |
| 4     | 14528.84 req/s | 0.2756 ms   | 0.2854 ms   | 0.2993 ms   | 0.2585 ms | 0.0292 ms    |
| 8     | 23467.29 req/s | 0.3364 ms   | 0.3509 ms   | 0.3724 ms   | 0.3257 ms | 0.0277 ms    |
| 16    | 29226.09 req/s | 0.5539 ms   | 0.5632 ms   | 0.5689 ms   | 0.5341 ms | 0.0327 ms    |
| 32    | 57494.86 req/s | 0.5625 ms   | 0.5716 ms   | 0.5966 ms   | 0.5168 ms | 0.0562 ms    |
| 64    | 86128.65 req/s | 0.7487 ms   | 0.7596 ms   | 0.7759 ms   | 0.6893 ms | 0.0731 ms    |

Observations:

GPU throughput increased from 3367.69 req/s at batch size 1 to 86128.65 req/s at batch size 64.

GPU p95 latency increased from 0.3067 ms at batch size 1 to 0.7596 ms at batch size 64.

Batch size 4 produced the lowest GPU p95 latency in this run at 0.2854 ms.

Host to device transfer overhead stayed small, ranging from 0.0277 ms to 0.0731 ms p95.

Takeaway:

The CUDA benchmark shows a large throughput and latency advantage over CPU execution for this synthetic inference workload. Larger batches dramatically improve throughput while p95 latency rises gradually.

## CPU vs GPU Comparison

| Batch | CPU throughput | GPU throughput | Throughput speedup | CPU p95   | GPU p95   | p95 latency improvement |
| ----- | -------------- | -------------- | ------------------ | --------- | --------- | ----------------------- |
| 1     | 653.93 req/s   | 3367.69 req/s  | 5.15x              | 1.7620 ms | 0.3067 ms | 5.74x lower             |
| 2     | 2276.32 req/s  | 7244.33 req/s  | 3.18x              | 0.9328 ms | 0.2859 ms | 3.26x lower             |
| 4     | 3036.62 req/s  | 14528.84 req/s | 4.78x              | 1.7259 ms | 0.2854 ms | 6.05x lower             |
| 8     | 3918.51 req/s  | 23467.29 req/s | 5.99x              | 2.7475 ms | 0.3509 ms | 7.83x lower             |
| 16    | 6861.52 req/s  | 29226.09 req/s | 4.26x              | 3.2259 ms | 0.5632 ms | 5.73x lower             |
| 32    | 9724.32 req/s  | 57494.86 req/s | 5.91x              | 3.9512 ms | 0.5716 ms | 6.91x lower             |
| 64    | 12676.54 req/s | 86128.65 req/s | 6.79x              | 5.7289 ms | 0.7596 ms | 7.54x lower             |

Key results:

At batch size 1, GPU throughput was about 5.15x higher than CPU throughput.

At batch size 64, GPU throughput was about 6.79x higher than CPU throughput.

At batch size 64, GPU p95 latency was about 7.54x lower than CPU p95 latency.

The best low latency GPU result occurred at batch size 4.

The best throughput GPU result occurred at batch size 64.

## Overall Takeaway

This benchmark establishes the first complete CPU vs CUDA inference performance comparison for the lab.

The results show that CUDA execution provides significantly higher throughput and lower p95 latency for this synthetic inference workload.

The benchmark also demonstrates the main inference batching tradeoff:

Higher batch sizes improve throughput, but they increase per batch latency and tail latency.

This experiment provides the foundation for future model serving, request queueing, CUDA kernel, and C++ runtime path experiments.

## Notes

Absolute timings vary by machine, driver version, GPU model, CPU load, power mode, scheduler behavior, PyTorch version, and benchmark configuration.

These results were collected from a local benchmark run and should be interpreted as machine-specific measurements rather than universal performance claims.
