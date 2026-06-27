import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

def percentile(values, p):
    return float(np.percentile(values, p))

def benchmark_cpui(iterations,  size):
    latencies_ms = []

    a = torch.randn(size, size)
    b = torch.randn(size, size)

    for _ in range(iterations):
        start = time.perf_counter()
        c = torch.matmul(a, b)
        _ = c[0,0].item()
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000)

    return latencies_ms

def benchmark_gpu(iterations, size):
    if not torch.cuda.is_available():
        return None
    
    device = torch.device("cuda")

    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)

    transfer_to_gpu_ms = []
    compute_ms = []
    transfer_to_cpu_ms = []
    end_to_end_ms = []

    for _ in range(10):
        a = a_cpu.to(device)
        b = b_cpu.to(device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        _ = c.cpu()


    for _ in range(iterations):
        start_total = time.perf_counter()

        start = time.perf_counter()
        a = a_cpu.to(device)
        b = b_cpu.to(device)
        torch.cuda.synchronize()
        end = time.perf_counter()
        transfer_to_gpu_ms.append((end - start) * 1000)

        start = time.perf_counter()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.perf_counter()
        compute_ms.append((end - start) * 1000)

        start = time.perf_counter()
        _ = c.cpu()
        torch.cuda.synchronize()
        end = time.perf_counter()
        transfer_to_cpu_ms.append((end - start) * 1000)

        end_total = time.perf_counter()
        end_to_end_ms.append((end_total - start_total) * 1000)

        return {
            "transfer_to_gpu_ms": transfer_to_gpu_ms,
            "compute_ms": compute_ms,
            "transfer_to_cpu_ms": transfer_to_cpu_ms,
            "end_to_end_ms": end_to_end_ms
        }
    
    def summarize(name, latencies_ms):
        return {
            "name": name,
            "samples": len(latencies_ms),
            "mean_ms": float(np.mean(latencies_ms)),
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "p99": percentile(latencies_ms, 99),
            "min_ms": float(np.min(latencies_ms)),
            "max_ms": float(np.max(latencies_ms))
        }
    
    def main():
        iterations = 100
        size = 1024

        results_dir  = Path("results")
        results_dir.mkdir(exist_ok=True)

        rows = []

        print("Running CPU benchmark")
        cpu_latencies = benchmark_cpui(iterations, size)
        rows.append(summarize("cpu_matmul_end_to_end", cpu_latencies))

        print("Running GPU benchmark")
        gpu_results = benchmark_gpu(iterations, size)

        if gpu_results is None:
            print("CUDA is not available. Skipping GPU benchmark.")
        else:
            rows.append(summarize("gpu_matmul_transfer_to_gpu", gpu_results["transfer_to_gpu_ms"]))
            rows.append(summarize("gpu_matmul_compute", gpu_results["compute_ms"]))
            rows.append(summarize("gpu_matmul_transfer_to_cpu", gpu_results["transfer_to_cpu_ms"]))
            rows.append(summarize("gpu_matmul_end_to_end", gpu_results["end_to_end_ms"]))


        df = pd.DataFrame(rows)
        output_path = results_dir / "001_cpu_vs_gpu_summary.csv"
        df.to_csv(output_path, index=False)

        print()
        print(df.to_string(index=False))
        print()
        print(f"Benchmark summary saved to {output_path}")
        
    if __name__ == "__main__":
        main()