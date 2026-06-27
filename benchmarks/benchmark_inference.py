from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.benchmark_runner import BenchmarkConfig, run_batch_sweep, run_single_benchmark


def _parse_batch_sizes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AI inference performance benchmarks.")

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-sizes", default="")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--measure-transfer", action="store_true")
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    config = BenchmarkConfig(
        device=args.device,
        batch_size=args.batch_size,
        warmup_iterations=args.warmup,
        measurement_iterations=args.iterations,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        depth=args.depth,
        measure_transfer=args.measure_transfer,
    )

    if args.batch_sizes:
        result = run_batch_sweep(config, _parse_batch_sizes(args.batch_sizes))
    else:
        result = run_single_benchmark(config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    resolved_device = result.get("resolved_device", result.get("results", [{}])[0].get("resolved_device", args.device))

    print("AI Inference Performance Benchmark")
    print(f"Device: {resolved_device}")
    print(f"Output: {output_path}")

    if result.get("experiment") == "batch_size_sweep":
        print("Batch size sweep completed.")
    else:
        end_to_end = result["stages"]["end_to_end_ms"]
        print(f"Batch size: {args.batch_size}")
        print(f"Throughput: {end_to_end['throughput_req_s']} req/s")
        print(f"p50: {end_to_end['p50_ms']} ms")
        print(f"p95: {end_to_end['p95_ms']} ms")
        print(f"p99: {end_to_end['p99_ms']} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
