from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _print_result_row(result: dict) -> None:
    batch_size = result["config"]["batch_size"]
    end_to_end = result["stages"]["end_to_end_ms"]
    model = result["stages"]["model_ms"]
    transfer = result["stages"]["host_to_device_ms"]

    print(
        f"{batch_size:>5} | "
        f"{end_to_end['throughput_req_s']:>12.2f} | "
        f"{end_to_end['p50_ms']:>9.4f} | "
        f"{end_to_end['p95_ms']:>9.4f} | "
        f"{end_to_end['p99_ms']:>9.4f} | "
        f"{model['p95_ms']:>11.4f} | "
        f"{transfer['p95_ms']:>14.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark JSON output.")
    parser.add_argument("--input", required=True)

    args = parser.parse_args()
    data = _load(Path(args.input))

    print("AI Inference Performance Summary")
    print(f"Input: {args.input}")

    if data.get("experiment") == "batch_size_sweep":
        print(f"Device: {data['resolved_device']}")
        print()
        print("batch | throughput/s |    p50 ms |    p95 ms |    p99 ms | model p95 | transfer p95")
        print("------|--------------|-----------|-----------|-----------|-----------|--------------")

        for result in data["results"]:
            _print_result_row(result)
    else:
        print(f"Device: {data['resolved_device']}")
        print()
        print("batch | throughput/s |    p50 ms |    p95 ms |    p99 ms | model p95 | transfer p95")
        print("------|--------------|-----------|-----------|-----------|-----------|--------------")
        _print_result_row(data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
