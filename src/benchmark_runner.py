from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter_ns

import torch

from src.device_utils import get_device_info, resolve_device, synchronize_device
from src.metrics import summarize_latencies
from src.models import create_model


@dataclass(frozen=True)
class BenchmarkConfig:
    device: str
    batch_size: int
    warmup_iterations: int
    measurement_iterations: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    depth: int
    measure_transfer: bool
    seed: int = 1337


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000.0


def run_single_benchmark(config: BenchmarkConfig) -> dict:
    device = resolve_device(config.device)

    torch.manual_seed(config.seed)

    model = create_model(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        depth=config.depth,
    ).to(device)

    model.eval()

    cpu_input = torch.randn(config.batch_size, config.input_dim, device="cpu")
    device_input = cpu_input.to(device)

    end_to_end_ms: list[float] = []
    model_ms: list[float] = []
    transfer_ms: list[float] = []

    with torch.inference_mode():
        for _ in range(config.warmup_iterations):
            _ = model(device_input)
            synchronize_device(device)

        for _ in range(config.measurement_iterations):
            if device == "cuda" and config.measure_transfer:
                synchronize_device(device)

                transfer_start = perf_counter_ns()
                x = cpu_input.to(device, non_blocking=False)
                synchronize_device(device)
                transfer_end = perf_counter_ns()

                _ = model(x)
                synchronize_device(device)
                model_end = perf_counter_ns()

                transfer_ms.append(_elapsed_ms(transfer_start, transfer_end))
                model_ms.append(_elapsed_ms(transfer_end, model_end))
                end_to_end_ms.append(_elapsed_ms(transfer_start, model_end))
            else:
                synchronize_device(device)

                start = perf_counter_ns()
                _ = model(device_input)
                synchronize_device(device)
                end = perf_counter_ns()

                elapsed = _elapsed_ms(start, end)

                transfer_ms.append(0.0)
                model_ms.append(elapsed)
                end_to_end_ms.append(elapsed)

    return {
        "benchmark": "single_inference_run",
        "config": asdict(config),
        "resolved_device": device,
        "device_info": get_device_info(device),
        "stages": {
            "end_to_end_ms": summarize_latencies(end_to_end_ms, config.batch_size),
            "model_ms": summarize_latencies(model_ms, config.batch_size),
            "host_to_device_ms": summarize_latencies(transfer_ms, config.batch_size),
        },
    }


def run_batch_sweep(base_config: BenchmarkConfig, batch_sizes: list[int]) -> dict:
    results = []

    for batch_size in batch_sizes:
        config = BenchmarkConfig(
            device=base_config.device,
            batch_size=batch_size,
            warmup_iterations=base_config.warmup_iterations,
            measurement_iterations=base_config.measurement_iterations,
            input_dim=base_config.input_dim,
            hidden_dim=base_config.hidden_dim,
            output_dim=base_config.output_dim,
            depth=base_config.depth,
            measure_transfer=base_config.measure_transfer,
            seed=base_config.seed,
        )

        result = run_single_benchmark(config)
        results.append(result)

    return {
        "experiment": "batch_size_sweep",
        "requested_device": base_config.device,
        "resolved_device": results[0]["resolved_device"] if results else base_config.device,
        "batch_sizes": batch_sizes,
        "results": results,
    }
