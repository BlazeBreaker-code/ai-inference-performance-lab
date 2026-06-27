from time import perf_counter, sleep
from statistics import median
from rich.console import Console
from rich.table import Table

console = Console()

def percentile(values: list[float], p:float) -> float:
    sorted_values = sorted(values)
    index  = int((len(sorted_values) - 1) * p)
    return sorted_values[index]

def fake_model_request() -> float:
    start = perf_counter()

    sleep(0.03) # fake tokenization
    sleep(0.05) # fake queue/scheduling
    sleep(0.12) # fake GPU inference

    return ((perf_counter() - start) * 1000) # return latency in ms

def main() -> None:
    latencies_ms = []

    for _ in range(20):
        latencies_ms.append(fake_model_request())

    table = Table(title="Fake Inference Benchmark")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Requests", str(len(latencies_ms)))
    table.add_row("p50 Latency (ms)", f"{median(latencies_ms):.2f} ms")
    table.add_row("p95 Latency (ms)", f"{percentile(latencies_ms, 0.95):.2f} ms")
    table.add_row("p99 Latency (ms)", f"{percentile(latencies_ms, 0.99):.2f} ms")

    console.print(table)

if __name__ == "__main__":
    main()