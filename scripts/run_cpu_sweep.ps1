$ErrorActionPreference = "Stop"

Write-Host "===== Running CPU batch sweep ====="

python benchmarks/benchmark_inference.py `
  --device cpu `
  --batch-sizes 1,2,4,8,16,32,64 `
  --warmup 25 `
  --iterations 200 `
  --input-dim 1024 `
  --hidden-dim 2048 `
  --output-dim 512 `
  --depth 4 `
  --output results/cpu_batch_sweep.json

python scripts/summarize_results.py --input results/cpu_batch_sweep.json
