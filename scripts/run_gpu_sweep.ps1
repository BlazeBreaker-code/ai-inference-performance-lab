$ErrorActionPreference = "Stop"

Write-Host "===== Checking CUDA availability ====="

python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"

if ($LASTEXITCODE -ne 0) {
  Write-Host "CUDA is not available on this machine. Skipping GPU sweep."
  exit 0
}

Write-Host "===== Running GPU batch sweep ====="

python benchmarks/benchmark_inference.py `
  --device cuda `
  --batch-sizes 1,2,4,8,16,32,64 `
  --warmup 25 `
  --iterations 200 `
  --input-dim 1024 `
  --hidden-dim 2048 `
  --output-dim 512 `
  --depth 4 `
  --measure-transfer `
  --output results/gpu_batch_sweep.json`

python scripts/summarize_results.py --input results/gpu_batch_sweep.json
