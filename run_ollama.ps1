# 一键在本地 Ollama 上跑完整实验（Windows PowerShell）
# 用法：  .\run_ollama.ps1                       默认 qwen2.5:7b，100 道题
#         .\run_ollama.ps1 -Models qwen2.5:7b,llama3.1:8b -N 200
param(
    [string[]]$Models = @("qwen2.5:7b"),
    [int]$N = 100,
    [int]$Workers = 2
)
$ErrorActionPreference = "Stop"

try { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 | Out-Null }
catch { Write-Error "连不上 Ollama（http://localhost:11434）。请先安装并启动 Ollama：https://ollama.com/download"; exit 1 }

python -m pip install -r requirements.txt --quiet

foreach ($m in $Models) {
    Write-Host "`n=== 评分者：$m ===" -ForegroundColor Cyan
    ollama pull $m
    python -m syco.run --backend ollama --model $m --n $N --workers $Workers
    $safe = ($m -replace '[^A-Za-z0-9._-]+', '_')
    python -m syco.analyze "results/raw_ollama_$safe.jsonl"
}
Write-Host "`n完成。报告在 results\report_*.md，图在 results\chart_*.png" -ForegroundColor Green
