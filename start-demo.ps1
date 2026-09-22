param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$billingPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $billingPython)) {
    throw 'Set up Python first: python -m venv .venv, then .venv/Scripts/python.exe -m pip install -r requirements-demo.txt'
}
$env:APP_MODE = 'demo'
Write-Host "Billing Copilot: http://127.0.0.1:$Port"
Write-Host 'Keep this terminal open while presenting. Press Ctrl+C to stop.'
& $billingPython -m uvicorn app.api:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
