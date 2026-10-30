$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10+ is required. Install Python and rerun this script."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Node.js 18+ and npm are required. Install Node.js and rerun this script."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

$python = Join-Path $root ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt
& $python -m pip install -r api\requirements.txt

Set-Location (Join-Path $root "frontend")
npm install

Write-Host ""
Write-Host "GEOSAT setup complete. Start the app with .\run_geosat.ps1"
