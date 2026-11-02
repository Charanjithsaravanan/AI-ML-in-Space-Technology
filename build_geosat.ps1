$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
Remove-Item (Join-Path $root "dist\geosat") -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $root "dist\geosat.exe") -Force -ErrorAction SilentlyContinue

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Local environment not found. Run .\setup_geosat.ps1 first."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Node.js and npm are required to build the dashboard."
}

$python = Join-Path $root ".venv\Scripts\python.exe"
& $python -m pip install -r build-requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Installing build dependencies failed." }

Push-Location (Join-Path $root "frontend")
try {
    npm ci
    if ($LASTEXITCODE -ne 0) { throw "Installing frontend dependencies failed." }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "Building the dashboard failed." }
} finally {
    Pop-Location
}

& $python -m PyInstaller --noconfirm --clean geosat.spec
if ($LASTEXITCODE -ne 0) { throw "Building the GEOSAT executable failed." }

Write-Host ""
Write-Host "Build complete: dist\geosat\geosat.exe"
Write-Host "Run it with: .\dist\geosat\geosat.exe"
