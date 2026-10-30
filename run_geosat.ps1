$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Local environment not found. Run .\setup_geosat.ps1 first."
}
if (-not (Test-Path (Join-Path $root "runs\best.pt"))) {
    throw "Missing trained model: runs\best.pt"
}
if (-not (Test-Path (Join-Path $root "frontend\node_modules"))) {
    throw "Frontend dependencies not found. Run .\setup_geosat.ps1 first."
}

$api = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "api.app:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $root -PassThru

try {
    $healthy = $false
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2
            if ($health.status -eq "ok") {
                $healthy = $true
                break
            }
        } catch {
            if ($api.HasExited) {
                throw "GEOSAT API exited before becoming ready."
            }
        }
    }
    if (-not $healthy) {
        throw "GEOSAT API did not become ready at http://127.0.0.1:8000/health"
    }

    Write-Host "GEOSAT API is ready at http://127.0.0.1:8000"
    Write-Host "Starting dashboard at http://localhost:3000"
    Write-Host "Press Ctrl+C to stop both services."
    Start-Process "http://localhost:3000"
    Set-Location (Join-Path $root "frontend")
    npm run dev
} finally {
    if ($api -and -not $api.HasExited) {
        Stop-Process -Id $api.Id
    }
}
