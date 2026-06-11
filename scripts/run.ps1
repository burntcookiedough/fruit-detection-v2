<#
.SYNOPSIS
    Fruit Detector v2 — Runner script (Windows).
.DESCRIPTION
    Loads .env, dispatches on FRUIT_RUN_MODE or CLI args,
    auto-detects installed package vs source fallback.
.PARAMETER Arguments
    Arguments to pass to the fruit-detect CLI.
    If empty, dispatches on FRUIT_RUN_MODE from .env.
.EXAMPLE
    .\scripts\run.ps1 train --epochs 10
    .\scripts\run.ps1 infer --image photo.jpg
    .\scripts\run.ps1                     # dispatches on FRUIT_RUN_MODE
    .\scripts\run.ps1 --help
.NOTES
    Environment Variables:
      FRUIT_RUN_MODE  — auto-dispatch mode (train|image|infer|webcam|verify|export|analyze)
      PYTHON_BIN      — Python interpreter (default: python)
    Exit Codes:
      0  — success
      1  — general error
      2  — Python not found
      3  — unknown FRUIT_RUN_MODE
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"

# ── Constants ────────────────────────────────────────────────
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

$ValidModes = @{
    "train"   = @("train")
    "image"   = @("infer")
    "infer"   = @("infer")
    "webcam"  = @("webcam")
    "verify"  = @("verify")
    "export"  = @("export")
    "analyze" = @("analyze")
}

# ── Logging ──────────────────────────────────────────────────
function Write-Info  { param([string]$Message) Write-Host "[run.ps1] $Message" -ForegroundColor Cyan }
function Write-Warn  { param([string]$Message) Write-Host "[run.ps1] $Message" -ForegroundColor Yellow }
function Write-Err   { param([string]$Message) Write-Host "[run.ps1] $Message" -ForegroundColor Red }

# ── Load .env ────────────────────────────────────────────────
function Import-DotEnv {
    $envFile = Join-Path $RootDir ".env"
    if (-not (Test-Path $envFile)) { return }

    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $parts = $line -split "=", 2
            $key = $parts[0].Trim()
            $val = $parts[1].Trim().Trim('"').Trim("'")
            # Only set if not already defined (env vars take precedence)
            if (-not [Environment]::GetEnvironmentVariable($key)) {
                [Environment]::SetEnvironmentVariable($key, $val)
            }
        }
    }
}

# ── Validate Python ──────────────────────────────────────────
function Get-PythonBinary {
    $pythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

    try {
        & $pythonBin --version 2>&1 | Out-Null
    } catch {
        Write-Err "Python not found. Set PYTHON_BIN or install Python 3.10+."
        exit 2
    }

    return $pythonBin
}

# ── Resolve FRUIT_RUN_MODE → CLI args ────────────────────────
function Resolve-ModeToArgs {
    param([string]$Mode)

    if ($ValidModes.ContainsKey($Mode)) {
        return $ValidModes[$Mode]
    }

    Write-Err "Unknown FRUIT_RUN_MODE: $Mode"
    Write-Err "Valid modes: $($ValidModes.Keys -join ', ')"
    exit 3
}

# ── Execute via installed package or source fallback ─────────
function Invoke-FruitDetect {
    param([string]$PythonBin, [string[]]$CliArgs)

    # Prefer installed entry point
    try {
        & $PythonBin -c "import fruit_detector" 2>$null
        if ($LASTEXITCODE -eq 0) {
            & fruit-detect @CliArgs
            exit $LASTEXITCODE
        }
    } catch {}

    # Fallback: run from source tree
    Write-Warn "Package not installed - running from source"
    $env:PYTHONPATH = "$RootDir\src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })
    & $PythonBin -m fruit_detector.cli @CliArgs
    exit $LASTEXITCODE
}

# ── Main ─────────────────────────────────────────────────────
Import-DotEnv
$pythonBin = Get-PythonBinary

# Auto-dispatch on FRUIT_RUN_MODE when no args given
if (-not $Arguments -or $Arguments.Count -eq 0) {
    $mode = $env:FRUIT_RUN_MODE
    if ($mode) {
        $Arguments = Resolve-ModeToArgs -Mode $mode
        Write-Info "FRUIT_RUN_MODE=$mode -> fruit-detect $Arguments"
    } else {
        $Arguments = @("--help")
    }
}

Invoke-FruitDetect -PythonBin $pythonBin -CliArgs $Arguments
