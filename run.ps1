$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

$EnvFile = Join-Path $RootDir ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $parts = $line.Split("=", 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if (-not [Environment]::GetEnvironmentVariable($key, "Process")) {
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

if (-not $env:PYTHON_BIN) {
    $env:PYTHON_BIN = "python"
}

if (-not $env:FRUIT_RUN_MODE) {
    $env:FRUIT_RUN_MODE = "image"
}

$pythonCmd = Get-Command $env:PYTHON_BIN -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python was not found. Set PYTHON_BIN or install Python 3.10+."
    exit 1
}

if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$RootDir;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $RootDir
}

& $env:PYTHON_BIN -c "import importlib.util, sys; missing=[m for m in ['torch','torchvision','PIL','timm'] if importlib.util.find_spec(m) is None]; print('ERROR: Missing Python packages: ' + ', '.join(missing), file=sys.stderr) if missing else None; print('Install dependencies with: python -m pip install -r requirements.txt', file=sys.stderr) if missing else None; sys.exit(1 if missing else 0)"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

switch ($env:FRUIT_RUN_MODE.ToLowerInvariant()) {
    { $_ -in @("image", "inference") } {
        & $env:PYTHON_BIN run_inference.py @args
        exit $LASTEXITCODE
    }
    "webcam" {
        & $env:PYTHON_BIN webcam_inference.py @args
        exit $LASTEXITCODE
    }
    "train" {
        & $env:PYTHON_BIN train.py @args
        exit $LASTEXITCODE
    }
    "verify" {
        & $env:PYTHON_BIN verify.py @args
        exit $LASTEXITCODE
    }
    "export" {
        & $env:PYTHON_BIN export.py @args
        exit $LASTEXITCODE
    }
    default {
        Write-Error "Unsupported FRUIT_RUN_MODE='$env:FRUIT_RUN_MODE'. Supported modes: image, webcam, train, verify, export."
        exit 1
    }
}
