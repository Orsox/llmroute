param(
    [switch]$NoTray,
    [switch]$Foreground
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python not found at: $pythonExe"
    exit 1
}

if (-not (Test-Path (Join-Path $projectRoot "llmrouter\__main__.py"))) {
    Write-Error "Package entrypoint not found at: llmrouter\\__main__.py"
    exit 1
}

$arguments = @("-m", "llmrouter")
if (-not $NoTray) {
    $arguments += "--tray"
}

if ($Foreground) {
    Set-Location $projectRoot
    & $pythonExe @arguments
    exit $LASTEXITCODE
}

Start-Process -FilePath $pythonExe -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden | Out-Null
Write-Output "LM Router started. Tray mode: $([bool](-not $NoTray))"
