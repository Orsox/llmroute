[CmdletBinding()]
param(
    [string]$RouterUrl = "http://127.0.0.1:12345",
    [string]$DeepModel = "",
    [string]$BearerToken = "",
    [int]$TimeoutSec = 90,
    [int]$RouterLogTail = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = Join-Path $projectRoot "outputs"
$sessionLogPath = Join-Path $outputDir ("deep_connection_test_" + $timestamp + ".log")
$routerLogPath = Join-Path $projectRoot "logs/router.log"

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    $line = "{0} [{1}] {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Level, $Message
    Write-Output $line
    Add-Content -Path $sessionLogPath -Value $line
}

function Load-DotEnv {
    param([string]$Path)
    $result = @{}
    if (-not (Test-Path $Path)) {
        return $result
    }
    foreach ($line in Get-Content $Path) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) { continue }
        if ($trimmed.StartsWith("#")) { continue }
        $eqPos = $trimmed.IndexOf("=")
        if ($eqPos -le 0) { continue }
        $key = $trimmed.Substring(0, $eqPos).Trim()
        $value = $trimmed.Substring($eqPos + 1).Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $result[$key] = $value
    }
    return $result
}

function Is-TrueValue {
    param([string]$Value)
    if ($null -eq $Value) { return $false }
    $v = $Value.Trim().ToLowerInvariant()
    return $v -in @("1", "true", "yes", "on")
}

function Mask-Secret {
    param([string]$Value)
    if ([string]::IsNullOrEmpty($Value)) { return "<empty>" }
    if ($Value.Length -le 8) { return ("*" * $Value.Length) }
    return $Value.Substring(0, 4) + "..." + $Value.Substring($Value.Length - 4)
}

function Write-RequestLogLines {
    param(
        [string]$RequestId
    )
    if (-not (Test-Path $routerLogPath)) {
        Write-Log "Router logfile nicht gefunden: $routerLogPath" "WARN"
        return
    }

    $matches = Select-String -Path $routerLogPath -Pattern $RequestId
    if ($matches.Count -gt 0) {
        Write-Log "Gefundene Router-Logzeilen fuer request-id '$RequestId':"
        foreach ($m in $matches) {
            Write-Log ("ROUTER: " + $m.Line)
        }
        return
    }

    Write-Log "Keine Router-Logzeilen mit request-id '$RequestId' gefunden. Zeige die letzten $RouterLogTail Zeilen." "WARN"
    $tail = Get-Content $routerLogPath -Tail $RouterLogTail
    foreach ($line in $tail) {
        Write-Log ("ROUTER: " + $line)
    }
}

function Invoke-CurlRequest {
    param(
        [string]$Method,
        [string]$Uri,
        [hashtable]$Headers,
        [string]$Body = "",
        [int]$TimeoutSec = 30
    )

    $headerFile = New-TemporaryFile
    $bodyFile = New-TemporaryFile
    $requestFile = $null
    try {
        $args = @(
            "-sS",
            "-X", $Method,
            $Uri,
            "--max-time", [string]$TimeoutSec,
            "-D", $headerFile.FullName,
            "-o", $bodyFile.FullName,
            "-w", "%{http_code}"
        )

        foreach ($key in $Headers.Keys) {
            $args += @("-H", ($key + ": " + [string]$Headers[$key]))
        }

        if (-not [string]::IsNullOrWhiteSpace($Body)) {
            $requestFile = New-TemporaryFile
            Set-Content -Path $requestFile.FullName -Value $Body -Encoding utf8
            $args += @("--data-binary", ("@" + $requestFile.FullName))
        }

        $statusText = & curl.exe @args
        if ($LASTEXITCODE -ne 0) {
            throw "curl.exe failed with exit code $LASTEXITCODE"
        }

        $headersMap = @{}
        foreach ($line in Get-Content $headerFile.FullName) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            if ($line -match "^\s*HTTP/") { continue }
            $idx = $line.IndexOf(":")
            if ($idx -le 0) { continue }
            $name = $line.Substring(0, $idx).Trim()
            $value = $line.Substring($idx + 1).Trim()
            if (-not [string]::IsNullOrWhiteSpace($name)) {
                $headersMap[$name] = $value
            }
        }

        return @{
            StatusCode = [int]([string]$statusText).Trim()
            Headers = $headersMap
            Content = (Get-Content $bodyFile.FullName -Raw)
        }
    } finally {
        Remove-Item -Path $headerFile.FullName -ErrorAction SilentlyContinue
        Remove-Item -Path $bodyFile.FullName -ErrorAction SilentlyContinue
        if ($null -ne $requestFile) {
            Remove-Item -Path $requestFile.FullName -ErrorAction SilentlyContinue
        }
    }
}

Write-Log "Deep-Connection-Test gestartet."
Write-Log "Router URL: $RouterUrl"
Write-Log "Session-Log: $sessionLogPath"

$dotenvPath = Join-Path $projectRoot ".env"
$dotenv = Load-DotEnv -Path $dotenvPath

$deepEnabled = ""
if ($dotenv.ContainsKey("DEEP_ENABLED")) {
    $deepEnabled = [string]$dotenv["DEEP_ENABLED"]
} elseif ($env:DEEP_ENABLED) {
    $deepEnabled = [string]$env:DEEP_ENABLED
}

$deepApiKey = ""
if ($dotenv.ContainsKey("DEEP_API_KEY")) {
    $deepApiKey = [string]$dotenv["DEEP_API_KEY"]
} elseif ($env:DEEP_API_KEY) {
    $deepApiKey = [string]$env:DEEP_API_KEY
}

if ([string]::IsNullOrWhiteSpace($DeepModel)) {
    if ($dotenv.ContainsKey("DEEP_MODEL_ID") -and -not [string]::IsNullOrWhiteSpace([string]$dotenv["DEEP_MODEL_ID"])) {
        $DeepModel = [string]$dotenv["DEEP_MODEL_ID"]
    } elseif ($env:DEEP_MODEL_ID) {
        $DeepModel = [string]$env:DEEP_MODEL_ID
    } else {
        $DeepModel = "gpt-5.4-mini"
    }
}

Write-Log "DEEP_ENABLED: $deepEnabled"
Write-Log ("DEEP_ENABLED aktiv: " + (Is-TrueValue $deepEnabled))
Write-Log ("DEEP_API_KEY: " + (Mask-Secret $deepApiKey))
Write-Log "DeepModel: $DeepModel"

if (-not (Is-TrueValue $deepEnabled)) {
    Write-Log "DEEP_ENABLED ist nicht aktiv. Setze in .env: DEEP_ENABLED=true" "ERROR"
    exit 1
}
if ([string]::IsNullOrWhiteSpace($deepApiKey)) {
    Write-Log "DEEP_API_KEY fehlt. Setze in .env: DEEP_API_KEY=<key>" "ERROR"
    exit 1
}

$healthHeaders = @{}
if (-not [string]::IsNullOrWhiteSpace($BearerToken)) {
    $healthHeaders["Authorization"] = "Bearer $BearerToken"
}

try {
    $healthReqId = "deep-health-" + [Guid]::NewGuid().ToString("N").Substring(0, 12)
    $healthHeaders["x-request-id"] = $healthReqId
    $health = Invoke-CurlRequest -Method "GET" -Uri ($RouterUrl.TrimEnd("/") + "/healthz") -Headers $healthHeaders -TimeoutSec 10
    if ($health.StatusCode -lt 200 -or $health.StatusCode -ge 300) {
        throw ("Healthcheck returned HTTP " + [string]$health.StatusCode + " with body: " + $health.Content)
    }
    Write-Log ("Healthcheck OK: HTTP " + [string]$health.StatusCode)
} catch {
    Write-Log ("Healthcheck fehlgeschlagen: " + $_.Exception.Message) "ERROR"
    exit 1
}

$requestId = "deep-test-" + [Guid]::NewGuid().ToString("N").Substring(0, 12)
$headers = @{
    "Content-Type" = "application/json"
    "x-request-id" = $requestId
}
if (-not [string]::IsNullOrWhiteSpace($BearerToken)) {
    $headers["Authorization"] = "Bearer $BearerToken"
}

$payload = @{
    model = $DeepModel
    max_tokens = 400
    temperature = 0
    messages = @(
        @{
            role = "user"
            content = "Bitte bewerte eine mehrstufige Architektur-Entscheidung mit Trade-offs, Compliance-Regeln und Risikoanalyse. Gib eine klare Empfehlung mit Begruendung."
        }
    )
}

$uri = $RouterUrl.TrimEnd("/") + "/v1/chat/completions"
Write-Log "Sende Testrequest an $uri (request-id: $requestId)"

try {
    $body = $payload | ConvertTo-Json -Depth 8
    $response = Invoke-CurlRequest -Method "POST" -Uri $uri -Headers $headers -Body $body -TimeoutSec $TimeoutSec
    Write-Log ("HTTP Status: " + [string]$response.StatusCode)

    $selectedModel = [string]$response.Headers["x-router-selected-model"]
    $judgeModel = [string]$response.Headers["x-router-judge-model"]
    $routeReason = [string]$response.Headers["x-router-reason"]
    $usedFallback = [string]$response.Headers["x-router-fallback"]

    Write-Log ("x-router-selected-model: " + $selectedModel)
    Write-Log ("x-router-judge-model: " + $judgeModel)
    Write-Log ("x-router-reason: " + $routeReason)
    Write-Log ("x-router-fallback: " + $usedFallback)

    if ($response.StatusCode -lt 200 -or $response.StatusCode -ge 300) {
        Write-Log ("Nicht erfolgreicher HTTP-Status. Antwort: " + $response.Content) "ERROR"
        Write-RequestLogLines -RequestId $requestId
        Write-Log "DEEP CONNECTION TEST: FAIL" "ERROR"
        exit 1
    }

    if (-not [string]::IsNullOrWhiteSpace($response.Content)) {
        try {
            $json = $response.Content | ConvertFrom-Json
            $text = ""
            if ($json.choices -and $json.choices.Count -gt 0) {
                $choice = $json.choices[0]
                if ($choice.message -and $choice.message.content) {
                    $text = [string]$choice.message.content
                } elseif ($choice.text) {
                    $text = [string]$choice.text
                }
            }
            if (-not [string]::IsNullOrWhiteSpace($text)) {
                $preview = $text
                if ($preview.Length -gt 300) {
                    $preview = $preview.Substring(0, 300) + "..."
                }
                Write-Log ("Antwort-Preview: " + $preview)
            }
        } catch {
            Write-Log "Antwort konnte nicht als JSON geparst werden." "WARN"
        }
    }

    Write-RequestLogLines -RequestId $requestId

    if ($selectedModel -eq $DeepModel) {
        Write-Log "DEEP CONNECTION TEST: PASS"
        exit 0
    }

    Write-Log "DEEP CONNECTION TEST: FAIL - Request wurde nicht auf Deep geroutet." "ERROR"
    exit 2
} catch {
    Write-Log ("Request fehlgeschlagen: " + $_.Exception.Message) "ERROR"
    Write-RequestLogLines -RequestId $requestId
    Write-Log "DEEP CONNECTION TEST: FAIL" "ERROR"
    exit 1
}
