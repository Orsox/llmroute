param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("claim", "list", "commit", "complete")]
    [string]$Command,

    [string]$Agent = "One of Five",
    [string]$Project = "",
    [string]$Status = "",
    [ValidateSet("project", "created_at", "updated_at", "priority", "status")]
    [string]$SortBy = "project",
    [string]$BaseBranch = "main",
    [string]$WorktreeRoot = ".worktrees",
    [int]$IssueId = 0,
    [string]$Message = "",
    [string]$CommitHash = ""
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python not found at $pythonExe"
    exit 1
}

$args = @("-m", "llmrouter.issue_agent", $Command)

switch ($Command) {
    "claim" {
        $args += @("--agent", $Agent, "--base-branch", $BaseBranch, "--worktree-root", $WorktreeRoot)
        if ($Project) { $args += @("--project", $Project) }
    }
    "list" {
        $args += @("--sort-by", $SortBy)
        if ($Project) { $args += @("--project", $Project) }
        if ($Status) { $args += @("--status", $Status) }
    }
    "commit" {
        if ($IssueId -le 0) {
            Write-Error "IssueId must be set for commit."
            exit 1
        }
        if (-not $Message) {
            Write-Error "Message must be set for commit."
            exit 1
        }
        $args += @("--issue-id", "$IssueId", "--message", $Message)
    }
    "complete" {
        if ($IssueId -le 0) {
            Write-Error "IssueId must be set for complete."
            exit 1
        }
        $args += @("--issue-id", "$IssueId")
        if ($CommitHash) { $args += @("--commit-hash", $CommitHash) }
    }
}

Set-Location $projectRoot
& $pythonExe @args
exit $LASTEXITCODE
