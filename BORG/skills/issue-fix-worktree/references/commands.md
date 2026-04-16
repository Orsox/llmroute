# Commands

## Claim the next issue

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agent_issue_worktree.ps1 -Command claim -Agent "Three of Five"
```

Shell:

```bash
./scripts/agent_issue_worktree.sh claim --agent "Three of Five"
```

## List issues by project

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agent_issue_worktree.ps1 -Command list -SortBy project
```

## Create the review commit

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agent_issue_worktree.ps1 -Command commit -IssueId <id> -Message "fix(issue-<id>): <summary>"
```

## Mark the issue done

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agent_issue_worktree.ps1 -Command complete -IssueId <id> -CommitHash <sha>
```
