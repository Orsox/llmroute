---
name: issue-fix-worktree
description: Fix repository issues through the local BORG workflow using the project issue tracker, dedicated git worktrees, and the Borg collective designations One of Five through Five of Five. Use when Codex needs to claim an issue, prepare a worktree, implement a fix, run focused verification, commit the change, and update issue state for handoff or closure.
---

# Issue Fix Worktree

Follow the Borg workflow in `BORG/workflows/issue-fix-worktree.yaml`.

Use the roster in `BORG/agents/enterprise-collective.yaml` to choose the active designation.

When you need exact commands, read `references/commands.md`.

## Procedure

1. Read the issue details from the local issue tracker or `/admin/issues`.
2. If you are `Three of Five`, claim the issue and create the worktree before editing anything.
3. Work only inside the issue worktree once it exists.
4. Make the smallest change that resolves the issue and matches local patterns.
5. Run focused verification for the touched behavior.
6. If the fix is ready, create a scoped commit and move the issue to `review`.
7. If the fix is accepted, mark the issue `done` and record the final commit hash.

## Required handoff data

- issue id
- project key
- active designation
- branch name
- worktree path
- verification performed
- commit hash when review is requested

## Guardrails

- Do not edit from the main checkout after the worktree is created.
- Do not claim a second issue until the first issue has a handoff state.
- Keep commit messages scoped to the issue id.
- Update the issue tracker whenever ownership or status changes.
