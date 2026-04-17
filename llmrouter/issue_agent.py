from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .issues import IssueClaimRequest, IssueStore, IssueUpdateRequest
from .shared import PROJECT_ROOT


def _run_git(args: list[str], *, cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _claim(args: argparse.Namespace) -> int:
    store = IssueStore()
    claimed = store.claim_next_issue(
        IssueClaimRequest(agent_name=args.agent, project_key=args.project or None)
    )
    if claimed is None:
        print("No open issue available.")
        return 1

    branch_name = store.issue_branch_name(claimed, args.agent)
    worktree_root = (PROJECT_ROOT / args.worktree_root).resolve()
    worktree_root.mkdir(parents=True, exist_ok=True)
    worktree_path = worktree_root / f"issue-{claimed['id']}-{args.agent.lower().replace(' ', '-')}"
    _run_git(["worktree", "add", "-b", branch_name, str(worktree_path), args.base_branch], cwd=PROJECT_ROOT)
    updated = store.update_issue(
        claimed["id"],
        IssueUpdateRequest(
            status="in_progress",
            agent_name=args.agent,
            branch_name=branch_name,
            worktree_path=str(worktree_path),
        ),
    )
    print(json.dumps(updated, ensure_ascii=True, indent=2))
    return 0


def _list(args: argparse.Namespace) -> int:
    store = IssueStore()
    issues = store.list_issues(
        project_key=args.project or "",
        status=args.status or "",
        sort_by=args.sort_by,
    )
    print(json.dumps(issues, ensure_ascii=True, indent=2))
    return 0


def _commit(args: argparse.Namespace) -> int:
    store = IssueStore()
    issue = store.get_issue(args.issue_id)
    worktree_path_value = (issue.get("worktree_path") or "").strip()
    if not worktree_path_value:
        raise RuntimeError(f"Issue {args.issue_id} has no worktree_path")
    worktree_path = Path(worktree_path_value)
    _run_git(["add", "-A"], cwd=worktree_path)
    status = _run_git(["status", "--short"], cwd=worktree_path)
    if not status.strip():
        raise RuntimeError("No changes to commit.")
    _run_git(["commit", "-m", args.message], cwd=worktree_path)
    commit_hash = _run_git(["rev-parse", "HEAD"], cwd=worktree_path)
    updated = store.update_issue(
        args.issue_id,
        IssueUpdateRequest(status="review", commit_hash=commit_hash),
    )
    print(json.dumps(updated, ensure_ascii=True, indent=2))
    return 0


def _complete(args: argparse.Namespace) -> int:
    store = IssueStore()
    updated = store.update_issue(
        args.issue_id,
        IssueUpdateRequest(status="done", commit_hash=args.commit_hash or None),
    )
    print(json.dumps(updated, ensure_ascii=True, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent issue worktree helper")
    sub = parser.add_subparsers(dest="command", required=True)

    claim = sub.add_parser("claim", help="Claim next open issue and create a worktree")
    claim.add_argument("--agent", required=True, help="Borg-style designation, e.g. 'One of Five'")
    claim.add_argument("--project", default="", help="Optional project key filter")
    claim.add_argument("--base-branch", default="main", help="Base branch for new worktree branch")
    claim.add_argument("--worktree-root", default=".worktrees", help="Directory for worktrees")
    claim.set_defaults(func=_claim)

    list_cmd = sub.add_parser("list", help="List issues")
    list_cmd.add_argument("--project", default="", help="Optional project key filter")
    list_cmd.add_argument("--status", default="", help="Optional status filter")
    list_cmd.add_argument("--sort-by", default="project", choices=["project", "created_at", "updated_at", "priority", "status"])
    list_cmd.set_defaults(func=_list)

    commit = sub.add_parser("commit", help="Commit changes for a claimed issue")
    commit.add_argument("--issue-id", type=int, required=True)
    commit.add_argument("--message", required=True)
    commit.set_defaults(func=_commit)

    complete = sub.add_parser("complete", help="Mark an issue as done")
    complete.add_argument("--issue-id", type=int, required=True)
    complete.add_argument("--commit-hash", default="")
    complete.set_defaults(func=_complete)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
