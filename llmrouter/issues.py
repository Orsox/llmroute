from __future__ import annotations

from .shared import *
from .shared import _utc_now_iso


ISSUES_DB_PATH = Path(os.getenv("ROUTER_ISSUES_DB_PATH", "logs/router_issues.sqlite"))
ISSUE_STATUSES = {"open", "in_progress", "review", "done"}
ISSUE_PRIORITIES = {"low": 1, "medium": 2, "high": 3, "critical": 4}
ISSUE_SORT_COLUMNS = {
    "created_at": "created_at DESC, id DESC",
    "updated_at": "updated_at DESC, id DESC",
    "project": "project_key COLLATE NOCASE ASC, created_at DESC, id DESC",
    "priority": "priority_rank DESC, project_key COLLATE NOCASE ASC, created_at DESC, id DESC",
    "status": "status COLLATE NOCASE ASC, project_key COLLATE NOCASE ASC, created_at DESC, id DESC",
}


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return cleaned or "issue"


class IssueCreateRequest(BaseModel):
    project_key: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=240)
    description: str = ""
    priority: Literal["low", "medium", "high", "critical"] = "medium"


class IssueUpdateRequest(BaseModel):
    project_key: Optional[str] = Field(default=None, min_length=1, max_length=120)
    title: Optional[str] = Field(default=None, min_length=1, max_length=240)
    description: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high", "critical"]] = None
    status: Optional[Literal["open", "in_progress", "review", "done"]] = None
    agent_name: Optional[str] = None
    branch_name: Optional[str] = None
    worktree_path: Optional[str] = None
    commit_hash: Optional[str] = None


class IssueClaimRequest(BaseModel):
    agent_name: str = Field(min_length=1, max_length=120)
    project_key: Optional[str] = Field(default=None, min_length=1, max_length=120)
    status: Literal["in_progress", "review"] = "in_progress"
    branch_name: Optional[str] = None
    worktree_path: Optional[str] = None


class IssueStore:
    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = (PROJECT_ROOT / (db_path or ISSUES_DB_PATH)).resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        if not self._initialized:
            self._ensure_schema(conn)
            self._initialized = True
        return conn

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_key TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open',
                priority TEXT NOT NULL DEFAULT 'medium',
                priority_rank INTEGER NOT NULL DEFAULT 2,
                agent_name TEXT,
                branch_name TEXT,
                worktree_path TEXT,
                commit_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                claimed_at TEXT,
                completed_at TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_issues_project ON issues(project_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_issues_updated ON issues(updated_at DESC)")
        conn.commit()

    @staticmethod
    def _row_to_issue(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "project_key": row["project_key"],
            "title": row["title"],
            "description": row["description"],
            "status": row["status"],
            "priority": row["priority"],
            "agent_name": row["agent_name"],
            "branch_name": row["branch_name"],
            "worktree_path": row["worktree_path"],
            "commit_hash": row["commit_hash"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "claimed_at": row["claimed_at"],
            "completed_at": row["completed_at"],
        }

    def create_issue(self, payload: IssueCreateRequest) -> dict[str, Any]:
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO issues (
                        project_key, title, description, status, priority, priority_rank,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, 'open', ?, ?, ?, ?)
                    """,
                    (
                        payload.project_key.strip(),
                        payload.title.strip(),
                        (payload.description or "").strip(),
                        payload.priority,
                        ISSUE_PRIORITIES[payload.priority],
                        now,
                        now,
                    ),
                )
                conn.commit()
                return self.get_issue(int(cursor.lastrowid))
            finally:
                conn.close()

    def get_issue(self, issue_id: int) -> dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM issues WHERE id = ?", (int(issue_id),)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"Issue {issue_id} not found")
        return self._row_to_issue(row)

    def list_issues(
        self,
        *,
        project_key: str = "",
        status: str = "",
        sort_by: str = "project",
    ) -> list[dict[str, Any]]:
        sort_sql = ISSUE_SORT_COLUMNS.get(sort_by, ISSUE_SORT_COLUMNS["project"])
        where: list[str] = []
        params: list[Any] = []
        if project_key.strip():
            where.append("project_key = ?")
            params.append(project_key.strip())
        if status.strip():
            where.append("status = ?")
            params.append(status.strip())
        sql = "SELECT * FROM issues"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {sort_sql}"
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_issue(row) for row in rows]
        finally:
            conn.close()

    def grouped_issues(
        self,
        *,
        status: str = "",
    ) -> list[dict[str, Any]]:
        issues = self.list_issues(status=status, sort_by="project")
        grouped: list[dict[str, Any]] = []
        current_project = None
        current_items: list[dict[str, Any]] = []
        for issue in issues:
            if issue["project_key"] != current_project:
                if current_project is not None:
                    grouped.append({"project_key": current_project, "issues": current_items})
                current_project = issue["project_key"]
                current_items = []
            current_items.append(issue)
        if current_project is not None:
            grouped.append({"project_key": current_project, "issues": current_items})
        return grouped

    def project_keys(self) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT project_key FROM issues ORDER BY project_key COLLATE NOCASE ASC"
            ).fetchall()
            return [str(row["project_key"]) for row in rows]
        finally:
            conn.close()

    def update_issue(self, issue_id: int, payload: IssueUpdateRequest) -> dict[str, Any]:
        current = self.get_issue(issue_id)
        updates: dict[str, Any] = {}
        if payload.project_key is not None:
            updates["project_key"] = payload.project_key.strip()
        if payload.title is not None:
            updates["title"] = payload.title.strip()
        if payload.description is not None:
            updates["description"] = payload.description.strip()
        if payload.priority is not None:
            updates["priority"] = payload.priority
            updates["priority_rank"] = ISSUE_PRIORITIES[payload.priority]
        if payload.status is not None:
            updates["status"] = payload.status
            if payload.status == "done":
                updates["completed_at"] = _utc_now_iso()
        for key in ("agent_name", "branch_name", "worktree_path", "commit_hash"):
            value = getattr(payload, key)
            if value is not None:
                updates[key] = value.strip() if isinstance(value, str) else value
        if not updates:
            return current
        updates["updated_at"] = _utc_now_iso()
        assignments = ", ".join(f"{column} = ?" for column in updates)
        params = list(updates.values()) + [int(issue_id)]
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(f"UPDATE issues SET {assignments} WHERE id = ?", params)
                conn.commit()
            finally:
                conn.close()
        return self.get_issue(issue_id)

    def claim_next_issue(self, payload: IssueClaimRequest) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                where = ["status = 'open'"]
                params: list[Any] = []
                if payload.project_key and payload.project_key.strip():
                    where.append("project_key = ?")
                    params.append(payload.project_key.strip())
                row = conn.execute(
                    f"""
                    SELECT * FROM issues
                    WHERE {' AND '.join(where)}
                    ORDER BY priority_rank DESC, created_at ASC, id ASC
                    LIMIT 1
                    """,
                    params,
                ).fetchone()
                if row is None:
                    return None
                issue_id = int(row["id"])
                now = _utc_now_iso()
                conn.execute(
                    """
                    UPDATE issues
                    SET status = ?, agent_name = ?, branch_name = ?, worktree_path = ?,
                        claimed_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        payload.status,
                        payload.agent_name.strip(),
                        (payload.branch_name or "").strip() or None,
                        (payload.worktree_path or "").strip() or None,
                        now,
                        now,
                        issue_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return self.get_issue(issue_id)

    def issue_branch_name(self, issue: dict[str, Any], agent_name: str) -> str:
        project_slug = _slugify(issue["project_key"])
        title_slug = _slugify(issue["title"])[:40]
        agent_slug = _slugify(agent_name)[:24]
        return f"borg/{agent_slug}/{project_slug}/issue-{issue['id']}-{title_slug}"
