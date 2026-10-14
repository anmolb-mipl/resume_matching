import json
import re
import sys
import sqlite3
import hashlib

from pathlib import Path
from datetime import datetime

db_path = Path("candidates.db")
section_dir = Path("section")
text_dir = Path("extracted_texts")


SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    cv_id        TEXT PRIMARY KEY,
    file_hash    TEXT UNIQUE NOT NULL,
    status       TEXT NOT NULL,            -- 'ok' or 'needs_review'
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS personal_details (
    cv_id   TEXT PRIMARY KEY REFERENCES candidates(cv_id) ON DELETE CASCADE,
    name    TEXT,
    email   TEXT,
    phone   TEXT,
    links   TEXT                            -- stored as a JSON list
);

CREATE TABLE IF NOT EXISTS sections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    cv_id         TEXT NOT NULL REFERENCES candidates(cv_id) ON DELETE CASCADE,
    section_name  TEXT NOT NULL,
    text          TEXT NOT NULL
); 
CREATE INDEX IF NOT EXISTS idx_sections_cv ON sections(cv_id);
"""

def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row            # lets us read columns by name
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite ignores foreign keys unless asked
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)

def text_hash(text: str) -> str:
    """Fingerprint of the CV text. Same text -> same hash -> duplicate."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add_candidate(result: dict, file_hash: str) -> str:
    """
    Save one split CV. Returns 'added' or 'duplicate'.
    `result` is the dict produced by section_splitter.split_cv().
    """
    cv_id = result["cv_id"]
    with connect() as conn:
        # 1. Duplicate check: same content already stored?
        if conn.execute("SELECT 1 FROM candidates WHERE file_hash = ?", (file_hash,)).fetchone():
            return "duplicate"

        # 2. Different CV with the same file name? Make the id unique.
        if conn.execute("SELECT 1 FROM candidates WHERE cv_id = ?", (cv_id,)).fetchone():
            cv_id = f"{cv_id}_{file_hash[:6]}"

        conn.execute(
            "INSERT INTO candidates (cv_id, file_hash, status, created_at) VALUES (?, ?, ?, ?)",
            (cv_id, file_hash, result["status"], datetime.now().isoformat(timespec="seconds")),
        )
        p = result["personal"]
        conn.execute(
            "INSERT INTO personal_details (cv_id, name, email, phone, links) VALUES (?, ?, ?, ?, ?)",
            (cv_id, p["name"], p["email"], p["phone"], json.dumps(p["links"])),
        )
        conn.executemany(
            "INSERT INTO sections (cv_id, section_name, text) VALUES (?, ?, ?)",
            [(cv_id, name, body) for name, body in result["sections"].items()],
        )
    return "added"


def get_sections_for_matching() -> list[dict]:
    """
    Everything the embedding step is allowed to see.
    NOTE: no personal details here, by design.
    """
    with connect() as conn:
        rows = conn.execute(
            """SELECT s.cv_id, s.section_name, s.text
               FROM sections s JOIN candidates c ON c.cv_id = s.cv_id
               WHERE c.status = 'ok'"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_candidate(cv_id: str) -> dict | None:
    """Full record incl. personal details, for showing results to a recruiter."""
    with connect() as conn:
        c = conn.execute("SELECT * FROM candidates WHERE cv_id = ?", (cv_id,)).fetchone()
        if not c:
            return None
        p = conn.execute("SELECT * FROM personal_details WHERE cv_id = ?", (cv_id,)).fetchone()
        s = conn.execute("SELECT section_name, text FROM sections WHERE cv_id = ?", (cv_id,)).fetchall()
    return {
        "cv_id": cv_id,
        "status": c["status"],
        "personal": {**dict(p), "links": json.loads(p["links"])} if p else {},
        "sections": {r["section_name"]: r["text"] for r in s},
    }


def list_candidates() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT c.cv_id, c.status, p.name, p.email,
                      (SELECT COUNT(*) FROM sections s WHERE s.cv_id = c.cv_id) AS n_sections
               FROM candidates c LEFT JOIN personal_details p ON p.cv_id = c.cv_id
               ORDER BY c.created_at"""
        ).fetchall()
    return [dict(r) for r in rows]


def delete_candidate(cv_id: str) -> bool:
    """Remove a candidate and all their data (ON DELETE CASCADE clears the other tables)."""
    with connect() as conn:
        cur = conn.execute("DELETE FROM candidates WHERE cv_id = ?", (cv_id,))
    return cur.rowcount > 0


def load_all() -> None:
    """Read every JSON produced by section_splitter.py and store it."""
    files = sorted(section_dir.glob("*.json"))
    if not files:
        print(f"No JSON files in {section_dir}/. Run section_splitter.py first.")
        sys.exit(1)

    counts = {"added": 0, "duplicate": 0}
    for path in files:
        result = json.loads(path.read_text(encoding="utf-8"))
        txt_path = text_dir / f"{path.stem}.txt"
        source = txt_path.read_text(encoding="utf-8") if txt_path.exists() else json.dumps(result["sections"])
        outcome = add_candidate(result, text_hash(source))
        counts[outcome] += 1
        print(f"[{outcome.upper():9}] {path.name:30} status: {result['status']}")
    print(f"\nDone: {counts['added']} added, {counts['duplicate']} duplicates skipped.")


def main():
    init_db()
    if "--list" in sys.argv:
        rows = list_candidates()
        if not rows:
            print("Database is empty.")
        for r in rows:
            print(f"{r['cv_id']:30} {r['status']:13} {r['n_sections']} sections   {r['name']}  {r['email']}")
    elif "--delete" in sys.argv:
        cv_id = sys.argv[sys.argv.index("--delete") + 1]
        print("Deleted." if delete_candidate(cv_id) else "No such cv_id.")
    else:
        load_all()


if __name__ == "__main__":
    main()
