"""
Workspaces: one per upload session, so a search only ever sees the CVs added in that session.

Each workspace is a folder with its OWN database, search index and files:

    workspaces/session_20261014_101500/
        candidates.db       stored candidates
        chroma_store/       search index
        uploads/            copies of the uploaded CVs
        extracted_texts/    extracted text

use_workspace() points the project's modules at that folder. Nothing from an older session,
or from the test database you built earlier, can appear in the results.
"""

import shutil
import time
from pathlib import Path

import build_index
import database
import ingest

ROOT = Path(__file__).resolve().parent / "workspaces"


def _point(module, names: tuple, value) -> None:
    """Set the first of these module-level names that exists. Fails loudly if none does,
    because silently searching the wrong database would defeat the whole point."""
    for name in names:
        if hasattr(module, name):
            setattr(module, name, value)
            return
    raise RuntimeError(f"{module.__name__}.py has none of {names}, so the session cannot be isolated.")


def new_workspace() -> Path:
    ROOT.mkdir(exist_ok=True)
    stamp = time.strftime("session_%Y%m%d_%H%M%S")
    path, n = ROOT / stamp, 1
    while path.exists():
        n += 1
        path = ROOT / f"{stamp}_{n}"
    path.mkdir()
    return path


def use_workspace(path) -> None:
    """Point database, search index and file folders at this workspace."""
    path = Path(path)
    _point(database, ("db_path", "DB_PATH"), path / "candidates.db")
    _point(build_index, ("CHROMA_DIR", "chroma_dir"), path / "chroma_store")
    _point(ingest, ("UPLOAD_DIR", "upload_dir"), path / "uploads")
    _point(ingest, ("TEXT_DIR", "text_dir"), path / "extracted_texts")
    database.init_db()


def delete_workspace(path) -> bool:
    """Delete one session's data. Refuses to touch anything outside the workspaces folder."""
    path = Path(path).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError("Refusing to delete a folder outside the workspaces folder.")
    for import_path in ("chromadb.api.shared_system_client", "chromadb.api.client"):
        try:                                   # release ChromaDB's file handles (matters on Windows)
            module = __import__(import_path, fromlist=["SharedSystemClient"])
            module.SharedSystemClient.clear_system_cache()
            break
        except Exception:
            continue
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()
