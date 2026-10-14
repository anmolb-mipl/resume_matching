import hashlib
import re
from pathlib import Path
 
from cv_extracter import extract_text, make_cv_id

from section_splitter import split_cv
import database

UPLOAD_DIR = Path("uploads")
TEXT_DIR = Path("extracted_texts")
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_BYTES = 10 * 1024 * 1024              # 10 MB per CV
SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")


def report(filename, status, cv_id=None, needs_review=False, message=""):
    return {"file": filename, "status": status, "cv_id": cv_id,
            "needs_review": needs_review, "message": message}


def ingest_file(path: Path, cv_id: str | None = None, display_name: str | None = None) -> dict:
    """Extract, split and store one saved CV file. Never raises: errors come back in the report."""
    display = display_name or path.name
    try:
        text, method = extract_text(path)
        if not text.strip():
            raise ValueError("no text could be extracted (a scanned file needs OCR)")
        if cv_id is None:
            cv_id = make_cv_id(path, hashlib.sha256(path.read_bytes()).hexdigest()[:8])

        result = split_cv(text, cv_id)
        outcome = database.add_candidate(result, database.text_hash(text))
        if outcome == "duplicate":
            return report(display, "duplicate", cv_id, message="Same text as a CV that is already stored.")

        TEXT_DIR.mkdir(exist_ok=True)
        (TEXT_DIR / f"{cv_id}.txt").write_text(text, encoding="utf-8")
        review = result["status"] == "needs_review"
        msg = f"extracted with {method}"
        if review:
            msg += "; flagged for review (no Skills or Experience section found)"
        return report(display, "added", cv_id, review, msg)
    except Exception as e:
        return report(display, "failed", cv_id, message=str(e))


def process_upload(filename: str | None, data: bytes) -> dict:
    """Validate and save one uploaded file, then ingest it."""
    name = Path(filename or "cv").name            # drop any folder parts a client might send
    ext = Path(name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return report(name, "rejected", message=f"Unsupported file type '{ext}'. Allowed: .pdf, .docx, .txt")
    if not data:
        return report(name, "rejected", message="The file is empty.")
    if len(data) > MAX_BYTES:
        return report(name, "rejected", message="The file is larger than 10 MB.")

    h = hashlib.sha256(data).hexdigest()[:8]
    cv_id = make_cv_id(Path(name), h)
    UPLOAD_DIR.mkdir(exist_ok=True)
    saved = UPLOAD_DIR / f"{cv_id}{ext}"
    if saved.exists():
        return report(name, "duplicate", cv_id, message="This exact file was already uploaded.")

    saved.write_bytes(data)
    rep = ingest_file(saved, cv_id, display_name=name)
    if rep["status"] in ("failed", "duplicate"):
        saved.unlink(missing_ok=True)             # keep uploads/ in step with the database
    return rep


def update_index() -> str:
    """Embed any CVs that are in the database but not yet in ChromaDB."""
    import build_index                            # imported late: loads the ML model
    try:
        build_index.build_index()
        return "index updated"
    except SystemExit:
        return "nothing to index"


def delete_cv(cv_id: str) -> bool:
    """Remove a candidate everywhere: database, search index, extracted text, uploaded file."""
    if not SAFE_ID.fullmatch(cv_id):
        return False
    found = database.delete_candidate(cv_id)
    try:
        import build_index
        build_index.delete_from_index(cv_id)
    except Exception:
        pass                                      # index may not exist yet
    (TEXT_DIR / f"{cv_id}.txt").unlink(missing_ok=True)
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.glob(f"{cv_id}.*"):
            f.unlink(missing_ok=True)
    return found
