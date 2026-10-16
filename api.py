
"""
Step 6a: FastAPI backend.

Install:
    python -m pip install fastapi uvicorn python-multipart

Run:
    python -m uvicorn api:app --reload --port 8000

Open:
    http://127.0.0.1:8000/docs

Endpoints:
    GET    /health
    POST   /cvs/upload
    GET    /cvs
    GET    /cvs/{cv_id}
    DELETE /cvs/{cv_id}
    POST   /search
    POST   /index/rebuild

Search uses:
    - ChromaDB for local vector retrieval
    - Local cross-encoder for reranking
    - Gemini for optional contextual candidate evaluation

SECURITY:
    This API has no authentication and returns candidate personal information.
    Keep it on your own computer. Do not expose it to a network.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

import database
import ingest
from search_rank import rank_candidates


# ---------------------------------------------------------
# Application startup
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Make sure database tables exist.
    database.init_db()

    # Make sure the upload directory exists.
    ingest.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Load the local embedding model at startup.
    import build_index

    build_index.get_model()

    # Try to preload the local cross-encoder.
    # If it fails to load, the server can still start.
    try:
        import rerank

        rerank.get_reranker()
    except Exception as e:
        print(
            f"Reranker not loaded ({e}). "
            "Searches may use first-pass results."
        )

    yield


app = FastAPI(
    title="CV Matcher API",
    description=(
        "Resume matching with ChromaDB retrieval, "
        "local cross-encoder reranking, and optional "
        "Gemini contextual evaluation."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

CV_ID = PathParam(
    pattern=r"^[A-Za-z0-9_-]+$",
    max_length=100,
)


class SearchRequest(BaseModel):
    jd_text: str = Field(
        min_length=3,
        description=(
            "Job description, role requirements, "
            "or a list of relevant skills."
        ),
    )

    top_n: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum number of candidates to return.",
    )

    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Hide candidates below this score.",
    )

    rerank: bool = Field(
        default=True,
        description="Use the local cross-encoder for reranking.",
    )

    contextual: bool = Field(
        default=True,
        description=(
            "Use Gemini for candidate-level contextual evaluation. "
            "Set false to use local matching only."
        ),
    )


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "candidates": len(database.list_candidates()),
    }


# ---------------------------------------------------------
# Upload CVs
# ---------------------------------------------------------

@app.post("/cvs/upload")
def upload_cvs(files: list[UploadFile] = File(...)):
    reports = []

    for uploaded_file in files:
        try:
            filename = uploaded_file.filename or "unnamed_file"

            # Read at most one byte above the configured limit,
            # allowing ingest.process_upload to detect oversized files.
            file_bytes = uploaded_file.file.read(
                ingest.MAX_BYTES + 1
            )

            report = ingest.process_upload(
                filename,
                file_bytes,
            )

            reports.append(report)

        except Exception as e:
            reports.append(
                {
                    "filename": uploaded_file.filename,
                    "status": "failed",
                    "message": str(e),
                }
            )

        finally:
            uploaded_file.file.close()

    added = [
        report
        for report in reports
        if report.get("status") == "added"
    ]

    index_message = (
        ingest.update_index()
        if added
        else "no new CVs to index"
    )

    return {
        "results": reports,
        "summary": {
            "added": len(added),
            "duplicates": sum(
                report.get("status") == "duplicate"
                for report in reports
            ),
            "failed": sum(
                report.get("status") == "failed"
                for report in reports
            ),
            "rejected": sum(
                report.get("status") == "rejected"
                for report in reports
            ),
            "needs_review": sum(
                bool(report.get("needs_review", False))
                for report in reports
            ),
        },
        "index": index_message,
    }


# ---------------------------------------------------------
# List all candidates
# ---------------------------------------------------------

@app.get("/cvs")
def list_cvs():
    return database.list_candidates()


# ---------------------------------------------------------
# Get one candidate
# ---------------------------------------------------------

@app.get("/cvs/{cv_id}")
def get_cv(cv_id: str = CV_ID):
    candidate = database.get_candidate(cv_id)

    if not candidate:
        raise HTTPException(
            status_code=404,
            detail="No such candidate.",
        )

    return candidate


# ---------------------------------------------------------
# Delete a candidate
# ---------------------------------------------------------

@app.delete("/cvs/{cv_id}")
def delete_cv(cv_id: str = CV_ID):
    deleted = ingest.delete_cv(cv_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="No such candidate.",
        )

    return {
        "deleted": cv_id,
    }


# ---------------------------------------------------------
# Search and rank candidates
# ---------------------------------------------------------

@app.post("/search")
def search(req: SearchRequest):
    try:
        output = rank_candidates(
            req.jd_text,
            req.top_n,
            req.min_score,
            req.rerank,
            use_context=req.contextual,
        )

    except TypeError as e:
        # Usually means search_rank.rank_candidates does not yet
        # accept the use_context parameter.
        raise HTTPException(
            status_code=500,
            detail=(
                "The search_rank.py integration does not match api.py. "
                "Ensure rank_candidates() accepts use_context. "
                f"Details: {e}"
            ),
        ) from e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {e}",
        ) from e

    if output.get("error"):
        error_message = output["error"]

        code = (
            409
            if "index is empty" in error_message.lower()
            else 422
        )

        raise HTTPException(
            status_code=code,
            detail=error_message,
        )

    return output


# ---------------------------------------------------------
# Rebuild the vector index
# ---------------------------------------------------------

@app.post("/index/rebuild")
def rebuild_index():
    import build_index

    try:
        build_index.build_index(rebuild=True)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Index rebuild failed: {e}",
        ) from e

    return {
        "index": "rebuilt",
    }
