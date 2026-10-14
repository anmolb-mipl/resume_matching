"""
Step 6a: FastAPI backend.

Install:   pip install fastapi uvicorn python-multipart
Run:       python -m uvicorn api:app --port 8000
Try it:    open http://127.0.0.1:8000/docs   (interactive page: upload CVs, run a search)

Endpoints
    GET    /health            is the server up, how many CVs are stored
    POST   /cvs/upload        upload one or more CVs (pdf, docx, txt)
    GET    /cvs               list stored candidates
    GET    /cvs/{cv_id}       one candidate in full
    DELETE /cvs/{cv_id}       delete a candidate everywhere
    POST   /search            send a JD, get the ranked top candidates
    POST   /index/rebuild     re-embed everything (after changing the splitter, say)

SECURITY: this has no login yet and returns names and emails. Run it only on your own
computer (the default address 127.0.0.1 is local-only). Do not expose it to a network.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

import database
import ingest
from search_rank import rank_candidates


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()                    # make sure the tables exist
    ingest.UPLOAD_DIR.mkdir(exist_ok=True)
    import build_index
    build_index.get_model()               # load the embedding model now, so the first search is fast
    try:
        import rerank
        rerank.get_reranker()             # same for the reranker; a failed download must not stop the server
    except Exception as e:
        print(f"Reranker not loaded ({e}). Searches will use first-pass results until it loads.")
    yield


app = FastAPI(title="CV Matcher API", lifespan=lifespan)

CV_ID = PathParam(pattern=r"^[A-Za-z0-9_-]+$", max_length=100)   # blocks odd characters in URLs


class SearchRequest(BaseModel):
    jd_text: str = Field(min_length=3, description="Job description or a list of skills. "
                         "You may also write 'I need top 5 candidates'.")
    top_n: int | None = Field(None, ge=1, le=100, description="How many candidates to return")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Hide candidates below this score")
    rerank: bool = Field(True, description="Re-read the shortlist with a cross-encoder for better accuracy")


@app.get("/health")
def health():
    return {"status": "ok", "candidates": len(database.list_candidates())}


# Plain `def` (not `async def`) on purpose: these calls are slow and blocking, so FastAPI
# runs them in a worker thread instead of freezing the whole server.

@app.post("/cvs/upload")
def upload_cvs(files: list[UploadFile] = File(...)):
    reports = [ingest.process_upload(f.filename, f.file.read(ingest.MAX_BYTES + 1)) for f in files]
    added = [r for r in reports if r["status"] == "added"]
    index_message = ingest.update_index() if added else "no new CVs to index"
    return {
        "results": reports,
        "summary": {
            "added": len(added),
            "duplicates": sum(r["status"] == "duplicate" for r in reports),
            "failed": sum(r["status"] == "failed" for r in reports),
            "rejected": sum(r["status"] == "rejected" for r in reports),
            "needs_review": sum(r["needs_review"] for r in reports),
        },
        "index": index_message,
    }


@app.get("/cvs")
def list_cvs():
    return database.list_candidates()


@app.get("/cvs/{cv_id}")
def get_cv(cv_id: str = CV_ID):
    candidate = database.get_candidate(cv_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="No such candidate.")
    return candidate


@app.delete("/cvs/{cv_id}")
def delete_cv(cv_id: str = CV_ID):
    if not ingest.delete_cv(cv_id):
        raise HTTPException(status_code=404, detail="No such candidate.")
    return {"deleted": cv_id}


@app.post("/search")
def search(req: SearchRequest):
    output = rank_candidates(req.jd_text, req.top_n, req.min_score, req.rerank)
    if output["error"]:
        code = 409 if "index is empty" in output["error"] else 422
        raise HTTPException(status_code=code, detail=output["error"])
    return output


@app.post("/index/rebuild")
def rebuild_index():
    import build_index
    build_index.build_index(rebuild=True)
    return {"index": "rebuilt"}
