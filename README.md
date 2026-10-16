# Resume matcher: contextual semantic upgrade

This is an **overlay** for your existing resume-matching project, not a replacement for the whole project.
Copy `semantic_evaluator.py` into the same folder as `search_rank.py`, then replace your existing
`search_rank.py` with the version included here. Keep your existing `database.py`, `experience.py`,
`jd_processor.py`, `rerank.py`, `build_index.py`, `ingest.py`, `section_splitter.py`, and `cv_extracter.py`.

## What it adds

- Keeps ChromaDB + `all-MiniLM-L6-v2` retrieval local.
- Keeps the cross-encoder reranker local.
- Uses Gemini for candidate-level reasoning over the full available resume sections.
- Returns overall fit, strengths, transferable skills, gaps, requirement assessments, and evidence quotes.
- Excludes personal/contact sections from the Gemini prompt.
- Verifies generated evidence quotes against the resume text and downgrades unverifiable quotes.
- Includes all database candidates with status `ok` in first-pass scoring, even when no chunk was retrieved for them.
- If Gemini is unavailable or fails, it falls back to local ranking and includes a warning.

## 1. Install dependencies

Activate the project's virtual environment, then run:

```powershell
python -m pip install -U google-genai pydantic
```

Keep your existing packages (`chromadb`, `sentence-transformers`, `fastapi`, `uvicorn`, etc.).

## 2. Add environment variables

Create or update `.env` in the project root:

```dotenv
GEMINI_API_KEY=put_your_real_key_here
GEMINI_MODEL=gemini-3.5-flash
```

Do not commit `.env` to GitHub. Make sure `.env` is in `.gitignore`.
Get an API key through Google AI Studio. API availability, quota, pricing and model availability
depend on your account/region and can change; verify current details in the official Gemini docs.

## 3. Run from the project root

Local ranking + Gemini contextual evaluation:

```powershell
python search_rank.py --file jd_sample.txt --top 5
```

Local-only comparison:

```powershell
python search_rank.py --file jd_sample.txt --top 5 --no-context
```

Disable local cross-encoder reranking too:

```powershell
python search_rank.py --file jd_sample.txt --top 5 --no-rerank --no-context
```

If you recently changed chunking or embedding settings, rebuild the index:

```powershell
python build_index.py --rebuild
```

You do **not** need to rebuild the index just to add Gemini evaluation because it reads the stored
resume sections from SQLite. If you change embedding model/chunking, rebuild it.

## 4. FastAPI integration

Your existing API can continue to call `rank_candidates(...)` without changes because the new
`use_context` argument defaults to `True`. To expose a per-request toggle, add the `contextual`
field and replace the `/search` endpoint using `api_integration_snippet.py`.

Run:

```powershell
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`, upload CVs, then POST `/search`.
Example request body:

```json
{
  "jd_text": "Backend engineer. Build REST APIs, deploy cloud services, write tests, and monitor production systems.",
  "top_n": 5,
  "min_score": 0.0,
  "rerank": true,
  "contextual": true
}
```

## How the score works

The final score for candidates successfully evaluated by Gemini is:

`0.55 * local_score + 0.45 * (Gemini overall_fit / 100)`

This is a **ranking heuristic**, not a calibrated probability of job success. Tune it on a labelled
validation set. In contextual mode, Gemini evaluates up to `max(10, min(20, top_n * 2))` candidates
from the local ranking; only those candidates are returned for the final contextual shortlist.
This controls API usage. Set `contextual=false` / use `--no-context` for local-only results.

## Important limitations and safe use

- Review rankings before making hiring decisions. The tool should assist recruiters, not make
  automatic rejection or hiring decisions.
- Resume claims may be incomplete or inaccurate. Gemini's reasoning is not proof of a skill.
- The local cross-encoder's score is a relevance signal, not a calibrated probability.
- Gemini sends resume section text to Google's API. Get appropriate permission and follow your
  organization's privacy/data-retention requirements before using real CVs.
- No personal details are intentionally included in the Gemini prompt, but resume sections may
  contain personal information accidentally. Inspect extraction/section splitting and consider
  redaction before production use.
- This version does not use protected characteristics for scoring. Audit outputs for bias and
  test with a representative labelled dataset before deployment.
- API failures fall back to local ranking; inspect the `warning` and `contextualized` fields in API
  responses.
