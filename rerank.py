"""
Step 5b: reranking with a cross-encoder.

First pass (search_rank.py):  each requirement and each CV chunk are embedded SEPARATELY and
compared by cosine similarity. Fast, but rough.

Rerank (this file): a cross-encoder reads the requirement and the CV text TOGETHER, as one input,
and scores how well the text satisfies the requirement. Slower, so it only runs on a shortlist.

    first pass: all candidates -> shortlist (top ~20) -> cross-encoder -> final order

Runs locally (no data leaves your computer). First use downloads the model (roughly 90 MB).
"""

import math

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # small and fast; swap for a larger one later
MIN_SHORTLIST = 20              # always rerank at least this many candidates...
CHUNKS_PER_REQUIREMENT = 3      # ...looking at each candidate's best 3 chunks per requirement
STRONG = 0.50                   # reranker probability: at or above = strong match
WEAK = 0.20                     # below this = requirement looks missing (starting guesses: tune!)

_model = None


def get_reranker():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder   # imported late: slow to load
        print(f"Loading reranker {RERANK_MODEL} ...")
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def to_probability(raw) -> list[float]:
    """
    Cross-encoders may return raw scores (any range) or probabilities, depending on the
    library version. Anything outside 0..1 is squashed with a sigmoid so we always get 0..1.
    """
    values = [float(x) for x in raw]
    if values and all(0.0 <= v <= 1.0 for v in values):
        return values
    return [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, v)))) for v in values]


def label(p: float) -> str:
    return "strong" if p >= STRONG else "partial" if p >= WEAK else "missing"


def shortlist_size(top_n: int) -> int:
    # a little extra room, because duplicates of the same person get hidden later
    return max(MIN_SHORTLIST, top_n * 3)


def rerank(queries: list[dict], hits: list[list[dict]], first_pass: list[dict], top_n: int) -> list[dict]:
    """
    queries / hits:  from jd_processor and search_rank.retrieve
    first_pass:      candidates from search_rank.score_candidates, best first
    Returns the shortlisted candidates, re-scored and re-ordered, in the same structure.
    """
    shortlist = first_pass[:shortlist_size(top_n)]
    wanted = {c["cv_id"] for c in shortlist}

    # Each shortlisted candidate's best few chunks (by first-pass similarity) for every requirement
    candidates_chunks: dict[tuple, list] = {}
    for qi, query_hits in enumerate(hits):
        per_candidate: dict[str, list] = {}
        for h in query_hits:
            if h["cv_id"] in wanted:
                per_candidate.setdefault(h["cv_id"], []).append(h)
        for cv_id, chunk_list in per_candidate.items():
            chunk_list.sort(key=lambda h: -h["sim"])
            candidates_chunks[(qi, cv_id)] = chunk_list[:CHUNKS_PER_REQUIREMENT]

    pairs, index = [], []
    for (qi, cv_id), chunk_list in candidates_chunks.items():
        for h in chunk_list:
            pairs.append((queries[qi]["text"], h["text"]))
            index.append((qi, cv_id, h))

    probabilities = to_probability(get_reranker().predict(pairs, batch_size=32)) if pairs else []

    best: dict[tuple, tuple] = {}
    for (qi, cv_id, h), p in zip(index, probabilities):
        if (qi, cv_id) not in best or p > best[(qi, cv_id)][0]:
            best[(qi, cv_id)] = (p, h)

    total_weight = sum(q["weight"] for q in queries) or 1.0
    reranked = []
    for first_rank, cand in enumerate(shortlist, 1):
        details, weighted = [], 0.0
        for qi, q in enumerate(queries):
            p, h = best.get((qi, cand["cv_id"]), (0.0, None))
            before = cand["details"][qi]
            weighted += q["weight"] * p
            details.append({
                **before,
                "first_pass_sim": before["sim"],
                "sim": p,                                   # from here on, "sim" = reranker score
                "label": label(p),
                "section": h["section"] if h else before["section"],
                "text": h["text"] if h else before["text"],  # evidence = chunk the reranker liked best
            })
        reranked.append({"cv_id": cand["cv_id"], "score": weighted / total_weight,
                         "first_pass_score": cand["score"], "first_pass_rank": first_rank,
                         "details": details})

    reranked.sort(key=lambda r: -r["score"])
    return reranked
