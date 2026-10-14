"""
Step 5: rank candidates against a job description.

Usage:
    python search_rank.py --file jd_sample.txt
    python search_rank.py "Python, SQL, AWS, Docker"
    python search_rank.py --file jd_sample.txt --top 5          # only the 5 best
    python search_rank.py --file jd_sample.txt --min-score 0.45 # only candidates above a score
    python search_rank.py --file jd_sample.txt --details        # full per-requirement breakdown
    python search_rank.py --file jd_sample.txt --no-rerank      # first pass only, to compare

How it works:
 1. jd_processor splits the JD into separate requirements (each with a weight).
 2. Each requirement is searched against every stored CV chunk (cosine similarity).
 3. For every candidate we keep their BEST matching chunk per requirement.
 4. Candidate score = weighted average of those best similarities.
 5. Requirements with a low best similarity are listed as weak / missing.

Personal details are looked up only AFTER ranking, for display. They never affect the score.
"""

import re
import sys
from pathlib import Path

import database
from jd_processor import process_jd

DEFAULT_TOP = 10        # used when neither --top nor the JD text asks for a number
MAX_RESULTS = 2000     # chunks fetched per query; high enough to see every candidate at this scale
STRONG = 0.50          # similarity at or above this = strong match
WEAK = 0.30            # below this = requirement looks missing (starting guesses, tune with evaluate.py)


# ---------- 1. Retrieval (needs chromadb + sentence-transformers) ----------

def retrieve(queries: list[dict]):
    """Return, for each query, a list of chunk hits: {cv_id, section, text, sim}."""
    import build_index
    collection = build_index.get_collection()
    total = collection.count()
    if total == 0:
        return None
    vectors = build_index.embed([q["text"] for q in queries])
    result = collection.query(query_embeddings=vectors, n_results=min(total, MAX_RESULTS))

    all_hits = []
    for metas, docs, dists in zip(result["metadatas"], result["documents"], result["distances"]):
        all_hits.append([
            {"cv_id": m["cv_id"], "section": m["section"], "text": d,
             "sim": max(0.0, 1.0 - dist)}        # cosine distance -> similarity, floor at 0
            for m, d, dist in zip(metas, docs, dists)
        ])
    return all_hits


# ---------- 2. Scoring (plain Python, no ML) ----------

def label(sim: float) -> str:
    return "strong" if sim >= STRONG else "partial" if sim >= WEAK else "missing"


def score_candidates(queries: list[dict], hits: list[list[dict]]) -> list[dict]:
    """
    queries: from jd_processor.   hits[i]: chunk hits for queries[i].
    Returns candidates sorted by score, each with a per-requirement breakdown.
    """
    # best chunk per (candidate, query)
    best: dict[str, list] = {}
    for qi, query_hits in enumerate(hits):
        for h in query_hits:
            per_query = best.setdefault(h["cv_id"], [None] * len(queries))
            if per_query[qi] is None or h["sim"] > per_query[qi]["sim"]:
                per_query[qi] = h

    total_weight = sum(q["weight"] for q in queries) or 1.0
    ranked = []
    for cv_id, per_query in best.items():
        weighted, details = 0.0, []
        for q, h in zip(queries, per_query):
            sim = h["sim"] if h else 0.0
            weighted += q["weight"] * sim
            details.append({
                "requirement": q["text"], "kind": q["kind"], "weight": q["weight"],
                "sim": sim, "label": label(sim),
                "section": h["section"] if h else None,
                "text": h["text"] if h else "",
            })
        ranked.append({"cv_id": cv_id, "score": weighted / total_weight, "details": details})

    ranked.sort(key=lambda r: -r["score"])
    return ranked


# ---------- 3. Put it together ----------

def rank_candidates(jd_text: str, top_n: int | None = None, min_score: float = 0.0,
                    use_rerank: bool = True) -> dict:
    jd = process_jd(jd_text)
    if not jd["queries"]:
        return {"jd": jd, "results": [], "hidden_duplicates": 0, "total_scored": 0,
                "error": "Could not find any requirements in that text."}

    hits = retrieve(jd["queries"])
    if hits is None:
        return {"jd": jd, "results": [], "hidden_duplicates": 0, "total_scored": 0,
                "error": "The index is empty. Run build_index.py first."}

    # Priority: --top on the command line, then 'top N' written in the JD, then the default.
    if top_n:
        source = "command line"
    elif jd.get("top_n"):
        top_n, source = jd["top_n"], "your JD text"
    else:
        top_n, source = DEFAULT_TOP, "default"

    scored = score_candidates(jd["queries"], hits)      # every candidate is scored internally...
    total_scored = len(scored)

    # Second pass: a cross-encoder re-reads the shortlist. If it can't run, keep the first-pass order.
    reranked, warning = False, None
    if use_rerank:
        try:
            import rerank
            scored = rerank.rerank(jd["queries"], hits, scored, top_n)
            reranked = True
        except Exception as e:
            warning = f"Reranking was skipped ({e}). Showing first-pass results."

    results, seen_emails, hidden = [], set(), 0
    for r in scored:                                    # ...but we only keep the best few
        if r["score"] < min_score or len(results) >= top_n:
            break
        # Personal details are fetched only now, for display and duplicate detection.
        cand = database.get_candidate(r["cv_id"])
        personal = cand["personal"] if cand else {}
        r["name"] = personal.get("name")
        r["email"] = personal.get("email")

        key = (r["email"] or "").strip().lower()
        if key and key in seen_emails:                  # same person uploaded twice
            hidden += 1
            continue
        if key:
            seen_emails.add(key)
        results.append(r)

    return {"jd": jd, "results": results, "hidden_duplicates": hidden,
            "total_scored": total_scored, "top_n": top_n, "top_n_source": source,
            "reranked": reranked, "warning": warning, "error": None}


def print_results(output: dict, details: bool = False) -> None:
    if output["error"]:
        print(output["error"])
        return
    jd = output["jd"]
    print(f"Role: {jd['title']}   Years mentioned in JD: {jd['years_mentioned']}")
    print(f"Searched {output['total_scored']} candidates against {len(jd['queries'])} requirements. "
          f"Showing the best {len(output['results'])} (limit {output['top_n']}, from {output['top_n_source']}).")
    print("Second pass: reranked with a cross-encoder." if output.get("reranked")
          else "First pass only (no reranking).")
    if output.get("warning"):
        print("WARNING: " + output["warning"])
    if output["hidden_duplicates"]:
        print(f"({output['hidden_duplicates']} duplicate CV(s) of the same person hidden)")
    print()

    if not output["results"]:
        print("No candidate reached the minimum score.")
        return

    for rank, r in enumerate(output["results"], 1):
        d = r["details"]
        covered = [x for x in d if x["label"] != "missing"]
        missing = [x for x in d if x["label"] == "missing"]
        moved = f"   [first pass: #{r['first_pass_rank']}]" if "first_pass_rank" in r else ""
        print(f"{rank}. {r['name'] or r['cv_id']}   score {r['score']:.2f}   ({r['email'] or 'no email'}){moved}")
        print(f"     covers {len(covered)} of {len(d)} requirements")
        if missing:
            print("     missing: " + "; ".join(x["requirement"][:45] for x in missing))
        top = max(d, key=lambda x: x["sim"])
        if top["text"]:
            print(f"     best evidence ({top['section']}): \"{top['text'].replace(chr(10), ' ')[:110]}\"")
        if details:
            for x in d:
                print(f"       {x['label']:8} {x['sim']:.2f}  {x['requirement'][:60]}")
        print()


def pop_flag(args: list, name: str, cast, default):
    """Remove `--name value` from args and return the value."""
    if name in args:
        i = args.index(name)
        value = cast(args[i + 1])
        del args[i:i + 2]
        return value
    return default


def normalise_flags(args: list) -> list:
    """Accept '--top5' and '--top=5' as well as '--top 5' (same for --min-score)."""
    out = []
    for a in args:
        m = re.fullmatch(r"(--[a-z][a-z-]*?)=?(\d+(?:\.\d+)?)", a)
        out.extend(m.groups() if m else [a])
    return out


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    args = normalise_flags(args)
    details = "--details" in args
    if details:
        args.remove("--details")
    no_rerank = "--no-rerank" in args
    if no_rerank:
        args.remove("--no-rerank")
    top_n = pop_flag(args, "--top", int, None)
    min_score = pop_flag(args, "--min-score", float, 0.0)

    unknown = [a for a in args if a.startswith("--") and a != "--file"]
    if unknown:
        print(f"Unknown option: {unknown[0]}   (valid: --file, --top N, --min-score X, --details, --no-rerank)")
        sys.exit(1)

    if args[0] == "--file":
        text = Path(args[1]).read_text(encoding="utf-8")
    else:
        text = " ".join(args)
    print_results(rank_candidates(text, top_n, min_score, use_rerank=not no_rerank), details)


if __name__ == "__main__":
    main()
