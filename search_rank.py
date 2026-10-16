"""Rank candidates against a job description using local retrieval/reranking and optional Gemini context."""
import re
import sys
from pathlib import Path

import database
import experience
from jd_processor import process_jd

DEFAULT_TOP = 10
MAX_RESULTS = 2000
STRONG = 0.50
WEAK = 0.30


def retrieve(queries: list[dict]):
    """Return per-query chunk hits: {cv_id, section, text, sim}."""
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
             "sim": max(0.0, 1.0 - float(dist))}
            for m, d, dist in zip(metas, docs, dists)
        ])
    return all_hits


def label(sim: float) -> str:
    return "strong" if sim >= STRONG else "partial" if sim >= WEAK else "missing"


def score_candidates(queries: list[dict], hits: list[list[dict]], profiles: dict | None = None,
                     candidate_ids: set[str] | None = None) -> list[dict]:
    """Score every eligible candidate, including candidates with no retrieved hits."""
    best: dict[str, list] = {
        cv_id: [None] * len(queries) for cv_id in (candidate_ids or set())
    }
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
            factor, note = 1.0, ""
            if q.get("min_years"):
                profile = (profiles or {}).get(cv_id)
                if profile is None:
                    factor, note = experience.UNVERIFIED_FACTOR, "no experience details available"
                else:
                    factor, note = experience.years_factor(q, profile)
            sim *= factor
            weighted += q["weight"] * sim
            details.append({
                "requirement": q["text"], "kind": q["kind"], "weight": q["weight"],
                "sim": sim, "label": label(sim),
                "section": h["section"] if h else None,
                "text": h["text"] if h else "",
                "years_factor": factor, "note": note,
            })
        ranked.append({"cv_id": cv_id, "score": weighted / total_weight, "details": details})
    ranked.sort(key=lambda r: -r["score"])
    return ranked


def _contextualize(jd_text: str, queries: list[dict], scored: list[dict], top_n: int) -> list[dict]:
    """Evaluate a local shortlist with Gemini, then blend semantic judgment with local reranker score."""
    import semantic_evaluator
    shortlist_count = min(len(scored), max(10, min(20, top_n * 2)))
    shortlist = scored[:shortlist_count]
    contextual = []
    for rank, candidate in enumerate(shortlist, 1):
        record = database.get_candidate(candidate["cv_id"])
        sections = record.get("sections", {}) if record else {}
        evaluation = semantic_evaluator.evaluate_candidate(jd_text, queries, sections)

        # Blend local evidence matching and candidate-level context. These are ranking scores,
        # not calibrated probabilities. Tune weights using a labelled evaluation set.
        local_score = float(candidate["score"])
        context_score = float(evaluation["overall_fit"]) / 100.0
        final_score = 0.55 * local_score + 0.45 * context_score

        updated = dict(candidate)
        updated["first_pass_score"] = candidate["score"]
        updated["local_score"] = round(local_score, 4)
        updated["context_score"] = round(context_score, 4)
        updated["score"] = final_score
        updated["contextual_evaluation"] = evaluation
        updated["contextual_rank_before_blend"] = rank
        # Attach Gemini assessments to matching requirement details for API/UI consumers.
        assessments = {a["requirement"].casefold(): a for a in evaluation.get("assessments", [])}
        for detail in updated["details"]:
            a = assessments.get(detail["requirement"].casefold())
            if a:
                detail["context_status"] = a["status"]
                detail["context_explanation"] = a["explanation"]
                detail["context_evidence"] = a["evidence_quote"]
        contextual.append(updated)
    contextual.sort(key=lambda r: -r["score"])
    return contextual


def rank_candidates(jd_text: str, top_n: int | None = None, min_score: float = 0.0,
                    use_rerank: bool = True, use_context: bool = True) -> dict:
    jd = process_jd(jd_text)
    if not jd["queries"]:
        return {"jd": jd, "results": [], "hidden_duplicates": 0, "total_scored": 0,
                "error": "Could not find any requirements in that text."}

    hits = retrieve(jd["queries"])
    if hits is None:
        return {"jd": jd, "results": [], "hidden_duplicates": 0, "total_scored": 0,
                "error": "The index is empty. Run build_index.py first."}

    if top_n:
        source = "command line or API"
    elif jd.get("top_n"):
        top_n, source = jd["top_n"], "your JD text"
    else:
        top_n, source = DEFAULT_TOP, "default"

    eligible = {c["cv_id"] for c in database.list_candidates() if c.get("status") == "ok"}
    profiles = None
    if any(q.get("min_years") for q in jd["queries"]):
        profiles = experience.load_profiles(eligible)

    scored = score_candidates(jd["queries"], hits, profiles, eligible)
    total_scored = len(scored)
    warning = None
    reranked = False
    if use_rerank:
        try:
            import rerank
            scored = rerank.rerank(jd["queries"], hits, scored, top_n)
            reranked = True
        except Exception as e:
            warning = f"Reranking was skipped ({e}). Showing first-pass results."

    contextualized = False
    if use_context:
        try:
            scored = _contextualize(jd_text, jd["queries"], scored, top_n)
            contextualized = True
        except Exception as e:
            warning = (warning + " " if warning else "") + (
                f"Gemini contextual evaluation was skipped ({type(e).__name__}: {e}). "
                "Showing local ranking only."
            )

    results, seen_emails, hidden = [], set(), 0
    for r in scored:
        if r["score"] < min_score or len(results) >= top_n:
            break
        cand = database.get_candidate(r["cv_id"])
        personal = cand["personal"] if cand else {}
        r["name"] = personal.get("name")
        r["email"] = personal.get("email")
        key = (r["email"] or "").strip().lower()
        if key and key in seen_emails:
            hidden += 1
            continue
        if key:
            seen_emails.add(key)
        results.append(r)

    return {"jd": jd, "results": results, "hidden_duplicates": hidden,
            "total_scored": total_scored, "top_n": top_n, "top_n_source": source,
            "reranked": reranked, "contextualized": contextualized,
            "warning": warning, "error": None}


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
    print("Gemini contextual evaluation enabled." if output.get("contextualized")
          else "Gemini contextual evaluation not applied.")
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
        print(f"   covers {len(covered)} of {len(d)} requirements")
        if r.get("contextual_evaluation"):
            ev = r["contextual_evaluation"]
            print(f"   context fit: {ev['overall_fit']}/100 — {ev['summary']}")
            if ev.get("strengths"):
                print("   strengths: " + "; ".join(ev["strengths"][:3]))
            if ev.get("gaps"):
                print("   gaps: " + "; ".join(ev["gaps"][:3]))
        if missing:
            print("   missing/weak local matches: " + "; ".join(x["requirement"][:45] for x in missing))
        top = max(d, key=lambda x: x["sim"]) if d else None
        if top and top["text"]:
            print(f"   best local evidence ({top['section']}): \"{top['text'].replace(chr(10), ' ')[:110]}\"")
        if details:
            for x in d:
                print(f"      {x['label']:8} {x['sim']:.2f}  {x['requirement'][:60]}"
                      + (f"   [{x['note']}]" if x.get("note") else ""))
        print()


def pop_flag(args: list, name: str, cast, default):
    if name in args:
        i = args.index(name)
        value = cast(args[i + 1])
        del args[i:i + 2]
        return value
    return default


def normalise_flags(args: list) -> list:
    out = []
    for arg in args:
        m = re.fullmatch(r"(--[a-z][a-z-]*?)=?(\d+(?:\.\d+)?)", arg)
        out.extend(m.groups() if m else [arg])
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
    no_context = "--no-context" in args
    if no_context:
        args.remove("--no-context")
    top_n = pop_flag(args, "--top", int, None)
    min_score = pop_flag(args, "--min-score", float, 0.0)
    unknown = [a for a in args if a.startswith("--") and a != "--file"]
    if unknown:
        print(f"Unknown option: {unknown[0]} (valid: --file, --top N, --min-score X, --details, --no-rerank, --no-context)")
        sys.exit(1)
    if args[0] == "--file":
        if len(args) < 2:
            raise SystemExit("--file requires a path")
        text = Path(args[1]).read_text(encoding="utf-8")
    else:
        text = " ".join(args)
    print_results(rank_candidates(text, top_n, min_score, use_rerank=not no_rerank,
                                  use_context=not no_context), details)


if __name__ == "__main__":
    main()
