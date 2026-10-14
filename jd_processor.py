"""
Step 4: turn a job description (or a plain skills list) into search queries.

Why: one embedding of a whole JD blurs everything together. Instead we split it into
separate requirements, search for each one, and later score a candidate on how well
they cover ALL of them (and show which ones are missing).

Usage:
    python jd_processor.py "Python, SQL, AWS, Docker"
    python jd_processor.py --file jd.txt

Output: {"title": ..., "years_mentioned": [...], "queries": [{"text", "weight", "kind"}, ...]}
No ML libraries needed: this step is plain text processing.
"""

import json
import re
import sys
from pathlib import Path

MAX_QUERIES = 15

# Weight = how much a requirement counts in the final score
WEIGHTS = {"role": 1.0, "skill": 1.0, "must": 1.0, "duty": 0.7, "nice": 0.5, "overall": 1.0}

HEADINGS = {
    "must": {"requirements", "required skills", "qualifications", "required qualifications",
             "minimum qualifications", "basic qualifications", "must have", "what you need",
             "what we are looking for", "skills required", "key skills", "skills", "who you are",
             "technical skills", "what you bring"},
    "nice": {"nice to have", "preferred", "preferred qualifications", "good to have", "bonus",
             "bonus points", "desirable", "plus", "preferred skills"},
    "duty": {"responsibilities", "key responsibilities", "what you will do", "duties",
             "the role", "role", "about the role", "job description", "what you ll do"},
    "ignore": {"about us", "about the company", "who we are", "benefits", "perks",
               "what we offer", "why join us", "equal opportunity", "how to apply",
               "compensation", "salary", "about"},
}

BULLET_RE = re.compile(r"^\s*(?:[-*\u2022\u25cf\u2013]|\d+[.)])\s+")
YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years|yrs)", re.I)
TITLE_RE = re.compile(r"^(?:job title|position|role)\s*[:\-]\s*(.+)$", re.I)


# ---------- helpers ----------

def norm(line: str) -> str:
    line = line.lower().replace("'", " ").replace("\u2019", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", line)).strip()


def classify_heading(line: str):
    """Return 'must' / 'nice' / 'duty' / 'ignore' if the line is a section heading, else None."""
    s = line.strip()
    if not s or BULLET_RE.match(s) or len(s.split()) > 6:
        return None
    n = norm(s)
    for kind, names in HEADINGS.items():
        if n in names:
            return kind
        if s.endswith(":") and any(n.startswith(name + " ") for name in names):
            return kind
    return None


def clean_item(text: str) -> str:
    text = BULLET_RE.sub("", text.strip())
    return re.sub(r"\s+", " ", text).strip(" .;")


def split_skills(text: str) -> list[str]:
    parts = re.split(r"[,;|\n]", text)
    out = []
    for p in parts:
        p = re.sub(r"^(?:and|or|&)\s+", "", p.strip(), flags=re.I).strip(" .")
        if p and len(p.split()) <= 5:
            out.append(p)
    return out


def looks_like_skill_list(text: str) -> bool:
    return text.count(",") >= 2 and len(text.split()) <= 25 and not text.rstrip().endswith(".")


def sentences(paragraph: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]



NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20}
_NUM = r"(\d{1,3}|" + "|".join(NUM_WORDS) + r")"
_NOUN = r"(?:candidates?|applicants?|resumes?|cvs?|profiles?|people|matches)"
# "I need top 5 candidates", "give me the best three applicants", "shortlist 7 candidates"
_ASK = (r"(?:\b(?:please\s+|kindly\s+|(?:can|could)\s+you\s+)?(?:i\s+)?(?:need|want|give\s+me|show\s+me|find\s+me|get\s+me|shortlist|list)\s+"
        r"(?:me\s+)?(?:the\s+)?(?:(?:top|best)\s+)?" + _NUM + r"\s+(?:(?:top|best)\s+)?" + _NOUN + r")")
# "top 5 candidates", "best 3 applicants"
_TOP = r"(?:\b(?:top|best)\s+" + _NUM + r"\s+" + _NOUN + r")"
TOP_N_RE = re.compile(f"{_ASK}|{_TOP}", re.I)


def extract_top_n(text: str):
    """Find an instruction like 'I need top 5 candidates'. Returns (n or None, text without it)."""
    m = TOP_N_RE.search(text)
    if not m:
        return None, text
    value = next(g for g in m.groups() if g)            # whichever number group matched
    n = int(value) if value.isdigit() else NUM_WORDS[value.lower()]
    cleaned = TOP_N_RE.sub("", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    lines = [l.rstrip(" -\u2013:;,") for l in cleaned.splitlines()]
    cleaned = "\n".join(lines).strip(" \n-\u2013:;,.?!")     # also drops a dangling final period
    return max(1, min(n, 100)), cleaned

# ---------- main logic ----------

def _process_jd(raw: str) -> dict:
    raw = raw.strip()
    if not re.search(r"[A-Za-z]{2,}", raw):           # no real words at all (e.g. "!!!" or "123")
        return {"title": None, "years_mentioned": [], "queries": []}
    lines = [l for l in raw.splitlines()]
    queries: list[dict] = []

    def add(text: str, kind: str):
        text = text.strip()
        if text and text.lower() not in {q["text"].lower() for q in queries}:
            queries.append({"text": text, "weight": WEIGHTS[kind], "kind": kind})

    # --- Case 1: a short skills list, e.g. "Python, SQL, AWS, Docker" ---
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) <= 3 and looks_like_skill_list(" ".join(non_empty)):
        skills = split_skills(" ".join(non_empty))
        for s in skills:
            add(f"Experience with {s}", "skill")
        if len(skills) >= 2:
            add("Candidate with experience in " + ", ".join(skills), "overall")
        return {"title": None, "years_mentioned": [], "queries": queries[:MAX_QUERIES]}

    # --- Case 2: a very short free-text request, e.g. "backend developer with FastAPI" ---
    if len(raw.split()) <= 8 and len(non_empty) == 1:
        add(raw.rstrip("."), "role")
        return {"title": None, "years_mentioned": [], "queries": queries}

    # --- Case 3: a full job description ---
    title = None
    for l in non_empty[:3]:
        m = TITLE_RE.match(l.strip())
        if m:
            title = m.group(1).strip()
            break
    if not title and non_empty:
        first = non_empty[0].strip()
        if len(first.split()) <= 10 and not BULLET_RE.match(first) and not classify_heading(first) \
                and not first.endswith((".", ":")):
            title = first
    if title:
        add(f"Experience working as {title}", "role")

    years = [int(y) for y in YEARS_RE.findall(raw)]

    current = "duty"                    # text before any heading counts as general duties
    ignoring = False
    paragraph: list[str] = []

    def flush_paragraph():
        if paragraph and not ignoring:
            for s in sentences(" ".join(paragraph)):
                handle_unit(clean_item(s), current)
        paragraph.clear()

    def handle_unit(unit: str, kind: str):
        if len(unit.split()) < 3:
            return
        colon = unit.split(":", 1)
        if len(colon) == 2 and looks_like_skill_list(colon[1]):
            unit = colon[1]
        if looks_like_skill_list(unit):
            for s in split_skills(unit):
                add(f"Experience with {s}", "must" if kind == "must" else kind)
        else:
            add(unit, kind)

    for line in lines:
        if not line.strip():
            flush_paragraph()
            continue
        if title and line.strip() == title:
            continue
        if TITLE_RE.match(line.strip()):
            continue
        heading = classify_heading(line)
        if heading:
            flush_paragraph()
            ignoring = heading == "ignore"
            if not ignoring:
                current = heading
            continue
        if ignoring:
            continue
        if BULLET_RE.match(line):
            flush_paragraph()
            handle_unit(clean_item(line), current)
        else:
            paragraph.append(line.strip())
    flush_paragraph()

    # Most important requirements first, then cap the number of queries
    queries.sort(key=lambda q: -q["weight"])
    return {"title": title, "years_mentioned": years, "queries": queries[:MAX_QUERIES]}


def process_jd(raw: str) -> dict:
    """Public entry point: also reads 'I need top N candidates' from the text."""
    top_n, cleaned = extract_top_n(raw.strip())
    result = _process_jd(cleaned)
    result["top_n"] = top_n
    return result


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    text = Path(args[1]).read_text(encoding="utf-8") if args[0] == "--file" else " ".join(args)
    result = process_jd(text)
    print(f"Title: {result['title']}   Years mentioned: {result['years_mentioned']}   "
          f"Top N requested: {result['top_n']}\n")
    for q in result["queries"]:
        print(f"[{q['kind']:7} w={q['weight']}] {q['text']}")
    print(f"\n{len(result['queries'])} queries")


if __name__ == "__main__":
    main()
