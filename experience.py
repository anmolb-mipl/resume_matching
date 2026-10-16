"""
Experience facts: the part of "meaning" that similarity search cannot do.

An embedding model sees "3 years of Python" and "10 years of Python" as nearly the same text.
So years are handled here, with arithmetic:

  CV side   read the dates in each job ("Jan 2023 - Present") and work out
            - total years of experience (overlapping jobs are not double counted)
            - years per skill (the time spent in jobs whose text mentions that skill)
  JD side   turn "3+ years of Python" into a number (3) and a scope ("Python"),
            and keep the sentence itself ("Python experience") for the similarity search.
  Scoring   a requirement's score is multiplied by min(1, years_have / years_needed).
            Two years against five required gives x0.4. Meeting the number changes nothing.

Everything here is plain Python (no ML libraries) and shows its working in a short note.
Dates given as a year only ("2019 - 2021") are treated as mid-year, so that is two years.
"""

import re
from datetime import date

FLOOR = 0.2               # a shortfall never reduces a requirement below 20% of its score
UNVERIFIED_FACTOR = 0.8   # the CV mentions the skill/years but gives no dates to check them

# ---------------------------------------------------------------- dates

MONTH_WORDS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_DATE_RAW = r"(?:[A-Za-z]{3,9}\.?,?\s+)?(?:19|20)\d{2}|\d{1,2}\s*[/.-]\s*(?:19|20)\d{2}"
_NOW = r"present|current(?:ly)?|now|till\s+date|to\s+date|ongoing|today"
RANGE_RE = re.compile(
    rf"(?P<start>{_DATE_RAW})\s*(?:-|\u2013|\u2014|to|until|till)\s*(?P<end>{_DATE_RAW}|{_NOW})", re.I)
BULLET_RE = re.compile(r"^\s*[-*+\u2022\u25cf\u2013]\s+")


def _today() -> date:
    return date.today()


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _parse_point(raw: str, today: date) -> int | None:
    """'Jan 2023' / '06/2010' / '2019' / 'Present'  ->  month number (year*12 + month-1)."""
    raw = raw.strip()
    if re.fullmatch(_NOW, raw, re.I):
        return _month_index(today.year, today.month)
    m = re.fullmatch(r"(\d{1,2})\s*[/.-]\s*((?:19|20)\d{2})", raw)
    if m and 1 <= int(m.group(1)) <= 12:
        return _month_index(int(m.group(2)), int(m.group(1)))
    year = re.search(r"(?:19|20)\d{2}", raw)
    if not year:
        return None
    words = re.findall(r"[A-Za-z]+", raw[:year.start()])
    month = MONTH_WORDS.get(words[-1].lower()) if words else None
    return _month_index(int(year.group()), month or 7)          # year only -> mid-year


def _interval(match, today: date):
    start = _parse_point(match.group("start"), today)
    end = _parse_point(match.group("end"), today)
    now = _month_index(today.year, today.month)
    if start is None or end is None:
        return None
    end = min(end, now)                                          # a job cannot run past today
    if start > now or start < _month_index(today.year - 60, 1):
        return None
    if end <= start:
        end = start + 6 if end == start or (end // 12 == start // 12) else None
        if end is None:
            return None
    return start, end


def union_months(intervals) -> int:
    """Total months covered, counting overlapping jobs only once."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def split_roles(experience_text: str, today: date) -> list[dict]:
    """
    Cut an Experience section into jobs. A line holding a date range starts a new job.
    If that line is only a date, the line above it (the job title) joins the new job.
    """
    roles: list[dict] = []
    preamble: list[str] = []
    for line in experience_text.splitlines():
        if not line.strip():
            continue
        m = None if BULLET_RE.match(line) else RANGE_RE.search(line)
        interval = _interval(m, today) if m else None
        if interval:
            leftover = (line[:m.start()] + " " + line[m.end():]).strip(" ()-|,")
            new = {"start": interval[0], "end": interval[1], "lines": [line]}
            if len(leftover.split()) < 3:                        # a date-only line: title is above it
                if roles and len(roles[-1]["lines"]) > 1 and not BULLET_RE.match(roles[-1]["lines"][-1]):
                    new["lines"].insert(0, roles[-1]["lines"].pop())
                elif not roles and preamble:
                    new["lines"].insert(0, preamble.pop())
            roles.append(new)
        elif roles:
            roles[-1]["lines"].append(line)
        else:
            preamble.append(line)
    return [{"start": r["start"], "end": r["end"], "text": " ".join(r["lines"])} for r in roles]


CLAIM_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b(?:\s+of)?(?:\s+[A-Za-z/&-]+){0,3}?\s+(?:experience|exp\b)", re.I)


def build_profile(sections: dict, today: date | None = None) -> dict:
    """Facts about one candidate, worked out from their stored CV sections."""
    today = today or _today()
    experience = sections.get("experience", "")
    roles = split_roles(experience, today)
    computed = union_months([(r["start"], r["end"]) for r in roles]) / 12 if roles else None
    claims = [float(c) for c in CLAIM_RE.findall(sections.get("summary", "") + "\n" + experience)]
    return {"computed_years": computed, "claimed_years": max(claims) if claims else None,
            "roles": roles, "text": "\n".join(sections.values())}


def load_profiles(cv_ids, today: date | None = None) -> dict:
    """Profiles for many candidates from the CURRENT database (the active workspace)."""
    import database
    wanted, by_cv = set(cv_ids), {}
    with database.connect() as conn:
        rows = conn.execute("SELECT cv_id, section_name, text FROM sections").fetchall()
    for r in rows:
        if r["cv_id"] in wanted:
            by_cv.setdefault(r["cv_id"], {})[r["section_name"]] = r["text"]
    return {cv: build_profile(sections, today) for cv, sections in by_cv.items()}


# ---------------------------------------------------------------- skills inside jobs

STOP = {"a", "an", "the", "of", "in", "with", "using", "on", "as", "for", "to", "and", "or", "at", "by",
        "is", "are", "be", "have", "has", "strong", "good", "solid", "proven", "excellent"}
GENERIC = {"software", "development", "developer", "developing", "engineering", "engineer", "professional",
           "relevant", "industry", "work", "working", "hands", "hands-on", "related", "field", "technical",
           "experience", "exp", "similar", "role", "roles", "position", "building", "shipping", "production",
           "designing", "delivering", "total", "overall", "commercial", "years", "year", "applications",
           "application", "systems", "system"}

ALIAS_GROUPS = [
    {"javascript", "js"}, {"typescript", "ts"}, {"postgresql", "postgres", "psql"}, {"kubernetes", "k8s"},
    {"machine learning", "ml"}, {"natural language processing", "nlp"}, {"amazon web services", "aws"},
    {"node.js", "nodejs", "node"}, {"c++", "cpp"}, {"scikit-learn", "sklearn"},
    {"ci/cd", "cicd", "continuous integration"}, {"golang"}, {"mongodb", "mongo"},
]


# Working with these tools means working in the skill on the left, even if a job never says so.
# Extend it as you meet new cases.
IMPLIED = {
    "python": ["pytorch", "tensorflow", "scikit-learn", "sklearn", "pandas", "numpy", "django", "flask",
               "fastapi", "spacy", "hugging face", "sentence-transformers", "pyspark"],
    "java": ["spring", "spring boot", "hibernate", "maven"],
    "javascript": ["react", "angular", "vue", "node.js", "next.js", "express", "typescript"],
    "sql": ["postgresql", "mysql", "sql server", "oracle", "sqlite"],
    "aws": ["ec2", "s3", "lambda", "sagemaker", "eks", "cloudwatch"],
    "machine learning": ["pytorch", "tensorflow", "scikit-learn", "model training", "deep learning", "nlp",
                         "hugging face", "classification model", "recommendation system"],
    "docker": ["kubernetes", "helm"],
    "kubernetes": ["helm", "eks", "argocd"],
}


def content_terms(phrase: str) -> list[str]:
    words = [w.strip(".") for w in re.findall(r"[A-Za-z0-9+#.\-/]+", phrase.lower())]
    return [w for w in words if w and w not in STOP and w not in GENERIC]


def _variants(term: str) -> set[str]:
    for group in ALIAS_GROUPS:
        if term in group:
            return set(group)
    return {term}


def _term_pattern(term: str):
    body = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(r"(?<![A-Za-z0-9+#])" + body + r"(?:s|es)?(?![A-Za-z0-9+#])", re.I)


def make_matcher(phrase: str):
    """A function that says whether a piece of text mentions this skill (or its usual aliases)."""
    terms = content_terms(phrase)
    if not terms:
        return None
    key = " ".join(terms)
    forms = _variants(key) | set(IMPLIED.get(key, []))
    patterns = [_term_pattern(v) for v in forms]
    stems = [t[:5] for t in terms]

    def match(text: str) -> bool:
        if any(p.search(text) for p in patterns):
            return True
        if len(terms) > 1:                                  # all key words present, in any order
            words = {w[:5] for w in re.findall(r"[a-z0-9+#]+", text.lower())}
            return all(s in words for s in stems)
        return False
    return match


def skill_years(profile: dict, phrase: str):
    """(years in dated jobs that mention the skill or None, mentioned anywhere in the CV)."""
    matcher = make_matcher(phrase)
    if matcher is None:
        return None, False
    spans = [(r["start"], r["end"]) for r in profile["roles"] if matcher(r["text"])]
    if spans:
        return union_months(spans) / 12, True
    return None, matcher(profile["text"])


# ---------------------------------------------------------------- the JD side

_LEAD = r"(?:(?:minimum|min\.?|at\s+least|over|more\s+than)\s+(?:of\s+)?)?"
_YEARS = r"(?P<n>\d{1,2})(?:\s*(?:-|\u2013|to)\s*\d{1,2})?\s*\+?\s*(?:years?|yrs?)\b"
SKILL_EXP_RE = re.compile(
    _LEAD + _YEARS + r"\s+(?:of\s+)?(?P<skill>[A-Za-z0-9+#./ -]{1,40}?)\s+(?:experience|exp\b\.?)", re.I)
GENERIC_EXP_RE = re.compile(
    _LEAD + _YEARS + r"(?:\s+of)?(?:\s+(?:relevant|professional|work|working|hands-on|industry|total|overall|solid|proven))*"
    r"(?:\s+(?P<exp>experience|exp\b\.?))?", re.I)
_DANGLING = re.compile(r"\b(?:with|of|in|and|having|for|having)\s*$", re.I)


def is_skill_phrase(phrase: str) -> bool:
    """A single clear skill or topic ('Python', 'machine learning'), not a list or a sentence."""
    p = phrase.strip(" .")
    if not p or re.search(r"\b(?:and|or)\b|[,;&]", p, re.I) or len(p.split()) > 4:
        return False
    return bool(content_terms(p))


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .,;:-")
    text = _DANGLING.sub("", text).strip(" .,;:-")
    return text[:1].upper() + text[1:] if text else text


def parse_years_requirement(unit: str, keep_experience_word: bool = True):
    """
    'Python developer with 3 years of experience'  ->  ('Python developer with experience', 3, None)
    '3+ years of Python experience'                ->  ('Python experience', 3, 'Python')
    '4+ years of experience in machine learning'   ->  ('Experience in machine learning', 4, 'machine learning')
    Returns (text for the similarity search, years needed, skill the years apply to or None = overall).
    If the text states no years, returns (unit, None, None).
    """
    m = SKILL_EXP_RE.search(unit)
    if m and content_terms(m.group("skill")):                  # '3 years of Python experience'
        skill = m.group("skill").strip()
        word = " experience " if keep_experience_word else " "
        text = _tidy(unit[:m.start()] + f" {skill}{word}" + unit[m.end():])
        return text, int(m.group("n")), skill if is_skill_phrase(skill) else None

    m = GENERIC_EXP_RE.search(unit)
    if not m:
        return unit, None, None
    left, right = unit[:m.start()], unit[m.end():]
    scope_text = re.sub(r"^\s*(?:in|with|using|on|of|as)\s+", "", right.strip(), flags=re.I).strip(" .")
    scope = scope_text if is_skill_phrase(scope_text) else None
    text = _tidy(left + (" experience " if m.group("exp") and keep_experience_word else " ") + right)
    if len(text.split()) < 2:
        text = "Relevant professional work experience"
    return text, int(m.group("n")), scope


# ---------------------------------------------------------------- scoring

def years_factor(query: dict, profile: dict):
    """(multiplier 0.2..1.0, short explanation) for one requirement and one candidate."""
    need = query.get("min_years")
    if not need:
        return 1.0, ""
    scope = query.get("years_scope")
    if scope:
        have, mentioned = skill_years(profile, scope)
        if have is None:
            if mentioned:
                total = profile["computed_years"] if profile["computed_years"] is not None else profile["claimed_years"]
                if total is not None and total < need:          # a skill cannot be older than the whole career
                    return max(FLOOR, total / need), \
                        f"needs {need}+ yrs of {scope}: only about {total:.1f} yrs of experience in total"
                return UNVERIFIED_FACTOR, f"needs {need}+ yrs of {scope}: mentioned, but no dated job shows how long"
            return FLOOR, f"needs {need}+ yrs of {scope}: not found in this CV"
        what = f"yrs of {scope}"
    else:
        have, what = profile["computed_years"], "yrs of experience"
        if have is None:
            if profile["claimed_years"] is None:
                return UNVERIFIED_FACTOR, f"needs {need}+ yrs: no job dates or stated total found"
            have, what = profile["claimed_years"], "yrs of experience (stated by the candidate)"
    ratio = have / need
    if ratio >= 1:
        return 1.0, f"meets it: about {have:.1f} {what} vs {need}+ needed"
    return max(FLOOR, ratio), f"short: about {have:.1f} {what} vs {need}+ needed"
