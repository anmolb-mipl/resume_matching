import json
import re
import sys
from pathlib import Path

input_dir = Path("extracted_texts")
output_dir = Path("section")


#section keywords
SECTION_ALIASES = {
    "summary": ["summary", "professional summary", "profile", "career objective",
                "objective", "about me", "about", "career summary"],
    "skills": ["skills", "technical skills", "key skills", "core skills",
               "core competencies", "skills and tools", "tech stack", "technologies"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment history", "work history", "career history", "internships",
                   "where i have worked"],
    "projects": ["projects", "personal projects", "academic projects", "key projects"],
    "education": ["education", "academic background", "academic qualifications",
                  "qualifications", "educational background"],
    "certifications": ["certifications", "certificates", "licenses and certifications",
                       "certifications and licenses", "courses", "training"],
    "achievements": ["achievements", "awards", "honors", "accomplishments", "awards and honors"],
}

#coversion of aliases to canonical section names
HEADING_LOOKUP = {alias: canon for canon, aliases in SECTION_ALIASES.items() for alias in aliases}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
LINK_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com|gitlab\.com)/[^\s|,]+", re.I)

def normalise_heading(heading:str) -> str:
    heading = heading.lower().replace("&", " and ")
    heading = re.sub(r"[^a-z ]", " ", heading)
    return re.sub(r"\s+", " ", heading).strip()

def detect_heading(line:str):
    stripped= line.strip()
    if not stripped or len(stripped.split()) > 5:
        return None
    return HEADING_LOOKUP.get(normalise_heading(stripped))

def extract_personal(header_lines: list[str]) -> tuple[dict, list[str]]:
    """Pull contact details out of the top block. Returns (personal, leftover_lines)."""
    personal = {"name": None, "email": None, "phone": None, "links": []}
    leftover = []

    for i, line in enumerate(header_lines):
        if i == 0 and not (EMAIL_RE.search(line) or PHONE_RE.search(line)):
            personal["name"] = line.strip()      # first line is usually the name
            continue

        has_contact = False
        email = EMAIL_RE.search(line)
        if email:
            personal["email"] = personal["email"] or email.group()
            has_contact = True
        for m in PHONE_RE.finditer(line):
            digits = re.sub(r"\D", "", m.group())
            if 10 <= len(digits) <= 13:          # avoids matching year ranges
                personal["phone"] = personal["phone"] or m.group().strip()
                has_contact = True
        for m in LINK_RE.finditer(line):
            personal["links"].append(m.group())
            has_contact = True

        if not has_contact:
            leftover.append(line)                # not contact info, keep as CV text

    return personal, leftover


def split_cv(text: str, cv_id: str) -> dict:
    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    header: list[str] = []
    current = None

    for line in lines:
        heading = detect_heading(line)
        if heading:
            current = heading
            sections.setdefault(current, [])
        elif current is None:
            if line.strip():
                header.append(line)             # text before the first heading
        else:
            sections[current].append(line)

    personal, leftover = extract_personal(header)

    # Header text that isn't contact info (e.g. a tagline) is treated as summary
    if leftover:
        sections.setdefault("summary", [])
        sections["summary"] = leftover + sections["summary"]

    cleaned = {name: "\n".join(body).strip() for name, body in sections.items()}
    cleaned = {name: body for name, body in cleaned.items() if body}

    # Flag CVs we couldn't understand instead of silently scoring them badly
    status = "ok"
    if not cleaned:
        status = "needs_review"
        cleaned = {"other": text.strip()}        # keep everything so nothing is lost
    elif "skills" not in cleaned and "experience" not in cleaned:
        status = "needs_review"

    return {"cv_id": cv_id, "status": status, "personal": personal, "sections": cleaned}


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else input_dir
    files = [target] if target.is_file() else sorted(target.glob("*.txt"))
    if not files:
        print("No .txt files found. Run cv_extractor.py first.")
        sys.exit(1)

    output_dir.mkdir(exist_ok=True)
    for path in files:
        result = split_cv(path.read_text(encoding="utf-8"), cv_id=path.stem)
        (output_dir / f"{path.stem}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        found = ", ".join(result["sections"].keys())
        print(f"[{result['status'].upper():12}] {path.name:35} sections: {found}")


if __name__ == "__main__":
    main()