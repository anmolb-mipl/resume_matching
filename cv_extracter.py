"""Usage: python cv_extracter.py <folder_or_file>"""
import re
import sys
import hashlib

from pathlib import Path
import pdfplumber
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

MIN_CHARS_PER_PAGE =50  # if <= this consider as scanned pdf
output_dir =Path("extracted_texts")

# Read pdf files(normal not scanned)
def read_pdf(path: Path) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)

# Read scanned images
def read_pdf_ocr(path: Path) -> str:
     import pytesseract
     from pdf2image import convert_from_path
     pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

     images = convert_from_path(
        str(path),
        poppler_path=r"C:\poppler\poppler-26.09.0\Library\bin",
    )
     return "\n".join([pytesseract.image_to_string(image) for image in images])

# Read docx files
def read_docx(path: Path) -> str:
    """Read paragraphs AND tables in the order they appear in the document."""
    doc = Document(path)
    parts = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            parts.append(Paragraph(child, doc).text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                cells, seen = [], set()
                for cell in row.cells:
                    if id(cell._tc) not in seen:          # merged cells repeat; keep one
                        seen.add(id(cell._tc))
                        cells.append(cell.text.strip())
                parts.append(" ".join(c for c in cells if c))
    return "\n".join(parts)

# Read txt files
def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")



# Cleaning
def clean_text(text: str) -> str:
        text = text.replace("\u2022", "-").replace("\u25cf", "-")
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def extract_text(path:Path) -> tuple[str, str]:
     ext = path.suffix.lower()
     if ext == ".pdf":
          text =read_pdf(path)
          with pdfplumber.open(path) as pdf:
            n_pages = max(len(pdf.pages), 1)
            if len(text.strip()) < MIN_CHARS_PER_PAGE * n_pages:
                try:
                    return clean_text(read_pdf_ocr(path)), "pdf+ocr"
                except Exception as e:
                    raise ValueError(f"scanned PDF, OCR failed: {e}")
            return clean_text(text), "pdf"
 
     if ext == ".docx":
        return clean_text(read_docx(path)), "docx"
 
     if ext == ".txt":
        return clean_text(read_txt(path)), "txt"
 
     raise ValueError(f"Unsupported file type: {ext}")


def file_hash(path):
       return hashlib.sha256(path.read_bytes()).hexdigest()[:8]

def make_cv_id(path, h):
       safe = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "cv"
       return f"{safe}_{h}"

# Collecting the files
def collect_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    exts = {".pdf", ".docx", ".txt"}
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in exts)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    files = collect_files(Path(sys.argv[1]))
    if not files:
        print("No CV files found.")
        sys.exit(1)

    output_dir.mkdir(exist_ok=True)
    ok, skipped, failed = 0, 0, []

    for path in files:
        try:
            h = file_hash(path)
            # Same bytes already extracted (this run or an earlier one)? Skip.
            if any(output_dir.glob(f"*_{h}.txt")):
                print(f"[SKIP] {path.name:40} duplicate of an already extracted file")
                skipped += 1
                continue

            text, method = extract_text(path)
            if not text:
                raise ValueError("no text extracted")

            cv_id = make_cv_id(path, h)
            (output_dir / f"{cv_id}.txt").write_text(text, encoding="utf-8")
            print(f"[OK]   {path.name:40} -> {cv_id}.txt  ({method}, {len(text)} chars)")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {path.name:40} {e}")
            failed.append(path.name)

    print(f"\nDone: {ok} extracted, {skipped} duplicates skipped, {len(failed)} failed.")
    if failed:
        print("Failed files:", ", ".join(failed))

        
if __name__ == "__main__":
    main()