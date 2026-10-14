"""
Step 6b: Streamlit app.

Install:   pip install -r requirements.txt
Run:       python -m streamlit run app.py        (opens http://localhost:8501)

How it works
  1. Add CVs: upload files, or point at a folder on this computer. Every batch starts a
     fresh session with its own database and search index (see workspace.py).
  2. Write the job description (or a skills list). A search only looks at THIS session's CVs.
  3. Read the best matches, with the evidence behind each score.

The app calls the project's modules directly, so the FastAPI server (api.py) is not needed.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

import ingest
import workspace
from jd_processor import process_jd
from search_rank import rank_candidates

ICON = {"strong": "✅", "partial": "🟡", "missing": "❌"}
DEFAULT_COUNT = 5


# ---------- state and helpers ----------

def init_state() -> None:
    defaults = {"workspace": None, "reports": [], "results": None, "uploader_key": 0, "notice": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner="Loading the AI models (the first time can take a few minutes) ...")
def load_models():
    """Load the embedding model and the reranker once, not on every click."""
    import build_index
    build_index.get_model()
    try:
        import rerank
        rerank.get_reranker()
        return None
    except Exception as e:
        return f"The reranker could not load ({e}). Searches will use first-pass results."


def read_cv_folder(folder: str, recursive: bool) -> list:
    """Find CV files in a folder. Returns [(file name, function that reads its bytes)]."""
    path = Path(folder.strip().strip('"')).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"'{path}' is not a folder on this computer.")
    candidates = path.rglob("*") if recursive else path.glob("*")
    files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in ingest.ALLOWED_EXTENSIONS)
    return [(p.name, p.read_bytes) for p in files]


def process_batch(items: list) -> None:
    """Read, split and index a batch of CVs inside the current session's workspace."""
    if st.session_state.workspace is None:
        st.session_state.workspace = str(workspace.new_workspace())
    workspace.use_workspace(st.session_state.workspace)

    bar = st.progress(0.0, text="Reading CVs ...")
    new_reports = []
    for i, (name, get_bytes) in enumerate(items, 1):
        try:
            new_reports.append(ingest.process_upload(name, get_bytes()))
        except Exception as e:
            new_reports.append(ingest.report(name, "failed", message=str(e)))
        bar.progress(i / len(items), text=f"Reading CVs ... {i} of {len(items)}")

    if any(r["status"] == "added" for r in new_reports):
        bar.progress(1.0, text="Building the search index (this can take a minute) ...")
        ingest.update_index()
    bar.empty()

    st.session_state.reports.extend(new_reports)
    st.session_state.results = None          # old results no longer match the loaded CVs


def session_counts() -> dict:
    reports = st.session_state.reports
    status = Counter(r["status"] for r in reports)
    return {
        "searchable": sum(1 for r in reports if r["status"] == "added" and not r["needs_review"]),
        "added": status["added"],
        "duplicates": status["duplicate"],
        "problems": status["failed"] + status["rejected"],
        "needs_review": sum(1 for r in reports if r["needs_review"]),
    }


def end_session(delete_data: bool) -> None:
    ws = st.session_state.workspace
    deleted = None
    if delete_data and ws:
        deleted = workspace.delete_workspace(ws)
    st.session_state.workspace = None
    st.session_state.reports = []
    st.session_state.results = None
    st.session_state.uploader_key += 1
    if delete_data:
        st.session_state.notice = ("This session's data was deleted." if deleted else
                                   f"Could not delete everything. You can delete the folder by hand: {ws}")
    else:
        st.session_state.notice = "Started a new session. The previous CVs will not be searched."


# ---------- screen sections ----------

def sidebar() -> None:
    with st.sidebar:
        st.header("Session")
        counts = session_counts()
        if st.session_state.workspace is None:
            st.info("No CVs loaded yet.")
        else:
            st.metric("CVs ready to search", counts["searchable"])
            st.caption(f"Session folder: {Path(st.session_state.workspace).name}")
        st.toggle("Rerank for better accuracy", value=True, key="use_rerank",
                  help="A second, slower model re-reads the best candidates. Turn off to compare.")
        st.divider()
        if st.button("Start a new session", key="new_session",
                     disabled=st.session_state.workspace is None,
                     help="Clears the current CVs from the search. The files stay on disk."):
            end_session(delete_data=False)
            st.rerun()
        sure = st.checkbox("I want to permanently delete this session's CVs", key="confirm_delete")
        if st.button("Delete this session's data", key="delete_session",
                     disabled=not (sure and st.session_state.workspace)):
            end_session(delete_data=True)
            st.rerun()
        st.caption("CVs are processed on this computer. The AI models are downloaded once "
                   "from Hugging Face; no CV text is sent anywhere.")


def step_add_cvs() -> None:
    st.header("1. Add your CVs")
    tab_files, tab_folder = st.tabs(["Upload files", "Use a folder on this computer"])

    with tab_files:
        st.caption("Tip: open your CV folder in the file picker and press Ctrl+A to select every CV in it.")
        uploaded = st.file_uploader("CV files (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"],
                                    accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
        label = f"Process {len(uploaded)} CV(s)" if uploaded else "Process CVs"
        if st.button(label, key="process_files", type="primary", disabled=not uploaded):
            process_batch([(f.name, f.getvalue) for f in uploaded])
            st.session_state.uploader_key += 1
            st.rerun()

    with tab_folder:
        folder = st.text_input("Folder path", key="folder_path", placeholder=r"C:\Users\you\Desktop\cvs")
        recursive = st.checkbox("Include subfolders", value=True, key="folder_recursive")
        if st.button("Process folder", key="process_folder", type="primary", disabled=not folder.strip()):
            try:
                found = read_cv_folder(folder, recursive)
            except Exception as e:
                st.error(str(e))
                found = None
            if found is not None and not found:
                st.warning("No PDF, DOCX or TXT files were found in that folder.")
            elif found:
                process_batch(found)
                st.rerun()

    reports = st.session_state.reports
    if reports:
        c = session_counts()
        a, b, d, e = st.columns(4)
        a.metric("Ready to search", c["searchable"])
        b.metric("Duplicates skipped", c["duplicates"])
        d.metric("Need a manual look", c["needs_review"])
        e.metric("Could not read", c["problems"])
        if c["needs_review"]:
            names = ", ".join(r["file"] for r in reports if r["needs_review"])
            st.warning(f"These CVs could not be understood well, so they are NOT part of the search: {names}")
        with st.expander("Details for each file"):
            st.dataframe(pd.DataFrame(reports)[["file", "status", "needs_review", "message"]])


def step_describe_job() -> None:
    st.header("2. Describe the job")
    searchable = session_counts()["searchable"]
    jd = st.text_area("Job description, or the skills you need", height=220, key="jd_text",
                      placeholder="Paste a full job description, or just: Python, SQL, AWS. I need top 5 candidates")
    with st.expander("Options"):
        count = st.number_input("Number of candidates to show", min_value=1, max_value=50,
                                value=DEFAULT_COUNT, step=1, key="n_candidates")
        use_jd_number = st.checkbox("If my text says e.g. 'I need top 3 candidates', use that number",
                                    value=True, key="use_jd_number")
        min_score = st.slider("Hide candidates scoring below", 0.0, 1.0, 0.0, 0.05, key="min_score",
                              help="Reranked scores are on a different scale from first-pass scores. "
                                   "Leave at 0 unless you have a reason.")
    if searchable == 0:
        st.info("Add CVs in step 1 first. The search only looks at the CVs in this session.")

    if st.button("Find best candidates", key="search", type="primary",
                 disabled=searchable == 0 or len(jd.strip()) < 3):
        asked = process_jd(jd)["top_n"] if use_jd_number else None
        top_n = asked or int(count)
        with st.spinner("Searching and reranking ..."):
            st.session_state.results = rank_candidates(
                jd, top_n, float(min_score), use_rerank=bool(st.session_state.get("use_rerank", True)))


def results_table(output: dict) -> pd.DataFrame:
    rows = []
    for rank, r in enumerate(output["results"], 1):
        missing = [d["requirement"] for d in r["details"] if d["label"] == "missing"]
        rows.append({"Rank": rank, "Name": r["name"] or r["cv_id"], "Email": r["email"] or "",
                     "Score": round(r["score"], 3),
                     "Covers": f"{len(r['details']) - len(missing)} of {len(r['details'])}",
                     "Missing requirements": "; ".join(missing)})
    return pd.DataFrame(rows)


def details_table(candidate: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "": ICON.get(d["label"], ""),
        "Requirement": d["requirement"],
        "Score": round(d["sim"], 2),
        "Section": d["section"] or "",
        "Best matching text": (d["text"] or "").replace("\n", " ")[:200],
    } for d in candidate["details"]])


def show_results() -> None:
    output = st.session_state.results
    if output is None:
        return
    st.header("3. Best matches")
    if output["error"]:
        st.error(output["error"])
        return
    if output.get("warning"):
        st.warning(output["warning"])

    queries = output["jd"]["queries"]
    how = "reranked with a second model" if output.get("reranked") else "first pass only"
    st.caption(f"Compared {output['total_scored']} CVs against {len(queries)} requirements ({how}). "
               f"Showing {len(output['results'])}.")
    if output["hidden_duplicates"]:
        st.info(f"{output['hidden_duplicates']} duplicate CV(s) of the same person were hidden.")
    if not output["results"]:
        st.info("No candidate reached the minimum score.")
        return

    for rank, r in enumerate(output["results"], 1):
        missing = [d["requirement"] for d in r["details"] if d["label"] == "missing"]
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.subheader(f"{rank}. {r['name'] or r['cv_id']}")
            left.caption(r["email"] or "no email found")
            right.metric("Score", f"{r['score']:.2f}")
            st.progress(min(max(float(r["score"]), 0.0), 1.0))
            st.write(f"Covers **{len(r['details']) - len(missing)} of {len(r['details'])}** requirements"
                     + (f"  \nLooks missing: {'; '.join(missing)}" if missing else ""))
            with st.expander("Why this candidate? See the evidence"):
                st.dataframe(details_table(r))

    table = results_table(output)
    st.download_button("Download this list as CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name="top_candidates.csv", mime="text/csv", key="download_csv")
    with st.expander("Requirements I searched for"):
        st.dataframe(pd.DataFrame(queries)[["text", "kind", "weight"]])
    st.caption("Scores show how closely a CV's text matches the requirements. They support a "
               "decision, they do not make it. Always read the evidence and the CV itself.")


# ---------- the page ----------

def main() -> None:
    st.set_page_config(page_title="CV Matcher", page_icon="📄", layout="wide")
    init_state()
    model_warning = load_models()
    if st.session_state.workspace:
        workspace.use_workspace(st.session_state.workspace)     # re-applied on every rerun

    sidebar()
    st.title("CV Matcher")
    st.caption("Add a batch of CVs, describe the job, and get the best matches with the evidence.")
    if st.session_state.notice:
        st.success(st.session_state.notice)
        st.session_state.notice = None
    if model_warning:
        st.warning(model_warning)

    step_add_cvs()
    step_describe_job()
    show_results()


if __name__ == "__main__":
    main()
