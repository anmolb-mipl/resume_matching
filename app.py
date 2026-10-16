
"""
Step 6b: Streamlit app.

Install:
    python -m pip install -r requirements.txt
    python -m pip install google-genai pydantic

Run:
    python -m streamlit run app.py

How it works:
  1. Upload CVs or load them from a folder on this computer.
  2. Each batch uses the current session's workspace and index.
  3. Enter a job description and choose local or Gemini-assisted matching.
  4. Review ranked candidates, scores, evidence, strengths, and gaps.

Matching pipeline:
  - ChromaDB: local vector retrieval
  - Cross-encoder: local reranking
  - Gemini: optional candidate-level contextual evaluation

Privacy:
  - With contextual evaluation disabled, this app does not send CV text
    to Gemini.
  - With contextual evaluation enabled, resume section text is sent to
    Google Gemini for evaluation. The evaluator excludes personal/contact
    sections, but other sections may still contain identifying information.
  - Do not upload CVs to a hosted service without appropriate permission.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

import ingest
import workspace
from jd_processor import process_jd
from search_rank import rank_candidates


ICON = {
    "strong": "✅",
    "partial": "🟡",
    "missing": "❌",
}

CONTEXT_STATUS_ICON = {
    "direct": "✅",
    "transferable": "🔄",
    "partial": "🟡",
    "missing": "❌",
    "unverified": "❔",
}

DEFAULT_COUNT = 5


# ---------------------------------------------------------
# State and helpers
# ---------------------------------------------------------

def init_state() -> None:
    defaults = {
        "workspace": None,
        "reports": [],
        "results": None,
        "uploader_key": 0,
        "notice": None,
        "use_rerank": True,
        "use_context": True,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(
    show_spinner="Loading the AI models (the first time can take a few minutes) ..."
)
def load_models():
    """Load the local embedding model and cross-encoder once."""
    import build_index

    build_index.get_model()

    try:
        import rerank

        rerank.get_reranker()
        return None
    except Exception as e:
        return (
            f"The reranker could not load ({e}). "
            "Searches will use first-pass results until it is available."
        )


def read_cv_folder(folder: str, recursive: bool) -> list:
    """Find supported CV files in a folder."""
    path = Path(folder.strip().strip('"')).expanduser()

    if not path.is_dir():
        raise FileNotFoundError(
            f"'{path}' is not a folder on this computer."
        )

    candidates = path.rglob("*") if recursive else path.glob("*")

    files = sorted(
        p
        for p in candidates
        if p.is_file()
        and p.suffix.lower() in ingest.ALLOWED_EXTENSIONS
    )

    return [(p.name, p.read_bytes) for p in files]


def process_batch(items: list) -> None:
    """Read, split and index CVs in the current session workspace."""
    if not items:
        st.warning("No files were selected.")
        return

    if st.session_state.workspace is None:
        st.session_state.workspace = str(workspace.new_workspace())

    workspace.use_workspace(st.session_state.workspace)

    bar = st.progress(0.0, text="Reading CVs ...")
    new_reports = []

    for i, (name, get_bytes) in enumerate(items, 1):
        try:
            new_reports.append(
                ingest.process_upload(name, get_bytes())
            )
        except Exception as e:
            new_reports.append(
                ingest.report(
                    name,
                    "failed",
                    message=str(e),
                )
            )

        bar.progress(
            i / len(items),
            text=f"Reading CVs ... {i} of {len(items)}",
        )

    if any(r["status"] == "added" for r in new_reports):
        bar.progress(
            1.0,
            text="Building the search index (this can take a minute) ...",
        )
        ingest.update_index()

    bar.empty()

    st.session_state.reports.extend(new_reports)

    # Old rankings no longer represent the current candidate set.
    st.session_state.results = None


def session_counts() -> dict:
    reports = st.session_state.reports
    status = Counter(r["status"] for r in reports)

    return {
        "searchable": sum(
            1
            for r in reports
            if r["status"] == "added"
            and not r.get("needs_review", False)
        ),
        "added": status["added"],
        "duplicates": status["duplicate"],
        "problems": status["failed"] + status["rejected"],
        "needs_review": sum(
            1 for r in reports if r.get("needs_review", False)
        ),
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
        st.session_state.notice = (
            "This session's data was deleted."
            if deleted
            else (
                "Could not delete everything. "
                f"You can delete the folder manually: {ws}"
            )
        )
    else:
        st.session_state.notice = (
            "Started a new session. "
            "The previous CVs will not be searched."
        )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

def sidebar() -> None:
    with st.sidebar:
        st.header("Session")

        counts = session_counts()

        if st.session_state.workspace is None:
            st.info("No CVs loaded yet.")
        else:
            st.metric(
                "CVs ready to search",
                counts["searchable"],
            )
            st.caption(
                "Session folder: "
                f"{Path(st.session_state.workspace).name}"
            )

        st.toggle(
            "Rerank for better accuracy",
            value=True,
            key="use_rerank",
            help=(
                "Use the local cross-encoder to re-read and rerank "
                "the strongest candidate matches."
            ),
        )

        st.toggle(
            "Gemini contextual evaluation",
            value=True,
            key="use_context",
            help=(
                "Use Gemini to evaluate shortlisted CVs against the "
                "overall job requirements. Resume section text is "
                "sent to Google Gemini when this is enabled."
            ),
        )

        if st.session_state.use_context:
            st.caption(
                "Gemini is enabled. Resume section text from shortlisted "
                "candidates may be sent to Google's API. The evaluator "
                "excludes personal/contact sections by name, but other "
                "sections can still contain identifying information."
            )
        else:
            st.caption(
                "Gemini is disabled. Matching uses local retrieval and "
                "the optional local cross-encoder."
            )

        st.divider()

        if st.button(
            "Start a new session",
            key="new_session",
            disabled=st.session_state.workspace is None,
            help=(
                "Clears the current CVs from the search. "
                "The files stay on disk."
            ),
        ):
            end_session(delete_data=False)
            st.rerun()

        sure = st.checkbox(
            "I want to permanently delete this session's CVs",
            key="confirm_delete",
        )

        if st.button(
            "Delete this session's data",
            key="delete_session",
            disabled=not (sure and st.session_state.workspace),
        ):
            end_session(delete_data=True)
            st.rerun()


# ---------------------------------------------------------
# Step 1: Add CVs
# ---------------------------------------------------------

def step_add_cvs() -> None:
    st.header("1. Add your CVs")

    tab_files, tab_folder = st.tabs(
        ["Upload files", "Use a folder on this computer"]
    )

    with tab_files:
        st.caption(
            "Tip: select multiple files in the file picker using Ctrl+A."
        )

        uploaded = st.file_uploader(
            "CV files (PDF, DOCX or TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )

        label = (
            f"Process {len(uploaded)} CV(s)"
            if uploaded
            else "Process CVs"
        )

        if st.button(
            label,
            key="process_files",
            type="primary",
            disabled=not uploaded,
        ):
            process_batch(
                [(f.name, f.getvalue) for f in uploaded]
            )
            st.session_state.uploader_key += 1
            st.rerun()

    with tab_folder:
        folder = st.text_input(
            "Folder path",
            key="folder_path",
            placeholder=r"C:\Users\you\Desktop\cvs",
        )

        recursive = st.checkbox(
            "Include subfolders",
            value=True,
            key="folder_recursive",
        )

        if st.button(
            "Process folder",
            key="process_folder",
            type="primary",
            disabled=not folder.strip(),
        ):
            try:
                found = read_cv_folder(folder, recursive)
            except Exception as e:
                st.error(str(e))
                found = None

            if found is not None and not found:
                st.warning(
                    "No PDF, DOCX or TXT files were found in that folder."
                )
            elif found:
                process_batch(found)
                st.rerun()

    reports = st.session_state.reports

    if reports:
        counts = session_counts()

        a, b, c, d = st.columns(4)

        a.metric("Ready to search", counts["searchable"])
        b.metric("Duplicates skipped", counts["duplicates"])
        c.metric("Need a manual look", counts["needs_review"])
        d.metric("Could not read", counts["problems"])

        if counts["needs_review"]:
            names = ", ".join(
                r["file"]
                for r in reports
                if r.get("needs_review", False)
            )

            st.warning(
                "These CVs could not be understood well, so they are "
                f"NOT part of the search: {names}"
            )

        with st.expander("Details for each file"):
            report_df = pd.DataFrame(reports)

            wanted_columns = [
                col
                for col in [
                    "file",
                    "status",
                    "needs_review",
                    "message",
                ]
                if col in report_df.columns
            ]

            st.dataframe(
                report_df[wanted_columns],
                use_container_width=True,
            )


# ---------------------------------------------------------
# Step 2: Describe the job and search
# ---------------------------------------------------------

def step_describe_job() -> None:
    st.header("2. Describe the job")

    searchable = session_counts()["searchable"]

    jd = st.text_area(
        "Job description, or the skills you need",
        height=220,
        key="jd_text",
        placeholder=(
            "Paste a full job description, or just: "
            "Python, SQL, AWS. I need top 5 candidates"
        ),
    )

    with st.expander("Search options"):
        count = st.number_input(
            "Number of candidates to show",
            min_value=1,
            max_value=50,
            value=DEFAULT_COUNT,
            step=1,
            key="n_candidates",
        )

        use_jd_number = st.checkbox(
            "If my text says e.g. 'I need top 3 candidates', use that number",
            value=True,
            key="use_jd_number",
        )

        min_score = st.slider(
            "Hide candidates scoring below",
            0.0,
            1.0,
            0.0,
            0.05,
            key="min_score",
            help=(
                "Scores are ranking signals, not probabilities. "
                "The score scale may differ depending on the matching "
                "stages used. Leave at 0 unless you have a reason."
            ),
        )

        st.write(
            "**Matching pipeline:** ChromaDB retrieval → "
            "optional local cross-encoder → optional Gemini evaluation."
        )

    if searchable == 0:
        st.info(
            "Add CVs in step 1 first. The search only looks at CVs "
            "in this session."
        )

    if st.button(
        "Find best candidates",
        key="search",
        type="primary",
        disabled=searchable == 0 or len(jd.strip()) < 3,
    ):
        asked = (
            process_jd(jd)["top_n"]
            if use_jd_number
            else None
        )

        top_n = asked or int(count)

        # Re-apply the current session workspace before using the
        # database and index modules.
        if st.session_state.workspace:
            workspace.use_workspace(st.session_state.workspace)

        try:
            with st.spinner(
                "Retrieving candidates, reranking and evaluating ..."
            ):
                st.session_state.results = rank_candidates(
                    jd,
                    top_n,
                    float(min_score),
                    use_rerank=bool(
                        st.session_state.get("use_rerank", True)
                    ),
                    use_context=bool(
                        st.session_state.get("use_context", True)
                    ),
                )
        except Exception as e:
            st.session_state.results = {
                "error": f"Search failed: {e}",
                "results": [],
                "hidden_duplicates": 0,
                "total_scored": 0,
                "jd": {"queries": []},
            }


# ---------------------------------------------------------
# Results tables
# ---------------------------------------------------------

def results_table(output: dict) -> pd.DataFrame:
    rows = []

    for rank, candidate in enumerate(output["results"], 1):
        details = candidate.get("details", [])

        missing = [
            d["requirement"]
            for d in details
            if d.get("label") == "missing"
        ]

        short = [
            d.get("note", "")
            for d in details
            if d.get("years_factor", 1.0) < 1.0
            and d.get("note")
        ]

        evaluation = candidate.get("contextual_evaluation") or {}

        rows.append(
            {
                "Rank": rank,
                "Name": candidate.get("name") or candidate["cv_id"],
                "Email": candidate.get("email") or "",
                "Final score": round(candidate.get("score", 0.0), 3),
                "Local score": round(
                    candidate.get(
                        "local_score",
                        candidate.get("score", 0.0),
                    ),
                    3,
                ),
                "Gemini fit / 100": evaluation.get("overall_fit", ""),
                "Covers locally": (
                    f"{len(details) - len(missing)} of {len(details)}"
                ),
                "Missing local matches": "; ".join(missing),
                "Experience check": "; ".join(short),
            }
        )

    return pd.DataFrame(rows)


def details_table(candidate: dict) -> pd.DataFrame:
    rows = []

    for detail in candidate.get("details", []):
        rows.append(
            {
                "Local status": ICON.get(
                    detail.get("label", ""),
                    "",
                ),
                "Requirement": detail.get("requirement", ""),
                "Local score": round(
                    detail.get("sim", 0.0),
                    2,
                ),
                "Section": detail.get("section") or "",
                "Experience check": detail.get("note", ""),
                "Gemini status": detail.get("context_status", ""),
                "Gemini explanation": detail.get(
                    "context_explanation",
                    "",
                ),
                "Gemini evidence": detail.get(
                    "context_evidence",
                    "",
                ),
                "Best local matching text": (
                    detail.get("text", "")
                    .replace("\n", " ")[:300]
                ),
            }
        )

    return pd.DataFrame(rows)


def contextual_assessments_table(evaluation: dict) -> pd.DataFrame:
    """Format the structured Gemini requirement assessments."""
    rows = []

    for assessment in evaluation.get("assessments", []):
        status = assessment.get("status", "")

        rows.append(
            {
                "Status": (
                    f"{CONTEXT_STATUS_ICON.get(status, '')} {status}"
                ),
                "Requirement": assessment.get("requirement", ""),
                "Explanation": assessment.get("explanation", ""),
                "Resume evidence": assessment.get("evidence_quote", ""),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Step 3: Display results
# ---------------------------------------------------------

def show_contextual_evaluation(candidate: dict) -> None:
    evaluation = candidate.get("contextual_evaluation")

    if not evaluation:
        return

    st.markdown("#### Gemini contextual assessment")

    overall_fit = evaluation.get("overall_fit")

    if overall_fit is not None:
        st.metric(
            "Contextual fit",
            f"{overall_fit}/100",
            help=(
                "Gemini's evidence-based contextual assessment. "
                "This is not a probability of being hired."
            ),
        )

    if evaluation.get("summary"):
        st.write(evaluation["summary"])

    strengths = evaluation.get("strengths", [])
    transferable = evaluation.get("transferable_skills", [])
    gaps = evaluation.get("gaps", [])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Strengths**")

        if strengths:
            for item in strengths:
                st.markdown(f"- {item}")
        else:
            st.caption("No strengths were returned.")

        st.markdown("**Transferable skills**")

        if transferable:
            for item in transferable:
                st.markdown(f"- {item}")
        else:
            st.caption("No transferable skills were returned.")

    with col2:
        st.markdown("**Gaps or concerns**")

        if gaps:
            for item in gaps:
                st.markdown(f"- {item}")
        else:
            st.caption("No gaps were returned.")

    assessments = contextual_assessments_table(evaluation)

    if not assessments.empty:
        with st.expander("Gemini assessment for each requirement"):
            st.dataframe(
                assessments,
                use_container_width=True,
                hide_index=True,
            )


def show_results() -> None:
    output = st.session_state.results

    if output is None:
        return

    st.header("3. Best matches")

    if output.get("error"):
        st.error(output["error"])
        return

    if output.get("warning"):
        st.warning(output["warning"])

    queries = output.get("jd", {}).get("queries", [])

    how = (
        "reranked with the local cross-encoder"
        if output.get("reranked")
        else "first pass only"
    )

    contextualized = output.get("contextualized", False)

    context_text = (
        "Gemini contextual evaluation applied"
        if contextualized
        else "Gemini contextual evaluation not applied"
    )

    st.caption(
        f"Compared {output.get('total_scored', 0)} CVs against "
        f"{len(queries)} requirements ({how}). "
        f"{context_text}. "
        f"Showing {len(output.get('results', []))} candidates."
    )

    if output.get("hidden_duplicates", 0):
        st.info(
            f"{output['hidden_duplicates']} duplicate CV(s) of the "
            "same person were hidden."
        )

    candidates = output.get("results", [])

    if not candidates:
        st.info("No candidate reached the minimum score.")
        return

    for rank, candidate in enumerate(candidates, 1):
        details = candidate.get("details", [])

        missing = [
            d["requirement"]
            for d in details
            if d.get("label") == "missing"
        ]

        with st.container(border=True):
            left, right = st.columns([4, 1])

            left.subheader(
                f"{rank}. "
                f"{candidate.get('name') or candidate['cv_id']}"
            )

            left.caption(
                candidate.get("email") or "No email found"
            )

            right.metric(
                "Final score",
                f"{candidate.get('score', 0.0):.2f}",
            )

            score = float(candidate.get("score", 0.0))
            st.progress(min(max(score, 0.0), 1.0))

            local_score = candidate.get("local_score")

            if local_score is not None:
                st.caption(
                    f"Local score: {local_score:.3f} · "
                    "Final score may blend local and contextual scores."
                )

            covered = len(details) - len(missing)

            st.write(
                f"Covers **{covered} of {len(details)}** "
                "requirements by local matching."
            )

            if missing:
                st.write(
                    "**Weak or missing local matches:** "
                    + "; ".join(missing)
                )

            short = [
                d.get("note", "")
                for d in details
                if d.get("years_factor", 1.0) < 1.0
                and d.get("note")
            ]

            if short:
                st.caption(
                    "Experience check: " + "; ".join(short)
                )

            evaluation = candidate.get("contextual_evaluation")

            if evaluation:
                show_contextual_evaluation(candidate)

            with st.expander(
                "Why this candidate? See matching evidence"
            ):
                table = details_table(candidate)

                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True,
                )

    table = results_table(output)

    st.download_button(
        "Download this list as CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="top_candidates.csv",
        mime="text/csv",
        key="download_csv",
    )

    with st.expander("Requirements extracted from the job description"):
        if queries:
            query_df = pd.DataFrame(queries)

            columns = [
                column
                for column in [
                    "text",
                    "kind",
                    "weight",
                    "min_years",
                    "years_scope",
                ]
                if column in query_df.columns
            ]

            st.dataframe(
                query_df[columns],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No requirements were returned.")

    st.caption(
        "Scores are decision-support signals, not probabilities of "
        "success or hiring recommendations. Review the original CV and "
        "supporting evidence before making a hiring decision."
    )


# ---------------------------------------------------------
# Main page
# ---------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="CV Matcher",
        page_icon="📄",
        layout="wide",
    )

    init_state()

    model_warning = load_models()

    # Re-apply the current workspace on every Streamlit rerun.
    if st.session_state.workspace:
        workspace.use_workspace(
            st.session_state.workspace
        )

    sidebar()

    st.title("CV Matcher")

    st.caption(
        "Add a batch of CVs, describe the job, and compare candidates "
        "using local matching with optional Gemini contextual evaluation."
    )

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
