"""Phase 6 Streamlit dashboard for the continual-learning study."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.engine import RecommendationEngine, RecommendationRequest

from dashboard_data import (
    GLOBAL_FIGURES,
    PHASE5_ROOT,
    available_sections,
    caution_notes,
    comparison_table,
    dataset_figure_map,
    datasets_from_summary,
    load_pareto,
    load_report_text,
    load_summary,
    missing_artifacts,
    top_findings,
)


st.set_page_config(
    page_title="CL Decision Support Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .main {
            background:
                radial-gradient(circle at top left, rgba(36, 99, 235, 0.10), transparent 32%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 28%),
                linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 48%, #0ea5e9 100%);
            border-radius: 24px;
            padding: 28px 30px;
            color: white;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.20);
            margin-bottom: 1.25rem;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.1rem;
            letter-spacing: -0.02em;
        }
        .hero p {
            margin: 0;
            font-size: 1rem;
            opacity: 0.92;
            max-width: 860px;
        }
        .metric-card, .finding-card, .recommend-card {
            background: rgba(255,255,255,0.86);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 20px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
            height: 100%;
        }
        .recommend-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(239,246,255,0.90));
        }
        .section-note {
            color: #334155;
            font-size: 0.95rem;
            margin-bottom: 0.6rem;
        }
        .pill {
            display: inline-block;
            font-size: 0.78rem;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.24);
            margin-right: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _summary_df() -> pd.DataFrame:
    return load_summary()


@st.cache_data(show_spinner=False)
def _pareto_df() -> pd.DataFrame:
    return load_pareto()


@st.cache_data(show_spinner=False)
def _report_text() -> str:
    return load_report_text()


def _warning_banner() -> None:
    missing = missing_artifacts()
    if not missing:
        return
    st.warning(
        "Some Phase 5 artifacts are missing. Parts of the dashboard may be unavailable.\n\n"
        "Regenerate them with `python Project/experiments/run_phase5.py`.",
        icon="⚠️",
    )
    with st.expander("Missing artifact details"):
        for path in missing:
            st.code(path)


def _hero(findings: list[dict[str, str]]) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="pill">Deadline MVP</div>
            <div class="pill">Local Demo</div>
            <div class="pill">Phase 5 Driven</div>
            <h1>Continual Learning Decision Support Dashboard</h1>
            <p>
                A faculty-facing interface for reviewing the current comparative study,
                inspecting empirical trade-offs, and generating recommendation outputs from
                the Phase 5 analysis pipeline without recomputing experiments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if findings:
        cols = st.columns(len(findings))
        for col, finding in zip(cols, findings):
            with col:
                st.markdown(
                    f"""
                    <div class="finding-card">
                        <h4 style="margin:0 0 0.45rem 0;">{finding['title']}</h4>
                        <div style="color:#334155;font-size:0.95rem;">{finding['body']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _sidebar(summary_df: pd.DataFrame) -> RecommendationRequest:
    datasets = datasets_from_summary(summary_df) or ["split_mini_imagenet"]
    with st.sidebar:
        st.header("Recommendation Inputs")
        dataset = st.selectbox(
            "Dataset",
            options=datasets,
            index=datasets.index("split_mini_imagenet") if "split_mini_imagenet" in datasets else 0,
        )
        memory_budget_mb = st.slider("Memory Budget (MB)", min_value=16, max_value=4096, value=250, step=16)
        compute_budget = st.selectbox("Compute Budget", options=["low", "medium", "high"], index=1)
        acceptable_forgetting = st.slider("Acceptable Forgetting", min_value=0, max_value=100, value=20, step=1)
        task_similarity = st.selectbox("Task Similarity", options=["low", "medium", "high"], index=1)
        joint_retraining_allowed = st.toggle("Allow Joint Retraining", value=False)

        st.caption("Recommendations are deadline-valid and will be revalidated after the post-deadline rerun campaign.")

        return RecommendationRequest(
            dataset=dataset,
            memory_budget_mb=float(memory_budget_mb),
            compute_budget=compute_budget,
            acceptable_forgetting=float(acceptable_forgetting),
            task_similarity=task_similarity,
            joint_retraining_allowed=joint_retraining_allowed,
        )


def _format_shortlist(shortlist: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(shortlist, start=1):
        rows.append(
            {
                "Rank": idx,
                "Method": item["method"],
                "Score": item["score"],
                "Avg Accuracy": item["avg_accuracy_mean"],
                "Forgetting": item["forgetting_mean"],
                "Runtime (h)": item["runtime_hours_mean"],
                "Estimated Memory (MB)": item["estimated_memory_mb"],
                "Rationale": " | ".join(item["reasons"]),
            }
        )
    return pd.DataFrame(rows)


def _metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.82rem;color:#475569;margin-bottom:0.35rem;">{label}</div>
            <div style="font-size:1.55rem;font-weight:700;color:#0f172a;">{value}</div>
            <div style="margin-top:0.35rem;color:#475569;font-size:0.86rem;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _recommendation_panel(summary_df: pd.DataFrame, request: RecommendationRequest) -> None:
    st.markdown('<div class="section-note">Decision support from the current Phase 5 recommendation engine.</div>', unsafe_allow_html=True)
    if summary_df.empty:
        st.info("Summary artifacts are unavailable, so no recommendation can be generated yet.")
        return

    engine = RecommendationEngine(summary_df)
    result = engine.recommend(request, top_k=3)
    best = result["shortlist"][0]

    col_left, col_right = st.columns([1.2, 1.0], gap="large")
    with col_left:
        st.markdown(
            f"""
            <div class="recommend-card">
                <div style="font-size:0.85rem;color:#334155;">Recommended Method</div>
                <h2 style="margin:0.2rem 0 0.35rem 0;">{result['recommended_method']}</h2>
                <div style="color:#334155;font-size:0.98rem;">{result['rationale']}</div>
                <div style="margin-top:0.8rem;color:#334155;font-size:0.92rem;">
                    Current recommendations are deadline-valid and will be revalidated after the full rerun campaign.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mcols = st.columns(4)
        with mcols[0]:
            _metric_card("Average Accuracy", f"{best['avg_accuracy_mean']:.2f}%", "Higher is better.")
        with mcols[1]:
            _metric_card("Forgetting", f"{best['forgetting_mean']:.2f}", "Lower means better retention.")
        with mcols[2]:
            _metric_card("Runtime", f"{best['runtime_hours_mean']:.2f} h", "Mean observed runtime in the archive.")
        with mcols[3]:
            _metric_card("Estimated Memory", f"{best['estimated_memory_mb']:.1f} MB", "Estimated replay/storage footprint.")

    with col_right:
        st.subheader("Shortlist")
        st.dataframe(_format_shortlist(result["shortlist"]), use_container_width=True, hide_index=True)

        dataset_rows = summary_df[
            (summary_df["dataset"].astype(str) == request.dataset)
            & (summary_df["method"] == result["recommended_method"])
        ].copy()
        if "is_primary_run" in dataset_rows.columns:
            dataset_rows = dataset_rows[dataset_rows["is_primary_run"] == True]  # noqa: E712
        if not dataset_rows.empty:
            row = dataset_rows.sort_values(["avg_accuracy_mean", "seeds"], ascending=[False, False]).iloc[0]
            st.caption(
                f"Supporting summary: {row['method']} on {row['dataset']} with {int(row['seeds'])} seeds, "
                f"AA {row['avg_accuracy_mean']:.2f}, forgetting {row['forgetting_mean']:.2f}, "
                f"runtime {row['runtime_hours_mean']:.2f} h."
            )


def _comparison_panel(summary_df: pd.DataFrame, request: RecommendationRequest, pareto_df: pd.DataFrame) -> None:
    if summary_df.empty:
        st.info("Comparison tables are unavailable until Phase 5 summary artifacts are present.")
        return

    controls = st.columns([1.0, 1.0, 2.0])
    with controls[0]:
        primary_only = st.toggle("Primary Runs Only", value=True)
    with controls[1]:
        show_notes = st.toggle("Show Notes Column", value=False)

    table = comparison_table(summary_df, request.dataset, primary_only=primary_only)
    if not show_notes and not table.empty and "Notes" in table.columns:
        table = table.drop(columns=["Notes"])

    st.dataframe(
        table.style.highlight_max(subset=["Avg Accuracy"], color="#dbeafe").highlight_min(subset=["Forgetting"], color="#dcfce7"),
        use_container_width=True,
        hide_index=True,
    )

    if not pareto_df.empty:
        subset = pareto_df[pareto_df["dataset"].astype(str) == request.dataset]
        if not subset.empty:
            st.caption(f"Pareto candidates for {request.dataset}: {', '.join(subset['method'].astype(str).tolist())}")


def _render_figure(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {path.name}", icon="🖼️")


def _visuals_panel(request: RecommendationRequest) -> None:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Cross-Dataset Heatmaps")
        _render_figure(GLOBAL_FIGURES["avg_accuracy_heatmap"], "Phase 5 average accuracy heatmap")
        _render_figure(GLOBAL_FIGURES["forgetting_heatmap"], "Phase 5 forgetting heatmap")

    with right:
        st.subheader(f"{request.dataset} Trade-Offs")
        figs = dataset_figure_map(request.dataset)
        _render_figure(figs["accuracy_vs_buffer"], f"{request.dataset}: accuracy vs buffer size")
        _render_figure(figs["accuracy_vs_runtime"], f"{request.dataset}: accuracy vs runtime")


def _about_panel(summary_df: pd.DataFrame) -> None:
    sections = available_sections()
    report_text = _report_text()
    st.markdown('<div class="section-note">Project context, caution notes, and artifact provenance.</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1.4, 1.0], gap="large")
    with col_a:
        if sections.get("report") and report_text:
            st.markdown(report_text)
        else:
            st.info("The generated Phase 5 report is unavailable.")

    with col_b:
        st.subheader("Current Cautions")
        notes = caution_notes(summary_df)
        if notes:
            for note in notes:
                st.markdown(f"- {note}")
        else:
            st.write("No caution notes available.")

        st.subheader("Reproducibility")
        st.code("python Project/experiments/run_phase5.py")
        st.code(str(PHASE5_ROOT))
        st.caption("The dashboard is read-only and uses precomputed Phase 5 artifacts as its source of truth.")


def main() -> None:
    _inject_css()
    summary_df = _summary_df()
    pareto_df = _pareto_df()
    findings = top_findings(summary_df)

    _hero(findings)
    _warning_banner()
    request = _sidebar(summary_df)

    tabs = st.tabs(["Recommendation", "Method Comparison", "Dataset Visuals", "Report / About"])
    with tabs[0]:
        _recommendation_panel(summary_df, request)
    with tabs[1]:
        _comparison_panel(summary_df, request, pareto_df)
    with tabs[2]:
        _visuals_panel(request)
    with tabs[3]:
        _about_panel(summary_df)


if __name__ == "__main__":
    main()
