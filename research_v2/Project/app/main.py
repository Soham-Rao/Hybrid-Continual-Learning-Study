"""Streamlit entrypoint for the research workbench dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard_data import DATASET_LABELS, dataset_options, load_dashboard_bundle
from app.copilot_runtime import get_copilot_settings, get_copilot_status
from app.copilot_ui import init_copilot_state, render_copilot_panel
from app.dashboard_views import (
    render_artifact_library_tab,
    render_decision_tree_tab,
    inject_css,
    render_comparison_tab,
    render_dataset_visuals_tab,
    render_hero,
    render_missing_artifacts,
    render_recommendation_tab,
    render_report_tab,
)
from src.recommendation.engine import RecommendationRequest


st.set_page_config(
    page_title="CL Research Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _bundle():
    return load_dashboard_bundle()


def _sidebar(bundle) -> dict:
    datasets = dataset_options(bundle.recommendation_profiles if not bundle.recommendation_profiles.empty else bundle.summary)
    default_dataset = "split_mini_imagenet" if "split_mini_imagenet" in datasets else datasets[0]
    st.session_state.setdefault("dashboard_dataset", default_dataset)
    st.session_state.setdefault("dashboard_families", ["baseline", "hybrid"])
    st.session_state.setdefault("dashboard_include_joint", True)
    st.session_state.setdefault("dashboard_top_cluster_only", False)
    st.session_state.setdefault("dashboard_memory_budget_mb", 256)
    st.session_state.setdefault("dashboard_compute_budget", "medium")
    st.session_state.setdefault("dashboard_acceptable_forgetting", 20)
    st.session_state.setdefault("dashboard_task_similarity", "medium")
    st.session_state.setdefault("dashboard_joint_allowed", False)
    if st.session_state["dashboard_dataset"] not in datasets:
        st.session_state["dashboard_dataset"] = default_dataset
    with st.sidebar:
        st.title("Dashboard")
        st.caption("Research workbench powered by finalized study artifacts.")
        copilot_status = get_copilot_status()
        copilot_settings = get_copilot_settings()
        with st.expander("Copilot Status", expanded=False):
            if copilot_status.available and copilot_status.model_available:
                st.success(copilot_status.message)
            elif copilot_status.available:
                st.warning(copilot_status.message)
            else:
                st.error(copilot_status.message)
            st.caption(f"Default local model: `{copilot_settings.default_model}`")
            st.caption(
                "Track 1 foundation is wired: local model discovery, health checks, and safe offline fallback are ready."
            )
        st.subheader("Study Lens")
        dataset = st.selectbox(
            "Dataset",
            options=datasets,
            format_func=lambda item: DATASET_LABELS.get(str(item), str(item)),
            key="dashboard_dataset",
        )
        families = st.multiselect(
            "Method families",
            options=["baseline", "hybrid"],
            help="Use this to simplify the comparison panels without changing the underlying source tables.",
            key="dashboard_families",
        )
        include_joint = st.toggle(
            "Include joint training in general comparisons",
            help="Joint training remains excluded from Pareto views even when this toggle is on.",
            key="dashboard_include_joint",
        )
        top_cluster_only = st.toggle(
            "Show only top-cluster methods",
            help="Limits method views to methods that are not significantly worse than the dataset leader.",
            key="dashboard_top_cluster_only",
        )

        st.subheader("Recommendation Inputs")
        memory_budget_mb = st.slider(
            "Memory budget (MB)",
            min_value=16,
            max_value=4096,
            step=16,
            key="dashboard_memory_budget_mb",
        )
        compute_budget = st.selectbox(
            "Compute budget",
            options=["low", "medium", "high"],
            key="dashboard_compute_budget",
        )
        acceptable_forgetting = st.slider(
            "Acceptable forgetting",
            min_value=0,
            max_value=100,
            step=1,
            key="dashboard_acceptable_forgetting",
        )
        task_similarity = st.selectbox(
            "Task similarity",
            options=["low", "medium", "high"],
            key="dashboard_task_similarity",
        )
        joint_retraining_allowed = st.toggle(
            "Allow joint retraining",
            key="dashboard_joint_allowed",
        )

        with st.expander("Glossary", expanded=False):
            st.markdown(
                """
                - **AA**: Average Accuracy across tasks after sequential training.
                - **Forgetting**: How much earlier-task performance drops after later tasks arrive.
                - **Top non-inferior cluster**: Methods not significantly worse than the dataset leader after Holm correction.
                - **Proxy memory**: Estimated memory footprint from configuration; not measured peak VRAM.
                - **Ablation context**: Secondary evidence shown for mechanism and robustness context, not primary claims.
                """
            )

    return {
        "dataset": dataset,
        "families": families,
        "include_joint": include_joint,
        "top_cluster_only": top_cluster_only,
        "request": RecommendationRequest(
            dataset=dataset,
            memory_budget_mb=float(memory_budget_mb),
            compute_budget=str(compute_budget),
            acceptable_forgetting=float(acceptable_forgetting),
            task_similarity=str(task_similarity),
            joint_retraining_allowed=bool(joint_retraining_allowed),
        ),
    }


def main() -> None:
    inject_css()
    init_copilot_state()
    bundle = _bundle()
    render_missing_artifacts(bundle)
    render_hero(bundle)
    state = _sidebar(bundle)
    tabs = st.tabs(["Recommendation", "Decision Tree", "Method Comparison", "Dataset Visuals", "Report / About", "Library"])
    with tabs[0]:
        render_recommendation_tab(bundle, state["request"])
    with tabs[1]:
        render_decision_tree_tab(bundle, state["request"])
    with tabs[2]:
        render_comparison_tab(
            bundle,
            dataset=state["dataset"],
            families=state["families"],
            include_joint=state["include_joint"],
            top_cluster_only=state["top_cluster_only"],
        )
    with tabs[3]:
        render_dataset_visuals_tab(
            bundle,
            dataset=state["dataset"],
            families=state["families"],
            include_joint=state["include_joint"],
        )
    with tabs[4]:
        render_report_tab(bundle, dataset=state["dataset"], include_joint=state["include_joint"])
    with tabs[5]:
        render_artifact_library_tab(bundle, dataset=state["dataset"])

    render_copilot_panel(bundle, state["request"])


if __name__ == "__main__":
    main()
