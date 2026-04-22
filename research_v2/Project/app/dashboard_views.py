"""Streamlit view helpers for the research workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.recommendation.engine import RecommendationEngine, RecommendationRequest

from app.dashboard_charts import (
    build_ablation_memory_chart,
    build_ablation_runtime_chart,
    build_cross_dataset_heatmap,
    build_decision_tree_chart,
    build_friedman_rank_chart,
    build_grouped_metric_bars,
    build_matrix_heatmap,
    build_rank_slope_chart,
    build_recommendation_breakdown,
    build_robustness_chart,
    build_shortlist_chart,
    build_top_cluster_chart,
    build_tradeoff_scatter,
)
from app.dashboard_data import (
    ARTIFACT_LABELS,
    DATASET_LABELS,
    DashboardBundle,
    artifact_library_entries,
    artifact_status_rows,
    available_ablation_datasets,
    build_effect_matrix,
    build_pairwise_matrix,
    build_rank_dataframe,
    build_top_cluster_membership,
    case_study_options,
    comparison_table,
    build_decision_tree_rows,
    dataset_leader_rows,
    dataset_snapshot,
    filter_profiles,
    request_bucket_state,
    sanitize_user_text,
    static_dataset_figures,
    strip_markdown_section,
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(19, 78, 74, 0.12), transparent 24%),
                radial-gradient(circle at top right, rgba(30, 64, 175, 0.14), transparent 28%),
                linear-gradient(180deg, #f4f7fb 0%, #edf2f8 44%, #e7eef6 100%);
        }
        [data-testid="stSidebar"] {
            background: rgba(245, 248, 252, 0.94);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        .block-container {
            max-width: 1460px;
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        .hero-panel {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(14, 116, 144, 0.92) 56%, rgba(37, 99, 235, 0.92));
            border-radius: 28px;
            padding: 1.8rem 1.9rem;
            color: white;
            box-shadow: 0 24px 48px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            display: inline-block;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: rgba(224, 242, 254, 0.9);
            margin-bottom: 0.55rem;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.70);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 24px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
            height: 100%;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.74);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            padding: 0.9rem 1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        }
        .metric-label {
            color: #475569;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.55rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }
        .metric-note {
            color: #475569;
            font-size: 0.88rem;
            margin-top: 0.2rem;
        }
        .section-note {
            color: #334155;
            font-size: 0.95rem;
            margin: 0.1rem 0 0.7rem 0;
        }
        .how-to-read {
            color: #475569;
            font-size: 0.87rem;
            margin-top: 0.25rem;
            padding-left: 0.2rem;
        }
        .study-chip {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            margin: 0 0.35rem 0.35rem 0;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.24);
            color: white;
            font-size: 0.78rem;
        }
        .link-card {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            margin: 0.2rem 0.35rem 0.2rem 0;
            border-radius: 999px;
            background: rgba(226, 232, 240, 0.7);
            color: #0f172a;
            text-decoration: none;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        @media (prefers-color-scheme: dark) {
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 24%),
                    radial-gradient(circle at top right, rgba(37, 99, 235, 0.2), transparent 28%),
                    linear-gradient(180deg, #07111f 0%, #0b1727 48%, #101b2f 100%);
            }
            [data-testid="stSidebar"] {
                background: rgba(12, 19, 33, 0.96);
                border-right: 1px solid rgba(71, 85, 105, 0.45);
            }
            .glass-card, .metric-card {
                background: rgba(15, 23, 42, 0.74);
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: 0 16px 34px rgba(2, 6, 23, 0.26);
            }
            .metric-label, .metric-note, .section-note, .how-to-read {
                color: #cbd5e1;
            }
            .metric-value {
                color: #f8fafc;
            }
            .link-card {
                background: rgba(30, 41, 59, 0.9);
                color: #e2e8f0;
                border: 1px solid rgba(148, 163, 184, 0.24);
            }
            .stMarkdown, .stCaption, .stText, .stSubheader, .stHeader, .stExpander, label, p, span, div {
                color: inherit;
            }
            [data-baseweb="select"] > div,
            [data-baseweb="popover"] > div,
            [data-baseweb="input"] > div,
            .stButton > button,
            .stDownloadButton > button,
            [data-testid="stBaseButton-secondary"],
            [data-testid="stBaseButton-primary"] {
                background-color: #132238 !important;
                color: #f8fafc !important;
                border-color: rgba(148, 163, 184, 0.28) !important;
            }
            [data-baseweb="tag"] {
                background-color: #1e293b !important;
                color: #e2e8f0 !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_missing_artifacts(bundle: DashboardBundle) -> None:
    if not bundle.missing_primary and not bundle.missing_secondary:
        return
    if bundle.missing_primary:
        st.warning(
            "Some primary study artifacts are missing. The dashboard will stay read-only and show partial content where possible.",
            icon="⚠️",
        )
        with st.expander("Missing primary artifacts"):
            for key, path in bundle.__class__.__annotations__.items():
                pass
            for label in ["Study Summary", "Presentation Summary", "Dataset Leaders", "Pairwise Tests", "Effect Sizes", "Cross-Dataset Ranking", "Pareto Candidates", "Analysis Report", "Recommendation Profiles", "Recommendation Cases", "Recommendation Notes"]:
                st.markdown(f"- {label}")
    if bundle.missing_secondary:
        st.info("Some ablation context artifacts are missing. Primary-study analysis remains available.")
        with st.expander("Missing ablation context artifacts"):
            for label in ["Ablation Results", "Ablation Runtime Summary", "Ablation Memory Summary", "Resume Robustness Summary", "Ablation Resource Notes"]:
                st.markdown(f"- {label}")


def render_hero(bundle: DashboardBundle) -> None:
    methods = 0 if bundle.recommendation_profiles.empty else bundle.recommendation_profiles["method"].nunique()
    datasets = 0 if bundle.recommendation_profiles.empty else bundle.recommendation_profiles["dataset"].nunique()
    primary_rows = len(bundle.summary)
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-kicker">Research Workbench</div>
            <h1 style="margin:0 0 0.45rem 0; font-size:2.35rem;">Continual Learning Dashboard</h1>
            <p style="margin:0 0 0.9rem 0; max-width:940px; color:rgba(241,245,249,0.92);">
                A modern, read-only analysis workspace built entirely on finalized study evidence.
                Use it to inspect the primary study results, compare methods, browse ablation context,
                and explore recommendations without touching raw experiment logs.
            </p>
            <div>
                <span class="study-chip">{datasets} datasets</span>
                <span class="study-chip">{methods} methods</span>
                <span class="study-chip">{primary_rows} summary rows</span>
                <span class="study-chip">Primary matrix + ablation context</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_note(text: str) -> None:
    st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)


def render_how_to_read(text: str) -> None:
    st.markdown(f'<div class="how-to-read"><strong>How to read:</strong> {text}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plot(fig: object | None, note: str, how_to_read: str | None = None, key: str | None = None) -> None:
    render_note(note)
    if fig is None:
        st.info("This chart is unavailable for the current selection.")
        return
    st.plotly_chart(fig, width="stretch", key=key)
    if how_to_read:
        render_how_to_read(how_to_read)


def render_image(path: Path, note: str, caption: str, how_to_read: str | None = None) -> None:
    render_note(note)
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
        if how_to_read:
            render_how_to_read(how_to_read)
    else:
        st.info(f"Static fallback image not found: {path.name}")


def render_recommendation_tab(bundle: DashboardBundle, request: RecommendationRequest) -> None:
    render_note("This panel turns the recommendation profiles into an explainable recommendation for the current constraints.")
    if bundle.recommendation_profiles.empty:
        st.info("Recommendation profiles are unavailable.")
        return

    engine = RecommendationEngine(bundle.recommendation_profiles)
    result = engine.recommend(request, top_k=3)
    best = result["shortlist"][0]

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size:0.82rem; letter-spacing:0.08em; text-transform:uppercase; color:#f8fafc;">Recommended method</div>
                <h2 style="margin:0.15rem 0 0.35rem 0;">{best['method']}</h2>
                <div style="color:#f8fafc; font-size:1rem;">{result['rationale']}</div>
                <div style="margin-top:0.65rem; color:#e2e8f0; font-size:0.92rem;">
                    Why this is recommended: it gives the strongest constraint-adjusted balance of empirical accuracy,
                    retention, runtime fit, and memory fit for the selected dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cards = st.columns(4)
        with cards[0]:
            metric_card("Avg Accuracy", f"{best['avg_accuracy_mean']:.2f}", "Higher is better.")
        with cards[1]:
            metric_card("Forgetting", f"{best['forgetting_mean']:.2f}", "Lower means better retention.")
        with cards[2]:
            metric_card("Runtime (h)", f"{best['runtime_hours_mean']:.2f}", "Mean runtime for this method.")
        with cards[3]:
            metric_card("Memory Proxy", f"{best['estimated_memory_mb']:.2f} MB", "Estimated, not instrumented.")

        render_plot(
            build_recommendation_breakdown(best),
            "This score breakdown shows how recommendation components push the chosen method up or down.",
            "Positive bars help the recommendation; negative bars reflect penalties from tighter runtime, memory, or forgetting constraints.",
            key="recommendation_breakdown",
        )

    with right:
        shortlist_df = pd.DataFrame(
            [
                {
                    "Rank": idx,
                    "Method": item["method"],
                    "Score": item["score"],
                    "Avg Accuracy": item["avg_accuracy_mean"],
                    "Forgetting": item["forgetting_mean"],
                    "Runtime (h)": item["runtime_hours_mean"],
                    "Memory Proxy (MB)": item["estimated_memory_mb"],
                    "Leader": item["leader_flag"],
                    "Top Cluster": item["top_cluster_flag"],
                }
                for idx, item in enumerate(result["shortlist"], start=1)
            ]
        )
        st.subheader("Top-3 Shortlist")
        st.dataframe(shortlist_df, width="stretch", hide_index=True)

        case_ids = case_study_options(bundle.recommendation_cases, request.dataset)
        if case_ids:
            selected_case = st.selectbox("Case study browser", options=case_ids, index=0, key="case_browser")
            render_plot(
                build_shortlist_chart(bundle.recommendation_cases, selected_case),
                "This case-study view shows how the saved shortlist scored for one reproducible scenario.",
                "Hover to compare score, accuracy, forgetting, runtime, and memory proxy for the fixed case-study request.",
                key="case_shortlist_chart",
            )
            case_rows = bundle.recommendation_cases[bundle.recommendation_cases["case_id"] == int(selected_case)].copy()
            if not case_rows.empty:
                st.caption(case_rows.iloc[0]["summary_rationale"])


def render_decision_tree_tab(bundle: DashboardBundle, request: RecommendationRequest) -> None:
    render_note(
        "This decision-tree view adds an interactive recommendation layer on top of the same evidence and scoring used by the recommendation panel."
    )
    if bundle.recommendation_profiles.empty:
        st.info("Recommendation profiles are unavailable.")
        return

    tree_df = build_decision_tree_rows(bundle.recommendation_profiles, request.dataset)
    if tree_df.empty:
        st.info("No decision-tree data are available for the selected dataset.")
        return

    engine = RecommendationEngine(bundle.recommendation_profiles)
    result = engine.recommend(request, top_k=3)
    active_path = request_bucket_state(request)
    active_path["recommended_method"] = result["recommended_method"]
    current_path = pd.DataFrame(
        [
            {"Stage": "Dataset", "Current choice": DATASET_LABELS.get(str(request.dataset), str(request.dataset))},
            {"Stage": "Memory", "Current choice": f"{int(request.memory_budget_mb)} MB"},
            {"Stage": "Compute", "Current choice": str(request.compute_budget).title()},
            {"Stage": "Retention", "Current choice": f"Target <= {int(request.acceptable_forgetting or 0)}"},
            {"Stage": "Similarity", "Current choice": str(request.task_similarity).title()},
            {"Stage": "Joint", "Current choice": "Allowed" if request.joint_retraining_allowed else "Not allowed"},
            {"Stage": "Outcome", "Current choice": str(result['recommended_method'])},
        ]
    )

    components.html(
        build_decision_tree_chart(
            tree_df,
            f"Decision Flow for {DATASET_LABELS.get(str(request.dataset), str(request.dataset))}",
            active_path=active_path,
        ),
        height=790,
        scrolling=False,
    )

    render_how_to_read(
        "Green flows trace the current recommendation path under the active controls. Grey flows keep the broader decision context visible so you can compare nearby alternatives."
    )

    lower = st.columns([0.95, 1.05], gap="large")
    with lower[0]:
        render_note("This table shows the exact path currently traced through the alluvial decision flow.")
        st.dataframe(current_path, width="stretch", hide_index=True)
    with lower[1]:
        best = result["shortlist"][0]
        st.subheader("Active Recommendation at This Path")
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size:0.82rem; letter-spacing:0.08em; text-transform:uppercase; color:#f8fafc;">Current leaf</div>
                <h2 style="margin:0.15rem 0 0.35rem 0;">{best['method']}</h2>
                <div style="color:#f8fafc; font-size:0.98rem;">{result['rationale']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_plot(
            build_recommendation_breakdown(best),
            "This breakdown confirms that the tree and the recommendation panel are reading from the same scoring policy.",
            "The tree is only a visual decision layer; the underlying scores are still coming from the same recommendation engine.",
            key="decision_tree_breakdown",
        )


def render_comparison_tab(
    bundle: DashboardBundle,
    dataset: str,
    families: Sequence[str],
    include_joint: bool,
    top_cluster_only: bool,
) -> None:
    render_note("Use this workspace to compare methods inside one dataset without mixing primary-study claims with ablation context.")
    profiles = filter_profiles(
        bundle.recommendation_profiles,
        dataset=dataset,
        families=families,
        include_joint=include_joint,
        top_cluster_only=top_cluster_only,
    )
    if profiles.empty:
        st.info("No method rows match the current filters.")
        return

    local_controls = st.columns([1.1, 1.0, 1.0])
    with local_controls[0]:
        sort_label = st.selectbox(
            "Sort comparison table by",
            options=["Avg Accuracy", "Forgetting", "Runtime (h)", "Memory Proxy (MB)"],
            index=0,
            key="comparison_sort",
        )
    with local_controls[1]:
        show_effect_metric = st.selectbox(
            "Effect / significance metric",
            options=["avg_accuracy", "forgetting"],
            index=0,
            key="comparison_effect_metric",
        )
    show_notes = False

    snapshot = dataset_snapshot(bundle.recommendation_profiles, bundle.leaders, dataset, include_joint=include_joint)
    cards = st.columns(4)
    with cards[0]:
        metric_card("Dataset Leader", snapshot["best_method_label"] or "N/A", "Highest mean AA in this dataset.")
    with cards[1]:
        metric_card("Top Cluster Size", str(snapshot["top_cluster_size"]), "Methods not significantly worse than the leader.")
    with cards[2]:
        metric_card("Fastest Top Cluster", snapshot["fastest_top_cluster_label"] or "N/A", "Fastest method still in the top cluster.")
    with cards[3]:
        metric_card("Lightest Top Cluster", snapshot["lowest_memory_top_cluster_label"] or "N/A", "Lowest-memory top-cluster option.")

    table = comparison_table(
        bundle.recommendation_profiles,
        dataset=dataset,
        families=families,
        include_joint=include_joint,
        top_cluster_only=top_cluster_only,
        sort_by=sort_label,
    )
    if not show_notes and "Notes" in table.columns:
        table = table.drop(columns=["Notes"])
    st.subheader("Sortable comparison table")
    st.dataframe(table, width="stretch", hide_index=True)

    render_plot(
        build_grouped_metric_bars(
            profiles,
            ["avg_accuracy_mean", "forgetting_mean", "runtime_hours_mean", "estimated_memory_mb"],
        ),
        "This grouped bar view lets you compare accuracy, forgetting, runtime, and proxy memory together.",
        "Bars are method-level summaries; hover to see exact values for the current dataset.",
        key="comparison_grouped_bars",
    )

    scatters = st.columns(3, gap="large")
    with scatters[0]:
        render_plot(
            build_tradeoff_scatter(
                profiles,
                "forgetting_mean",
                "avg_accuracy_mean",
                "Accuracy vs Forgetting",
                "Forgetting",
                "Average Accuracy",
            ),
            "This scatter shows the core retention trade-off: stronger methods sit higher and further left.",
            "Each point is one method. Leader and top-cluster status are encoded in the symbol and shown again on hover.",
            key="comparison_acc_forgetting",
        )
    with scatters[1]:
        render_plot(
            build_tradeoff_scatter(
                profiles,
                "runtime_hours_mean",
                "avg_accuracy_mean",
                "Accuracy vs Runtime",
                "Runtime (hours)",
                "Average Accuracy",
            ),
            "This scatter shows how much accuracy each method buys relative to compute time.",
            "Look for methods high on the chart but not too far right if runtime matters.",
            key="comparison_acc_runtime",
        )
    with scatters[2]:
        render_plot(
            build_tradeoff_scatter(
                profiles,
                "estimated_memory_mb",
                "avg_accuracy_mean",
                "Accuracy vs Memory Proxy",
                "Estimated Memory (MB)",
                "Average Accuracy",
            ),
            "This scatter shows the empirical trade-off between method accuracy and storage footprint.",
            "Memory here is a proxy estimate, not measured peak VRAM; hover makes that explicit.",
            key="comparison_acc_memory",
        )

    matrices = st.columns(2, gap="large")
    with matrices[0]:
        render_plot(
            build_matrix_heatmap(
                build_pairwise_matrix(bundle.pairwise, dataset, show_effect_metric, "holm_adjusted_p_value"),
                f"Holm-Adjusted Pairwise p-values ({show_effect_metric})",
                "Adj. p",
                "Blues_r",
                zmin=0.0,
                zmax=1.0,
            ),
            "This significance matrix shows which method pairs remain different after Holm correction within this dataset.",
            "Lower p-values indicate stronger evidence of a difference. Use it alongside effect size, not by itself.",
            key="comparison_significance_matrix",
        )
    with matrices[1]:
        render_plot(
            build_matrix_heatmap(
                build_effect_matrix(bundle.effect_sizes, dataset, show_effect_metric),
                f"Rank-Biserial Effect Sizes ({show_effect_metric})",
                "Effect",
                "RdBu",
                zmin=-1.0,
                zmax=1.0,
                midpoint=0.0,
            ),
            "This effect-size matrix shows the direction and magnitude of pairwise differences for the selected metric.",
            "Positive values mean the row method outperformed the column method; negative values mean the reverse.",
            key="comparison_effect_matrix",
        )

    pareto_subset = bundle.pareto[bundle.pareto["dataset"].astype(str) == str(dataset)].copy()
    pareto_subset["method_label"] = pareto_subset["method"].map(lambda item: item if pd.isna(item) else item)
    render_plot(
        build_tradeoff_scatter(
            pareto_subset.assign(method_family="pareto"),
            "runtime_hours_mean",
            "avg_accuracy_mean",
            "Non-Joint Pareto Candidates",
            "Runtime (hours)",
            "Average Accuracy",
        ),
        "This Pareto view shows only non-joint methods that are not dominated on accuracy, runtime, and proxy memory.",
        "Joint training is intentionally excluded here to keep the Pareto frontier aligned with the study policy.",
        key="comparison_pareto",
    )

    with st.expander("Ablation Context Explorer", expanded=False):
        render_note("These charts are secondary context from ablations and should not be read as primary-study claims.")
        ablation_datasets = available_ablation_datasets(bundle)
        selected_ablation_dataset = dataset if dataset in ablation_datasets else (ablation_datasets[0] if ablation_datasets else dataset)
        if ablation_datasets:
            selected_ablation_dataset = st.selectbox(
                "Context dataset",
                options=ablation_datasets,
                index=ablation_datasets.index(selected_ablation_dataset),
                format_func=lambda item: DATASET_LABELS.get(str(item), str(item)),
                key="ablation_dataset_selector",
            )
            if selected_ablation_dataset != dataset:
                st.caption(
                    f"The selected dataset does not have ablation coverage yet, so this section is showing context from {DATASET_LABELS.get(selected_ablation_dataset, selected_ablation_dataset)}."
                )
        ablation_cols = st.columns(3, gap="large")
        with ablation_cols[0]:
            render_plot(
                build_ablation_runtime_chart(bundle.ablation_runtime, selected_ablation_dataset),
                "This chart compares how ablation variants changed total runtime for the selected dataset.",
                "The x-axis is the ablation variant; hover shows seeds completed and task-time details.",
                key="ablation_runtime",
            )
        with ablation_cols[1]:
            render_plot(
                build_ablation_memory_chart(bundle.ablation_memory, selected_ablation_dataset),
                "This proxy chart highlights memory-relevant ablation settings like buffer and batch size.",
                "It is a configuration proxy view, not measured VRAM telemetry.",
                key="ablation_memory",
            )
        with ablation_cols[2]:
            render_plot(
                build_robustness_chart(bundle.ablation_robustness, selected_ablation_dataset),
                "This robustness chart shows how resumed runs shifted relative to their primary-study counterpart.",
                "Points close to zero indicate resumed runs stayed close to the original result.",
                key="ablation_robustness",
            )
        if not bundle.ablation_current.empty:
            st.subheader("Ablation result table")
            st.dataframe(
                bundle.ablation_current[bundle.ablation_current["dataset"].astype(str) == str(selected_ablation_dataset)].copy(),
                width="stretch",
                hide_index=True,
            )


def render_dataset_visuals_tab(
    bundle: DashboardBundle,
    dataset: str,
    families: Sequence[str],
    include_joint: bool,
) -> None:
    render_note("This tab provides a broader visual read of the study: cross-dataset heatmaps, ranking shifts, and dataset-specific trade-offs.")
    profiles = filter_profiles(bundle.recommendation_profiles, dataset=dataset, families=families, include_joint=include_joint)
    if profiles.empty:
        st.info("No dataset visuals are available for the current filters.")
        return

    heatmaps = st.columns(2, gap="large")
    with heatmaps[0]:
        render_plot(
            build_cross_dataset_heatmap(bundle.recommendation_profiles, "avg_accuracy_mean"),
            "This heatmap shows where average accuracy is strongest across the full dataset × method matrix.",
            "Each tile is one dataset-method summary; darker/high-value tiles indicate stronger accuracy.",
            key="visuals_heatmap_aa",
        )
    with heatmaps[1]:
        render_plot(
            build_cross_dataset_heatmap(bundle.recommendation_profiles, "forgetting_mean"),
            "This heatmap shows where forgetting remains low or high across the full study.",
            "Lower forgetting is preferable, so read this as a retention heatmap rather than a performance leaderboard.",
            key="visuals_heatmap_forgetting",
        )

    snapshot = dataset_snapshot(bundle.recommendation_profiles, bundle.leaders, dataset, include_joint=include_joint)
    cards = st.columns(4)
    with cards[0]:
        metric_card("Best Method", snapshot["best_method_label"] or "N/A", "Highest AA for this dataset.")
    with cards[1]:
        metric_card("Top Cluster", str(snapshot["top_cluster_size"]), "How many methods remain statistically competitive.")
    with cards[2]:
        metric_card("Fastest Cluster Method", snapshot["fastest_top_cluster_label"] or "N/A", "Best time among top-cluster methods.")
    with cards[3]:
        metric_card("Lightest Cluster Method", snapshot["lowest_memory_top_cluster_label"] or "N/A", "Best memory proxy among top-cluster methods.")

    tradeoffs = st.columns(3, gap="large")
    with tradeoffs[0]:
        render_plot(
            build_tradeoff_scatter(
                profiles,
                "forgetting_mean",
                "avg_accuracy_mean",
                f"{DATASET_LABELS.get(dataset, dataset)}: Accuracy vs Forgetting",
                "Forgetting",
                "Average Accuracy",
            ),
            "This view highlights the selected dataset’s retention frontier.",
            "Hover to see whether a point is a leader or part of the top non-inferior cluster.",
            key="visuals_acc_forgetting",
        )
    with tradeoffs[1]:
        render_plot(
            build_tradeoff_scatter(
                profiles,
                "runtime_hours_mean",
                "avg_accuracy_mean",
                f"{DATASET_LABELS.get(dataset, dataset)}: Accuracy vs Runtime",
                "Runtime (hours)",
                "Average Accuracy",
            ),
            "This chart shows which methods are accuracy-efficient for the selected dataset.",
            "Methods high and left are especially attractive when compute is limited.",
            key="visuals_acc_runtime",
        )
    with tradeoffs[2]:
        render_plot(
            build_tradeoff_scatter(
                profiles,
                "estimated_memory_mb",
                "avg_accuracy_mean",
                f"{DATASET_LABELS.get(dataset, dataset)}: Accuracy vs Memory Proxy",
                "Estimated Memory (MB)",
                "Average Accuracy",
            ),
            "This view compares performance against storage footprint for the selected dataset.",
            "The memory axis is proxy-estimated from configuration, not measured peak VRAM.",
            key="visuals_acc_memory",
        )

    rank_df = build_rank_dataframe(bundle.recommendation_profiles, include_joint=include_joint)
    membership_df = build_top_cluster_membership(bundle.recommendation_profiles, include_joint=include_joint)
    lower = st.columns(2, gap="large")
    with lower[0]:
        render_plot(
            build_rank_slope_chart(rank_df),
            "This slope chart shows how method ordering changes from dataset to dataset.",
            "Lower lines are better because the y-axis is rank; steep changes suggest dataset sensitivity.",
            key="visuals_rank_slope",
        )
    with lower[1]:
        render_plot(
            build_top_cluster_chart(membership_df),
            "This chart shows which methods most often stay inside the dataset-level top cluster.",
            "Bars count how many datasets kept a method in the non-inferior cluster around the leader.",
            key="visuals_top_cluster",
        )

    with st.expander("Saved Figure Gallery", expanded=False):
        render_image(
            bundle.static_figures["avg_accuracy_heatmap"],
            "This is the saved static accuracy heatmap kept for archival parity with the generated study figures.",
            "Saved static average-accuracy heatmap",
        )
        render_image(
            bundle.static_figures["forgetting_heatmap"],
            "This is the saved static forgetting heatmap kept for archival parity with the generated study figures.",
            "Saved static forgetting heatmap",
        )
        figs = static_dataset_figures(dataset)
        render_image(
            figs["accuracy_vs_forgetting"],
            "This static figure mirrors the generated dataset-level accuracy-vs-forgetting trade-off plot.",
            f"{dataset}: static accuracy vs forgetting",
        )
        render_image(
            figs["accuracy_vs_runtime"],
            "This static figure mirrors the generated dataset-level accuracy-vs-runtime trade-off plot.",
            f"{dataset}: static accuracy vs runtime",
        )
        render_image(
            figs["accuracy_vs_memory"],
            "This static figure mirrors the generated dataset-level accuracy-vs-memory trade-off plot.",
            f"{dataset}: static accuracy vs memory",
        )


def render_report_tab(bundle: DashboardBundle, dataset: str, include_joint: bool) -> None:
    render_note("This tab turns the generated artifacts into a readable study brief with provenance and caution notes.")
    dynamic_leaders = dataset_leader_rows(bundle.recommendation_profiles, include_joint=include_joint)
    if not dynamic_leaders.empty:
        st.subheader("Dataset Leaders")
        for row in dynamic_leaders.itertuples(index=False):
            st.markdown(
                (
                    f"**{row.dataset_label}**: leader `{row.best_method}` with AA {row.best_avg_accuracy_mean:.4f}, "
                    f"forgetting {row.best_forgetting_mean:.4f}, runtime {row.best_runtime_hours_mean:.4f} h, "
                    f"memory proxy {row.best_estimated_memory_mb:.2f} MB. "
                    f"Top cluster: {row.top_cluster_methods}."
                )
            )
        render_how_to_read(
            "This summary follows the current joint-training toggle. Top cluster lists the methods that remain statistically competitive after applying the same view filter."
        )

    if bundle.report_text:
        st.subheader("Analysis Report")
        st.markdown(strip_markdown_section(sanitize_user_text(bundle.report_text), "## Dataset Leaders"))
    else:
        st.info("The analysis report is unavailable.")

    top = st.columns([1.0, 1.0], gap="large")
    with top[0]:
        render_plot(
            build_friedman_rank_chart(bundle.friedman),
            "This chart summarizes the cross-dataset Friedman ranking test, which is descriptive and secondary because only four datasets were used.",
            "Lower average rank is better. Treat this as a cross-dataset pattern summary, not the only decision rule.",
            key="report_friedman",
        )
    with top[1]:
        if bundle.recommendation_notes:
            st.subheader("Recommendation Notes")
            st.markdown(sanitize_user_text(bundle.recommendation_notes))

    highlights = st.columns(3, gap="large")
    leaders = dynamic_leaders.copy()
    if not leaders.empty:
        dataset_row = leaders[leaders["dataset"].astype(str) == str(dataset)]
        if dataset_row.empty:
            dataset_row = leaders.iloc[[0]]
        row = dataset_row.iloc[0]
        with highlights[0]:
            metric_card("Selected Dataset Leader", str(row["best_method_label"]), "Highest AA in the selected dataset for the current view.")
        with highlights[1]:
            metric_card("Leader AA", f"{float(row['best_avg_accuracy_mean']):.2f}", "Mean average accuracy.")
        with highlights[2]:
            metric_card("Leader Memory Proxy", f"{float(row['best_estimated_memory_mb']):.2f} MB", "Estimated, not instrumented.")

    if not bundle.ablation_runtime.empty or not bundle.ablation_robustness.empty:
        render_note("Ablation context is shown below as supporting evidence rather than as primary-study claims.")
        ablation_datasets = available_ablation_datasets(bundle)
        selected_ablation_dataset = dataset if dataset in ablation_datasets else (ablation_datasets[0] if ablation_datasets else dataset)
        if ablation_datasets and selected_ablation_dataset != dataset:
            st.caption(
                f"The selected dataset does not have ablation coverage yet, so this section is showing context from {DATASET_LABELS.get(selected_ablation_dataset, selected_ablation_dataset)}."
            )
        cols = st.columns(2, gap="large")
        with cols[0]:
            render_plot(
                build_ablation_runtime_chart(bundle.ablation_runtime, selected_ablation_dataset),
                "This runtime chart shows how representative ablation families changed compute cost in the selected dataset.",
                "Read this as supporting engineering context rather than direct evidence about the primary method leaderboard.",
                key="report_ablation_runtime",
            )
        with cols[1]:
            render_plot(
                build_robustness_chart(bundle.ablation_robustness, selected_ablation_dataset),
                "This robustness view summarizes how restart/resume runs deviated from the primary study runs.",
                "Near-zero changes suggest the resume path preserved the main result reasonably well.",
                key="report_ablation_robustness",
            )

    st.subheader("Artifact provenance")
    status = artifact_status_rows(bundle)
    st.dataframe(status, width="stretch", hide_index=True)
    st.caption("The dashboard is read-only and consumes only generated study artifacts, never raw experiment logs.")


def _render_artifact_preview(label: str, content: object) -> None:
    anchor = label.lower().replace(" ", "-")
    st.markdown(f"<a id='{anchor}'></a>", unsafe_allow_html=True)
    st.subheader(label)
    if isinstance(content, pd.DataFrame):
        st.dataframe(content, width="stretch", hide_index=True)
    elif isinstance(content, str):
        st.markdown(sanitize_user_text(content) or "_No preview available._")
    else:
        st.info("Preview unavailable.")


def render_artifact_library_tab(bundle: DashboardBundle, dataset: str) -> None:
    render_note("This page lets you preview the saved study artifacts without exposing internal file-system paths.")
    entries = artifact_library_entries(bundle)
    toc = []
    for entry in entries:
        anchor = entry["label"].lower().replace(" ", "-")
        toc.append(f'<a class="link-card" href="#{anchor}">{entry["label"]}</a>')
    st.markdown("".join(toc), unsafe_allow_html=True)

    dataset_cases = bundle.recommendation_cases[bundle.recommendation_cases["dataset"].astype(str) == str(dataset)].copy()
    previews = {
        "Study Summary": bundle.summary_pretty if not bundle.summary_pretty.empty else bundle.summary,
        "Presentation Summary": bundle.summary_pretty,
        "Dataset Leaders": bundle.leaders,
        "Pairwise Tests": bundle.pairwise,
        "Effect Sizes": bundle.effect_sizes,
        "Cross-Dataset Ranking": bundle.friedman,
        "Pareto Candidates": bundle.pareto,
        "Analysis Report": bundle.report_text,
        "Recommendation Profiles": bundle.recommendation_profiles,
        "Recommendation Cases": dataset_cases if not dataset_cases.empty else bundle.recommendation_cases,
        "Recommendation Notes": bundle.recommendation_notes,
        "Ablation Results": bundle.ablation_current,
        "Ablation Runtime Summary": bundle.ablation_runtime,
        "Ablation Memory Summary": bundle.ablation_memory,
        "Resume Robustness Summary": bundle.ablation_robustness,
        "Ablation Resource Notes": bundle.ablation_notes,
    }
    for entry in entries:
        _render_artifact_preview(entry["label"], previews.get(entry["label"], ""))
