"""Plotly chart builders for the research workbench."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from app.dashboard_data import DATASET_LABELS, METHOD_LABELS


FAMILY_COLORS = {
    "baseline": "#2563eb",
    "hybrid": "#f97316",
    "other": "#64748b",
}

METRIC_LABELS = {
    "avg_accuracy_mean": "Average Accuracy",
    "forgetting_mean": "Forgetting",
    "runtime_hours_mean": "Runtime (hours)",
    "estimated_memory_mb": "Memory Proxy (MB)",
}


def _plotly():
    import plotly.express as px
    import plotly.graph_objects as go

    return px, go


TEXT_POSITIONS = [
    "top center",
    "bottom center",
    "middle left",
    "middle right",
    "top left",
    "top right",
    "bottom left",
    "bottom right",
]


def _assign_text_positions(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")
    rounded = df.copy()
    rounded["_x_bucket"] = pd.to_numeric(rounded[x_col], errors="coerce").round(3)
    rounded["_y_bucket"] = pd.to_numeric(rounded[y_col], errors="coerce").round(3)
    positions = []
    seen: dict[tuple[float, float], int] = {}
    for _, row in rounded.iterrows():
        key = (float(row["_x_bucket"]) if pd.notna(row["_x_bucket"]) else float("nan"), float(row["_y_bucket"]) if pd.notna(row["_y_bucket"]) else float("nan"))
        idx = seen.get(key, 0)
        positions.append(TEXT_POSITIONS[idx % len(TEXT_POSITIONS)])
        seen[key] = idx + 1
    return pd.Series(positions, index=df.index)


def build_recommendation_breakdown(candidate: dict) -> object:
    px, _ = _plotly()
    components = candidate.get("score_components", {})
    df = pd.DataFrame(
        [
            {"component": name.replace("_", " ").title(), "value": value}
            for name, value in components.items()
        ]
    )
    fig = px.bar(
        df,
        x="value",
        y="component",
        orientation="h",
        color="value",
        color_continuous_scale="Tealrose",
        text="value",
        hover_data={"value": ":.2f"},
    )
    fig.update_layout(
        title="Constraint-Adjusted Recommendation Score Breakdown",
        height=420,
        margin=dict(l=10, r=10, t=60, b=20),
        coloraxis_showscale=False,
        yaxis_title="",
        xaxis_title="Score Contribution",
    )
    return fig


def build_shortlist_chart(cases_df: pd.DataFrame, case_id: int) -> object | None:
    if cases_df.empty:
        return None
    px, _ = _plotly()
    subset = cases_df[cases_df["case_id"] == int(case_id)].copy()
    if subset.empty:
        return None
    subset["method_label"] = subset["method"].map(lambda item: METHOD_LABELS.get(str(item), str(item)))
    fig = px.bar(
        subset.sort_values("score", ascending=True),
        x="score",
        y="method_label",
        orientation="h",
        color="rank",
        text="score",
        hover_data={
            "avg_accuracy_mean": ":.2f",
            "forgetting_mean": ":.2f",
            "runtime_hours_mean": ":.2f",
            "estimated_memory_mb": ":.2f",
            "rank": True,
        },
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        title=f"Case Study {case_id}: Top Recommendation Shortlist",
        height=360,
        margin=dict(l=10, r=10, t=60, b=20),
        coloraxis_showscale=False,
        yaxis_title="",
        xaxis_title="Score",
    )
    return fig


def build_grouped_metric_bars(df: pd.DataFrame, metrics: Sequence[str]) -> object | None:
    if df.empty:
        return None
    px, _ = _plotly()
    melt = df[["method_label", "method_family", *metrics]].melt(
        id_vars=["method_label", "method_family"],
        var_name="metric",
        value_name="value",
    )
    melt["metric_label"] = melt["metric"].map(lambda item: METRIC_LABELS.get(str(item), str(item)))
    fig = px.bar(
        melt,
        x="method_label",
        y="value",
        color="metric_label",
        barmode="group",
        hover_data={"method_family": True, "value": ":.3f"},
    )
    fig.update_layout(
        title="Method Comparison Across Selected Metrics",
        height=460,
        margin=dict(l=10, r=10, t=60, b=70),
        xaxis_title="Method",
        yaxis_title="Value",
        legend_title="Metric",
    )
    return fig


def build_tradeoff_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_title: str,
    y_title: str,
) -> object | None:
    if df.empty:
        return None
    _, go = _plotly()
    source = df.copy()
    source["status"] = source.apply(
        lambda row: "Leader" if bool(row.get("leader_flag", False))
        else "Top Cluster" if bool(row.get("top_cluster_flag", False))
        else "Other",
        axis=1,
    )
    source["text_position"] = _assign_text_positions(source, x_col, y_col)
    symbol_map = {"Leader": "diamond", "Top Cluster": "circle", "Other": "x"}
    fig = go.Figure()
    for row in source.itertuples(index=False):
        fig.add_trace(
            go.Scatter(
                x=[getattr(row, x_col)],
                y=[getattr(row, y_col)],
                mode="markers+text",
                text=[row.method_label],
                textposition=getattr(row, "text_position"),
                marker=dict(
                    size=11,
                    color=FAMILY_COLORS.get(str(row.method_family), FAMILY_COLORS["other"]),
                    symbol=symbol_map.get(str(row.status), "circle"),
                    line=dict(width=1, color="#ffffff"),
                ),
                name=str(row.method_label),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{x_title}: %{{x:.3f}}<br>"
                    f"{y_title}: %{{y:.3f}}<br>"
                    f"Leader: {bool(getattr(row, 'leader_flag', False))}<br>"
                    f"Top Cluster: {bool(getattr(row, 'top_cluster_flag', False))}<br>"
                    f"Runtime (h): {float(getattr(row, 'runtime_hours_mean', float('nan'))):.3f}<br>"
                    f"Memory Proxy (MB): {float(getattr(row, 'estimated_memory_mb', float('nan'))):.3f}<extra>{row.method_family}</extra>"
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        title=title,
        height=430,
        margin=dict(l=10, r=10, t=60, b=30),
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title="Method Family",
    )
    return fig


def build_matrix_heatmap(
    matrix: pd.DataFrame,
    title: str,
    colorbar_title: str,
    color_scale: str,
    zmin: float | None = None,
    zmax: float | None = None,
    midpoint: float | None = None,
) -> object | None:
    if matrix.empty:
        return None
    px, _ = _plotly()
    fig = px.imshow(
        matrix,
        text_auto=".2f",
        color_continuous_scale=color_scale,
        zmin=zmin,
        zmax=zmax,
        aspect="auto",
    )
    if midpoint is not None:
        fig.update_coloraxes(cmid=midpoint)
    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        coloraxis_colorbar_title=colorbar_title,
    )
    return fig


def build_cross_dataset_heatmap(profiles_df: pd.DataFrame, metric: str) -> object | None:
    if profiles_df.empty:
        return None
    px, _ = _plotly()
    source = profiles_df.copy()
    if "method_label" not in source.columns:
        source["method_label"] = source["method"].map(lambda item: METHOD_LABELS.get(str(item), str(item)))
    pivot = source.pivot(index="dataset", columns="method_label", values=metric)
    pivot = pivot.rename(index=lambda item: DATASET_LABELS.get(str(item), str(item)))
    pivot = pivot.sort_index()
    fig = px.imshow(
        pivot,
        text_auto=".2f",
        color_continuous_scale="RdYlGn_r" if metric == "forgetting_mean" else "Viridis",
        aspect="auto",
    )
    fig.update_layout(
        title=f"Cross-Dataset {METRIC_LABELS.get(metric, metric)} Heatmap",
        height=460,
        margin=dict(l=10, r=10, t=60, b=10),
        coloraxis_colorbar_title=METRIC_LABELS.get(metric, metric),
    )
    return fig


def build_rank_slope_chart(rank_df: pd.DataFrame) -> object | None:
    if rank_df.empty:
        return None
    px, _ = _plotly()
    fig = px.line(
        rank_df,
        x="dataset_label",
        y="rank",
        color="method_label",
        markers=True,
        hover_data={"avg_accuracy_mean": ":.3f", "top_cluster_flag": True},
    )
    fig.update_layout(
        title="Method Rank Changes Across Datasets",
        height=500,
        margin=dict(l=10, r=10, t=60, b=20),
        xaxis_title="Dataset",
        yaxis_title="Average-Accuracy Rank (Lower is Better)",
        legend_title="Method",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def build_top_cluster_chart(membership_df: pd.DataFrame) -> object | None:
    if membership_df.empty:
        return None
    px, _ = _plotly()
    fig = px.bar(
        membership_df,
        x="method_label",
        y="datasets_in_top_cluster",
        color="mean_rank",
        text="datasets_in_top_cluster",
        color_continuous_scale="Blues_r",
        hover_data={"mean_rank": ":.2f", "mean_avg_accuracy": ":.2f"},
    )
    fig.update_layout(
        title="Top-Cluster Membership Across Datasets",
        height=430,
        margin=dict(l=10, r=10, t=60, b=70),
        xaxis_title="Method",
        yaxis_title="Datasets in Top Cluster",
        coloraxis_colorbar_title="Mean Rank",
    )
    return fig


def build_friedman_rank_chart(friedman_df: pd.DataFrame) -> object | None:
    if friedman_df.empty:
        return None
    px, _ = _plotly()
    fig = px.bar(
        friedman_df.sort_values("average_rank"),
        x="method",
        y="average_rank",
        color="average_rank",
        text="average_rank",
        color_continuous_scale="Cividis_r",
        hover_data={"friedman_p_value": ":.6f", "dataset_count": True},
    )
    fig.update_layout(
        title="Cross-Dataset Friedman Average Ranks",
        height=400,
        margin=dict(l=10, r=10, t=60, b=60),
        xaxis_title="Method",
        yaxis_title="Average Rank (Lower is Better)",
        coloraxis_colorbar_title="Avg Rank",
    )
    return fig


def build_ablation_runtime_chart(runtime_df: pd.DataFrame, dataset: str) -> object | None:
    if runtime_df.empty:
        return None
    px, _ = _plotly()
    subset = runtime_df[runtime_df["dataset"].astype(str) == str(dataset)].copy()
    if subset.empty:
        return None
    subset["label"] = subset["method"].map(lambda item: METHOD_LABELS.get(str(item), str(item))) + " | " + subset["run_tag"].astype(str)
    fig = px.bar(
        subset.sort_values("mean_total_time_sec", ascending=False),
        x="label",
        y="mean_total_time_sec",
        color="ablation_family",
        hover_data={"seeds_completed": True, "mean_task_time_sec": ":.2f", "max_task_time_sec": ":.2f"},
    )
    fig.update_layout(
        title="Ablation Context: Runtime Sensitivity",
        height=420,
        margin=dict(l=10, r=10, t=60, b=80),
        xaxis_title="Ablation Variant",
        yaxis_title="Mean Total Time (s)",
        legend_title="Ablation Family",
    )
    return fig


def build_ablation_memory_chart(memory_df: pd.DataFrame, dataset: str) -> object | None:
    if memory_df.empty:
        return None
    px, _ = _plotly()
    subset = memory_df[memory_df["dataset"].astype(str) == str(dataset)].copy()
    if subset.empty:
        return None
    subset["label"] = subset["method"].map(lambda item: METHOD_LABELS.get(str(item), str(item))) + " | " + subset["run_tag"].astype(str)
    fig = px.scatter(
        subset,
        x="buffer_size",
        y="batch_size",
        color="ablation_family",
        size="seeds_present",
        hover_data={
            "label": True,
            "agem_mem_batch": True,
            "fisher_samples": True,
            "joint_replay_epochs": True,
        },
    )
    fig.update_layout(
        title="Ablation Context: Memory-Relevant Settings",
        height=420,
        margin=dict(l=10, r=10, t=60, b=20),
        xaxis_title="Replay Buffer Size (Proxy Driver)",
        yaxis_title="Batch Size",
        legend_title="Ablation Family",
    )
    return fig


def build_robustness_chart(robust_df: pd.DataFrame, dataset: str) -> object | None:
    if robust_df.empty:
        return None
    px, _ = _plotly()
    subset = robust_df[robust_df["dataset"].astype(str) == str(dataset)].copy()
    if subset.empty:
        return None
    subset["method_label"] = subset["method"].map(lambda item: METHOD_LABELS.get(str(item), str(item)))
    fig = px.scatter(
        subset,
        x="delta_forgetting",
        y="delta_avg_accuracy",
        color="method_label",
        symbol="restart_verified",
        hover_data={
            "seed": True,
            "stop_after_task": True,
            "log_starts": True,
            "delta_backward_transfer": ":.3f",
            "delta_forward_transfer": ":.3f",
        },
    )
    fig.update_layout(
        title="Ablation Context: Resume / Restart Robustness",
        height=420,
        margin=dict(l=10, r=10, t=60, b=20),
        xaxis_title="Change in Forgetting vs Primary Run",
        yaxis_title="Change in Average Accuracy vs Primary Run",
        legend_title="Method",
    )
    return fig
