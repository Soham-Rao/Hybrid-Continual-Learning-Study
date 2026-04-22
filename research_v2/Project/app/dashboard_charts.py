"""Plotly chart builders for the research workbench."""

from __future__ import annotations

import json
import math
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


ANNOTATION_OFFSETS = [
    (0, -26),
    (26, 0),
    (-26, 0),
    (0, 26),
    (24, -24),
    (-24, -24),
    (24, 24),
    (-24, 24),
    (40, -8),
    (-40, -8),
    (40, 18),
    (-40, 18),
]


def _status_symbol(status: str) -> str:
    symbol_map = {"Leader": "diamond", "Top Cluster": "circle", "Other": "x"}
    return symbol_map.get(str(status), "circle")


def _cluster_offsets(df: pd.DataFrame, x_col: str, y_col: str, *, log_x: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ax", "ay"])
    source = df.copy()
    x_values = pd.to_numeric(source[x_col], errors="coerce").fillna(0.0)
    y_values = pd.to_numeric(source[y_col], errors="coerce").fillna(0.0)
    if log_x:
        x_values = x_values.clip(lower=0.01).map(lambda item: math.log10(float(item)))

    x_span = max(float(x_values.max() - x_values.min()), 1e-6)
    y_span = max(float(y_values.max() - y_values.min()), 1e-6)
    x_norm = (x_values - float(x_values.min())) / x_span
    y_norm = (y_values - float(y_values.min())) / y_span
    x_bucket = (x_norm / 0.08).round().astype(int)
    y_bucket = (y_norm / 0.08).round().astype(int)

    rows = []
    seen: dict[tuple[int, int], int] = {}
    for idx, key in zip(source.index, zip(x_bucket, y_bucket)):
        offset_idx = seen.get(key, 0)
        ring = offset_idx // len(ANNOTATION_OFFSETS)
        dx, dy = ANNOTATION_OFFSETS[offset_idx % len(ANNOTATION_OFFSETS)]
        scale = 1 + (0.45 * ring)
        rows.append({"index": idx, "ax": int(dx * scale), "ay": int(dy * scale)})
        seen[key] = offset_idx + 1
    return pd.DataFrame(rows).set_index("index")


def _build_scatter_annotations(df: pd.DataFrame, x_col: str, y_col: str, *, log_x: bool = False) -> list[dict[str, object]]:
    if df.empty:
        return []
    offsets = _cluster_offsets(df, x_col, y_col, log_x=log_x)
    annotations: list[dict[str, object]] = []
    for row in df.itertuples():
        offset = offsets.loc[row.Index] if row.Index in offsets.index else pd.Series({"ax": 20, "ay": -20})
        annotations.append(
            {
                "x": getattr(row, x_col),
                "y": getattr(row, y_col),
                "text": str(row.method_label),
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 0.8,
                "arrowcolor": "rgba(100, 116, 139, 0.65)",
                "ax": int(offset["ax"]),
                "ay": int(offset["ay"]),
                "bgcolor": "rgba(248, 250, 252, 0.86)",
                "bordercolor": "rgba(148, 163, 184, 0.45)",
                "borderpad": 2,
                "font": {"size": 10, "color": "#0f172a"},
                "opacity": 0.94,
            }
        )
    return annotations


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
    log_x = x_col == "estimated_memory_mb"
    source["marker_color"] = source["method_family"].map(lambda item: FAMILY_COLORS.get(str(item), FAMILY_COLORS["other"]))
    source["marker_symbol"] = source["status"].map(_status_symbol)
    customdata = source[
        [
            "method_label",
            "status",
            "method_family",
            "runtime_hours_mean",
            "estimated_memory_mb",
            "leader_flag",
            "top_cluster_flag",
        ]
    ].to_numpy()
    fig = go.Figure(
        data=[
            go.Scatter(
                x=source[x_col],
                y=source[y_col],
                mode="markers",
                marker=dict(
                    size=12,
                    color=source["marker_color"],
                    symbol=source["marker_symbol"],
                    line=dict(width=1, color="#ffffff"),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{x_title}: %{{x:.3f}}<br>"
                    f"{y_title}: %{{y:.3f}}<br>"
                    "Status: %{customdata[1]}<br>"
                    "Family: %{customdata[2]}<br>"
                    "Runtime (h): %{customdata[3]:.3f}<br>"
                    "Memory Proxy (MB): %{customdata[4]:.3f}<br>"
                    "Leader: %{customdata[5]}<br>"
                    "Top Cluster: %{customdata[6]}<extra></extra>"
                ),
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=430,
        margin=dict(l=10, r=10, t=60, b=30),
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title="Method Family",
        annotations=_build_scatter_annotations(source.reset_index(), x_col, y_col, log_x=log_x),
    )
    if log_x:
        fig.update_xaxes(type="log")
    return fig


def build_decision_tree_chart(tree_df: pd.DataFrame, title: str, active_path: dict[str, object] | None = None) -> str:
    if tree_df.empty:
        return "<div>No decision-tree data are available.</div>"
    nodes: list[dict[str, object]] = []
    links: list[dict[str, object]] = []

    dataset_key = str(tree_df.iloc[0]["dataset"])
    dataset_label = str(tree_df.iloc[0]["dataset_label"])
    active_path = active_path or {}
    active_keys = {
        "memory": str(active_path.get("memory_bucket_key", "")),
        "compute": str(active_path.get("compute_bucket_key", "")),
        "forgetting": str(active_path.get("forgetting_bucket_key", "")),
        "similarity": str(active_path.get("similarity_bucket_key", "")),
        "joint": str(active_path.get("joint_bucket_key", "")),
        "method": str(active_path.get("recommended_method", "")),
    }

    def add_node(node_id: str, label: str, level_name: str, score: float, count: int, meta: dict[str, object], *, active: bool) -> None:
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "level": level_name,
                "score": float(score),
                "count": max(int(count), 1),
                "color": "#8fdcc9" if active else "#cbd5e1",
                "meta": meta,
            }
        )

    def add_link(source_id: str, target_id: str, value: int, score: float, *, active: bool) -> None:
        links.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "value": max(int(value), 1),
                "score": float(score),
                "color": "rgba(143, 220, 201, 0.88)" if active else "rgba(148, 163, 184, 0.46)",
            }
        )

    root_id = f"dataset::{dataset_key}"
    add_node(
        root_id,
        dataset_label,
        "dataset",
        float(tree_df["score"].mean()),
        int(len(tree_df)),
        {"dataset": dataset_key},
        active=True,
    )

    level_keys = [
        ("memory_bucket_label", "memory_bucket_key", "memory"),
        ("compute_bucket_label", "compute_bucket_key", "compute"),
        ("forgetting_bucket_label", "forgetting_bucket_key", "forgetting"),
        ("similarity_bucket_label", "similarity_bucket_key", "similarity"),
        ("joint_bucket_label", "joint_bucket_key", "joint"),
        ("method_label", "recommended_method", "method"),
    ]

    current_level = [(root_id, tree_df.copy(), {})]
    for label_col, key_col, level_name in level_keys:
        next_level: list[tuple[str, pd.DataFrame, dict[str, object]]] = []
        for parent_id, subset, path_meta in current_level:
            grouped = (
                subset.groupby([label_col, key_col], dropna=False)
                .agg(
                    score=("score", "mean"),
                    count=("case_id", "count"),
                    avg_accuracy=("avg_accuracy_mean", "mean"),
                    forgetting=("forgetting_mean", "mean"),
                    runtime=("runtime_hours_mean", "mean"),
                    memory=("estimated_memory_mb", "mean"),
                )
                .reset_index()
                .sort_values(["score", "avg_accuracy"], ascending=[False, False])
            )
            for row in grouped.itertuples(index=False):
                key = str(getattr(row, key_col))
                label = str(getattr(row, label_col))
                node_id = f"{parent_id}|{level_name}::{key}"
                node_meta = {
                    **path_meta,
                    "dataset": dataset_key,
                    f"{level_name}_key": key,
                    f"{level_name}_label": label,
                    "avg_accuracy_mean": float(row.avg_accuracy),
                    "forgetting_mean": float(row.forgetting),
                    "runtime_hours_mean": float(row.runtime),
                    "estimated_memory_mb": float(row.memory),
                }
                is_active = not active_keys.get(level_name) or active_keys[level_name] == key
                add_node(
                    node_id,
                    label,
                    level_name,
                    float(row.score),
                    int(row.count),
                    node_meta,
                    active=is_active,
                )
                add_link(parent_id, node_id, int(row.count), float(row.score), active=is_active)
                next_subset = subset[subset[key_col].astype(str) == key].copy()
                next_level.append((node_id, next_subset, {**path_meta, f"{level_name}_key": key, f"{level_name}_label": label}))
        current_level = next_level

    node_df = pd.DataFrame(nodes).drop_duplicates(subset=["id"]).reset_index(drop=True)
    meta_cols = sorted({key for meta in node_df["meta"] for key in meta.keys()})
    for column in meta_cols:
        node_df[column] = node_df["meta"].map(lambda item: item.get(column))
    node_index = {node_id: idx for idx, node_id in enumerate(node_df["id"].tolist())}
    link_df = pd.DataFrame(links)
    if link_df.empty:
        return None
    link_df["source"] = link_df["source_id"].map(node_index)
    link_df["target"] = link_df["target_id"].map(node_index)

    for column in [
        "dataset",
        "memory_key",
        "compute_key",
        "forgetting_key",
        "similarity_key",
        "joint_key",
        "method_key",
        "avg_accuracy_mean",
        "forgetting_mean",
        "runtime_hours_mean",
        "estimated_memory_mb",
    ]:
        if column not in node_df.columns:
            node_df[column] = ""

    node_customdata = node_df[
        [
            "level",
            "memory_key",
            "compute_key",
            "forgetting_key",
            "similarity_key",
            "joint_key",
            "method_key",
            "avg_accuracy_mean",
            "forgetting_mean",
            "runtime_hours_mean",
            "estimated_memory_mb",
        ]
    ].fillna("").to_numpy()

    graph_nodes = []
    stage_order = ["dataset", "memory", "compute", "forgetting", "similarity", "joint", "method"]
    stage_index = {name: idx for idx, name in enumerate(stage_order)}
    stage_width = 170
    node_width = 36
    y_padding = 8
    base_height = 760
    total_cases = max(float(tree_df["case_id"].nunique() if "case_id" in tree_df.columns else len(tree_df)), 1.0)
    max_stage_nodes = max(int((node_df["level"].astype(str) == level_name).sum()) for level_name in stage_order)
    global_scale = max((base_height - ((max_stage_nodes + 1) * y_padding)) / total_cases, 2.2)

    for level_name in stage_order:
        stage_subset = node_df[node_df["level"].astype(str) == level_name].copy()
        stage_subset = stage_subset.sort_values(["score", "label"], ascending=[False, True]).reset_index(drop=True)
        stage_total_height = float(stage_subset["count"].sum()) * global_scale + max(len(stage_subset) - 1, 0) * y_padding
        cursor_y = 30.0 + max((base_height - stage_total_height) / 2.0, 0.0)
        for _, row in stage_subset.iterrows():
            height = max(float(row["count"]) * global_scale, 10.0)
            graph_nodes.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "level": level_name,
                    "x": 30 + (stage_index[level_name] * stage_width),
                    "y": cursor_y,
                    "width": node_width,
                    "height": height,
                    "score": float(row["score"]),
                    "count": int(row["count"]),
                    "color": row["color"],
                    "meta": {
                        "avg_accuracy_mean": row.get("avg_accuracy_mean", ""),
                        "forgetting_mean": row.get("forgetting_mean", ""),
                        "runtime_hours_mean": row.get("runtime_hours_mean", ""),
                        "estimated_memory_mb": row.get("estimated_memory_mb", ""),
                    },
                }
            )
            cursor_y += height + y_padding

    node_layout = {item["id"]: item for item in graph_nodes}
    svg_width = ((len(stage_order) - 1) * stage_width) + 300
    svg_height = max(
        base_height + 90,
        int(max((node["y"] + node["height"]) for node in graph_nodes) + 80) if graph_nodes else base_height + 90,
    )

    paths = []
    link_cursor = {node_id: 0.0 for node_id in node_layout}
    for row in link_df.itertuples(index=False):
        source = node_layout.get(str(row.source_id))
        target = node_layout.get(str(row.target_id))
        if source is None or target is None:
            continue
        thickness = max(float(row.value) * global_scale, 4.0)
        sy = source["y"] + min(link_cursor[source["id"]] + (thickness / 2.0), max(source["height"] - 4.0, thickness / 2.0))
        ty = target["y"] + min(link_cursor[target["id"]] + (thickness / 2.0), max(target["height"] - 4.0, thickness / 2.0))
        link_cursor[source["id"]] += thickness
        link_cursor[target["id"]] += thickness
        x1 = source["x"] + source["width"]
        x2 = target["x"]
        cx1 = x1 + ((x2 - x1) * 0.42)
        cx2 = x1 + ((x2 - x1) * 0.58)
        path_d = f"M{x1:.2f},{sy:.2f} C{cx1:.2f},{sy:.2f} {cx2:.2f},{ty:.2f} {x2:.2f},{ty:.2f}"
        paths.append(
            {
                "id": f"link::{row.source_id}=>{row.target_id}",
                "source_id": str(row.source_id),
                "target_id": str(row.target_id),
                "d": path_d,
                "stroke": str(row.color),
                "stroke_width": thickness,
                "value": int(row.value),
                "score": float(row.score),
                "active": "143, 220, 201" in str(row.color),
            }
        )

    stage_titles = [
        {"label": "Dataset", "x": 30},
        {"label": "Memory", "x": 30 + stage_width},
        {"label": "Compute", "x": 30 + (stage_width * 2)},
        {"label": "Retention", "x": 30 + (stage_width * 3)},
        {"label": "Similarity", "x": 30 + (stage_width * 4)},
        {"label": "Joint", "x": 30 + (stage_width * 5)},
        {"label": "Method", "x": 30 + (stage_width * 6)},
    ]

    payload = {
        "title": title,
        "width": svg_width,
        "height": svg_height,
        "nodes": graph_nodes,
        "paths": paths,
        "stages": stage_titles,
    }

    return f"""
    <style>
      .alluvial-shell {{
        background: rgba(255,255,255,0.68);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 24px;
        padding: 14px 14px 10px 14px;
        box-shadow: 0 18px 40px rgba(15,23,42,0.08);
      }}
      .alluvial-title {{
        font-size: 1.02rem;
        font-weight: 700;
        color: #0f172a;
      }}
      .alluvial-subtitle, .alluvial-legend {{
        font-size: 0.82rem;
        color: #475569;
      }}
      .alluvial-chip {{
        width: 12px;
        height: 12px;
        border-radius: 999px;
        display: inline-block;
      }}
      .alluvial-frame {{
        width: 100%;
        height: 720px;
        overflow: hidden;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(248,250,252,0.96), rgba(241,245,249,0.96));
        cursor: grab;
      }}
      .alluvial-reset {{
        border: none;
        border-radius: 999px;
        background: #dbeafe;
        color: #1d4ed8;
        padding: 0.35rem 0.8rem;
        cursor: pointer;
        font-size: 0.82rem;
      }}
      @media (prefers-color-scheme: dark) {{
        .alluvial-shell {{
          background: rgba(15,23,42,0.82);
          border-color: rgba(148,163,184,0.18);
          box-shadow: 0 18px 40px rgba(2,6,23,0.34);
        }}
        .alluvial-title {{
          color: #f8fafc;
        }}
        .alluvial-subtitle, .alluvial-legend {{
          color: #cbd5e1;
        }}
        .alluvial-frame {{
          background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(22,33,54,0.98));
        }}
        .alluvial-reset {{
          background: #1e3a8a;
          color: #dbeafe;
        }}
      }}
    </style>
    <div class="alluvial-shell">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:8px;">
        <div>
          <div class="alluvial-title">{title}</div>
          <div class="alluvial-subtitle">Drag to pan. Use the mouse wheel or trackpad to zoom. Double-click reset if needed.</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <span class="alluvial-legend" style="display:inline-flex; align-items:center; gap:6px;"><span class="alluvial-chip" style="background:rgba(110,231,183,0.88);"></span>active path</span>
          <span class="alluvial-legend" style="display:inline-flex; align-items:center; gap:6px;"><span class="alluvial-chip" style="background:rgba(203,213,225,0.92);"></span>context flows</span>
          <span class="alluvial-legend" style="display:inline-flex; align-items:center; gap:6px;"><span class="alluvial-chip" style="background:rgba(251,191,36,0.95);"></span>clicked lineage</span>
          <button id="alluvial-reset" class="alluvial-reset">Reset view</button>
        </div>
      </div>
      <div id="alluvial-frame" class="alluvial-frame">
        <svg id="alluvial-svg" viewBox="0 0 {svg_width} {svg_height}" style="width:100%; height:100%; user-select:none;">
          <g id="alluvial-viewport"></g>
        </svg>
      </div>
    </div>
    <script>
    (function() {{
      const payload = {json.dumps(payload)};
      const svg = document.getElementById("alluvial-svg");
      const viewport = document.getElementById("alluvial-viewport");
      const frame = document.getElementById("alluvial-frame");
      const reset = document.getElementById("alluvial-reset");
      if (!svg || !viewport || !frame) return;

      const darkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      let scale = 1.0;
      let tx = 20;
      let ty = 24;
      let dragging = false;
      let lastX = 0;
      let lastY = 0;
      let velocityX = 0;
      let velocityY = 0;
      let momentumFrame = null;
      let lastMoveAt = 0;
      let selectedFlowId = null;
      const activeStroke = "rgba(110,231,183,0.88)";
      const contextStroke = "rgba(148,163,184,0.46)";
      const selectedStroke = "rgba(251,191,36,0.96)";
      const activeNodeFill = "#8fdcc9";
      const contextNodeFill = "#cbd5e1";
      const selectedNodeFill = "#fcd34d";

      function setTransform() {{
        viewport.style.transformOrigin = "0 0";
        viewport.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`;
      }}

      function resetView() {{
        scale = 1.0;
        tx = 20;
        ty = 24;
        velocityX = 0;
        velocityY = 0;
        selectedFlowId = null;
        if (momentumFrame) cancelAnimationFrame(momentumFrame);
        applySelection(new Set());
        setTransform();
      }}

      function stopMomentum() {{
        if (momentumFrame) {{
          cancelAnimationFrame(momentumFrame);
          momentumFrame = null;
        }}
      }}

      function startMomentum() {{
        stopMomentum();
        const friction = 0.92;
        const minVelocity = 0.12;
        function tick() {{
          tx += velocityX;
          ty += velocityY;
          velocityX *= friction;
          velocityY *= friction;
          setTransform();
          if (Math.abs(velocityX) > minVelocity || Math.abs(velocityY) > minVelocity) {{
            momentumFrame = requestAnimationFrame(tick);
          }} else {{
            momentumFrame = null;
          }}
        }}
        momentumFrame = requestAnimationFrame(tick);
      }}

      function titleElement(text) {{
        const t = document.createElementNS("http://www.w3.org/2000/svg", "title");
        t.textContent = text;
        return t;
      }}

      const incomingByTarget = new Map();
      const pathElements = new Map();
      const nodeElements = new Map();

      function registerIncoming(flow) {{
        if (!incomingByTarget.has(flow.target_id)) {{
          incomingByTarget.set(flow.target_id, []);
        }}
        incomingByTarget.get(flow.target_id).push(flow);
      }}

      function collectAncestorLinks(targetNodeId, selected) {{
        const incoming = incomingByTarget.get(targetNodeId) || [];
        incoming.forEach((flow) => {{
          if (!selected.has(flow.id)) {{
            selected.add(flow.id);
            collectAncestorLinks(flow.source_id, selected);
          }}
        }});
      }}

      function selectedNodeIds(linkIds) {{
        const ids = new Set();
        linkIds.forEach((linkId) => {{
          const flow = pathElements.get(linkId)?.__data;
          if (!flow) return;
          ids.add(flow.source_id);
          ids.add(flow.target_id);
        }});
        return ids;
      }}

      function applySelection(linkIds) {{
        const highlightedNodes = selectedNodeIds(linkIds);
        pathElements.forEach((pathEl, linkId) => {{
          const flow = pathEl.__data;
          const stroke = linkIds.size
            ? (linkIds.has(linkId) ? selectedStroke : contextStroke)
            : (flow.active ? activeStroke : contextStroke);
          const opacity = linkIds.size
            ? (linkIds.has(linkId) ? "0.98" : "0.18")
            : (flow.active ? "0.92" : "0.62");
          pathEl.setAttribute("stroke", stroke);
          pathEl.setAttribute("opacity", opacity);
        }});

        nodeElements.forEach((groupEl, nodeId) => {{
          const rect = groupEl.querySelector("rect");
          const label = groupEl.querySelector("text");
          const node = groupEl.__data;
          if (!rect || !label) return;
          const fill = linkIds.size
            ? (highlightedNodes.has(nodeId) ? selectedNodeFill : contextNodeFill)
            : (node.active ? activeNodeFill : contextNodeFill);
          rect.setAttribute("fill", fill);
          label.setAttribute("font-weight", highlightedNodes.has(nodeId) || node.active ? "700" : "500");
        }});
      }}

      payload.stages.forEach((stage) => {{
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", stage.x);
        text.setAttribute("y", 16);
        text.setAttribute("fill", darkMode ? "#e2e8f0" : "#334155");
        text.setAttribute("font-size", "13");
        text.setAttribute("font-weight", "700");
        text.textContent = stage.label;
        viewport.appendChild(text);
      }});

      payload.paths.forEach((flow) => {{
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.__data = flow;
        path.setAttribute("d", flow.d);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", flow.stroke);
        path.setAttribute("stroke-width", flow.stroke_width);
        path.setAttribute("stroke-linecap", "round");
        path.setAttribute("opacity", flow.active ? "0.92" : "0.62");
        path.appendChild(titleElement(`Flow count: ${{flow.value}} | Mean score: ${{flow.score.toFixed(2)}}`));
        path.style.cursor = "pointer";
        path.addEventListener("click", (event) => {{
          event.stopPropagation();
          if (selectedFlowId === flow.id) {{
            selectedFlowId = null;
            applySelection(new Set());
            return;
          }}
          const selected = new Set([flow.id]);
          collectAncestorLinks(flow.source_id, selected);
          selectedFlowId = flow.id;
          applySelection(selected);
        }});
        registerIncoming(flow);
        pathElements.set(flow.id, path);
        viewport.appendChild(path);
      }});

      payload.nodes.forEach((node) => {{
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.__data = {{
          id: node.id,
          active: node.color === activeNodeFill,
        }};
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", node.x);
        rect.setAttribute("y", node.y);
        rect.setAttribute("rx", "8");
        rect.setAttribute("ry", "8");
        rect.setAttribute("width", node.width);
        rect.setAttribute("height", node.height);
        rect.setAttribute("fill", node.color);
        rect.setAttribute("stroke", darkMode ? "rgba(226,232,240,0.16)" : "rgba(100,116,139,0.22)");
        rect.setAttribute("stroke-width", "1");

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", node.x + node.width + 8);
        label.setAttribute("y", node.y + Math.min(Math.max(node.height / 2, 14), node.height - 4));
        label.setAttribute("dominant-baseline", "middle");
        label.setAttribute("fill", darkMode ? "#f8fafc" : "#0f172a");
        label.setAttribute("font-size", "12");
        label.setAttribute("font-weight", node.color === "#0f766e" ? "700" : "500");
        label.textContent = node.label;

        group.appendChild(rect);
        group.appendChild(label);
        group.appendChild(titleElement(
          `${{node.label}}\\nStage: ${{node.level}}\\nMean accuracy: ${{Number(node.meta.avg_accuracy_mean || 0).toFixed(2)}}\\nMean forgetting: ${{Number(node.meta.forgetting_mean || 0).toFixed(2)}}\\nMean runtime (h): ${{Number(node.meta.runtime_hours_mean || 0).toFixed(2)}}\\nMean memory proxy (MB): ${{Number(node.meta.estimated_memory_mb || 0).toFixed(2)}}`
        ));
        nodeElements.set(node.id, group);
        viewport.appendChild(group);
      }});

      frame.addEventListener("wheel", (event) => {{
        event.preventDefault();
        const rect = svg.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        stopMomentum();
        const direction = event.deltaY < 0 ? 1.18 : 0.84;
        const nextScale = Math.min(Math.max(scale * direction, 0.3), 18.0);
        const worldX = (mouseX - tx) / scale;
        const worldY = (mouseY - ty) / scale;
        tx = mouseX - (worldX * nextScale);
        ty = mouseY - (worldY * nextScale);
        scale = nextScale;
        setTransform();
      }}, {{ passive: false }});

      frame.addEventListener("mousedown", (event) => {{
        stopMomentum();
        dragging = true;
        lastX = event.clientX;
        lastY = event.clientY;
        velocityX = 0;
        velocityY = 0;
        lastMoveAt = performance.now();
        frame.style.cursor = "grabbing";
      }});

      window.addEventListener("mousemove", (event) => {{
        if (!dragging) return;
        const now = performance.now();
        const dx = event.clientX - lastX;
        const dy = event.clientY - lastY;
        const dt = Math.max(now - lastMoveAt, 8);
        const panBoost = Math.min(3.0 + (scale * 0.22), 7.0);
        tx += dx * panBoost;
        ty += dy * panBoost;
        velocityX = (dx * panBoost) / dt * 22;
        velocityY = (dy * panBoost) / dt * 22;
        lastX = event.clientX;
        lastY = event.clientY;
        lastMoveAt = now;
        setTransform();
      }});

      window.addEventListener("mouseup", () => {{
        if (dragging) startMomentum();
        dragging = false;
        frame.style.cursor = "grab";
      }});

      frame.addEventListener("dblclick", resetView);
      if (reset) reset.addEventListener("click", resetView);
      applySelection(new Set());
      setTransform();
    }})();
    </script>
    """


def build_decision_flow_chart(tree_df: pd.DataFrame, title: str, active_path: dict[str, object] | None = None) -> str:
    if tree_df.empty:
        return "<div>No decision-tree data are available.</div>"

    active_path = active_path or {}
    active_keys = {
        "dataset": str(tree_df.iloc[0]["dataset"]),
        "memory": str(active_path.get("memory_bucket_key", "")),
        "compute": str(active_path.get("compute_bucket_key", "")),
        "forgetting": str(active_path.get("forgetting_bucket_key", "")),
        "similarity": str(active_path.get("similarity_bucket_key", "")),
        "joint": str(active_path.get("joint_bucket_key", "")),
        "method": str(active_path.get("recommended_method", "")),
    }
    stage_defs = [
        ("dataset", "dataset", "dataset_label"),
        ("memory", "memory_bucket_key", "memory_bucket_label"),
        ("compute", "compute_bucket_key", "compute_bucket_label"),
        ("forgetting", "forgetting_bucket_key", "forgetting_bucket_label"),
        ("similarity", "similarity_bucket_key", "similarity_bucket_label"),
        ("joint", "joint_bucket_key", "joint_bucket_label"),
        ("method", "recommended_method", "method_label"),
    ]
    active_node_color = "#8fdcc9"
    context_node_color = "#cbd5e1"
    active_line_color = "rgba(143, 220, 201, 0.92)"
    context_line_color = "rgba(148, 163, 184, 0.44)"

    def _stats(df: pd.DataFrame) -> dict[str, float]:
        return {
            "avg_accuracy_mean": float(df["avg_accuracy_mean"].mean()),
            "forgetting_mean": float(df["forgetting_mean"].mean()),
            "runtime_hours_mean": float(df["runtime_hours_mean"].mean()),
            "estimated_memory_mb": float(df["estimated_memory_mb"].mean()),
            "score": float(df["score"].mean()) if "score" in df.columns else 0.0,
        }

    node_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, object]] = []
    for stage_name, key_col, label_col in stage_defs:
        grouped = (
            tree_df.groupby([key_col, label_col], dropna=False)
            .apply(lambda df: pd.Series({**_stats(df), "count": int(len(df))}))
            .reset_index()
            .sort_values(["score", label_col], ascending=[False, True])
            .reset_index(drop=True)
        )
        for row in grouped.itertuples(index=False):
            key = str(getattr(row, key_col))
            node_rows.append(
                {
                    "id": f"{stage_name}::{key}",
                    "stage": stage_name,
                    "key": key,
                    "label": str(getattr(row, label_col)),
                    "count": int(row.count),
                    "score": float(row.score),
                    "avg_accuracy_mean": float(row.avg_accuracy_mean),
                    "forgetting_mean": float(row.forgetting_mean),
                    "runtime_hours_mean": float(row.runtime_hours_mean),
                    "estimated_memory_mb": float(row.estimated_memory_mb),
                    "color": active_node_color if active_keys.get(stage_name, "") == key else context_node_color,
                }
            )

    for idx in range(len(stage_defs) - 1):
        src_stage, src_key_col, src_label_col = stage_defs[idx]
        dst_stage, dst_key_col, dst_label_col = stage_defs[idx + 1]
        grouped = (
            tree_df.groupby([src_key_col, src_label_col, dst_key_col, dst_label_col], dropna=False)
            .apply(lambda df: pd.Series({"count": int(len(df)), "score": float(df["score"].mean()) if "score" in df.columns else 0.0}))
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            src_key = str(getattr(row, src_key_col))
            dst_key = str(getattr(row, dst_key_col))
            link_rows.append(
                {
                    "id": f"{src_stage}::{src_key}->{dst_stage}::{dst_key}",
                    "source_id": f"{src_stage}::{src_key}",
                    "target_id": f"{dst_stage}::{dst_key}",
                    "count": int(row.count),
                    "score": float(row.score),
                    "active": active_keys.get(src_stage, "") == src_key and active_keys.get(dst_stage, "") == dst_key,
                }
            )

    node_df = pd.DataFrame(node_rows)
    link_df = pd.DataFrame(link_rows)
    if node_df.empty or link_df.empty:
        return "<div>No decision-tree data are available.</div>"

    stage_order = [item[0] for item in stage_defs]
    stage_index = {name: idx for idx, name in enumerate(stage_order)}
    stage_width = 170
    node_width = 42
    min_gap = 6.0
    base_height = 780.0
    total_cases = max(float(len(tree_df)), 1.0)
    max_stage_nodes = max(int((node_df["stage"].astype(str) == stage).sum()) for stage in stage_order)
    global_scale = max((base_height - ((max_stage_nodes - 1) * min_gap)) / total_cases, 1.8)
    stage_span = (total_cases * global_scale) + ((max_stage_nodes - 1) * min_gap)

    graph_nodes: list[dict[str, object]] = []
    for stage_name in stage_order:
        stage_subset = node_df[node_df["stage"].astype(str) == stage_name].copy()
        stage_subset = stage_subset.sort_values(["score", "label"], ascending=[False, True]).reset_index(drop=True)
        gap = 0.0 if len(stage_subset) <= 1 else max((stage_span - (total_cases * global_scale)) / (len(stage_subset) - 1), min_gap)
        cursor_y = 36.0
        for row in stage_subset.itertuples(index=False):
            height = max(float(row.count) * global_scale, 10.0)
            graph_nodes.append(
                {
                    "id": row.id,
                    "stage": row.stage,
                    "key": row.key,
                    "label": row.label,
                    "x": 36 + (stage_index[str(row.stage)] * stage_width),
                    "y": cursor_y,
                    "width": node_width,
                    "height": height,
                    "count": int(row.count),
                    "score": float(row.score),
                    "color": row.color,
                    "avg_accuracy_mean": float(row.avg_accuracy_mean),
                    "forgetting_mean": float(row.forgetting_mean),
                    "runtime_hours_mean": float(row.runtime_hours_mean),
                    "estimated_memory_mb": float(row.estimated_memory_mb),
                }
            )
            cursor_y += height + gap

    node_layout = {node["id"]: node for node in graph_nodes}
    svg_width = ((len(stage_order) - 1) * stage_width) + 350
    svg_height = int(stage_span + 120)

    link_cursor = {node_id: 0.0 for node_id in node_layout}
    graph_paths: list[dict[str, object]] = []
    for row in link_df.itertuples(index=False):
        source = node_layout.get(str(row.source_id))
        target = node_layout.get(str(row.target_id))
        if source is None or target is None:
            continue
        thickness = max(float(row.count) * global_scale, 4.0)
        sy = source["y"] + min(link_cursor[source["id"]] + (thickness / 2.0), max(source["height"] - thickness / 2.0, thickness / 2.0))
        ty = target["y"] + min(link_cursor[target["id"]] + (thickness / 2.0), max(target["height"] - thickness / 2.0, thickness / 2.0))
        link_cursor[source["id"]] += thickness
        link_cursor[target["id"]] += thickness
        x1 = source["x"] + source["width"]
        x2 = target["x"]
        cx1 = x1 + ((x2 - x1) * 0.38)
        cx2 = x1 + ((x2 - x1) * 0.62)
        graph_paths.append(
            {
                "id": row.id,
                "source_id": row.source_id,
                "target_id": row.target_id,
                "d": f"M{x1:.2f},{sy:.2f} C{cx1:.2f},{sy:.2f} {cx2:.2f},{ty:.2f} {x2:.2f},{ty:.2f}",
                "stroke": active_line_color if bool(row.active) else context_line_color,
                "stroke_width": thickness,
                "count": int(row.count),
                "score": float(row.score),
                "active": bool(row.active),
            }
        )

    stage_titles = [
        {"label": "Dataset", "x": 36},
        {"label": "Memory", "x": 36 + stage_width},
        {"label": "Compute", "x": 36 + (stage_width * 2)},
        {"label": "Retention", "x": 36 + (stage_width * 3)},
        {"label": "Similarity", "x": 36 + (stage_width * 4)},
        {"label": "Joint", "x": 36 + (stage_width * 5)},
        {"label": "Method", "x": 36 + (stage_width * 6)},
    ]

    payload = {
        "title": title,
        "width": svg_width,
        "height": svg_height,
        "nodes": graph_nodes,
        "paths": graph_paths,
        "stages": stage_titles,
    }

    return f"""
    <style>
      .alluvial-shell {{
        background: rgba(255,255,255,0.68);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 24px;
        padding: 14px 14px 10px 14px;
        box-shadow: 0 18px 40px rgba(15,23,42,0.08);
      }}
      .alluvial-title {{
        font-size: 1.02rem;
        font-weight: 700;
        color: #0f172a;
      }}
      .alluvial-subtitle, .alluvial-legend {{
        font-size: 0.82rem;
        color: #475569;
      }}
      .alluvial-chip {{
        width: 12px;
        height: 12px;
        border-radius: 999px;
        display: inline-block;
      }}
      .alluvial-frame {{
        width: 100%;
        height: 720px;
        overflow: hidden;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(248,250,252,0.96), rgba(241,245,249,0.96));
        cursor: grab;
      }}
      .alluvial-reset {{
        border: none;
        border-radius: 999px;
        background: #dbeafe;
        color: #1d4ed8;
        padding: 0.35rem 0.8rem;
        cursor: pointer;
        font-size: 0.82rem;
      }}
      @media (prefers-color-scheme: dark) {{
        .alluvial-shell {{
          background: rgba(15,23,42,0.82);
          border-color: rgba(148,163,184,0.18);
          box-shadow: 0 18px 40px rgba(2,6,23,0.34);
        }}
        .alluvial-title {{
          color: #f8fafc;
        }}
        .alluvial-subtitle, .alluvial-legend {{
          color: #cbd5e1;
        }}
        .alluvial-frame {{
          background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(22,33,54,0.98));
        }}
        .alluvial-reset {{
          background: #1e3a8a;
          color: #dbeafe;
        }}
      }}
    </style>
    <div class="alluvial-shell">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:8px;">
        <div>
          <div class="alluvial-title">{title}</div>
          <div class="alluvial-subtitle">Drag to pan. Use the mouse wheel or trackpad to zoom. Double-click reset if needed.</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <span class="alluvial-legend" style="display:inline-flex; align-items:center; gap:6px;"><span class="alluvial-chip" style="background:rgba(143,220,201,0.92);"></span>active path</span>
          <span class="alluvial-legend" style="display:inline-flex; align-items:center; gap:6px;"><span class="alluvial-chip" style="background:rgba(203,213,225,0.92);"></span>context flows</span>
          <span class="alluvial-legend" style="display:inline-flex; align-items:center; gap:6px;"><span class="alluvial-chip" style="background:rgba(251,191,36,0.95);"></span>clicked lineage</span>
          <button id="alluvial-reset" class="alluvial-reset">Reset view</button>
        </div>
      </div>
      <div id="alluvial-frame" class="alluvial-frame">
        <svg id="alluvial-svg" viewBox="0 0 {svg_width} {svg_height}" style="width:100%; height:100%; user-select:none;">
          <g id="alluvial-viewport"></g>
        </svg>
      </div>
    </div>
    <script>
    (function() {{
      const payload = {json.dumps(payload)};
      const svg = document.getElementById("alluvial-svg");
      const viewport = document.getElementById("alluvial-viewport");
      const frame = document.getElementById("alluvial-frame");
      const reset = document.getElementById("alluvial-reset");
      if (!svg || !viewport || !frame) return;

      const darkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      const activeStroke = "rgba(143,220,201,0.92)";
      const contextStroke = "rgba(148,163,184,0.44)";
      const selectedStroke = "rgba(251,191,36,0.96)";
      const activeNodeFill = "#8fdcc9";
      const contextNodeFill = "#cbd5e1";
      const selectedNodeFill = "#fcd34d";
      let scale = 1.0;
      let tx = 20;
      let ty = 24;
      let dragging = false;
      let lastX = 0;
      let lastY = 0;
      let velocityX = 0;
      let velocityY = 0;
      let momentumFrame = null;
      let lastMoveAt = 0;

      function setTransform() {{
        viewport.style.transformOrigin = "0 0";
        viewport.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`;
      }}

      function resetView() {{
        scale = 1.0;
        tx = 20;
        ty = 24;
        velocityX = 0;
        velocityY = 0;
        if (momentumFrame) cancelAnimationFrame(momentumFrame);
        applySelection(new Set());
        setTransform();
      }}

      function stopMomentum() {{
        if (momentumFrame) {{
          cancelAnimationFrame(momentumFrame);
          momentumFrame = null;
        }}
      }}

      function startMomentum() {{
        stopMomentum();
        const friction = 0.92;
        const minVelocity = 0.12;
        function tick() {{
          tx += velocityX;
          ty += velocityY;
          velocityX *= friction;
          velocityY *= friction;
          setTransform();
          if (Math.abs(velocityX) > minVelocity || Math.abs(velocityY) > minVelocity) {{
            momentumFrame = requestAnimationFrame(tick);
          }} else {{
            momentumFrame = null;
          }}
        }}
        momentumFrame = requestAnimationFrame(tick);
      }}

      function titleElement(text) {{
        const t = document.createElementNS("http://www.w3.org/2000/svg", "title");
        t.textContent = text;
        return t;
      }}

      const incomingByTarget = new Map();
      const pathElements = new Map();
      const nodeElements = new Map();

      function registerIncoming(link) {{
        if (!incomingByTarget.has(link.target_id)) incomingByTarget.set(link.target_id, []);
        incomingByTarget.get(link.target_id).push(link);
      }}

      function collectAncestorLinks(targetNodeId, selected) {{
        const incoming = incomingByTarget.get(targetNodeId) || [];
        incoming.forEach((link) => {{
          if (!selected.has(link.id)) {{
            selected.add(link.id);
            collectAncestorLinks(link.source_id, selected);
          }}
        }});
      }}

      function selectedNodeIds(linkIds) {{
        const ids = new Set();
        linkIds.forEach((linkId) => {{
          const link = pathElements.get(linkId)?.__data;
          if (!link) return;
          ids.add(link.source_id);
          ids.add(link.target_id);
        }});
        return ids;
      }}

      function applySelection(linkIds) {{
        const highlightedNodes = selectedNodeIds(linkIds);
        pathElements.forEach((pathEl, linkId) => {{
          const link = pathEl.__data;
          const color = linkIds.size ? (linkIds.has(linkId) ? selectedStroke : contextStroke) : (link.active ? activeStroke : contextStroke);
          const opacity = linkIds.size ? (linkIds.has(linkId) ? "1" : "0.16") : (link.active ? "0.92" : "0.56");
          pathEl.setAttribute("stroke", color);
          pathEl.setAttribute("opacity", opacity);
        }});
        nodeElements.forEach((groupEl, nodeId) => {{
          const rect = groupEl.querySelector("rect");
          const label = groupEl.querySelector("text");
          const node = groupEl.__data;
          if (!rect || !label) return;
          const fill = linkIds.size ? (highlightedNodes.has(nodeId) ? selectedNodeFill : contextNodeFill) : (node.color === activeNodeFill ? activeNodeFill : contextNodeFill);
          rect.setAttribute("fill", fill);
          label.setAttribute("font-weight", highlightedNodes.has(nodeId) || fill === activeNodeFill ? "700" : "500");
        }});
      }}

      payload.stages.forEach((stage) => {{
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", stage.x);
        text.setAttribute("y", 16);
        text.setAttribute("fill", darkMode ? "#e2e8f0" : "#334155");
        text.setAttribute("font-size", "13");
        text.setAttribute("font-weight", "700");
        text.textContent = stage.label;
        viewport.appendChild(text);
      }});

      payload.paths.forEach((flow) => {{
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.__data = flow;
        path.setAttribute("d", flow.d);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", flow.stroke);
        path.setAttribute("stroke-width", flow.stroke_width);
        path.setAttribute("stroke-linecap", "round");
        path.setAttribute("opacity", flow.active ? "0.92" : "0.56");
        path.appendChild(titleElement(`Flow count: ${{flow.count}} | Mean score: ${{flow.score.toFixed(2)}}`));
        path.style.cursor = "pointer";
        path.addEventListener("click", (event) => {{
          event.stopPropagation();
          const selected = new Set([flow.id]);
          collectAncestorLinks(flow.source_id, selected);
          applySelection(selected);
        }});
        registerIncoming(flow);
        pathElements.set(flow.id, path);
        viewport.appendChild(path);
      }});

      payload.nodes.forEach((node) => {{
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.__data = node;
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", node.x);
        rect.setAttribute("y", node.y);
        rect.setAttribute("rx", "8");
        rect.setAttribute("ry", "8");
        rect.setAttribute("width", node.width);
        rect.setAttribute("height", node.height);
        rect.setAttribute("fill", node.color);
        rect.setAttribute("stroke", darkMode ? "rgba(226,232,240,0.16)" : "rgba(100,116,139,0.22)");
        rect.setAttribute("stroke-width", "1");

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", node.x + node.width + 8);
        label.setAttribute("y", node.y + Math.min(Math.max(node.height / 2, 14), node.height - 4));
        label.setAttribute("dominant-baseline", "middle");
        label.setAttribute("fill", darkMode ? "#f8fafc" : "#0f172a");
        label.setAttribute("font-size", "12");
        label.setAttribute("font-weight", node.color === activeNodeFill ? "700" : "500");
        label.textContent = node.label;

        group.appendChild(rect);
        group.appendChild(label);
        group.appendChild(titleElement(
          `${{node.label}}\\nStage: ${{node.stage}}\\nMean accuracy: ${{Number(node.avg_accuracy_mean || 0).toFixed(2)}}\\nMean forgetting: ${{Number(node.forgetting_mean || 0).toFixed(2)}}\\nMean runtime (h): ${{Number(node.runtime_hours_mean || 0).toFixed(2)}}\\nMean memory proxy (MB): ${{Number(node.estimated_memory_mb || 0).toFixed(2)}}`
        ));
        group.style.cursor = "pointer";
        group.addEventListener("click", (event) => {{
          event.stopPropagation();
          const selected = new Set();
          collectAncestorLinks(node.id, selected);
          applySelection(selected);
        }});
        nodeElements.set(node.id, group);
        viewport.appendChild(group);
      }});

      frame.addEventListener("click", (event) => {{
        if (event.target === frame || event.target === svg) {{
          applySelection(new Set());
        }}
      }});

      frame.addEventListener("wheel", (event) => {{
        event.preventDefault();
        const rect = svg.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        stopMomentum();
        const direction = event.deltaY < 0 ? 1.18 : 0.84;
        const nextScale = Math.min(Math.max(scale * direction, 0.3), 18.0);
        const worldX = (mouseX - tx) / scale;
        const worldY = (mouseY - ty) / scale;
        tx = mouseX - (worldX * nextScale);
        ty = mouseY - (worldY * nextScale);
        scale = nextScale;
        setTransform();
      }}, {{ passive: false }});

      frame.addEventListener("mousedown", (event) => {{
        stopMomentum();
        dragging = true;
        lastX = event.clientX;
        lastY = event.clientY;
        velocityX = 0;
        velocityY = 0;
        lastMoveAt = performance.now();
        frame.style.cursor = "grabbing";
      }});

      window.addEventListener("mousemove", (event) => {{
        if (!dragging) return;
        const now = performance.now();
        const dx = event.clientX - lastX;
        const dy = event.clientY - lastY;
        const dt = Math.max(now - lastMoveAt, 8);
        const panBoost = Math.min(3.0 + (scale * 0.22), 7.0);
        tx += dx * panBoost;
        ty += dy * panBoost;
        velocityX = (dx * panBoost) / dt * 22;
        velocityY = (dy * panBoost) / dt * 22;
        lastX = event.clientX;
        lastY = event.clientY;
        lastMoveAt = now;
        setTransform();
      }});

      window.addEventListener("mouseup", () => {{
        if (dragging) startMomentum();
        dragging = false;
        frame.style.cursor = "grab";
      }});

      frame.addEventListener("dblclick", resetView);
      if (reset) reset.addEventListener("click", resetView);
      applySelection(new Set());
      setTransform();
    }})();
    </script>
    """


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
