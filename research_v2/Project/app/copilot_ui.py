"""UI helpers for the dashboard copilot panel."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.copilot_prompt_templates import build_prompt_templates
from app.copilot_runtime import get_copilot_engine, get_copilot_settings, get_copilot_status
from app.dashboard_data import DATASET_LABELS
from src.copilot import InferredSettingsResult
from src.recommendation.engine import RecommendationEngine, RecommendationRequest


def init_copilot_state() -> None:
    st.session_state.setdefault("copilot_open", False)
    st.session_state.setdefault("copilot_messages", [])
    st.session_state.setdefault("copilot_draft", "")
    st.session_state.setdefault("copilot_last_inferred", None)


def infer_copilot_intent(text: str) -> str:
    lowered = text.lower()
    settings_terms = [
        "gpu",
        "vram",
        "ram",
        "memory",
        "compute",
        "retrain",
        "retention",
        "forgetting",
        "laptop",
        "cpu",
        "cifar",
        "mnist",
        "imagenet",
    ]
    if any(term in lowered for term in settings_terms):
        return "infer_settings"
    return "explain_recommendation"


def _append_message(role: str, content: str) -> None:
    st.session_state["copilot_messages"] = [
        *st.session_state.get("copilot_messages", []),
        {"role": role, "content": content},
    ]


def _format_inferred_settings(result: InferredSettingsResult) -> str:
    lines = [
        "I parsed your description into a proposed dashboard configuration.",
        "",
        "Recommendation source note: this is a heuristic settings suggestion, not a direct experimental result.",
        "",
        f"- Dataset: {DATASET_LABELS.get(result.request.dataset, result.request.dataset)}",
        f"- Memory budget: {int(result.request.memory_budget_mb)} MB",
        f"- Compute budget: {result.request.compute_budget}",
        f"- Acceptable forgetting: {int(result.request.acceptable_forgetting or 0)}",
        f"- Task similarity: {result.request.task_similarity}",
        f"- Joint retraining allowed: {result.request.joint_retraining_allowed}",
        "",
        "Assumptions:",
    ]
    for item in result.assumptions:
        lines.append(f"- {item}")
    if result.scope_notes:
        lines.append("")
        lines.append("Scope notes:")
        for item in result.scope_notes:
            lines.append(f"- {item}")
    if result.clarification_questions:
        lines.append("")
        lines.append("Follow-up questions:")
        for item in result.clarification_questions:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("Use `Apply inferred settings` if you want these values pushed into the dashboard controls.")
    return "\n".join(lines)


def _apply_inferred_settings() -> None:
    result = st.session_state.get("copilot_last_inferred")
    if not result:
        return
    request = result.request
    st.session_state["dashboard_dataset"] = request.dataset
    st.session_state["dashboard_memory_budget_mb"] = int(request.memory_budget_mb)
    st.session_state["dashboard_compute_budget"] = request.compute_budget
    st.session_state["dashboard_acceptable_forgetting"] = int(request.acceptable_forgetting or 0)
    st.session_state["dashboard_task_similarity"] = request.task_similarity
    st.session_state["dashboard_joint_allowed"] = request.joint_retraining_allowed
    _append_message("assistant", "Applied the inferred settings to the dashboard controls.")
    st.rerun()


def handle_copilot_message(bundle, request: RecommendationRequest, text: str, *, force_mode: str | None = None) -> None:
    engine = get_copilot_engine()
    intent = force_mode or infer_copilot_intent(text)
    _append_message("user", text)
    if intent == "infer_settings":
        result = engine.infer_settings(text, current_request=request)
        st.session_state["copilot_last_inferred"] = result
        _append_message("assistant", _format_inferred_settings(result))
        return

    explanation = engine.explain_recommendation(bundle.recommendation_profiles, request, query=text)
    response_parts = [
        explanation.summary,
        "",
        explanation.recommendation_source_note,
        explanation.source_disclosure,
        explanation.uncertainty_note,
    ]
    if explanation.claim_guardrail_note:
        response_parts.extend(["", explanation.claim_guardrail_note])
    response_parts.extend(["", explanation.explanation])
    if explanation.evidence_snippets:
        evidence_preview = "\n".join(f"- {snippet}" for snippet in explanation.evidence_snippets)
        response_parts.extend(["", "Evidence snippets:", evidence_preview])
    response = "\n".join(response_parts)
    _append_message("assistant", response)


def render_copilot_panel(bundle, request: RecommendationRequest) -> None:
    init_copilot_state()
    settings = get_copilot_settings()
    status = get_copilot_status()
    current_recommendation = None
    if not bundle.recommendation_profiles.empty:
        current_recommendation = RecommendationEngine(bundle.recommendation_profiles).recommend(request, top_k=1)["recommended_method"]

    if not st.session_state["copilot_open"]:
        with st.container():
            st.markdown(
                """
                <style>
                .copilot-handle-note {
                    writing-mode: vertical-rl;
                    transform: rotate(180deg);
                    letter-spacing: 0.08em;
                    color: #e2e8f0;
                    font-size: 0.8rem;
                    margin: 0 auto 0.6rem auto;
                    text-align: center;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div class='copilot-handle-note'>Copilot</div>", unsafe_allow_html=True)
            if st.button("Open", key="copilot_open_button", use_container_width=True):
                st.session_state["copilot_open"] = True
                st.rerun()
        return

    st.markdown("### Copilot")
    if status.available and status.model_available:
        st.caption(f"Local model ready: {settings.default_model}")
    elif status.available:
        st.warning(status.message)
    else:
        st.info(status.message)

    controls = st.columns(2, gap="small")
    with controls[0]:
        if st.button("Explain recommendation", key="copilot_explain_reco", use_container_width=True):
            handle_copilot_message(bundle, request, "Explain why this recommendation fits my current settings.", force_mode="explain_recommendation")
            st.rerun()
    with controls[1]:
        if st.button("Explain chart", key="copilot_explain_chart", use_container_width=True):
            _append_message(
                "assistant",
                "Chart-specific interpretation is available in a limited first pass. Mention the chart title or the active tab, and I will interpret it from the current dataset context and study evidence.",
            )
            st.rerun()

    lower_controls = st.columns(2, gap="small")
    with lower_controls[0]:
        apply_disabled = st.session_state.get("copilot_last_inferred") is None
        if st.button("Apply inferred settings", key="copilot_apply_settings", use_container_width=True, disabled=apply_disabled):
            _apply_inferred_settings()
    with lower_controls[1]:
        if st.button("Dismiss", key="copilot_dismiss", use_container_width=True):
            st.session_state["copilot_open"] = False
            st.rerun()

    st.caption("Prompt ideas")
    templates = build_prompt_templates(request, recommended_method=current_recommendation)
    for row_start in range(0, len(templates), 2):
        cols = st.columns(2, gap="small")
        for col, template in zip(cols, templates[row_start : row_start + 2]):
            with col:
                if st.button(template.label, key=f"copilot_template_{template.key}", use_container_width=True, help=template.help_text):
                    st.session_state["copilot_draft"] = template.prompt
                    st.rerun()

    for message in st.session_state.get("copilot_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    draft = st.text_area(
        "Ask the copilot",
        key="copilot_draft",
        placeholder="Ask why this method was recommended, or describe your hardware and constraints.",
        height=120,
    )
    send_cols = st.columns(2, gap="small")
    with send_cols[0]:
        if st.button("Send", key="copilot_send", use_container_width=True) and draft.strip():
            handle_copilot_message(bundle, request, draft.strip())
            st.session_state["copilot_draft"] = ""
            st.rerun()
    with send_cols[1]:
        if st.button("Clear chat", key="copilot_clear", use_container_width=True):
            st.session_state["copilot_messages"] = []
            st.session_state["copilot_last_inferred"] = None
            st.rerun()
