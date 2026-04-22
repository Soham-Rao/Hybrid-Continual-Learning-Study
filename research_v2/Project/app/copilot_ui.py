"""UI helpers for the dashboard copilot panel."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Thread
from collections.abc import Iterable

import streamlit as st
import streamlit.components.v1 as components
from app.copilot_prompt_templates import build_prompt_templates
from app.copilot_runtime import get_copilot_engine, get_copilot_settings, get_copilot_status
from app.dashboard_data import DATASET_LABELS
from src.copilot import InferredSettingsResult
from src.copilot.knowledge_base import (
    CHART_ALIASES,
    ChartExplanationFacts,
    HARDWARE_CARDS,
    SETTINGS_QUERY_TERMS,
    chart_explanation_draft,
    chart_focus_from_text,
)
from src.recommendation.engine import RecommendationEngine, RecommendationRequest


@dataclass
class _PendingCopilotResult:
    response_text: str | None = None
    inferred_result: InferredSettingsResult | None = None
    error_text: str | None = None


def init_copilot_state() -> None:
    st.session_state.setdefault("copilot_open", False)
    st.session_state.setdefault("copilot_messages", [])
    st.session_state.setdefault("copilot_last_inferred", None)
    st.session_state.setdefault("copilot_draft_widget", "")
    st.session_state.setdefault("copilot_draft_next", None)
    st.session_state.setdefault("copilot_submit_text", None)
    st.session_state.setdefault("copilot_submit_mode", None)
    st.session_state.setdefault("copilot_post_apply_refresh", False)
    st.session_state.setdefault("copilot_pending_dashboard_request", None)


def _looks_like_settings_query(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in SETTINGS_QUERY_TERMS):
        return True
    return any(alias in lowered for alias in HARDWARE_CARDS)


def infer_copilot_intent(text: str) -> str:
    lowered = text.lower()
    chart_terms = [
        "chart",
        "plot",
        "graph",
        "trade-off",
        "tradeoff",
        "dashboard showing",
        "visible chart",
        "current chart",
        "scatter",
        "heatmap",
    ]
    chart_terms.extend(alias for aliases in CHART_ALIASES.values() for alias in aliases)
    if any(term in lowered for term in chart_terms):
        return "interpret_chart"
    if _looks_like_settings_query(text):
        return "infer_settings"
    return "explain_recommendation"


def _chart_focus_from_text(text: str) -> str:
    return chart_focus_from_text(text)


def _append_message(role: str, content: str) -> None:
    st.session_state["copilot_messages"] = [
        *st.session_state.get("copilot_messages", []),
        {"role": role, "content": content},
    ]


def _prepare_draft_widget_state() -> None:
    next_value = st.session_state.pop("copilot_draft_next", None)
    if next_value is not None:
        st.session_state["copilot_draft_widget"] = next_value


def _queue_prompt_submission(text: str, *, force_mode: str | None = None) -> None:
    cleaned = text.strip()
    if not cleaned:
        return
    st.session_state["copilot_submit_text"] = cleaned
    st.session_state["copilot_submit_mode"] = force_mode
    st.session_state["copilot_draft_next"] = ""


def _queue_submit_from_widget() -> None:
    _queue_prompt_submission(st.session_state.get("copilot_draft_widget", ""))


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
    st.session_state["copilot_pending_dashboard_request"] = result.request
    _append_message("assistant", "Applied the inferred settings to the dashboard controls.")
    st.rerun()


def _build_copilot_response(bundle, request: RecommendationRequest, text: str, *, force_mode: str | None = None) -> str:
    intent = force_mode or infer_copilot_intent(text)
    if intent == "infer_settings":
        engine = get_copilot_engine()
        result = engine.infer_settings(text, current_request=request)
        return _format_inferred_settings(result)
    if intent == "interpret_chart":
        recommendation = RecommendationEngine(bundle.recommendation_profiles).recommend(request, top_k=3)
        shortlist = recommendation["shortlist"]
        winner = shortlist[0]
        dataset_df = bundle.recommendation_profiles[
            bundle.recommendation_profiles["dataset"].astype(str) == str(request.dataset)
        ].copy()
        best_accuracy = dataset_df.sort_values("avg_accuracy_mean", ascending=False).iloc[0]
        lowest_forgetting = dataset_df.sort_values("forgetting_mean", ascending=True).iloc[0]
        fastest = dataset_df.sort_values("runtime_hours_mean", ascending=True).iloc[0]
        smallest_memory = dataset_df.sort_values("estimated_memory_mb", ascending=True).iloc[0]
        chart_focus = _chart_focus_from_text(text)
        alt_text = ", ".join(
            f"{item['method']} (acc {item['avg_accuracy_mean']:.2f}, forget {item['forgetting_mean']:.2f}, runtime {item['runtime_hours_mean']:.2f}h, memory {item['estimated_memory_mb']:.1f} MB)"
            for item in shortlist[1:3]
        ) or "nearby alternatives"
        return chart_explanation_draft(
            ChartExplanationFacts(
                chart_focus=chart_focus,
                dataset=str(request.dataset),
                winner=winner,
                best_accuracy=best_accuracy.to_dict(),
                lowest_forgetting=lowest_forgetting.to_dict(),
                fastest=fastest.to_dict(),
                smallest_memory=smallest_memory.to_dict(),
                shortlist_summary=alt_text,
            )
        )

    engine = get_copilot_engine()
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
    return "\n".join(response_parts)


def _compute_copilot_result(
    bundle,
    request: RecommendationRequest,
    text: str,
    *,
    force_mode: str | None = None,
) -> _PendingCopilotResult:
    engine = get_copilot_engine()
    intent = force_mode or infer_copilot_intent(text)
    if intent == "infer_settings":
        result = engine.infer_settings(text, current_request=request)
        return _PendingCopilotResult(
            response_text=_format_inferred_settings(result),
            inferred_result=result,
        )
    return _PendingCopilotResult(
        response_text=_build_copilot_response(bundle, request, text, force_mode=force_mode)
    )


def _run_copilot_in_background(
    bundle,
    request: RecommendationRequest,
    text: str,
    *,
    force_mode: str | None = None,
) -> tuple[Event, _PendingCopilotResult]:
    outcome = _PendingCopilotResult()
    done = Event()

    def _worker() -> None:
        try:
            result = _compute_copilot_result(bundle, request, text, force_mode=force_mode)
            outcome.response_text = result.response_text
            outcome.inferred_result = result.inferred_result
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            outcome.error_text = f"I hit an internal copilot error while preparing the reply: {exc}"
        finally:
            done.set()

    Thread(target=_worker, daemon=True).start()
    return done, outcome


def _text_stream(text: str) -> Iterable[str]:
    normalized = text.replace("\r\n", "\n")
    chunk_size = 20
    for idx in range(0, len(normalized), chunk_size):
        yield normalized[idx : idx + chunk_size]
        time.sleep(0.01)


def _inject_copilot_shell_css(*, pinned: bool) -> None:
    st.markdown(
        """
        <style>
        .copilot-panel-anchor {
            width: 0;
            height: 0;
            overflow: hidden;
            display: block;
        }
        .codex-copilot-shell {
            position: fixed !important;
            top: 5.4rem !important;
            right: 0.8rem !important;
            width: min(29rem, calc(100vw - 1.6rem)) !important;
            max-height: calc(100vh - 6.2rem) !important;
            padding: 0.9rem 0.95rem 1rem 0.95rem !important;
            border-radius: 1.15rem !important;
            border: 1px solid rgba(148, 163, 184, 0.22) !important;
            background: rgba(15, 23, 42, 0.96) !important;
            box-shadow: 0 18px 60px rgba(2, 6, 23, 0.42) !important;
            backdrop-filter: blur(14px) !important;
            z-index: 9999 !important;
            transform: translateX(calc(100% - 1.15rem));
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
            overflow: hidden auto !important;
        }
        .codex-copilot-shell:hover,
        .codex-copilot-shell:focus-within,
        .codex-copilot-shell.pinned {
            transform: translateX(0);
            box-shadow: 0 24px 72px rgba(2, 6, 23, 0.52) !important;
            border-color: rgba(94, 234, 212, 0.3) !important;
        }
        .codex-copilot-shell .copilot-peek {
            position: absolute;
            left: -2.2rem;
            top: 5.2rem;
            width: 2rem;
            padding: 0.55rem 0.2rem;
            border-radius: 1rem 0 0 1rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-right: none;
            background: rgba(15, 23, 42, 0.97);
            color: #dbeafe;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-align: center;
            writing-mode: vertical-rl;
            text-transform: uppercase;
            transform: rotate(180deg);
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.28);
            pointer-events: none;
        }
        .copilot-chat-scroll {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 0.95rem;
            background: rgba(15, 23, 42, 0.28);
            padding: 0.2rem 0.25rem 0.1rem 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        f"""
        <script>
        const doc = window.parent.document;
        const anchors = doc.querySelectorAll('.copilot-panel-anchor');
        const anchor = anchors[anchors.length - 1];
        if (anchor) {{
          const root = anchor.closest('[data-testid="stVerticalBlock"]');
          if (root) {{
            root.classList.add('codex-copilot-shell');
            if ({str(pinned).lower()}) {{
              root.classList.add('pinned');
            }} else {{
              root.classList.remove('pinned');
            }}
          }}
        }}
        </script>
        """,
        height=0,
        width=0,
    )


def render_copilot_panel(bundle, request: RecommendationRequest) -> None:
    init_copilot_state()
    _prepare_draft_widget_state()
    _inject_copilot_shell_css(pinned=bool(st.session_state.get("copilot_open")))

    settings = get_copilot_settings()
    status = get_copilot_status()
    pending_text = st.session_state.pop("copilot_submit_text", None)
    pending_mode = st.session_state.pop("copilot_submit_mode", None)

    current_recommendation = None
    if not bundle.recommendation_profiles.empty:
        current_recommendation = RecommendationEngine(bundle.recommendation_profiles).recommend(request, top_k=1)["recommended_method"]

    panel = st.container()
    with panel:
        st.markdown(
            f"<div class='copilot-panel-anchor'></div><div class='copilot-peek'>Copilot</div>",
            unsafe_allow_html=True,
        )
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
                _queue_prompt_submission(
                    "Explain why this recommendation fits my current settings.",
                    force_mode="explain_recommendation",
                )
                st.rerun()
        with controls[1]:
            if st.button("Explain chart", key="copilot_explain_chart", use_container_width=True):
                _queue_prompt_submission(
                    "Explain the currently visible chart in the active dashboard context.",
                    force_mode="interpret_chart",
                )
                st.rerun()

        lower_controls = st.columns(2, gap="small")
        with lower_controls[0]:
            apply_disabled = st.session_state.get("copilot_last_inferred") is None
            if st.button("Apply inferred settings", key="copilot_apply_settings", use_container_width=True, disabled=apply_disabled):
                _apply_inferred_settings()
        with lower_controls[1]:
            label = "Auto-hide" if st.session_state.get("copilot_open") else "Pin open"
            if st.button(label, key="copilot_pin_toggle", use_container_width=True):
                st.session_state["copilot_open"] = not st.session_state.get("copilot_open", False)
                st.rerun()

        st.caption("Prompt ideas")
        templates = build_prompt_templates(request, recommended_method=current_recommendation)
        for row_start in range(0, len(templates), 2):
            cols = st.columns(2, gap="small")
            for col, template in zip(cols, templates[row_start : row_start + 2]):
                with col:
                    if st.button(
                        template.label,
                        key=f"copilot_template_{template.key}",
                        use_container_width=True,
                        help=template.help_text,
                    ):
                        st.session_state["copilot_draft_next"] = template.prompt
                        st.rerun()

        st.markdown("<div class='copilot-chat-scroll'>", unsafe_allow_html=True)
        message_host = st.container(height=360, border=False)
        with message_host:
            if pending_text:
                _append_message("user", pending_text)

            for message in st.session_state.get("copilot_messages", []):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if pending_text:
                with st.chat_message("assistant"):
                    thinking_placeholder = st.empty()
                    done_event, outcome = _run_copilot_in_background(
                        bundle,
                        request,
                        pending_text,
                        force_mode=pending_mode,
                    )
                    dot_cycle = (".", "..", "...")
                    cycle_index = 0
                    while not done_event.is_set():
                        dots = dot_cycle[cycle_index % len(dot_cycle)]
                        thinking_placeholder.markdown(f"_Thinking{dots}_")
                        cycle_index += 1
                        time.sleep(0.18)
                    response_text = outcome.error_text or outcome.response_text or "I couldn't generate a reply just now."
                    if outcome.inferred_result is not None:
                        st.session_state["copilot_last_inferred"] = outcome.inferred_result
                        st.session_state["copilot_post_apply_refresh"] = True
                    thinking_placeholder.empty()
                    final_text = st.write_stream(_text_stream(response_text))
                _append_message("assistant", final_text if isinstance(final_text, str) else response_text)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.form("copilot_chat_form", clear_on_submit=False):
            st.text_area(
                "Ask the copilot",
                key="copilot_draft_widget",
                placeholder="Ask why this method was recommended, or describe your hardware and constraints.",
                height=120,
            )
            send_cols = st.columns(2, gap="small")
            with send_cols[0]:
                st.form_submit_button("Send", on_click=_queue_submit_from_widget, use_container_width=True)
            with send_cols[1]:
                clear_clicked = st.form_submit_button("Clear chat", use_container_width=True)
        if clear_clicked:
            st.session_state["copilot_messages"] = []
            st.session_state["copilot_last_inferred"] = None
            st.session_state["copilot_draft_next"] = ""
            st.rerun()

    if st.session_state.get("copilot_post_apply_refresh"):
        st.session_state["copilot_post_apply_refresh"] = False
        st.rerun()
