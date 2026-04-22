# Research Workbench Copilot — Release Gate

## Scope Of This Gate
This document closes the release-gate track for the dashboard copilot. It records what is verified, what is intentionally limited, and what remains a known limitation rather than a blocker.

## Release Checks

### 1. Recommendation Results Are Not Mutated Silently
Status: PASS

What is verified:
- recommendation explanations do not change the deterministic recommendation output
- inferred settings are only suggestions until the user explicitly presses `Apply inferred settings`
- the apply flow now uses a deferred handoff into the sidebar initialization path, which avoids Streamlit widget-state mutation errors

### 2. Explanations Stay Aligned With Study Artifacts
Status: PASS

What is verified:
- recommendation explanations are built from grounded current-study context first
- the model is used primarily as a response composer on top of a grounded draft
- responses disclose evidence basis, uncertainty, and recommendation source of truth
- unsupported significance-style claims are guarded

### 3. The Right-Side Copilot Panel Works Across The Dashboard
Status: PASS WITH UI CAVEAT

What is verified:
- the copilot is available alongside Recommendation, Decision Tree, Method Comparison, Dataset Visuals, Report/About, and Library
- chat state persists across reruns and tab changes
- prompt-template helpers, explanation actions, and settings-apply actions are integrated

Known limitation:
- the right-side drawer is implemented as a Streamlit overlay approximation rather than a native framework sidebar drawer
- this is acceptable for the current release, but should remain documented as a UI implementation caveat

### 4. Local Ollama-Only Usage Is Viable
Status: PASS

What is verified:
- local Ollama is reachable on the current machine
- default selected model `qwen2.5:7b-instruct` is installed
- offline fallback behavior still exists if Ollama or the model becomes unavailable

### 5. Capabilities, Boundaries, And Known Limitations Are Documented
Status: PASS

This is documented in:
- `docs/copilot_scope_lock.md`
- `docs/copilot_implementation_plan.md`
- `docs/copilot_evaluation.md`
- this release-gate document

## First-Release Capabilities
- grounded explanation of the current recommendation
- comparison against nearby alternatives
- natural-language-to-settings inference with explicit assumptions
- explicit apply flow for pushing inferred settings into the dashboard
- chart interpretation for named chart families covered by the knowledge base
- prompt-template helpers above the chat box
- local-model-backed wording with deterministic fallback behavior

## First-Release Boundaries
- the deterministic recommendation engine remains the only recommendation authority
- the assistant may explain and suggest, but it must not silently override the dashboard state
- external knowledge is secondary to local study artifacts and local project documentation
- chart interpretation is knowledge-base driven for the chart families currently encoded; it is not full DOM/chart-object introspection

## Known Limitations
- the visible streaming experience is UI-driven; backend model transport is not yet true token streaming
- the right-side drawer is a Streamlit overlay approximation, not a native right sidebar component
- settings inference is heuristic and should remain framed as approximate when hardware or benchmarks fall outside the exact tested study scope
- chart explanations are strongest when the chart family is named clearly in the prompt

## Release Verdict
The copilot is acceptable for first release.

Reason:
- core trust boundaries are in place
- local usage is viable
- explanations are grounded enough for the intended dashboard-assistant role
- settings inference is explicit and confirmation-gated
- major known limitations are documented rather than hidden
