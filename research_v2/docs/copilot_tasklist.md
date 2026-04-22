# Research Workbench Copilot — Task List

**Purpose**
This track covers the conversational assistant that lives inside the dashboard as a persistent right-side copilot. It is intentionally separate from the main study checklist so the research execution plan stays clean.

**Initial Product Goal**
- Build a local, grounded assistant that helps users understand recommendations, interpret results, translate natural-language constraints into dashboard settings, and navigate the evidence base without replacing the deterministic recommendation engine.

## Track 0 — Scope Lock And UX Definition
- [x] Lock the copilot scope as an assistant layer on top of the existing dashboard, not a replacement for the recommendation engine
- [x] Lock the interaction model for the hidden right-edge hover panel and slide-in chat experience
- [x] Lock the first-release modes: explain recommendation, natural-language-to-settings, chart/result interpretation
- [x] Decide whether the first release is chat-only or includes lightweight action buttons like “Apply inferred settings”
- [x] Define the trust policy: what the copilot may explain, infer, summarize, and what it must never claim without evidence

## Track 1 — Local Model Integration
- [x] Choose the default local Ollama model for the first release
- [x] Keep model selection configurable so multiple local models can be tested
- [x] Build a small local inference wrapper for Ollama requests, timeouts, retries, and health checks
- [x] Add a clear offline / unavailable-model fallback state in the UI
- [x] Define prompt budget and context-size policy so the assistant stays responsive

## Track 2 — Knowledge Layer
- [x] Build a grounded retrieval layer over current study artifacts
- [x] Ingest recommendation summaries, study tables, reports, and ablation outputs from the current workspace
- [x] Add support for using literature-survey material and relevant markdown/doc files across both `v1_deadline_prototype` and `research_v2`
- [x] Define a controlled policy for optional internet-backed lookups when external literature or definitions are needed
- [x] Add metadata so retrieved evidence can be labeled as empirical result, design note, literature note, or external source

## Track 3 — Explain Recommendation Mode
- [ ] Use current dashboard state plus retrieved evidence to explain why the recommended method fits the current constraints
- [ ] Explain tradeoffs in conceptual language, not just engine score components
- [ ] Compare the winner against the top alternatives using grounded evidence
- [ ] Distinguish clearly between empirical evidence and reasoned inference
- [ ] Add a concise “why this works here” summary block for the current recommendation

## Track 4 — Natural Language To Settings
- [ ] Parse freeform user descriptions of hardware, constraints, and goals into structured dashboard settings
- [ ] Show the inferred settings and assumptions before applying them
- [ ] Add a confirmation step for mutating sliders, toggles, or dropdowns
- [ ] Support vague hardware descriptions like old GPUs, laptop-only setups, and overnight retraining tolerance
- [ ] Preserve user trust by surfacing what was assumed versus what was directly stated

## Track 5 — Follow-Up Questioning
- [ ] Add a lightweight clarification mode when the user request is too ambiguous
- [ ] Limit clarifying questions to high-value gaps only
- [ ] Use follow-up answers to refine inferred settings or the explanation context
- [ ] Avoid turning the copilot into a long generic chat unless that is explicitly needed

## Track 6 — Dashboard Integration
- [ ] Add a right-edge hidden hover handle that reveals the copilot across all tabs
- [ ] Keep the copilot persistent across Recommendation, Method Comparison, Dataset Visuals, Report/About, Library, and Decision Tree views
- [ ] Preserve chat state across tab switches
- [ ] Add quick actions for “Explain this recommendation”, “Explain this chart”, and “Apply inferred settings”
- [ ] Add dismiss / collapse behavior that feels lightweight and non-intrusive

## Track 7 — Prompt Templates And Guided Prompts
- [ ] Add hoverable prompt-template chips above the chat box
- [ ] Include templates for explanation, constraint translation, comparison, chart interpretation, and literature-backed questions
- [ ] Make prompt suggestions context-aware to the active tab when possible
- [ ] Keep templates short, practical, and phrased for strong grounded responses
- [ ] Test that templates improve output quality without making the UI feel cluttered

## Track 8 — Safety, Grounding, And Claim Control
- [ ] Force the copilot to treat the deterministic engine as the recommendation source of truth
- [ ] Prevent unsupported claims about scientific results, significance, or literature
- [ ] Surface uncertainty when evidence is thin or mixed
- [ ] Label when a response is using external internet material versus local study evidence
- [ ] Add citation or evidence snippets for high-stakes explanations when practical

## Track 9 — Evaluation
- [ ] Define qualitative acceptance criteria for clarity, grounding, and usefulness
- [ ] Create test prompts for recommendation explanation quality
- [ ] Create test prompts for natural-language-to-settings quality
- [ ] Create test prompts for chart and report interpretation quality
- [ ] Check latency, failure handling, and model-availability behavior on this machine

## Track 10 — Release Gate
- [ ] Verify the copilot does not alter core recommendation results unless the user explicitly applies inferred settings
- [ ] Verify explanations stay aligned with actual study artifacts
- [ ] Verify the right-side panel works cleanly across the full dashboard
- [ ] Verify local Ollama-only usage is viable before considering broader extensions
- [ ] Document the copilot’s capabilities, boundaries, and known limitations

## Completion Rule
This copilot track should be considered ready when:
- a local Ollama-backed assistant works reliably inside the dashboard
- recommendation explanations are grounded in current study evidence
- natural-language constraint parsing can safely suggest or apply settings
- prompt-template helpers improve user prompting quality
- the assistant clearly distinguishes evidence, inference, and external knowledge

**Track 0 output**
- Scope and UX lock recorded in `docs/copilot_scope_lock.md`
