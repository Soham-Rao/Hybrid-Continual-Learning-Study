# Research Workbench Copilot — Implementation Plan

## Summary
Add a persistent conversational copilot to the dashboard as a hidden right-edge panel that slides in on hover. The copilot should help users understand recommendations, interpret charts and study outputs, translate natural-language descriptions into dashboard settings, and navigate the evidence base in a more human way.

This assistant should be:
- local-first
- grounded in current study artifacts
- explanatory rather than authoritative
- tightly integrated with the existing deterministic recommendation engine

The copilot is not meant to replace the engine or the analytical views. It is meant to make them easier to understand and use.

## Track 0 Decisions Already Locked
The following decisions are now locked for the first implementation pass:
- the copilot is an assistant layer, not a recommendation-engine replacement
- the panel uses a right-edge hover or click reveal pattern
- the first release includes three modes: explain recommendation, natural-language-to-settings, and chart/result interpretation
- the first release is chat-first with lightweight optional actions only
- inferred settings must be confirmed before they are applied
- one shared conversation thread is used across the dashboard in the first release
- the deterministic engine remains the sole recommendation authority

See `docs/copilot_scope_lock.md` for the full Track 0 contract.

## Product Goals
1. Explain why a recommendation makes sense conceptually, not just numerically.
2. Translate natural-language user descriptions into structured dashboard settings.
3. Help users interpret charts, tradeoffs, rankings, and report summaries.
4. Surface evidence from current results, prior notes, literature surveys, and relevant local documents.
5. Stay trustworthy by separating empirical evidence, design reasoning, and external knowledge.

## Core Product Principles
- The recommendation engine remains the source of truth for chosen methods.
- The copilot may explain and interpret recommendations, but should not silently override them.
- The copilot should never present unsupported scientific claims as fact.
- The copilot should make assumptions visible before mutating dashboard controls.
- The interface should feel lightweight, modern, and always available without overwhelming the main workspace.

## UX Design

### Interaction Model
- A slim hover target lives on the right edge of the dashboard.
- Hovering or clicking reveals a slide-in copilot panel.
- The panel remains available across all tabs and remembers conversation context.
- The panel can be collapsed without losing the active chat state.

### Major Modes
#### 1. Explain Recommendation
The copilot should explain:
- why the recommended method fits the current settings
- what tradeoffs it is making
- why nearby alternatives were not chosen
- what the method is doing conceptually in continual-learning terms

This explanation should use:
- engine outputs
- study summaries
- method traits
- dataset characteristics
- relevant ablation/context evidence

#### 2. Natural Language To Settings
The user should be able to say things like:
- “I have an old GPU and limited memory”
- “I want something stable and cheap to run”
- “I can retrain overnight if it helps”

The copilot should:
- infer likely dashboard settings
- display what assumptions were made
- ask a follow-up only when necessary
- offer a one-click or explicit confirmation to apply those settings

#### 3. Chart And Report Interpretation
The copilot should be able to answer:
- what a chart means
- why a method appears where it does
- what “top cluster” means
- what tradeoff a Pareto view is showing
- how a report statement relates to the evidence

### Prompt Helpers Above The Chat Box
Add hoverable prompt-template chips above the chat input to help users get better output.

Examples:
- `Explain why this method fits my current settings`
- `Compare the top recommendation with the next two alternatives`
- `I have an old laptop GPU and low memory. Suggest settings`
- `Interpret this chart in simple terms`
- `What tradeoff is this plot showing?`
- `Explain this recommendation using study results and method intuition`
- `What would change if I cared more about forgetting than runtime?`
- `Use relevant literature or project notes to explain this method`

These should be:
- short
- easy to reuse
- context-sensitive where possible
- visible as suggestions, not mandatory workflows

First-release approval:
- prompt-template chips are in scope for the initial build
- they should appear above the chat box as hoverable suggestions/examples
- they may prefill or inject text, but should not auto-send prompts
- future chat refinement should support streamed token-by-token output instead of only full responses appearing at once

## Knowledge Sources

### Local Structured Study Artifacts
The copilot should retrieve from:
- current summary tables
- recommendation profiles
- case-study outputs
- report files
- statistical summaries
- ablation summaries
- dataset leader outputs
- chart-related CSV and markdown artifacts

### Local Project Documentation
The copilot should also be able to use:
- markdown, notes, and design documents across both `research_v2` and `v1_deadline_prototype`
- literature-survey notes and research writeups stored anywhere under this workspace
- method explanations, task lists, project scope notes, and comparative documents across both versions

### External Knowledge
The plan should allow optional use of:
- internet-backed sources for literature clarification, definitions, or method background
- retrieved external sources only when needed and labeled clearly as external

Important policy:
- local project artifacts should be preferred for project-specific claims
- literature surveys and local notes should be preferred over freeform model memory
- external internet material should be clearly distinguished from empirical study evidence

## Architecture

### 1. UI Layer
Location:
- dashboard app under `research_v2/Project/app/`

Responsibilities:
- render the hover-reveal copilot panel
- show chat history
- show prompt-template chips
- show mode-aware quick actions
- expose optional “Apply inferred settings” actions
- pass current dashboard context into the copilot backend

First-release UI policy:
- keep one shared thread across tabs
- preserve panel state when the user changes tabs
- keep apply-style actions explicit and confirmable

### 2. Copilot Orchestration Layer
Responsibilities:
- receive current dashboard context
- classify the user request type
- collect the right evidence bundle
- build prompts for the local LLM
- enforce trust and response policies
- return response text plus any optional structured actions

Likely request categories:
- explain_recommendation
- compare_methods
- interpret_chart
- interpret_report
- infer_settings
- clarify_user_constraints
- literature_or_background_explanation

### 3. Retrieval Layer
Responsibilities:
- gather relevant local evidence for the current query
- fetch rows or snippets from structured artifacts
- search markdown and project notes across `research_v2` and `v1_deadline_prototype`
- optionally route to internet-backed retrieval when explicitly needed

Potential sources:
- CSV summaries
- markdown reports
- analysis notes
- task lists
- local literature-survey documents
- v1 and v2 design notes

Recommended strategy:
- simple structured retrieval first
- local text retrieval second
- optional external retrieval last

Track 2 implementation lock:
- structured current-study artifacts are now treated as first-class empirical evidence
- local markdown/text documents across both `research_v2` and `v1_deadline_prototype` are searchable as grounded context
- retrieval results should carry evidence labels such as `empirical_result`, `design_note`, `literature_note`, and `external_source`
- optional internet-backed retrieval remains policy-gated and off by default

### 4. Deterministic Action Layer
Responsibilities:
- convert inferred settings into structured dashboard state
- return recommended changes without applying them silently
- validate values before pushing them into the UI

This layer should remain separate from the LLM so that:
- the LLM proposes
- the app validates
- the user confirms

First-release lock:
- settings application is allowed only as an explicit confirmed action
- no autonomous multi-step mutations are allowed

### 5. Local Model Layer
Use Ollama as the first backend.

Recommended initial models to test:
- `qwen2.5:7b-instruct`
- `llama3.1:8b-instruct-q4_K_M`

Suggested default:
- `qwen2.5:7b-instruct`

Why:
- strong instruction following
- good structured reasoning for local use
- appropriate size for this machine compared with larger alternatives

Keep the selected model configurable in settings or environment config.

Track 1 implementation lock:
- default first-release model: `qwen2.5:7b-instruct`
- configurable via environment instead of hard-coding a single model choice
- local availability should be surfaced in the dashboard UI before the full chat panel is built
- prompt budget and context-item limits should be enforced in the local client/config layer

## Suggested File/Module Structure

### App-side UI
- `research_v2/Project/app/copilot_panel.py`
- `research_v2/Project/app/copilot_ui.py`
- `research_v2/Project/app/copilot_prompt_templates.py`

### Backend / orchestration
- `research_v2/Project/src/copilot/engine.py`
- `research_v2/Project/src/copilot/prompts.py`
- `research_v2/Project/src/copilot/policies.py`
- `research_v2/Project/src/copilot/context_builder.py`
- `research_v2/Project/src/copilot/actions.py`

### Retrieval / knowledge
- `research_v2/Project/src/copilot/retrieval.py`
- `research_v2/Project/src/copilot/local_docs_index.py`
- `research_v2/Project/src/copilot/artifact_queries.py`
- `research_v2/Project/src/copilot/external_sources.py`

### Ollama integration
- `research_v2/Project/src/copilot/ollama_client.py`

### Tests
- `research_v2/Project/tests/test_copilot_prompts.py`
- `research_v2/Project/tests/test_copilot_retrieval.py`
- `research_v2/Project/tests/test_copilot_actions.py`
- `research_v2/Project/tests/test_copilot_panel.py`

## Request Flow

### Explain Recommendation Flow
1. User opens the copilot and asks why a method was recommended.
2. App sends current dashboard state and current recommendation context.
3. Retrieval layer gathers:
   - current recommendation row
   - nearby alternatives
   - relevant dataset summary
   - method traits
   - useful report/ablation snippets if relevant
4. Prompt builder asks the LLM for:
   - a conceptual explanation
   - tradeoffs
   - comparison against alternatives
   - clear distinction between evidence and inference
5. Copilot returns grounded natural-language output.

Track 3 implementation lock:
- explain-recommendation mode now has a dedicated backend path
- it builds a grounded context from the current request, shortlist, method card, and retrieved evidence
- it uses Ollama when available and falls back to a deterministic grounded explanation when Ollama is offline
- the explanation path is UI-ready but not yet exposed as the right-edge chat panel

### Natural Language To Settings Flow
1. User describes constraints in free text.
2. Copilot classifies intent as `infer_settings`.
3. LLM proposes a structured interpretation.
4. Action layer validates or normalizes those values.
5. UI shows:
   - inferred settings
   - assumptions
   - apply/cancel option
6. On approval, dashboard controls update.

Track 4 implementation lock:
- the first settings-inference backend uses deterministic heuristics and validation as the primary parser
- it returns a structured proposal plus visible assumptions rather than mutating controls directly
- every inferred-settings result is confirmation-gated before application
- vague hardware descriptions such as older GPUs, laptop-only setups, and retraining tolerance are mapped conservatively
- broader hardware descriptions such as additional GPUs, CPU-only setups, CPU tiers, and RAM amounts are interpreted heuristically with explicit scope notes when the mapping is only approximate
- if the request references hardware, datasets, or backbone choices outside the exact evaluated study scope, the copilot should say so clearly and frame the result as an estimated suggestion rather than a directly validated one

### Chart Interpretation Flow
1. User asks about a chart while on a given tab.
2. App passes the active tab, chart type, filters, and selected dataset/method context.
3. Retrieval layer gathers the relevant summary rows and report snippets.
4. LLM returns a simple interpretation grounded in visible evidence.

### Clarification Flow
1. User gives an ambiguous or underspecified natural-language request.
2. The settings-inference layer identifies only the highest-value missing signals.
3. The copilot returns at most a few targeted clarification questions instead of expanding into a long generic chat.
4. Follow-up answers refine the proposed settings while preserving visible assumptions and confirmation gating.

Track 5 implementation lock:
- clarification is now generated only when important setting gaps remain
- the question set is intentionally capped and focused on the biggest unresolved decision variables
- follow-up prompting should refine the inferred settings rather than opening a freeform broad conversation

### Dashboard Integration Flow
1. The user opens the copilot from the slim right-side reveal handle.
2. The panel stays available while the main dashboard tabs continue to drive the core workspace.
3. The copilot keeps its conversation state across tab switches and reruns.
4. Quick actions can trigger explanation, chart-help entry points, and settings application.
5. Settings application remains confirmation-gated and updates the dashboard controls only when explicitly applied.

Track 6 implementation lock:
- the first visible copilot UI is now integrated as a collapsible right-side panel shell
- the panel persists alongside Recommendation, Decision Tree, Method Comparison, Dataset Visuals, Report/About, and Library
- chat state survives tab switches through session state
- quick actions exist for explaining the recommendation, prompting chart help, applying inferred settings, dismissing the panel, and clearing chat
- the first Streamlit implementation uses a slim right-side reveal/toggle handle rather than a fully custom hover-only overlay

Track 7 implementation lock:
- prompt-template buttons now appear above the chat box
- they prefill the chat draft rather than sending immediately
- templates cover recommendation explanation, comparison, settings inference, chart interpretation, trade-offs, and literature-backed prompts
- template copy is contextualized with the current dataset and, when available, the current recommended method

## Prompt Design

### System-Level Guidance
Prompts should enforce:
- recommendation engine outputs are the authoritative recommendation source
- local study evidence should be preferred over model memory
- external knowledge must be labeled clearly
- explanations should be descriptive, grounded, and not overclaim
- assumptions must be surfaced when inferring settings

### Output Style Guidance
Prompt the model to:
- explain in clear, human language
- avoid empty generic ML buzzwords
- distinguish “observed in our study” from “conceptually expected”
- note uncertainty when evidence is limited

## Safety And Trust Policy
- Do not let the LLM silently set dashboard controls.
- Do not let the LLM invent significance or effect-size claims.
- Do not let the LLM replace the deterministic recommendation engine.
- Keep a visible distinction between:
  - empirical evidence from current study outputs
  - local project notes or literature surveys
  - internet-backed external sources

Additional Track 0 lock:
- if the evidence is insufficient, the assistant should say so instead of filling the gap with confident generic prose

Track 8 implementation lock:
- every explanation response now carries an explicit note that the deterministic recommendation engine remains the recommendation source of truth
- responses now disclose whether they are grounded in local study evidence, local notes/documents, or external material
- the copilot now surfaces an uncertainty note whenever retrieved evidence is sparse or mixed
- if generated wording drifts toward unsupported significance or literature-authority claims without matching retrieved evidence, the UI adds a claim-guardrail note
- evidence snippets are now attached to explanation responses so higher-stakes answers can point back to concrete retrieved context

## Suggested Delivery Order

### Milestone 1
Local chat panel with:
- Ollama connectivity
- simple prompt templates
- explain current recommendation only

### Milestone 2
Local retrieval from current study artifacts plus method traits

### Milestone 3
Natural-language-to-settings with confirmation before apply

### Milestone 4
Local doc retrieval across `research_v2` and `v1_deadline_prototype`

### Milestone 5
Optional literature / internet retrieval with explicit labeling

### Milestone 6
Chart/report interpretation and deeper cross-tab awareness

## Evaluation Plan

### Functional Checks
- Can the assistant explain the current recommendation in grounded language?
- Can it compare the top recommendation against alternatives?
- Can it infer settings from natural language without silently mutating them?
- Can it work across all dashboard tabs?
- Does it degrade gracefully when Ollama is unavailable?

### Quality Checks
- Are explanations aligned with actual study results?
- Are tradeoffs described accurately?
- Are assumptions clearly visible?
- Are local project docs and literature notes being used appropriately?
- Are external internet-sourced claims clearly labeled?

### UX Checks
- Does the panel stay lightweight and non-disruptive?
- Do prompt-template chips help users ask better questions?
- Does the assistant feel useful without becoming a distraction?

Track 9 implementation lock:
- qualitative acceptance criteria are now recorded explicitly for clarity, grounding, and usefulness
- manual prompt suites now exist for recommendation explanation, settings inference, and chart/report interpretation
- local runtime/model-availability viability has been checked on the current machine and recorded as part of the evaluation artifact

Track 10 implementation lock:
- the first release is now gated on explicit confirmation before any dashboard-setting mutation
- explanation release checks now treat the deterministic engine as the only recommendation authority
- the right-side panel is accepted for first release as a Streamlit overlay approximation, with that caveat documented
- known limitations are now recorded rather than left implicit

## Open Decisions
- Whether the first release should include settings-application actions or explanation only
- Whether internet access should be enabled in the first release or kept for a later stage
- Whether to maintain one shared conversation thread across tabs or one thread per context
- Whether to expose model choice in the UI or keep it hidden in config

## Recommended First Build
If we want the highest-value first version with the least risk:
- use a local Ollama model
- explain recommendations only
- ground responses in current study artifacts plus method cards
- add prompt-template chips above the chat box
- keep settings mutation and external retrieval for the next increment
