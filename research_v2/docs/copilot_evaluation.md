# Research Workbench Copilot — Evaluation

## Purpose
This document closes the evaluation track for the dashboard copilot. It records:
- qualitative acceptance criteria
- prompt suites for manual evaluation
- the intended pass criteria for clarity, grounding, and usefulness
- the local runtime/availability check for the current machine

## Qualitative Acceptance Criteria

### Clarity
- The answer should respond to the user’s actual intent, not a nearby intent.
- Hardware prompts should return settings guidance, not recommendation essays.
- Chart prompts should explain the named chart family when it can be inferred from the user text.
- Explanations should be understandable without reading source files or internal implementation details.

### Grounding
- Recommendation answers must preserve the deterministic engine as the source of truth.
- Recommendation explanations must stay tied to current study artifacts and retrieved evidence.
- Settings inference must stay rule-first and should surface assumptions explicitly.
- Chart explanations must stay tied to the dataset and chart family, not generic filler text.
- If evidence is thin or mixed, the answer must say so.

### Usefulness
- The answer should help the user take the next step: compare, interpret, or apply settings.
- Prompt-template chips should produce answers that are materially better than vague freeform prompting.
- The assistant should degrade safely when the local model is unavailable rather than failing silently.

## Manual Prompt Suites

### Recommendation Explanation
Use these prompts while varying datasets and current request settings:
- `Why is this method recommended right now?`
- `Compare the current recommendation with the next two alternatives.`
- `Use project notes to explain why this method works under these settings.`
- `If I care more about forgetting than runtime, what changes conceptually?`

Expected pass behavior:
- references the current winner and nearby alternatives
- keeps recommendation authority with the deterministic engine
- distinguishes empirical evidence from conceptual interpretation

### Natural-Language To Settings
Use these prompts:
- `I have a GT210.`
- `I have a GTX 1650 laptop with 16 GB RAM.`
- `CPU only, no retraining, I care about retention.`
- `I want something lightweight for Split CIFAR-100 and I can retrain overnight.`
- `I want to use Fashion MNIST with a ViT backbone.`

Expected pass behavior:
- converts the request into settings suggestions, not a recommendation essay
- states assumptions explicitly
- flags out-of-scope datasets/backbones clearly
- requires explicit apply before mutating dashboard controls

### Chart And Report Interpretation
Use these prompts:
- `Explain accuracy vs forgetting chart.`
- `Explain accuracy vs estimated memory chart.`
- `Explain the score breakdown.`
- `What does top cluster mean in this view?`
- `Explain the significance matrix.`
- `Explain the effect-size matrix.`
- `What is the rank slope chart saying?`
- `Explain the robustness chart.`

Expected pass behavior:
- identifies the chart family when named explicitly
- explains what the axes or matrix mean
- uses dataset-aware comparisons where appropriate
- does not confuse chart interpretation with settings inference

## Failure Conditions
- A hardware/settings prompt is answered as a recommendation explanation.
- A chart prompt is answered as a settings suggestion.
- The copilot silently mutates dashboard controls.
- The copilot claims statistical significance without retrieved support.
- The copilot cites external/background knowledge as if it were a study result.

## Local Runtime And Availability Check
Current local check on this machine:
- `ollama list` succeeded
- installed local models include:
  - `qwen2.5:7b-instruct`
  - `llama3.1:8b-instruct-q4_K_M`
- copilot runtime status check returned:
  - `available=True`
  - `model_available=True`
  - selected model: `qwen2.5:7b-instruct`

Interpretation:
- local Ollama-backed usage is viable on this machine for the first release
- offline fallback remains important because availability can still change between sessions

## Latency / UX Notes
- The visible reply path now shows the user message immediately and streams the assistant response into the chat panel.
- The current “thinking” indicator is UI-side and loops while the response is being prepared.
- The current visible streaming is UI-driven; the backend Ollama call is still single-shot rather than token-streaming from the model transport.

## Current Evaluation Verdict
- Recommendation explanation: acceptable for first release
- Natural-language-to-settings: acceptable for first release with heuristic caveats
- Chart interpretation: acceptable for the named chart families currently covered in the knowledge base
- Local model viability: acceptable for first release on the current machine
