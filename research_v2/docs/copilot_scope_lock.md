# Research Workbench Copilot — Scope Lock

## Status
Track 0 is locked for the first implementation pass. This document defines what the copilot is, what it is not, and how the first release should behave.

## Product Identity
The copilot is an assistant layer on top of the dashboard.

It is:
- a conversational explainer
- a grounded interface to the current study evidence
- a helper for translating natural-language constraints into dashboard settings
- a lightweight decision-support companion that stays subordinate to the deterministic engine

It is not:
- a replacement recommender
- a freeform research chatbot with unrestricted claims
- an autonomous control layer that silently changes the dashboard
- a substitute for the underlying study artifacts or statistical outputs

## First-Release Scope
The first release should support three user-facing modes:

### 1. Explain Recommendation
Allowed:
- explain why the current recommendation fits the current settings
- describe the main tradeoffs against nearby alternatives
- explain the method conceptually using grounded evidence plus method knowledge

Not allowed:
- invent stronger conclusions than the evidence supports
- imply that the LLM chose the method independently of the engine

### 2. Natural Language To Settings
Allowed:
- interpret freeform hardware/resource/goal descriptions
- propose structured dashboard settings
- show assumptions before applying anything

Not allowed:
- silently change sliders, toggles, dropdowns, or engine state
- treat vague user statements as precise facts without surfacing assumptions

### 3. Chart / Result Interpretation
Allowed:
- explain what current plots, tables, and report statements mean
- clarify terminology such as top cluster, forgetting, Pareto candidate, and effect size
- connect visible results to relevant study context

Not allowed:
- fabricate unobserved trends
- present descriptive plots as if they were inferential findings when they are not

## Interaction Model Lock
The copilot will use a hidden right-edge reveal pattern.

Locked behaviors:
- a slim hover target is visible on the right edge
- hover or click reveals a slide-in panel
- the panel remains available across all dashboard tabs
- collapse should hide the panel without losing conversation state
- the panel should feel secondary and non-disruptive to the research workspace

## First-Release Action Policy
The first release is **chat-first with lightweight optional actions**, not a fully action-driven agent.

Locked action policy:
- explanation responses are always safe to show directly
- inferred settings may be proposed in the response
- applying inferred settings must require explicit user confirmation
- there will be no autonomous multi-step actions in the first release

## Trust Policy Lock
The copilot must operate under the following trust rules:

### Recommendation Authority
- the deterministic recommendation engine is the sole recommendation authority
- the copilot may explain or contextualize the recommendation
- the copilot may not override the recommendation source of truth

### Evidence Hierarchy
Use sources in this order whenever possible:
1. current structured study artifacts
2. local project notes, literature surveys, and markdown/doc files in this workspace
3. clearly labeled external internet-backed sources
4. model prior knowledge only for general conceptual glue, never for unsupported project claims

### Response Boundaries
The copilot may:
- explain
- compare
- summarize
- infer likely settings with visible assumptions
- ask targeted clarifying questions when needed

The copilot may not:
- invent empirical results
- invent significance claims
- present literature claims as if they came from the current study
- silently mutate dashboard state
- blur the distinction between evidence and inference

## Prompt Template Policy
Prompt-template suggestions above the chat box are approved for the first release.

They should:
- help users ask better grounded questions
- be short and practical
- reflect the active tab when possible
- stay suggestive rather than forcing a workflow

Approved first-release prompt-template categories:
- explain recommendation
- compare alternatives
- infer settings from a natural-language description
- explain a chart
- explain a tradeoff
- use project notes or literature context to explain a method

## Conversation Policy
- maintain one shared conversation thread across the dashboard for the first release
- include lightweight contextual awareness of the active tab
- avoid creating separate hidden threads per tab in the initial version

## Model Policy
- use a local Ollama model for the first release
- keep model selection configurable
- default model selection is deferred to the model-integration track, but local inference is mandatory for the initial build

## Non-Goals For The First Release
- autonomous tool use outside dashboard-approved actions
- automatic internet retrieval by default
- long-memory personalized profiles
- agentic literature review workflows
- replacing the dashboard’s main comparison and report surfaces with chat

## Release Gate For Track 0
Track 0 should be considered complete when:
- the product identity is locked
- the first-release modes are locked
- the panel interaction model is locked
- the action policy is locked
- the trust policy is locked
- prompt-template behavior is approved for the first release
