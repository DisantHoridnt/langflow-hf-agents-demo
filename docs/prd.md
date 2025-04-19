# Product Requirements Document

**Project Name:** *HF‑Native ReAct & Plan‑and‑Execute Agent Components for Langflow*

---

## 1  Background & Problem Statement
Langflow’s built‑in "Agent" components rely on proprietary tool‑calling chat models (e.g., OpenAI). Users who deploy **open‑source LLMs hosted on Hugging Face Inference Endpoints** cannot build agents in a purely low‑code manner—they must fork the codebase or proxy the agent logic behind a micro‑service. We need native components that:

* Implement **LangChain ReAct** and **Plan‑and‑Execute (PnE)** paradigms.
* Accept **any `LLM` object**—particularly those produced by Langflow’s HuggingFace Inference component.
* Consume a **dynamic list of LangChain tools** connected on the canvas.
* Preserve the drag‑and‑drop UX and standard Langflow APIs (schema, metadata, FastAPI docs).

---

## 2  Goal & Success Metrics
| Goal | Metric | Target (v1) |
|------|--------|-------------|
| Drop‑in component discovery | Appears under *Agents* tab after restart | 100 % |
| Open‑source LLM parity | Works with ≥ 3 public HF models (Mistral‑7B, Phi‑3‑mini, Llava‑1.6) | Pass |
| Tool interoperability | Executes ≥ 5 built‑in tools (Calculator, SerpAPI, Python REPL, Vector Search, Bash) | Pass |
| Developer DX | Time to wire *ChatInput ➜ Agent ➜ ChatOutput* flow | ≤ 3 min |
| Stability | ≥ 95 % of runs complete without exception across 50 random prompts/tools | 95 %+ |

---

## 3  Scope
### 3.1 In Scope (v1)
* Two custom components: **`ReActAgentComponent`** and **`PlanExecuteAgentComponent`**.
* Support for Hugging Face Inference `LLM` input.
* Variadic **tools** list input.
* Configurable flags: `verbose`, `max_iterations`, `max_subtask_iterations`.
* README + example `.json` Langflow flow.
* Unit and smoke tests via **pytest**.

### 3.2 Out of Scope (future releases)
* Memory integration (buffer / summary).
* Streaming token output port.
* Advanced caching or rate‑limiting controls.

---

## 4  User Stories
1. **Langflow power user**: *I can drag “ReAct Agent” onto the canvas, connect a HF LLM and tools, and build a working flow without writing Python.*
2. **OSS developer**: *I can import the component via `LANGFLOW_COMPONENT_PATH` and avoid maintaining patches.*
3. **QA engineer**: *I can run `make test` and receive deterministic pass/fail for agent initialisation.*

---

## 5  Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| F‑1 | Components subclass `CustomComponent` for auto‑registration. | P0 |
| F‑2 | Expose inputs: `llm: LLM`, `tools: List[BaseTool]`, `verbose: bool`, `max_iterations` / `max_subtask_iterations`. | P0 |
| F‑3 | `build()` constructs: ReAct ⇒ `initialize_agent(…, AgentType.REACT_DESCRIPTION)`, PnE ⇒ `PlanAndExecute`. | P0 |
| F‑4 | Output must be an `AgentExecutor` consumable by Langflow’s runner / ChatOutput block. | P0 |
| F‑5 | Surface runtime errors via Langflow toast notifications. | P1 |
| F‑6 | Provide icons, colours, and group metadata matching Langflow design tokens. | P2 |

---

## 6  Non‑Functional Requirements
* **Performance:** First token < 15 s on 7‑B model via HF endpoint.
* **Security:** Only calls the HF endpoint specified by credentials; no additional outbound requests.
* **Extensibility:** New inputs (memory, callbacks) can be added with backward‑compatible defaults.
* **License:** MIT, consistent with Langflow core.

---

## 7  Design & Architecture Overview
```text
ChatInput ─▶ ReAct Agent ─▶ ChatOutput
                         ▲
                         └── tools[]

ChatInput ─▶ Plan&Execute Agent ─▶ ChatOutput
                               ▲
                               └── tools[]
```
* **Component Discovery:** Placed under `custom_components/`, loaded via `LANGFLOW_COMPONENT_PATH` env var.
* **LLM Input:** Accepts any subclass of `langchain.llms.base.LLM` (HF Inference output qualifies).
* **Tools Input:** Variadic; Langflow aggregates connected outputs into a Python `list`.
* **Execution:** Returns an `AgentExecutor` (or `PlanAndExecute` wrapper) handled natively by Langflow.

---

## 8  Implementation Plan & Milestones
| Date | Deliverable | Owner |
|------|-------------|-------|
| **T + 0** | PRD approval | Daniel |
| T + 3 d | Component code + README | Dev |
| T + 5 d | Unit tests (mock LLM/tool stubs) | Dev |
| T + 6 d | Manual QA with real HF models | QA |
| T + 7 d | Docs + sample flow | Dev |
| T + 8 d | OSS PR to langflow‑ai/langflow | Dev |
| T + 9 d | Community review & merge | Maintainers |
| **T + 10 d** | v1 GA | Daniel |

---

## 9  Acceptance Criteria
* Components visible in *Agents* tab after restart.
* Example flows run successfully:
  1. ReAct Agent with Calculator + Mistral‑7B.
  2. Plan‑and‑Execute Agent with SerpAPI + Phi‑3.
* All P0 functional requirements verified via CI tests.
* Documentation merged into upstream repo.

---

## 10  Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| HF endpoint latency causing timeouts | High | Document timeout param; suggest local model fallback. |
| Langflow internal API changes | Medium | Pin to Langflow ≥ 0.5.0; CI against nightly. |
| Planner recursion exceeds token budget | Medium | Default `max_subtask_iterations=5`; expose parameter. |
| PR rejected for style / tests | Low | Follow CONTRIBUTING; include full test suite. |

---

## 11  Open Questions
1. Should we add **memory** input in v1 or defer to v1.1?
2. Do we need a dedicated **streaming tokens** output port?
3. Any branding or icon guidelines to align with Langflow’s design system?