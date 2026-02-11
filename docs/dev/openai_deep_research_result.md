# Open-source Playwright browser environments for RL training of LLM web agents

## Executive assessment

Your requirements combine three properties that rarely co-occur in today’s open-source ecosystems: (i) a **Playwright-first browser controller**, (ii) **LLM-friendly, text-based page observations** (AXTree + HTML/markdown-style representations), and (iii) a **native async Python API** designed for **32–128 concurrent rollouts driven by `asyncio.gather()`**.

From a 2024–2025 research + GitHub scan, the main pattern is:

- The most **research-oriented web-agent environments** (WebArena-family, BrowserGym) are **Gym-like and synchronous**, designed around reproducibility and evaluation rather than asyncio-scale rollout throughput. citeturn10search1turn21view0turn22view2  
- The most **production/agent tooling libraries** (e.g., browser-use, Skyvern) emphasize **async orchestration and concurrent sessions**, but they are often **not Playwright-first “environments”** in the Gym sense, and some are **CDP-centric** (or “Playwright-compatible”) rather than Playwright-as-the-core abstraction. citeturn29search0turn34search2turn34search4turn12search1  
- RL-specific work in 2024–2025 tends to **build on WebArena/BrowserGym** rather than introduce a new open-source Playwright+async environment; the “environment innovation” is typically about *server-side reset*, *container scaling*, or *curriculum/data pipelines*. citeturn11search12turn35view0turn6search18  

**Closest match overall (least adaptation)**: **BrowserGym** (plus AgentLab tooling) because it has: gym-like loop, rich text-friendly observations (DOM + AXTree), and a structured “high-level action set” that already resembles tool calls (click/fill/scroll/press/goto), with an explicit extension point (`action_mapping`) to enforce structured/JSON input. The main gap is your #3: it is **not natively asyncio-based**. citeturn21view0turn19view3turn22view2turn26search13  

## Candidate environments and libraries

The projects below are the ones that surfaced as most relevant for *open-source web-agent browser control* and/or *RL training pipelines*:

- BrowserGym (ecosystem paper + GitHub packages, including open-ended tasks and multiple benchmarks). citeturn21view0turn22view2turn17view0  
- WebArena / VisualWebArena (canonical WebArena repository; VisualWebArena extends it for multimodal tasks). citeturn10search1turn6search20turn2view2  
- AgentLab (primarily an experimentation/benchmark harness; relevant because it targets parallel experiments and integrates multiple benchmarks under a common interface). citeturn17view0turn26search5turn26search13  
- browser-use (async agent tooling with a structured action registry; relevant for your asyncio concurrency requirement, but likely CDP-centric with optional Playwright integration, not a Playwright-first “env”). citeturn29search0turn33view0turn32search1turn34search4  
- Skyvern (open-source workflow automation with a Playwright-compatible SDK; closer to “agent platform” than RL environment). citeturn12search1turn12search19  
- RL/pipeline work that directly ties into browser environments: WebRL (targets WebArena), Go-Browse (uses BrowserGym to implement a web agent for data collection / training), WEBSERV (environment concept for scalable RL rollouts on WebArena-like tasks). citeturn35view0turn6search18turn11search12  
- NNetNav (2024–2025 web-agent data + training; explicitly notes migration to AgentLab + BrowserGym). citeturn37view0  

## Comparison table

Interpretation notes for this table:

- “Structured tool-call interface” is scored **strictly**: *typed/structured function calls or JSON-like tools* score higher than *string DSL parsing*.  
- “Arbitrary URLs / pywb” is scored as: can it navigate to an arbitrary `start_url` without assuming a fixed benchmark domain set, and can you plausibly point it at a local replay proxy. Most projects do not mention pywb explicitly; those entries are marked as “Likely / unverified” unless documented.

| Project | Uses Playwright? | Observation formats (text-first) | Async vs sync API | Concurrent sessions | Structured tool calls (vs free-form parsing) | Arbitrary URLs / pywb | Maintenance / last update signal | Evidence of RL-loop integration |
|---|---|---|---|---|---|---|---|---|
| BrowserGym | **Yes** (actions via Playwright; environment supports Playwright page object) citeturn19view0turn22view2 | **DOM + AXTree** as structured representations; screenshot also available citeturn21view0turn21view3 | **Sync** Gymnasium-style `reset()/step()` loop citeturn21view0turn22view2 | **Yes**, by running multiple env instances; ecosystem emphasizes standardized experimentation (AgentLab) citeturn17view0turn26search13 | **High-level action primitives**: `click`, `fill`, `scroll`, `press`, `goto`, etc., one action at a time; customizable `action_mapping` can parse JSON/CLI-like inputs into safe actions citeturn19view3turn19view0turn21view0 | **Yes** for arbitrary URLs via open-ended task `start_url`; pywb likely works but not documented citeturn22view2 | Repo shown as “updated 2 weeks ago” on org listing (as of Feb 2026) citeturn26search13 | **Yes, indirectly**: used as a base env in web-agent research and data collection (e.g., Go-Browse explicitly uses BrowserGym) citeturn6search18turn17view0 |
| AgentLab | Not a browser env itself; it is an experimentation framework around BrowserGym citeturn17view0turn26search5 | Inherits BrowserGym obs conventions (DOM/AXTree processing pipelines) citeturn17view0turn21view0 | Mostly **sync** / batch experimentation patterns (not an asyncio env) citeturn17view0 | **Designed for parallel large-scale experiments** (explicit motivation) citeturn17view0 | N/A (delegates to BrowserGym action spaces / agents; AgentLab provides agent tooling & trace analysis) citeturn17view0turn26search5 | N/A directly; depends on underlying BrowserGym tasks citeturn22view2 | Org listing shows AgentLab “Updated Feb 10, 2026” citeturn26search13 | Used for large-scale benchmark runs; not a drop-in RL env but can support data/trace generation citeturn17view0turn26search5 |
| WebArena | **Yes** (Playwright in setup; env modeled “like OpenAI Gym”) citeturn10search1turn7search1 | At least **accessibility tree**; README implies text observation can be “html, accessibility tree” citeturn2view1turn7search1 | **Sync** env loop (`reset/step`) shown in README citeturn7search1 | **Possible** by multiple env instances; not designed around asyncio primitives citeturn7search1 | Actions often expressed via helper like `create_id_based_action(...)` (string/DSL flavored, not strictly typed tools) citeturn7search1 | Primarily benchmark-tied (self-hosted sites), but can be pointed at URLs in demos; pywb not documented citeturn26search6turn2view1 | Org listing shows repo “Updated Nov 26, 2025” citeturn26search10 | Widely used in web-agent research; WebRL framework targets WebArena environment citeturn35view0turn7search1 |
| VisualWebArena | Built on WebArena; Playwright implied via WebArena base; designed for multimodal tasks citeturn6search20turn2view2 | Adds **visual observations** (screenshot) to WebArena-style tasks citeturn6search20 | **Sync** evaluation scripts pattern; environment-level async not surfaced citeturn2view2turn7search6 | Similar to WebArena: multi-instance possible but not asyncio-native citeturn2view2 | Same action interfaces as WebArena family (not strict typed tool calls) citeturn2view2 | Demo script explicitly allows `--start_url` on arbitrary websites (suggests arbitrary URL capability); pywb not documented citeturn2view2 | Maintenance not clearly signaled here; BrowserGym integrates VisualWebArena as a packaged benchmark citeturn22view2turn17view0 | Used mainly for evaluation; BrowserGym ecosystem unifies it for agent evaluation citeturn17view0turn22view2 |
| browser-use | Not clearly Playwright-first; docs describe **CDP-centric control** (“direct and full CDP control”), with a “Playwright integration” example citeturn34search4turn34search2turn34search1 | Provides a DOM-oriented representation and supports screenshot/tooling; tool registry includes screenshot/read-content style actions citeturn32search1turn33view0 | **Async-first** (`async def`, `await agent.run()`) citeturn33view0turn32search1 | **Explicit** “Parallel Agents” pattern with multiple `Browser(...)` instances under asyncio citeturn34search6turn33view0 | **Strong**: typed/structured actions via a registry + Pydantic param models; tools implemented as `async def` actions citeturn32search1 | Likely supports arbitrary URLs (navigate/search); pywb plausible if you point navigation at the replay proxy, but not documented citeturn33view0turn27search10 | Org listing shows “Updated Feb 10, 2026” citeturn29search0 | Not an RL env; mostly an agent framework. Useful as an asyncio tool-executor layer, but would require building the RL `step/reset` wrapper yourself citeturn33view0turn32search1 |
| Skyvern | “Playwright-compatible SDK” (agent platform / workflow automation) citeturn12search1turn12search4 | Not clearly AXTree/HTML-first; emphasizes LLM + computer vision workflow automation citeturn12search1 | Likely async/service-based; not a Gym env interface in sources reviewed citeturn12search1turn12search4 | Claims high parallelism as a platform concept; concrete RL-facing concurrency API not evidenced here citeturn12search1turn12search7 | Workflow/action abstraction appears higher-level than “env tool calls”; unclear how “structured” is exposed for RL citeturn12search1turn12search4 | “Works on any website” in product framing; pywb not documented citeturn12search7 | Org page shows repo “Updated Feb 10, 2026” citeturn12search19 | No RL training loop integration evidenced; more of an automation platform citeturn12search1turn12search4 |
| WEBSERV | Proposed as an RL-scalable browser-server environment; paper emphasizes scalable, isolated client-server containers and faster reset/launch citeturn11search12turn10search13 | Paper claims a “compact, site-agnostic browser environment” (details of exact obs format not verified in sources here) citeturn11search12 | Not enough public API detail in sources reviewed to classify as async/sync | Emphasizes **200+ concurrent containers on a single host** (paper claim) citeturn11search12 | Not enough public API detail in sources reviewed | Targets WebArena tasks (shopping CMS, Gitlab), but frames site-agnostic; pywb not mentioned citeturn11search12 | 2025 arXiv/OpenReview listing citeturn11search12turn11search4 | This is explicitly RL-focused (environment designed for RL rollouts), but open-source availability is not confirmed from sources reviewed citeturn11search12 |
| WebRL | RL framework targeting WebArena; provides training scripts, task generation, and interaction/eval instructions for WebArena(-Lite) citeturn35view0 | Inherits WebArena text-modal evaluation setup; obs format details not fully captured in sources here citeturn35view0 | Not an env; a training pipeline around an env citeturn35view0 | RL training is distributed/multinode in scripts (not asyncio browser sessions) citeturn35view0 | Depends on underlying env action interface | Benchmark-tied (WebArena sites) citeturn35view0turn26search6 | Active repo signals unknown in captured snippet; published as a framework with released checkpoints citeturn35view0 | **Yes** (it is an RL training loop), but it is not a reusable Playwright env abstraction itself citeturn35view0 |
| NNetNav | Focus is data + agent training; repo explicitly notes it now uses AgentLab + BrowserGym (therefore inherits BrowserGym environment choices) citeturn37view0 | Depends on BrowserGym/AgentLab pipeline after migration; exact obs options not detailed in snippet citeturn37view0turn21view0 | Not characterized as async env in reviewed snippet | Mentions running agents (`run_agent.py`) and jobs (`--n_jobs`), but concurrency model not specified in reviewed snippet citeturn37view0 | Depends on underlying env/agent prompt+action format | Designed for “any website” usage, and differentiates models for live sites vs WebArena citeturn37view0 | Repo released Nov 2024; major updates Feb 2025 (news section) citeturn37view0 | Yes in the sense of training web agents; environment reuse is via BrowserGym | citeturn37view0turn6search18 |

## Implications for an asyncio RL rollout architecture

### The crux: synchronous Gym environments vs asyncio-scale rollouts

Both BrowserGym and WebArena present **classic synchronous `reset()/step()`** APIs. citeturn21view0turn7search1turn22view2  

For your target pattern—**N independent trajectory coroutines**, each alternating between:
1) async LLM generation (remote call), and  
2) async environment tool execution (browser actions),  

the missing piece is an **async-friendly environment executor**.

BrowserGym *does* provide the key pieces that make this conversion tractable:

- **Tool-call-like action primitives** (e.g., `click`, `fill`, `scroll`, `press`, `goto`) that are explicitly “python function executing playwright code,” and are intended to be executed **one action at a time**. citeturn19view3turn19view0  
- A built-in extension point (`action_mapping`) intended to map user actions into executable code, including the explicit mention that you can define action spaces where actions are “in JSON” and converted into trusted code. citeturn19view0turn19view3  
- Multiple observation components (goal/chat, open tabs, DOM/AXTree/screenshot), which are already structured for multi-turn agent-environment loops. citeturn21view0turn21view3turn22view2  

### Sequential tool execution within a trajectory

Your requirement #4 (“fill then click, not concurrent”) is directly aligned with BrowserGym’s design: its high-level action set is shown as **single-action steps**, and observations include prior action history and errors. citeturn19view3turn19view0  

In contrast, browser-use supports “custom tools” and has a registry where actions can be marked as terminating a sequence, but you would still need to implement the exact “one action per environment step” semantics yourself if you want strict RL-style state transitions. citeturn32search1turn33view0  

### Concurrency across sessions

- browser-use is the clearest “async concurrency” reference point: its own docs show creating multiple `Browser(...)` instances and running them under asyncio, which matches your rollout model. citeturn34search6turn33view0turn32search1  
- BrowserGym and WebArena do not advertise asyncio concurrency; instead, their ecosystem story is about **parallel experiments** via experiment harnesses (AgentLab) or distributed pipelines—useful, but not the same as `asyncio.gather()` over 128 in-process sessions. citeturn17view0turn26search5turn7search1  

## Compatibility with arbitrary URLs and pywb replay

### Arbitrary URLs

Two environment families clearly support arbitrarily chosen URLs:

- BrowserGym’s “openended” environment explicitly takes a `start_url` parameter, and its “Goal/chat messages” design supports interactive chat-style loops (important for multi-turn agents). citeturn22view2turn21view3  
- VisualWebArena’s demo script exposes `--start_url` for running a task on arbitrary sites (not just the benchmark’s hosted domains). citeturn2view2  

This is a strong indicator that both stacks can operate outside hard-coded benchmark sites, even if the *task/evaluator* pieces in WebArena/VisualWebArena are benchmark-specific. citeturn26search6turn6search20turn22view2  

### pywb replay

None of the reviewed sources explicitly document **pywb** support. So any “works with pywb” conclusion is necessarily an inference from “can navigate to arbitrary URLs” and “handles normal HTTP(S) browsing.”

What you should expect in practice (based on how these frameworks work):

- **BrowserGym/WebArena** should be compatible with pywb replay as long as:
  - your replay proxy produces normal HTTP responses,
  - you ensure stable timing/waits (archived pages sometimes have different load behaviors), and
  - any environment-side DOM injection (e.g., adding element IDs like `bid`) does not break replayed scripts. BrowserGym explicitly injects a unique identifier attribute (`bid`) into the DOM/AXTree to enable robust element references. citeturn21view0  
- If you plan to use **Assistive representations** (AXTree/DOM dumps), archives sometimes produce different accessibility trees due to resource loading differences. This isn’t a blocker but can change observation distribution.

## Recommendations and least-adaptation paths

### Best starting point for your exact requirements

**BrowserGym (core + openended) is the closest starting point** for a research-grade RL environment that already has (a) the observations you want and (b) an action space that maps cleanly to structured tool calls. citeturn21view0turn22view2turn19view3turn26search13  

It matches your requirements most directly on:

- Playwright-based interaction primitives (click/fill/scroll/press/goto). citeturn19view3turn19view0  
- Multiple observation formats (DOM + AXTree; screenshot optional). citeturn21view0turn21view3  
- Multi-turn interaction loop (goal/chat messages are first-class in the observation design). citeturn21view3turn22view2  
- Arbitrary URL operation (`start_url` in openended). citeturn22view2  

The main gaps you must build:

- **Native asyncio env API**: BrowserGym’s interface is synchronous Gymnasium. citeturn21view0turn22view2  
- **Strict “structured tool call” IO**: by default, BrowserGym action inputs may be expressed as function-call-like strings (or even raw Python code), but it explicitly supports using an `action_mapping` to restrict and parse formats (including JSON-style action lists) into safe code. citeturn19view0turn19view3  

### Minimal adaptation blueprint for BrowserGym to fit `asyncio.gather()` rollouts

A practical “least rewrite” approach is:

- Treat each BrowserGym env as **a stateful worker object** bound to one trajectory coroutine.
- Wrap `env.reset()` and `env.step()` in `asyncio.to_thread(...)` (or an equivalent thread/process executor) so that your rollouts remain `await`-able while the underlying environment stays sync.
- Keep **one env per thread** (or one env per process) to avoid cross-session bleed. This preserves your sequential-within-session constraint naturally because a trajectory coroutine awaits each step before issuing the next tool call.

This is not as clean as a native Playwright async env, but it is the most direct way to get **32–128 concurrent rollouts** without rewriting the environment itself.

If you can afford a deeper refactor, the “best long-term” approach is to **rebuild the environment executor on Playwright’s async API**, while reusing BrowserGym’s design choices:

- DOM/AXTree extraction strategy + injected stable element IDs (`bid`). citeturn21view0turn21view3  
- High-level action set semantics and single-action-per-step constraint. citeturn19view3turn19view0  

### Notes on integrating with verl specifically

Your environment is inherently **stateful**: tool calls must operate on the same browser session across steps. A key friction point is that verl has had community discussion about browser GUI tools and stateful tool support:

- A verl discussion explicitly proposes adding a “Browser GUI tool” (Playwright/Selenium) and flags “lack of stateful tool support” as a risk. citeturn15view0  
- A related verl issue describes the architecture as treating each tool call as isolated (stateless), and proposes a **session-based tool execution model** to support browser-like stateful workflows. citeturn16view0  

So, even if your environment is perfect technically, you likely need one of:

- A **session-aware tool abstraction** (one tool session per rollout) consistent with the session model described in the verl issue, citeturn16view0  
- Or, run the browser environment outside verl’s core tool abstraction (i.e., your own async rollout loop that calls verl’s LLM server + your env executor), using verl mainly for optimization/training rather than being the orchestration layer for stateful tools.

### When browser-use is the better fit

If your overriding constraint is **true asyncio-native concurrency**, browser-use is the most “off-the-shelf” async executor among the reviewed options:

- Its quickstart and tool system are explicitly async. citeturn33view0turn32search1  
- It has explicit docs for running multiple agents in parallel with separate browser instances. citeturn34search6  
- Its action interface is strongly structured via a registry + typed parameter models. citeturn32search1  

But it likely fails your requirement #1 as stated, because it positions its low-level automation as **CDP-first** (“direct and full CDP control”), with Playwright as an integration path rather than the core. citeturn34search4turn34search2  

So browser-use is best viewed as:
- a reference implementation for **async tool execution, concurrency, and typed action schemas**, and
- potentially a component to reuse, but not a ready-made Playwright-based RL env.

### RL-specific environments to watch

Two research directions from 2024–2025 are particularly relevant if you want *true* large-scale RL rollouts (beyond what a single Playwright host process comfortably supports):

- **WEBSERV** proposes a browser-server environment explicitly targeting scalable RL training/eval, emphasizing deterministic reset and high throughput (paper claims 200+ concurrent containers on a single host). citeturn11search12  
- **WebRL** is an open RL training framework targeting WebArena/WebArena-Lite, which is valuable as a reference for how RL pipelines structure web-agent trajectories and rewards around these environments. citeturn35view0  

The main caveat is that neither of these, from the sources reviewed, clearly provides the exact **Playwright+async Python env abstraction** you want as a reusable “drop into verl” component. citeturn11search12turn35view0