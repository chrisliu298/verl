# Browser environments for RL training of LLM web agents

**No single open-source project satisfies all seven requirements today, but BrowserGym comes closest and BrowserAgent has proven the critical Playwright-at-scale pattern.** BrowserGym provides the richest observation space (accessibility tree, HTML, screenshots) with a proper Gymnasium API and arbitrary-URL support, but its sync-only Playwright API prevents native asyncio concurrency. BrowserAgent (TIGER-AI-Lab) demonstrated **64 concurrent Playwright browsers on a 32-CPU machine** integrated with verl-tool, proving the architecture is viable. The practical recommendation is to build a thin async Playwright wrapper that borrows BrowserGym's observation extraction and BrowserAgent's Ray+FastAPI parallelism pattern, targeting verl's AgentLoop interface.

---

## Detailed comparison of every environment evaluated

| Feature | **BrowserGym** | **WebArena** | **browser-use** | **BrowserAgent** | **Notte** | **OSWorld** | **WebVoyager** | **Mind2Web** |
|---|---|---|---|---|---|---|---|---|
| **Playwright-based** | ✅ Sync API | ✅ Sync + Async variant | ⚠️ CDP via Playwright | ✅ Custom Playwright | ✅ Playwright+CDP | ❌ PyAutoGUI | ❌ Selenium | N/A (dataset) |
| **AXTree observation** | ✅ CDP-enriched with bid/coords | ✅ Via Playwright | ❌ Custom indexed DOM | ✅ Playwright axtree | ❌ Custom "navigable map" | ⚠️ OS-level a11y | ❌ | ❌ |
| **HTML observation** | ✅ Full DOM + pruned HTML | ✅ | ❌ | ❌ Text axtree only | ❌ | ❌ | ❌ | ✅ Offline only |
| **Markdown observation** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Screenshot observation** | ✅ + Set-of-Marks | ✅ (VisualWebArena) | ✅ With element highlights | ❌ Text-only | ❌ | ✅ Primary modality | ✅ + SoM | ✅ Offline |
| **Async Python API** | ❌ Sync (Gymnasium) | ⚠️ AsyncScriptBrowserEnv exists | ✅ Fully asyncio-native | ⚠️ Ray+FastAPI server | ✅ Async sessions | ❌ | ❌ | N/A |
| **Concurrent sessions** | ✅ Via AgentLab (ray/joblib, 20-100) | ❌ Manual only | ⚠️ Experimental (~10) | ✅ 64 proven (Ray) | ✅ Via CDP remotes | ✅ Via multiple VMs | ❌ | N/A |
| **Structured tool calls** | ⚠️ Text function strings | ⚠️ Text-based ID actions | ✅ JSON dict actions | ⚠️ Schema-constrained text | ✅ /observe + /step API | ❌ PyAutoGUI strings | ❌ Free-form text | N/A |
| **Gym step/reset API** | ✅ Gymnasium-native | ✅ Custom Gym-like | ❌ Agent.run() loop | ✅ Via verl-tool | ❌ | ✅ Custom Gym-like | ❌ | N/A |
| **Arbitrary URLs / pywb** | ✅ openended mode | ❌ 6 Docker websites | ✅ Any URL | ❌ WebArena Wikipedia | ✅ Any URL | ✅ Any app | ✅ Live websites | ❌ |
| **verl integration** | ❌ | ❌ | ❌ | ✅ Via verl-tool | ❌ | ❌ | ❌ | N/A |
| **Used for RL training** | ✅ ARLAS, statistical study | ✅ WebRL, WebAgent-R1 | ❌ | ✅ SFT+RFT | ❌ | ❌ | ❌ | ❌ |
| **Last active** | Jan 2026 | Active | Feb 2026 | 2025 | Active | Active | Stale | Active |
| **Stars** | ~1.1k | ~1.3k | ~78k | Part of verl-tool | ~2k | ~2k | ~700 | ~800 |

---

## BrowserGym is the richest observation layer but lacks async

BrowserGym from ServiceNow Research is the de facto standard for web agent research, published in TMLR 2025. It models browser interaction as a POMDP with the most comprehensive observation space of any project evaluated. Its accessibility tree representation enriches every DOM element with a unique **bid** (BrowserGym ID), bounding-box coordinates, visibility flags, and clickability annotations — all extracted via Chrome DevTools Protocol. The pruned HTML variant strips non-interactive elements, reducing token count significantly.

The action space uses text-based function-call strings like `click('a1b2c3')`, `fill('a1b2c3', 'hello')`, `goto('https://...')`, `scroll('down')`, and `press('Enter')`. These are not structured JSON tool calls, but the functions are well-defined with typed parameters. The `action_mapping` is pluggable, so converting to structured tool-call format is straightforward — define a JSON schema per action, parse LLM output as JSON, and map to the existing action functions.

**The critical limitation is the sync-only API.** BrowserGym uses Playwright's synchronous API throughout its `env.py`, `observation.py`, and task implementations. Running 128 concurrent sessions via `asyncio.gather()` is impossible within a single process. AgentLab works around this with process-level parallelism (ray/joblib), supporting **20–100 parallel tasks**, but this consumes significantly more memory than async browser contexts would. The `openended` mode accepts any `start_url`, making it directly compatible with pywb by pointing to `http://localhost:8080/{collection}/{url}`.

---

## BrowserAgent proved 64 concurrent Playwright browsers with verl-tool

The BrowserAgent system from TIGER-AI-Lab (arXiv:2510.10666, TMLR 2025) is the most important reference architecture for connecting Playwright browsers to RL training. It runs **64 concurrent Playwright instances on a single 32-CPU machine**, achieving **50+ episodes per minute** — over an order of magnitude faster than serial execution.

The architecture uses a three-layer stack: **verl** (RL training) → **verl-tool** (tool-agent decoupling) → **custom Playwright tool** (browser interaction). A Ray-based tool server fronted by FastAPI handles all browser requests. Each episode gets a unique `trace_id` for session routing, and sessions terminate automatically on stop actions. The observation is a **text-based accessibility tree parsed by Playwright** with rule-based post-processing (merging consecutive text nodes).

BrowserAgent does not use BrowserGym — it implements a custom Playwright wrapper following verl-tool's `BaseTool` interface. Its training pipeline is SFT followed by rejection fine-tuning (RFT), not full on-policy RL like PPO or GRPO. But the environment infrastructure — the Ray+FastAPI server managing dozens of concurrent Playwright browsers — is directly reusable for any RL algorithm. The codebase lives at `github.com/TIGER-AI-Lab/BrowserAgent`.

---

## WebAgent-R1 achieved the strongest RL results with a simpler approach

WebAgent-R1 (EMNLP 2025) represents the current state-of-the-art for end-to-end multi-turn RL training of web agents. It boosted **Qwen-2.5-3B from 6.1% to 33.9%** and **Llama-3.1-8B from 8.5% to 44.8%** on WebArena-Lite using a straightforward M-GRPO algorithm with binary task-success rewards — no reward model, no curriculum, no replay buffer.

The key innovations are architectural: **asynchronous trajectory generation** spawns G independent browser instances per task, each producing a different trajectory for the GRPO group advantage computation. **Dynamic context compression** replaces earlier observations with simplified templates as the trajectory grows, preventing OOM during training. The training infrastructure uses **DeepSpeed ZeRO-3** on 8×A100 80GB GPUs, not verl. Observations are raw HTML content from WebArena's Playwright-based environment.

WebRL (ICLR 2025) from Tsinghua/Zhipu AI used a more complex approach — self-evolving curriculum plus a trained outcome-supervised reward model — to achieve 42.4% with Llama-3.1-8B on the same benchmark. Both papers confirm that **WebArena-Lite is the standard training ground** for web agent RL, with 647 training tasks and 165 evaluation tasks.

---

## verl's AgentLoop provides the right async abstraction

verl's multi-turn agent support (v0.7+) introduces `AgentLoopBase`, an async interface where each rollout coroutine calls an async LLM server and executes tool actions. The `AsyncLLMServerManager` provides least-request load balancing across vLLM/SGLang instances with sticky sessions for multi-turn conversations. All `AgentLoopWorker` coroutines run concurrently via asyncio, and results are gathered only when all prompts in a batch complete.

Custom tools implement `BaseTool` with `async create()` and `async execute()` methods. The tool returns `(observation, reward, info)` per step. Configuration is YAML-driven. The fully async trainer mode supports `AsyncPartialToolAgentLoop` for saving and resuming incomplete rollouts during parameter synchronization — critical for long-horizon browser tasks where a single trajectory may take minutes.

**No direct BrowserGym or Playwright integration exists in verl's core.** The verl-agent extension (github.com/langfengQ/verl-agent, ~1.3k stars) supports WebShop (a simplified text-based e-commerce simulator) but not real browser environments. Agent Lightning from Microsoft wraps verl with an OpenAI-compatible API, enabling any agent framework to be RL-trained, but has no browser-specific examples.

---

## pywb is fully compatible with Playwright via proxy mode

pywb (webrecorder/pywb) serves archived WARC files as a local HTTP server with two replay modes. **Proxy mode** is ideal for RL training: configure Playwright to route traffic through `http://localhost:8080`, and all requests serve deterministic archived content transparently. Each Playwright browser context can be configured with its own proxy settings independently.

```python
context = await browser.new_context(
    proxy={"server": "http://localhost:8080"},
    ignore_https_errors=True  # pywb generates on-the-fly HTTPS certs
)
page = await context.new_page()
await page.goto("http://example.com/")  # Served from WARC archive
```

**No existing work combines pywb with browser agent training** — this would be a novel contribution. The benefits are significant: **deterministic replay** (same WARC → same pages every time) enables reproducible RL episodes, **zero network latency** from local serving, **lightweight pages** (no ads, trackers, or external resources) allowing more concurrent sessions, and **multiple WARC collections** can serve as different training distributions. pywb handles concurrent connections well via gevent, and its URL-rewriting mode uses `wombat.js` for client-side JavaScript URL interception, though proxy mode provides better fidelity for complex SPAs.

---

## What no project provides: markdown observations

**None of the evaluated projects generate markdown representations of web pages.** The standard text-based observation formats are accessibility tree (BrowserGym, WebArena, BrowserAgent) and cleaned/pruned HTML (BrowserGym, WebArena, WebAgent-R1). The closest analog is browser-use's custom indexed DOM tree, which presents elements in a simplified text format. Generating markdown from HTML is technically straightforward using libraries like `markdownify` or `html2text`, but no browser environment integrates this as a first-class observation format. Adding markdown as an observation format to BrowserGym would require a modest modification — pipe the pruned HTML through an HTML-to-markdown converter during observation extraction.

---

## Requirement-by-requirement verdict for each project

| Requirement | **BrowserGym** | **browser-use** | **BrowserAgent** | **WebArena (native)** |
|---|---|---|---|---|
| 1. Playwright + structured tool calls | ⚠️ Playwright yes, text actions (easy to wrap) | ✅ CDP/Playwright + JSON actions | ✅ Playwright + schema actions | ⚠️ Playwright yes, text actions |
| 2. ≥2 observation formats | ✅ AXTree + HTML + screenshots | ❌ Custom DOM only (no axtree) | ⚠️ AXTree only | ✅ AXTree + HTML |
| 3. Async API, 32-128 sessions | ❌ Sync only; process parallelism | ✅ Async native; ~10 tested | ✅ 64 proven via Ray+FastAPI | ⚠️ Async variant exists; untested at scale |
| 4. Sequential within trajectory | ✅ Gym step() is sequential | ✅ Agent loop is sequential | ✅ Trace-ID sequential | ✅ Step() is sequential |
| 5. Multi-turn interaction | ✅ Step/reset loop | ⚠️ Agent.run() (no step API) | ✅ Via verl-tool AgentLoop | ✅ Step/reset loop |
| 6. pywb / arbitrary URLs | ✅ openended mode | ✅ Any URL | ❌ Tied to WebArena Wikipedia | ❌ Tied to 6 Docker sites |
| 7. Gym API + verl integration | ✅ Gymnasium; no verl yet | ❌ No Gym API; no verl | ✅ verl-tool integrated | ✅ Gym-like; no verl |

---

## Recommendation: BrowserGym's observations + async Playwright + verl-tool's architecture

The optimal path requires combining components from multiple projects rather than adopting any single one. Here is the concrete build plan, ordered by decreasing certainty:

**Start from BrowserGym's observation extraction.** Its CDP-based accessibility tree enrichment (bid annotations, coordinate extraction, visibility/clickability flags) and pruned HTML generation are battle-tested across thousands of research experiments. Extract the `browsergym.core.observation` module and adapt it to work with Playwright's async API. This is the highest-value reusable component.

**Use Playwright's async API directly for the browser backend.** Create a thin `AsyncBrowserEnv` class that launches a headless Chromium instance and manages N browser contexts via `asyncio`. Each context represents one RL trajectory session. The BrowserAgent paper proved 64 concurrent contexts on 32 CPUs; with pywb-served archived pages (lightweight, no external requests), **128 contexts is plausible on a 128GB+ / 32-core machine** in headless mode with resource blocking enabled.

**Implement the verl-tool BaseTool interface.** Define `async create()` to initialize a browser context + navigate to the starting URL, and `async execute()` to dispatch structured action dicts (click, fill, goto, scroll, press) and return the new observation + reward. The FastAPI+Ray server pattern from BrowserAgent provides the proven scaling approach if single-process asyncio hits limits.

**Add markdown observations** by piping BrowserGym's pruned HTML through `markdownify` or `html2text`. This is trivial to implement and provides a third text-based observation format alongside axtree and HTML.

**Wire pywb in proxy mode** for deterministic, reproducible training. Configure each Playwright context with `proxy={"server": "http://localhost:8080"}` to serve archived content. Multiple pywb instances or collections can serve different task distributions.

**Estimated effort breakdown:**
- Async port of BrowserGym observation extraction: **2–3 days** (replace sync Playwright calls with async equivalents in `observation.py` and the DOM marking functions)
- Thin async browser environment with step/reset: **1–2 days** (straightforward Playwright async wrapper)
- Structured tool-call action dispatch: **1 day** (map JSON action schema to Playwright async page methods)
- verl-tool BaseTool integration: **1–2 days** (implement create/execute following BrowserAgent's pattern)
- pywb proxy configuration: **< 1 day** (proxy kwarg on context creation)
- Markdown observation format: **< 1 day** (html2text on pruned HTML)
- Testing at 64–128 concurrent sessions: **1–2 days** (memory profiling, stability testing)

**Total: roughly 1–2 weeks** to build a production-quality async browser environment satisfying all seven requirements, integrated with verl for RL training. The alternative — trying to force BrowserGym's sync API into an async RL loop via process-level parallelism — is technically possible (AgentLab does it) but wastes memory, adds IPC complexity, and caps practical concurrency below what native async Playwright contexts can achieve.

## Conclusion

The browser-environment-for-RL space is fragmented: BrowserGym leads on observation quality and API design, browser-use leads on async and popularity, BrowserAgent leads on proven RL integration, and WebAgent-R1 leads on training results. No project unifies all of these. The key insight from this research is that **BrowserGym's observation extraction is the most valuable reusable component** — its CDP-enriched accessibility tree with bid annotations is used by nearly every serious web agent paper — but its sync Gymnasium wrapper is the wrong abstraction for highly parallel RL rollouts. The right architecture is BrowserAgent's Ray+FastAPI tool-server pattern backed by async Playwright contexts, feeding observations into verl's AgentLoop. The pywb integration for deterministic replay training is entirely unexplored in the literature and represents a genuine opportunity for reproducible, scalable web agent RL.