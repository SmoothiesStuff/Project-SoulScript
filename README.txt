########## SoulScript Demo ##########
Six-NPC LangGraph loop with local LLM, short-term/long-term memory, and dual Streamlit UIs.

#### Ollama setup in PowerShell ####
# Stop any existing runner
taskkill /IM ollama.exe /F

# Start a fresh session with CPU + smaller context
$env:OLLAMA_NUM_GPU = "0"
$env:OLLAMA_CONTEXT_LENGTH = "1024"   # try 512 if 1024 still fails
ollama serve


######### Run the Streamlit app #########
# from the repo root
streamlit run soulscript/demo/ui_streamlit_simple.py
streamlit run soulscript/demo/ui_streamlit.py

########## Quick Start ##########
1) Create/activate venv (uv shown):
   uv venv --python 3.11 .venv
   .\.venv\Scripts\Activate.ps1
2) Install deps:
   uv pip install -r soulscript/requirements.txt
3) Configure LLM in `soulscript/core/config.py`:
   - LLM_MODEL_NAME = a model you can load locally (e.g., "phi3:mini" or "llama3.2:1b")
   - LLM_BASE_URL = http://localhost:11434/v1
   - LLM_API_KEY = ollama
   - LLM_NUM_GPU = 0 to run CPU-only (set -1 to let Ollama auto-pick GPU if you want)
4) Start Ollama and pull the model:
   ollama serve
   ollama pull <model>
5) Run Streamlit UI:
   streamlit run soulscript/demo/ui_streamlit_simple.py


########## Architecture Overview ##########
- Build: `soulscript/demo/soulscript_demo.py` loads JSON seeds into `NPC` containers, registers them with `SimulationScheduler`, bootstraps relationships, and initializes SQLite-backed memory.
- Tick: `SimulationScheduler.step` (in `soulscript/core/scheduler.py`) decays relationships, shuffles active NPCs, runs node stack, collects `Decision` objects, and advances per-pair chat sessions.
- Nodes: `soulscript/core/runtime.py` defines LangGraph-style nodes (`node_idle`, `node_speak`, `node_decide_join`, `node_adjust_rel`) that call the policy layer, build LLM context, validate outputs, and apply mood/relationship effects.
- Policy: `soulscript/core/runtime.py` lists allowed `ActionType`s per node, prefetches tool data (inventory/location/schedule), and injects current targets based on focus/nearby NPCs.
- LLM & Validator: `soulscript/core/llm.py` wraps an Ollama/OpenAI-compatible client; `_validated_decision` in `runtime` reruns or falls back when the model emits junk or non-speak actions during sessions.
- Memory & Relationships: bundled in `soulscript/core/runtime.py` (ConversationMemory + RelationshipGraph) streaming into SQLite, summarizing, decaying, and nudging trait perception.
- UI: Streamlit UIs in `soulscript/demo/ui_streamlit.py` (tavern map) and `soulscript/demo/ui_streamlit_simple.py` (focused pair). Both rely on the same runtime and logging.

########## LangGraph Agent Flow (Streamlit demo) ##########
- Scheduler loop: `SimulationScheduler.step(world_context)` drives LangGraph-style nodes per NPC. `world_context` (from Streamlit) carries `gathering_spot`, `table_map`, `active_pairs`, optional `focus_target`, `allowed_npcs`, `defer_effects`, and sentiment hints from mood. The scheduler decays relationships, shuffles NPCs, prepares per-NPC context (nearby roster, session locks), then calls nodes.
- Context aggregation: `_build_llm_context` (in `runtime`) merges NPC profile (name/backstory/motivation/trait summaries + trait-scale legend), NPC state (mood + truth trait vector), short-term lines (last 4), recent thread, last partner line, long-term summary, global facts, interaction tone, partner id, and tool outputs (inventory/schedule/location) plus any `world_context` fields. This dictionary is the input to the LLM agent.
- Memory: `ConversationMemory.record` persists each `speak` line to SQLite with pruning (`INTERACTION_LINES_KEEP`). `context_bundle` returns `{short_term, long_term, global_facts}`; long-term summaries are built after `LONG_TERM_SUMMARY_TRIGGER` lines via `_summarize_pair`. `finalize_conversation` re-summarizes with the LLM summarizer for both directions, applies relationship deltas, and clears the history for that pair.
- Relationship graph: `RelationshipGraph` stores trust/affinity/trait perception per directed edge, decays toward neutral each tick, and clamps using soft limits. `_toward_truth_delta` nudges perceived traits toward the target's truth when they speak. Deferred payloads in `pending_effects` carry `{source_id,target_id,line,timestamp,trait_deltas}` until applied.
- Streamlit (simple): `_run_tick` passes `world_context={"focus_target": npc_b, "defer_effects": conversation_active, "allowed_npcs":[npc_a,npc_b]}` into the scheduler. While `conversation_active=True`, `pending_effects` accumulate; pressing "End conversation" calls `finalize_conversation` to apply summaries/relationship updates and reset the per-pair transcript.
- Streamlit (full tavern): `_maybe_run_tick` refreshes seating/active pairs, then calls `scheduler.step` with `table_map` + `active_pairs`. `_update_active_pairs` locks pairs sharing a table and marks `defer_effects` via `_build_npc_context` session logic; `_finalize_pair` runs `finalize_conversation` for a table when someone leaves or a session expires. UI panels pull live state from `scheduler.npcs`, `relationships`, and `active_conversations`.

########## Logging ##########
- Human-readable log: `logs/soulscript.log` (configurable via `LOG_TEXT_*` in `soulscript/core/config.py`) captures LLM inputs/outputs, dialogue lines, and summary/relationship updates. Truncated to keep the latest entries readable.
- SQLite-backed log: still stored via `soulscript/core/db.py` helpers for conversations, relationships, and event history.


########## Graph / Node Flow ##########
1) World build: seeds -> `NPCTruth` -> `NPC` -> `SimulationScheduler` registers NPCs, stores initial relationship edges, loads persisted mood/self-perception.
2) Tick start: `scheduler.step(world_context)` decays trust/affinity/traits toward neutral, shuffles allowed NPCs, and prepares per-NPC context (nearby roster, focus target, session turn locks, sentiment).
3) Node execution:
   - `node_idle`: runs when no active session; policy usually only allows `idle`; LLM may return `idle` and mood recovers slightly.
   - `node_speak`: always runs; requires a partner; validator enforces `speak` during active sessions; records dialogue, updates summaries/relationships or defers effects when `defer_effects` is set.
   - Optional nodes exist for `decide_join` and `adjust_relationship` (not wired in the simple UI loop).
4) Memory + summary: `ConversationMemory.record` persists lines and returns recent history; summaries are derived when enough lines exist or on conversation finalize.
5) Relationship effects: immediate or deferred adjustments apply trust/affinity deltas, trait perception nudges toward partner truth, and summary updates; decay runs each tick.
6) Session finalize: `SimulationScheduler.finalize_conversation` applies deferred effects for a pair, summarizes both directions, bumps mood, clears sessions, and resets per-pair transcripts.


########## Agent Components (with sample I/O) ##########
- Policy filter (`soulscript/core/runtime.py`): decides targets and allowed actions.
  Sample I/O:
  ```
  apply_policy(npc_bren, "speak",
    {"nearby_npcs": ["npc_cassia"], "focus_target": "npc_cassia", "session_turns_left": 2})
  -> allowed_actions = [
       Action(action_type=ActionType.SPEAK, target_id="npc_cassia"),
       Action(action_type=ActionType.IDLE, target_id="npc_cassia")
     ]
     tool_outputs = {"inventory": [...], "schedule": {...}}
  ```

- LLM decision + validator (`soulscript/core/runtime.py` + `soulscript/core/llm.py`): builds context bundle (profile, mood, short/long-term memory, global facts, last partner line), calls Ollama/OpenAI, and reruns/falls back if invalid.
  Sample I/O:
  ```
  _validated_decision(llm_client, "npc_cassia", llm_context, allowed_actions,
    {"session_turns_left": 3, "last_partner_line": "How was the road?"})
  Model raw -> {"action":"speak","line":"Quiet tonight, but the stew is warm.","reason":"Keep chat light."}
  Validator -> Decision(selected_action=speak@npc_marek, dialogue_line="Quiet tonight, but the stew is warm.", confidence=0.6)
  (If model returned non-speak or empty line, it retries once then falls back to a filtered speak line.)
  ```

- Conversation memory (`soulscript/core/runtime.py`): records and summarizes per pair with SQLite backing.
  Sample I/O:
  ```
  record("npc_anya","npc_bren","npc_anya","Nice night by the hearth.", timestamp)
  -> [ConversationItem(ts=..., speaker_id="npc_anya", text="Nice night by the hearth.")]
  context_bundle("npc_anya","npc_bren", existing_summary="")
  -> {"short_term": ["npc_anya: Nice night by the hearth."],
      "long_term": "",
      "global_facts": ["You are in a cozy fantasy tavern.", ...]}
  ```

- Relationship graph (`soulscript/core/runtime.py`): directed trust/affinity/trait perception with decay and summaries.
  Sample I/O:
  ```
  adjust_relation("npc_anya","npc_bren", trust_delta=2, affinity_delta=2,
                  trait_deltas={"kindness":1}, summary="I caught up with npc_bren.", timestamp=now)
  -> RelationshipEdge(trust=2, affinity=2, traits.kindness=1, summary="I caught up with npc_bren.")
  decay_all(now + 60s) -> trust/affinity drift toward neutral, traits drift toward 0.
  ```

- Nodes (`soulscript/core/runtime.py`): orchestrate policy + LLM + effects per state.
  Sample I/O:
  ```
  node_idle(npc, memory, rel_graph, llm, ctx={"nearby_npcs": ["npc_cassia"]})
  -> Decision(action=idle, reason="Resting", dialogue_line=None); mood +1

  node_speak(npc, memory, rel_graph, llm,
    ctx={"focus_target":"npc_cassia","target_truth":TraitVector(...),"defer_effects":True,"pending_effects":[]})
  -> Decision(action=speak@npc_cassia, line="Evening, need anything mended?")
  -> pending_effects append trust/affinity and trait deltas until finalize_conversation.
  ```

- Node inputs/outputs at a glance:
  - `node_idle(npc, memory, relationships, llm, context)` -> returns `{"decision": Decision, "tool_outputs": {...}}`; side effects: mood +1 when idling.
  - `node_speak(...)` -> `{"decision": Decision, "tool_outputs": {...}}`; side effects: records line to memory, queues relationship deltas in `pending_effects` when deferring, immediate summary/log update otherwise, mood +1.
  - `node_decide_join(...)` -> `{"decision": Decision, "tool_outputs": {...}}`; side effects: logs `join:<spot>` and mood +1 on join.
  - `node_adjust_rel(...)` -> `{"decision": Decision, "tool_outputs": {...}}`; side effects: direct trust/affinity adjustments from sentiment.

- Scheduler (`soulscript/core/scheduler.py`): tick orchestrator and session manager, importing all runtime helpers from `soulscript/core/runtime.py`.
  Sample I/O:
  ```
  scheduler.step({"allowed_npcs":["npc_anya","npc_bren"],"focus_target":"npc_bren","defer_effects":True})
  -> [Decision(idle@npc_anya), Decision(speak@npc_bren), Decision(idle@npc_bren), Decision(speak@npc_anya)]
  finalize_conversation() -> applies deferred effects, summarizes, clears session/history for the pair.
  ```

- LLM summarizer (`soulscript/core/llm.py`): refines long-term summaries after chats.
  Sample I/O:
  ```
  summarize_conversation("I often see Bren.", ["npc_anya: Quiet night.", "npc_bren: Storm coming."],
                         source_id="npc_anya", target_id="npc_bren", trust=4, affinity=4)
  -> "Bren mentioned a storm coming; I should check supplies soon."
  ```

- Tools + filters (`soulscript/core/runtime.py`): lightweight grounding helpers.
  Sample I/O:
  ```
  style_and_lore_filter("What the heck is that?") -> "What the ... is that?"
  location_of(profile) -> "tavern_common" (morning schedule slot)
  ```


########## What to Expect ##########
- Two NPCs (see `config.ACTIVE_NPCS`) cycle idle + speak actions in short chat sessions; trust/affinity/trait perception update per talk.
- Context fed to the LLM includes short-term lines, long-term summaries, global tavern facts, schedules, inventory, and last partner line.
- Streamlit UI shows NPC cards, pair perceptions, and transcript; SQLite persists mood/self-perception, relationships, conversations, and event log between runs.


########## Key Files ##########
- Config knobs: `soulscript/core/config.py` (models, sampling, action set, memory sizes, logging paths)
- LLM client: `soulscript/core/llm.py` (Ollama/OpenAI-compatible client with CPU/GPU toggle and fallback stub)
- Runtime bundle: `soulscript/core/runtime.py` (nodes, policy, tools, memory, relationships, relationship updates, text logging)
- Scheduler: `soulscript/core/scheduler.py`
- NPC container: `soulscript/core/npc.py`; types/enums in `soulscript/core/types.py`; helpers in `soulscript/core/agentic_helpers.py`
- UI: `soulscript/demo/ui_streamlit.py` (primary) and `soulscript/demo/ui_streamlit_simple.py` (lightweight)
- Seeds: `soulscript/demo/seeds/` (6 simplified NPCs)
