########## Vision ##########
SoulScript aims to give NPCs a believable semblance of a soul. Each agent mixes authored truth with evolving perceptions so relationships shift as the town breathes.

########## Utility ##########
For players SoulScript delivers cozy playthroughs where tavern regulars remember conversations, form opinions, and surprise you each run. For developers it offers configurable agents with JSON seeds, a LangGraph loop, and a live Streamlit monitor for rapid tuning.

########## Core Concepts ##########
Truth vs self vs others: profiles define canonical truth, self perception tracks bias, and perception of others lives in SQLite with trust, affinity, and trait guesses. Traits use ten bipolar axes (kindness, bravery, extraversion, ego, honesty, curiosity, discipline, patience, optimism, generosity) each clamped to −100..100. Conversations retain the last five lines per pair with a first person synopsis that updates on prune. Runtime state persists in SQLite so restarts keep memories and relationships aligned with truth drift rules.

########## LangGraph Flow ##########

    +------------------+           +----------------+
    |   NPC Truth      |           |  Player / NPC  |
    | self & others    |           |    Event       |
    +---------+--------+           +-------+--------+
              \                          /
               \                        /
                v                      v
           +----------- Build Context -----------+
           | gather state & memory, load sqlite  |
           +----------------+--------------------+
                            |
                            v
                 +-----------------------+
                 |   LangGraph Node      |
                 | policy + tools + LLM  |
                 +-----------+-----------+
                             |
                             v
                  +--------------------+
                  |  LLM select_action |
                  +--------------------+
                             |
                             v
                 +------------------------+
                 | Execute action & log   |
                 | update memory, sqlite  |
                 +------------------------+
                             |
                             v
                    (loop to next tick)

########## File Structure ##########
- soulscript/
  - __init__.py
  - core/
    - config.py
    - db.py
    - graph_nodes.py
    - llm.py
    - memory.py
    - npc.py
    - policy.py
    - relationships.py
    - scheduler.py
    - tools.py
    - types.py
  - demo/
    - soulscript_demo.py
    - ui_streamlit.py
    - seeds/
      - npc_*.json (20 seed profiles)
      - README.md
  - engine_api/
    - server.py
  - migrations/
    - README.md
  - tests/
    - test_traits.py
    - test_memory.py
    - test_relationships.py
    - test_policy.py
- .env.example
- soulscript/requirements.txt
- README.txt

########## Data Layer ##########
Static truth, self perception seeds, and initial relationships live in JSON under `soulscript/demo/seeds/`. SQLite stores runtime mutations in four tables: `npc_state` (self trait drift and mood), `relationships` (trust, affinity, trait perception, summary), `conversations` (last five lines per pair), and `event_log` (tick by tick actions). Migrations directory holds future schema evolution notes.

########## LLM Integration ##########
SoulScript speaks to Ollama through the OpenAI client.
1 Install Ollama from https://ollama.ai/download.
2 Pull the model: `ollama pull llama3:8b`.
3 Optional interactive test: `ollama run llama3:8b`.
4 Copy `.env.example` to `.env` and tweak if needed:
   - `LLM_BASE_URL=http://localhost:11434/v1`
   - `LLM_MODEL=llama3:8b`
`soulscript/core/llm.py` loads the .env file, calls the Ollama endpoint, and falls back to the deterministic stub when the service is offline.

########## Development Setup ##########
1 Create and activate a Python 3.11 virtual environment.
2 `pip install -r soulscript/requirements.txt`.
3 `pip install -e soulscript` (optional; `.env` loader makes it unnecessary but handy for IDEs).
4 Copy `.env.example` to `.env` in the repo root.
5 Start Ollama in another terminal (`ollama serve` or `ollama run llama3:8b`).
6 Launch the UI: `streamlit run soulscript/demo/ui_streamlit.py`.

########## NPC Seeds ##########
Twenty authored NPCs ship with the demo: ten locals with interlinked starting perceptions and ten visitors who arrive fresh. Each seed records truth traits, self bias, optional relationship summaries, inventory, and schedules so the simulation starts with personality-rich edges.

########## Roadmap ##########
- [x] Truth vs perception design
- [x] SQLite schema designed
- [x] Ollama integration (basic)
- [ ] Gossip/factions
- [ ] Engine bindings
- [ ] Procedural quests and story arcs

