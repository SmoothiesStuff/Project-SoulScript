########## SoulScript Demo ##########
# Two-NPC LangGraph loop with local LLM, short-term/long-term memory, and a simple Streamlit UI.

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
   - LLM_NUM_GPU = 0 to force CPU, -1 for Ollama auto
4) Start Ollama and pull the model:
   ollama serve
   ollama pull <model>
5) Run Streamlit UI:
   streamlit run soulscript/demo/ui_streamlit_simple.py
6) Run tests:
   python run_tests.py

########## What to Expect ##########
- Two NPCs (see `config.ACTIVE_NPCS`) cycle idle + speak actions.
- Context includes short-term lines, long-term summary, and global tavern facts.
- Logs and state are visible in the Streamlit UI; SQLite persists runtime data.

########## Key Files ##########
- Config knobs: `soulscript/core/config.py` (models, sampling, action set, memory sizes, logging paths)
- LLM client: `soulscript/core/llm.py` (Ollama/OpenAI-compatible client with CPU/GPU toggle)
- Memory & relationships: `soulscript/core/memory.py`, `soulscript/core/relationships.py`
- Scheduler & nodes: `soulscript/core/scheduler.py`, `soulscript/core/graph_nodes.py`, `soulscript/core/policy.py`
- UI: `soulscript/demo/ui_streamlit_simple.py`
- Tests: `soulscript/tests/`
