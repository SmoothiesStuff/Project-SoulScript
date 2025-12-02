from __future__ import annotations
import os

########## Core Config ##########
# Houses runtime constants for the SoulScript tavern demo.

########## Variable Controls ##########
# All tweakable knobs live here so you can tune the demo without code changes.

# LLM 
LLM_PROVIDER: str = "openrouter"  
#LLM_PROVIDER: str = "ollama"  

# Ollama defaults (CPU-only)
LLM_MODEL_NAME = "phi3:mini"  # small, 
#LLM_MODEL_NAME = "llama3.2:1b"             # alt tiny llama
#LLM_MODEL_NAME = "llama3.2:3b"            # slightly larger llama
#LLM_MODEL_NAME = "deepseek-r1:7b"         # heavier
LLM_BASE_URL: str = "http://localhost:11434/v1"
LLM_API_KEY: str = "ollama"
LLM_NUM_GPU: int = 0  # CPU only by default; set -1 for Ollama auto selection / causes lots of cuda problems
OLLAMA_OPTIONS: dict = {"num_gpu": 0, "gpu_layers": 0}

# OpenRouter defaults (override via env OPENROUTER_API_KEY when possible)
#LLM_OPENROUTER_MODEL: str = "x-ai/grok-4.1-fast:free"  
#LLM_OPENROUTER_MODEL: str = "qwen/qwen-2.5-72b-instruct:free"
#LLM_OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct:free"
LLM_OPENROUTER_MODEL: str = "openai/gpt-4o-mini"
LLM_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
LLM_OPENROUTER_API_KEY: str = os.getenv(
    "OPENROUTER_API_KEY",
    "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",
)
if LLM_PROVIDER.lower() == "openrouter" and LLM_OPENROUTER_API_KEY:
    os.environ.setdefault("OPENROUTER_API_KEY", LLM_OPENROUTER_API_KEY)

# Sampling per node/tool
ACTION_TEMPERATURE: float = 0.4
ACTION_TOP_P: float = 0.9
ACTION_MAX_TOKENS: int = 256

SUMMARY_TEMPERATURE: float = 0.5
SUMMARY_TOP_P: float = 0.9
SUMMARY_MAX_TOKENS: int = 120

VALIDATOR_TEMPERATURE: float = 0.2
VALIDATOR_TOP_P: float = 0.9
VALIDATOR_MAX_TOKENS: int = 64
RESPONSE_VALIDATION_ENABLED: bool = False
RESPONSE_VALIDATION_MAX_ATTEMPTS: int = 3

RELATIONSHIP_UPDATER_TEMPERATURE: float = 0.3
RELATIONSHIP_UPDATER_TOP_P: float = 0.9
RELATIONSHIP_UPDATER_MAX_TOKENS: int = 80

# Prompts per node/tool 
ACTION_SYSTEM_PROMPT: str = (
    "You decide NPC actions in a tavern sim. "
    "Keep dialogue natural; should match the NPC "
    "Reflect trait scale  in tone (axes show -100..100). "
    "Trust and affinity run from -100 (worst) to 100 (best), strangers start at 0. "
    "Reply directly to the partner's last line in one short sentence (<=30 words). "
    'Return only JSON: {\"action\": <number or \"speak\">, \"line\": \"...\", \"reason\": \"...\"}. '
    "Do not include any other fields or text. Do not write essays, tasks, or instructions."
)

SUMMARY_SYSTEM_PROMPT: str = (
    "You are updating a first-person relationship summary about the partner NPC. "
    "Keep it to one or two short sentences. "
    "Only mention details about the partner, not yourself. "
    "It should serve as a memory of the person as percieved"
)

VALIDATOR_SYSTEM_PROMPT: str = (
    "Validate the NPC action JSON. Ensure it matches allowed actions, uses speak when required, "
    "and has a short, natural line. Keep outputs lean; reject meta or prompts."
)

RELATIONSHIP_UPDATER_PROMPT: str = (
    "Given a conversation summary and trait similarity, produce small trust/affinity deltas based on the conversation"
    "in the range -3..3 that reflect rapport, mood, and conversational energy. Do not be too kind."
)

# NPC and interaction pacing
ACTIVE_NPCS: list[str] = ["riven", "mara", "vex", "sena", "thorne", "lyra"]
ACTION_SET: list[str] = ["idle", "speak"]  # restrict allowed actions for predictability
TICK_INTERVAL_SECONDS: float = 1.0
MAX_TICKS_PER_RUN: int = 300
INTERACTION_LINES_KEEP: int = 8
LONG_TERM_SUMMARY_TRIGGER: int = 4  # summarize after this many lines per pair

# Memory and global knowledge
GLOBAL_KNOWLEDGE: list[str] = [
    "You are in a cozy fantasy tavern.",
    "Half the people here are locals and half are visitors.",
    "Keep dialogue short and friendly.",
]

# Logging and debug
DEBUG_VERBOSE: bool = True # breaks out individual llm calls / responses
DEFAULT_DIALOGUE_EXPORT: str = "soulscript/demo/run_logs"
DEFAULT_DIALOGUE_FILENAME_TEMPLATE: str = "run_{timestamp}.jsonl"
DEFAULT_EVENT_LOG_EXPORT: str = "soulscript/demo/run_logs/events_{timestamp}.csv"
LOG_TEXT_ENABLED: bool = True  # toggle human-readable run log
LOG_TEXT_DIR: str = "logs"
LOG_TEXT_FILENAME: str = "soulscript.log"
LOG_TEXT_MAX_LINES: int = 800

from datetime import timedelta
from pathlib import Path

RANDOM_SEED: int = 202410

TRAIT_MIN: int = -100
TRAIT_MAX: int = 100
TRAIT_SOFT_LIMIT: int = 85
TRAIT_EVENT_DELTA: int = 5

MOOD_MIN: int = 0
MOOD_MAX: int = 100
INITIAL_MOOD: int = 55

RELATIONSHIP_TRUST_CLAMP: tuple[int, int] = (-50, 50)
RELATIONSHIP_AFFINITY_CLAMP: tuple[int, int] = (-50, 50)
RELATIONSHIP_EVENT_DELTA: int = 5
RELATIONSHIP_DECAY_PER_TICK: float = 0.5
RELATIONSHIP_SOFT_LIMIT: int = 90
RELATIONSHIP_COOLDOWN_SECONDS: float = 4.0
RELATIONSHIP_NEUTRAL: int = 0
RELATIONSHIP_TRUST_BASE_INCREMENT: int = 0  # steady per-interaction nudge

SHORT_TERM_MEMORY_SIZE: int = 12
SHORT_TERM_PROMPT_MAX_LINES: int = 6
SUMMARY_INTERVAL_TICKS: int = 15
SUMMARY_TTL_TICKS: int = 120
MEMORY_SALIENCE_BASE: float = 0.6
MEMORY_SALIENCE_TAG_BONUS: float = 0.4

STREAMLIT_DEFAULT_SPEED: float = 1.0
STREAMLIT_SPEED_OPTIONS: list[float] = [0.5, 1.0, 1.5, 2.0]
STREAMLIT_MAX_LOG_LINES: int = 200
STREAMLIT_VISITOR_TAG: str = "visitor"
STREAMLIT_LOCAL_TAG: str = "local"

RELATIONSHIP_DECAY_TIMESTEP = timedelta(seconds=45)

DB_FILE: str = str(Path("soulscript/runtime_data/tavern_state.sqlite"))
DB_ECHO: bool = False

CONVERSATION_KEEP: int = 5
CONVERSATION_SUMMARY_LENGTH: int = 2
CONVERSATION_LENGTH_TURNS: int = 10  # total turns per conversation before wrap-up
CONVERSATION_SESSION_TURNS: int = CONVERSATION_LENGTH_TURNS

# TODO: move hard coded Streamlit settings into a UI config module.
