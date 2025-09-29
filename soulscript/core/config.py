########## Core Config ##########
# Houses runtime constants for the SoulScript tavern demo.

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

RANDOM_SEED: int = 202410
TICK_INTERVAL_SECONDS: float = 1.0
MAX_TICKS_PER_RUN: int = 900

TRAIT_MIN: int = -100
TRAIT_MAX: int = 100
TRAIT_SOFT_LIMIT: int = 85
TRAIT_EVENT_DELTA: int = 5

MOOD_MIN: int = 0
MOOD_MAX: int = 100
INITIAL_MOOD: int = 55

RELATIONSHIP_TRUST_CLAMP: tuple[int, int] = (0, 100)
RELATIONSHIP_AFFINITY_CLAMP: tuple[int, int] = (0, 100)
RELATIONSHIP_EVENT_DELTA: int = 5
RELATIONSHIP_DECAY_PER_TICK: float = 0.5
RELATIONSHIP_SOFT_LIMIT: int = 90
RELATIONSHIP_COOLDOWN_SECONDS: float = 4.0
RELATIONSHIP_NEUTRAL: int = 50

SHORT_TERM_MEMORY_SIZE: int = 12
SUMMARY_INTERVAL_TICKS: int = 15
SUMMARY_TTL_TICKS: int = 120
MEMORY_SALIENCE_BASE: float = 0.6
MEMORY_SALIENCE_TAG_BONUS: float = 0.4

DEFAULT_DIALOGUE_EXPORT: str = "soulscript/demo/run_logs"
DEFAULT_DIALOGUE_FILENAME_TEMPLATE: str = "run_{timestamp}.jsonl"
DEFAULT_EVENT_LOG_EXPORT: str = "soulscript/demo/run_logs/events_{timestamp}.csv"

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

# TODO: move hard coded Streamlit settings into a UI config module.
