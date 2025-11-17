########## Streamlit Simple UI ##########
# Minimal panel to watch two NPCs interact and evolve.

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

# Ensure repo root is importable when launched via `streamlit run`.
root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from soulscript.core import config
from soulscript.core.npc import NPC
from soulscript.core.relationships import RelationshipGraph
from soulscript.core.scheduler import SimulationScheduler
from soulscript.core.types import Decision
from soulscript.demo.soulscript_demo import build_demo_world


########## Env Loader ##########
# Reads a simple .env file so Ollama settings are picked up.

def _load_env_file() -> None:
    """Load key=value pairs from an optional .env file."""

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_env_file()


########## Session Setup ##########
# Initializes scheduler and UI state.

def _init_session() -> None:
    """Ensure scheduler and buffers exist in session state."""

    if "scheduler" in st.session_state:
        return
    scheduler = build_demo_world()
    st.session_state.scheduler: SimulationScheduler = scheduler
    st.session_state.log_lines: List[Decision] = []
    st.session_state.focus_pair: Tuple[str, str] | None = None


########## Controls ##########
# Sidebar for pair selection and tick execution.

def _render_controls(npcs: Dict[str, NPC]) -> Tuple[str, str]:
    """Render pair selectors and action buttons."""

    npc_ids = list(npcs.keys())
    with st.sidebar:
        st.markdown("###### Pair Selection ########")
        left = st.selectbox("NPC A", npc_ids, index=0)
        right = st.selectbox("NPC B", npc_ids, index=1 if len(npc_ids) > 1 else 0)
        st.markdown("###### Actions ########")
        if st.button("Run one interaction"):
            st.session_state.focus_pair = (left, right)
            _run_tick(left, right)
        if st.button("Reset world"):
            _reset_world()
    return left, right


########## Tick Runner ##########
# Executes a single scheduler tick with an optional focus pair.

def _run_tick(npc_a: str, npc_b: str) -> None:
    """Run one tick and store decisions."""

    scheduler: SimulationScheduler = st.session_state.scheduler
    world_context: Dict[str, str] = {"focus_target": npc_b}
    decisions = scheduler.step(world_context)
    st.session_state.log_lines.extend(decisions)


def _reset_world() -> None:
    """Reset scheduler and logs."""

    scheduler = build_demo_world()
    st.session_state.scheduler = scheduler
    st.session_state.log_lines = []


########## Panels ##########
# Show states, conversation, and logs.

def _render_state(npcs: Dict[str, NPC], relationships: RelationshipGraph) -> None:
    """Display NPC cards and current relationship summary."""

    st.markdown("###### NPC States ########")
    cols = st.columns(len(npcs))
    for col, npc in zip(cols, npcs.values()):
        with col:
            st.write(
                {
                    "name": npc.profile.truth.name,
                    "mood": npc.mood,
                    "motivation": npc.profile.truth.motivation,
                    "role": npc.role,
                }
            )
    st.markdown("###### Relationship Snapshot ########")
    edges = []
    for npc_id in npcs:
        for target_id in npcs:
            if npc_id == target_id:
                continue
            edge = relationships.get_edge(npc_id, target_id)
            edges.append(
                {
                    "source": npc_id,
                    "target": target_id,
                    "trust": edge.trust,
                    "affinity": edge.affinity,
                    "summary": edge.summary,
                }
            )
    st.table(edges)


def _render_conversation(npc_a: str, npc_b: str) -> None:
    """Show the recent conversation and long-term summary."""

    scheduler: SimulationScheduler = st.session_state.scheduler
    memory = scheduler.conversation_memory.history(npc_a, npc_b)
    st.markdown("###### Recent Conversation ########")
    lines = [f"{item.speaker_id}: {item.text}" for item in memory]
    st.write(lines if lines else "No dialogue yet.")
    st.markdown("###### Long-Term Summary ########")
    edge = scheduler.relationships.get_edge(npc_a, npc_b)
    st.write(edge.summary or "No summary yet.")


def _render_logs() -> None:
    """Display the run log with reasons."""

    st.markdown("###### Decisions Log ########")
    rows = []
    for decision in st.session_state.log_lines[-50:]:
        rows.append(
            {
                "npc": decision.npc_id,
                "action": decision.selected_action.action_type.value,
                "target": decision.selected_action.target_id,
                "line": decision.dialogue_line,
                "reason": decision.reason,
            }
        )
    st.table(rows)


########## Main ##########
# Wire the UI together.

def main() -> None:
    """Streamlit entrypoint."""

    st.set_page_config(page_title="SoulScript Simple", layout="wide")
    _init_session()
    scheduler: SimulationScheduler = st.session_state.scheduler
    npc_a, npc_b = _render_controls(scheduler.npcs)
    _render_state(scheduler.npcs, scheduler.relationships)
    _render_conversation(npc_a, npc_b)
    _render_logs()


if __name__ == "__main__":
    main()
