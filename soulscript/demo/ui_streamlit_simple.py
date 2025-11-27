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
from soulscript.core.runtime import RelationshipGraph
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
    st.session_state.conversation_active: bool = False


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
        if st.button("Start conversation"):
            st.session_state.focus_pair = (left, right)
            st.session_state.conversation_active = True
            st.session_state.scheduler.pending_effects.clear()
            _run_tick(left, right, conversation_active=True)
        if st.button("Continue conversation", disabled=not st.session_state.conversation_active):
            _run_tick(left, right, conversation_active=True)
        if st.button("End conversation", disabled=not st.session_state.conversation_active):
            st.session_state.scheduler.finalize_conversation()
            st.session_state.conversation_active = False
            st.session_state.log_lines = []
        if st.button("Reset world"):
            _reset_world()
    return left, right


########## Tick Runner ##########
# Executes a single scheduler tick with an optional focus pair.

def _run_tick(npc_a: str, npc_b: str, conversation_active: bool = False) -> None:
    """Run one tick and store decisions."""

    scheduler: SimulationScheduler = st.session_state.scheduler
    world_context: Dict[str, str] = {
        "focus_target": npc_b,
        "defer_effects": conversation_active,
        "allowed_npcs": [npc_a, npc_b],
    }
    decisions = scheduler.step(world_context)
    st.session_state.log_lines.extend(decisions)


def _reset_world() -> None:
    """Reset scheduler and logs."""

    # 1 Drop persisted state so prompts do not reuse stale conversations.      # steps
    # 2 Recreate the world and clear in memory logs.                           # steps
    db_path = Path(config.DB_FILE)
    try:
        from soulscript.core import db

        engine = getattr(db, "_ENGINE", None)
        if engine is not None:
            try:
                with engine.begin() as connection:
                    connection.exec_driver_sql("DELETE FROM conversations")
                    connection.exec_driver_sql("DELETE FROM relationships")
                    connection.exec_driver_sql("DELETE FROM npc_state")
                    connection.exec_driver_sql("DELETE FROM event_log")
            except Exception:
                pass
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass
        db._ENGINE = None  # type: ignore[attr-defined]
        if db_path.exists():
            try:
                db_path.unlink()
            except PermissionError:
                try:
                    db_path.unlink(missing_ok=True)  # best effort on Windows locks
                except Exception:
                    pass
    except Exception:
        # If cleanup fails, proceed with a new scheduler so in-memory state is fresh.
        pass
    scheduler = build_demo_world()
    st.session_state.scheduler = scheduler
    st.session_state.log_lines = []
    st.session_state.conversation_active = False


########## Panels ##########
# Show states, conversation, and logs.

def _render_state(npcs: Dict[str, NPC], relationships: RelationshipGraph, left: str, right: str) -> None:
    """Display selected NPC cards and their view of each other."""

    st.markdown("###### NPC States ########")
    cols = st.columns(2)
    for col, npc_id in zip(cols, [left, right]):
        npc = npcs.get(npc_id)
        partner_id = right if npc_id == left else left
        if not npc or partner_id not in npcs:
            continue
        edge = relationships.get_edge(npc_id, partner_id)
        with col:
            st.write(
                {
                    "npc_id": npc.npc_id,
                    "name": npc.profile.truth.name,
                    "mood": npc.mood,
                    "motivation": npc.profile.truth.motivation,
                    "role": npc.role,
                    "traits_summary": npc.profile.truth.traits,
                    "trait_inputs": npc.profile.truth.trait_inputs,
                    "trust_to_partner": edge.trust,
                    "affinity_to_partner": edge.affinity,
                    "summary_about_partner": edge.summary,
                }
            )
    # Explicit pair perceptions both directions for clarity.
    st.markdown("###### Pair Perceptions ########")
    rows = []
    if left in npcs and right in npcs and left != right:
        edge_lr = relationships.get_edge(left, right)
        edge_rl = relationships.get_edge(right, left)
        rows.append(
            {
                "source": left,
                "target": right,
                "trust": edge_lr.trust,
                "affinity": edge_lr.affinity,
                "summary": edge_lr.summary,
            }
        )
        rows.append(
            {
                "source": right,
                "target": left,
                "trust": edge_rl.trust,
                "affinity": edge_rl.affinity,
                "summary": edge_rl.summary,
            }
        )
    st.table(rows if rows else [{"info": "Select two different NPCs to view perceptions."}])


def _render_conversation(npc_a: str, npc_b: str) -> None:
    """Show the recent conversation and long-term summary."""

    # Conversation view omitted; see transcript below.


def _render_logs() -> None:
    """Display the run log with reasons."""

    st.markdown("###### Conversation Transcript ########")
    transcript_lines = []
    for decision in st.session_state.log_lines[-200:]:
        if decision.selected_action.action_type.value != "speak":
            continue
        speaker = decision.npc_id
        line = decision.dialogue_line or ""
        transcript_lines.append(f"{speaker}: {line}")
    transcript_text = "\n".join(transcript_lines)
    st.text(transcript_text if transcript_lines else "No dialogue yet.")


########## Main ##########
# Wire the UI together.

def main() -> None:
    """Streamlit entrypoint."""

    st.set_page_config(page_title="SoulScript Simple", layout="wide")
    _init_session()
    scheduler: SimulationScheduler = st.session_state.scheduler
    npc_a, npc_b = _render_controls(scheduler.npcs)
    _render_state(scheduler.npcs, scheduler.relationships, npc_a, npc_b)
    _render_conversation(npc_a, npc_b)
    _render_logs()


if __name__ == "__main__":
    main()
