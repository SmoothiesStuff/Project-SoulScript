########## Streamlit UI ##########
# Presents the tavern simulation with controls, map, and live panels.

from __future__ import annotations

import math
import os
from datetime import datetime
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

import sys

########## Path Setup ##########
# Ensures project root is importable when streamlit runs standalone.

# 1 Resolve project root two levels up for package imports.                 # steps
root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    # 2 Insert the root ahead of site-packages when missing.                 # steps
    sys.path.insert(0, str(root_path))

########## Env Loader ##########
# Reads a simple .env file so Ollama settings are picked up.

def _load_env_file() -> None:
    """Load key=value pairs from an optional .env file."""

    # 1 Stop when the .env file is missing.                               # steps
    env_path = root_path / ".env"
    if not env_path.exists():
        return
    # 2 Read lines and export them to os.environ.                         # steps
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

from soulscript.core import config
from soulscript.core.npc import NPC
from soulscript.core.relationships import RelationshipGraph
from soulscript.core.scheduler import SimulationScheduler
from soulscript.core.types import Decision
from soulscript.demo.soulscript_demo import build_demo_world

ROLE_OPTIONS = ["all", config.STREAMLIT_LOCAL_TAG, config.STREAMLIT_VISITOR_TAG]
NPC_SYMBOLS = {"female": "triangle-up", "male": "square", "nonbinary": "circle"}
TRAIT_COLOR_ORDER = [
    ("bravery", "#e76f51"),
    ("kindness", "#f4a261"),
    ("curiosity", "#2a9d8f"),
    ("discipline", "#264653"),
    ("optimism", "#7bc8f6"),
    ("generosity", "#ffafcc"),
]
TABLE_RADIUS = 0.08
NPC_SPEED = 0.035
DWELL_PROBABILITY = 0.65


def _init_session() -> None:
    """Ensure Streamlit session state carries the scheduler and buffers."""

    # 1 Create scheduler and shared structures when the session starts.        # steps
    if "scheduler" not in st.session_state:
        scheduler = build_demo_world()
        st.session_state.scheduler = scheduler
        st.session_state.log_lines: List[Decision] = []
        st.session_state.running = False
        st.session_state.speed = config.STREAMLIT_DEFAULT_SPEED
        st.session_state.last_tick = 0.0
        st.session_state.role_filter = "all"
        st.session_state.speaker_filter = "all"
        st.session_state.pair_source = ""
        st.session_state.pair_target = ""
        st.session_state.selected_npc: Optional[str] = None
        st.session_state.npc_positions: Dict[str, Dict[str, float]] = {}
        st.session_state.table_positions: List[Tuple[float, float]] = _generate_tables()
        st.session_state.position_rng = random.Random(config.RANDOM_SEED)
        st.session_state.tavern_conversations: List[Dict[str, str]] = []


def _generate_tables() -> List[Tuple[float, float]]:
    """Lay out round tables in a gentle grid."""

    # 1 Prepare a small grid so tables feel evenly spaced.                     # steps
    rows = 2
    cols = 3
    spacing = 0.3
    offset_x = 0.2
    offset_y = 0.25
    positions: List[Tuple[float, float]] = []
    for row in range(rows):
        for col in range(cols):
            x = offset_x + col * spacing
            y = offset_y + row * spacing
            positions.append((x, y))
    return positions


def _render_sidebar_controls() -> None:
    """Build sidebar controls for the simulation."""

    # 1 Show buttons for run state and resetting the world.                    # steps
    with st.sidebar:
        st.markdown("### Controls")
        cols = st.columns(3)
        if cols[0].button("Start"):
            st.session_state.running = True
        if cols[1].button("Stop"):
            st.session_state.running = False
        if cols[2].button("Reset"):
            scheduler: SimulationScheduler = build_demo_world()
            st.session_state.scheduler = scheduler
            st.session_state.log_lines = []
            st.session_state.running = False
            st.session_state.last_tick = 0.0
            st.session_state.npc_positions.clear()
            st.session_state.table_positions = _generate_tables()
            st.session_state.selected_npc = None
            st.session_state.tavern_conversations.clear()
        speed = st.selectbox(
            "Speed",
            options=config.STREAMLIT_SPEED_OPTIONS,
            index=config.STREAMLIT_SPEED_OPTIONS.index(config.STREAMLIT_DEFAULT_SPEED),
        )
        st.session_state.speed = speed
        st.markdown("### Filters")
        st.session_state.role_filter = st.selectbox("View", ROLE_OPTIONS, index=0)
        scheduler: SimulationScheduler = st.session_state.scheduler
        npc_ids = [npc_id for npc_id in scheduler.npcs]
        speakers = ["all"] + npc_ids
        st.session_state.speaker_filter = st.selectbox("Speaker", speakers, index=0)


def _maybe_run_tick() -> None:
    """Advance the simulation when the session is marked running."""

    # 1 Check elapsed time versus the speed slider.                            # steps
    if not st.session_state.running:
        return
    interval = config.TICK_INTERVAL_SECONDS / max(st.session_state.speed, 0.1)
    now = time.time()
    should_advance = now - st.session_state.last_tick >= interval
    if not should_advance:
        return
    scheduler: SimulationScheduler = st.session_state.scheduler
    decisions = scheduler.step({"gathering_spot": "tavern_floor"})
    st.session_state.last_tick = now
    log_lines = st.session_state.log_lines
    for decision in decisions:
        log_lines.append(decision)
    while len(log_lines) > config.STREAMLIT_MAX_LOG_LINES:
        log_lines.pop(0)
    st.session_state.log_lines = log_lines
    _update_positions(scheduler)
    _record_conversations(decisions)
    time.sleep(0.05)
    st.experimental_rerun()


def _update_positions(scheduler: SimulationScheduler) -> None:
    """Maintain soft roaming for each NPC."""

    # 1 Create positions for new NPCs anchored near a table.                   # steps
    rng: random.Random = st.session_state.position_rng
    for npc_id, npc in scheduler.npcs.items():
        if npc_id not in st.session_state.npc_positions:
            table_x, table_y = rng.choice(st.session_state.table_positions)
            jitter_x = rng.uniform(-0.05, 0.05)
            jitter_y = rng.uniform(-0.05, 0.05)
            st.session_state.npc_positions[npc_id] = {
                "x": table_x + jitter_x,
                "y": table_y + jitter_y,
                "anchor_x": table_x,
                "anchor_y": table_y,
            }
    # 2 Randomly drift positions a little toward their anchor.                 # steps
    for npc_id, pos in st.session_state.npc_positions.items():
        anchor_x = pos["anchor_x"]
        anchor_y = pos["anchor_y"]
        current_x = pos["x"]
        current_y = pos["y"]
        move = rng.random() > DWELL_PROBABILITY
        if move:
            angle = rng.uniform(0, 2 * math.pi)
            step = NPC_SPEED * rng.uniform(0.3, 1.0)
            current_x += math.cos(angle) * step
            current_y += math.sin(angle) * step
        else:
            current_x += (anchor_x - current_x) * 0.1
            current_y += (anchor_y - current_y) * 0.1
        current_x = max(0.05, min(0.95, current_x))
        current_y = max(0.05, min(0.95, current_y))
        pos["x"] = current_x
        pos["y"] = current_y


def _record_conversations(decisions: List[Decision]) -> None:
    """Store recent conversation snippets for quick lookup."""

    # 1 Append speak actions with dialogue lines.                               # steps
    for decision in decisions:
        action = decision.selected_action
        if action.action_type == action.action_type.SPEAK and action.target_id:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "speaker": decision.npc_id,
                "target": action.target_id,
                "line": decision.dialogue_line or "(silent glance)",
                "reason": decision.reason,
            }
            st.session_state.tavern_conversations.append(entry)
    # 2 Clamp stored conversation history.                                     # steps
    while len(st.session_state.tavern_conversations) > 200:
        st.session_state.tavern_conversations.pop(0)


def _build_trait_color(npc: NPC) -> str:
    """Choose a color based on the most prominent truth trait."""

    # 1 Find the strongest trait in the author truth.                           # steps
    truth = npc.truth_vector()
    best_trait = "bravery"
    best_value = -float("inf")
    for trait, color in TRAIT_COLOR_ORDER:
        value = getattr(truth, trait)
        if value > best_value:
            best_value = value
            best_trait = trait
    for trait, color in TRAIT_COLOR_ORDER:
        if trait == best_trait:
            return color
    return "#cccccc"


def _build_tavern_figure(scheduler: SimulationScheduler) -> Optional[str]:
    """Render the tavern layout and handle click selection."""

    # 1 Gather table positions and convert to scatter markers.                  # steps
    table_x = [pos[0] for pos in st.session_state.table_positions]
    table_y = [pos[1] for pos in st.session_state.table_positions]
    table_trace = go.Scatter(
        x=table_x,
        y=table_y,
        mode="markers",
        marker=dict(size=220 * TABLE_RADIUS, color="#b08968", opacity=0.7),
        hoverinfo="skip",
        showlegend=False,
    )
    # 2 Build NPC scatter with custom symbols.                                  # steps
    npc_ids: List[str] = []
    npc_x: List[float] = []
    npc_y: List[float] = []
    npc_symbols: List[str] = []
    npc_colors: List[str] = []
    npc_names: List[str] = []
    for npc_id, npc in scheduler.npcs.items():
        if st.session_state.role_filter != "all" and npc.role != st.session_state.role_filter:
            continue
        pos = st.session_state.npc_positions.get(npc_id)
        if not pos:
            continue
        npc_ids.append(npc_id)
        npc_x.append(pos["x"])
        npc_y.append(pos["y"])
        npc_names.append(npc.profile.truth.name)
        gender_key = npc.profile.truth.sex.lower() if npc.profile.truth.sex else "unknown"
        symbol = NPC_SYMBOLS.get(gender_key, "circle")
        npc_symbols.append(symbol)
        npc_colors.append(_build_trait_color(npc))
    npc_trace = go.Scatter(
        x=npc_x,
        y=npc_y,
        mode="markers",
        marker=dict(size=20, color=npc_colors, symbol=npc_symbols, line=dict(color="#1f1f1f", width=1)),
        text=npc_names,
        customdata=npc_ids,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )
    fig = go.Figure(data=[table_trace, npc_trace])
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#f2e9e4",
        plot_bgcolor="#f2e9e4",
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
    )
    selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="tavern-map")
    if selected_points:
        custom = selected_points[0].get("customdata")
        if custom:
            st.session_state.selected_npc = custom
            return custom
    return st.session_state.selected_npc


def _render_detail_panel(scheduler: SimulationScheduler, selected_npc: Optional[str]) -> None:
    """Show character stats and conversations in the side panel."""

    # 1 Handle missing selection with a friendly hint.                          # steps
    if not selected_npc or selected_npc not in scheduler.npcs:
        st.markdown("Select a character on the map to view their story.")
        return
    npc = scheduler.npcs[selected_npc]
    st.markdown(f"### {npc.profile.truth.name}")
    st.caption(f"Role: {npc.role} | Motivation: {npc.profile.truth.motivation}")
    st.write(npc.profile.truth.backstory)
    trait_rows = []
    truth = npc.truth_vector()
    self_traits = npc.serialize_self_traits()
    for trait_name, _color in TRAIT_COLOR_ORDER:
        trait_rows.append(
            {
                "Trait": trait_name.capitalize(),
                "Truth": getattr(truth, trait_name),
                "Self": self_traits.get(trait_name, 0),
            }
        )
    st.dataframe(pd.DataFrame(trait_rows), hide_index=True, use_container_width=True)
    latest = _find_latest_conversation(selected_npc)
    if latest:
        target_id = latest["target"] if latest["speaker"] == selected_npc else latest["speaker"]
        st.markdown("#### Recent Conversation")
        st.write(f"**Line:** {latest['line']}")
        st.caption(f"Reason: {latest['reason']}")
        if target_id in scheduler.npcs:
            other = scheduler.npcs[target_id]
            st.markdown(f"**Talking with:** {other.profile.truth.name}")
            st.caption(other.profile.truth.backstory[:160] + "...")
    else:
        st.markdown("No recent dialogue yet. Let's wait for the chatter.")


def _find_latest_conversation(npc_id: str) -> Optional[Dict[str, str]]:
    """Locate the latest conversation entry involving the NPC."""

    # 1 Scan from the end for a quick match.                                    # steps
    for entry in reversed(st.session_state.tavern_conversations):
        if entry["speaker"] == npc_id or entry["target"] == npc_id:
            return entry
    return None


def _render_log_section(log_lines: List[Decision]) -> None:
    """Show a filtered slice of the dialogue log."""

    # 1 Convert captured decisions into a tidy DataFrame.                      # steps
    records: List[Dict[str, str]] = []
    speaker_filter = st.session_state.speaker_filter
    for decision in log_lines[-120:]:
        if speaker_filter != "all" and decision.npc_id != speaker_filter:
            continue
        dialogue = decision.dialogue_line or ""
        records.append(
            {
                "NPC": decision.npc_id,
                "Action": decision.selected_action.action_type.value,
                "Target": decision.selected_action.target_id or "",
                "Line": dialogue,
                "Why": decision.reason,
            }
        )
    if records:
        frame = pd.DataFrame(records)
        st.dataframe(frame, hide_index=True, use_container_width=True)
    else:
        st.write("No activity yet. Click Start to begin.")


def _render_main_layout() -> None:
    """Compose the map and detail panel layout."""

    scheduler: SimulationScheduler = st.session_state.scheduler
    col_map, col_detail = st.columns([3, 2])
    with col_map:
        st.markdown("### Tavern Floor")
        selected = _build_tavern_figure(scheduler)
        if selected:
            st.caption(f"Selected: {selected}")
    with col_detail:
        st.markdown("### Character Details")
        _render_detail_panel(scheduler, st.session_state.selected_npc)


def main() -> None:
    """Streamlit app entrypoint."""

    st.set_page_config(page_title="SoulScript Tavern", layout="wide")
    _init_session()
    _render_sidebar_controls()
    _render_main_layout()
    st.markdown("### Live Log")
    _render_log_section(st.session_state.log_lines)
    _maybe_run_tick()


if __name__ == "__main__":
    main()



