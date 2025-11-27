########## Streamlit UI ##########
# Presents the tavern simulation with controls, map, and live panels.

from __future__ import annotations

import math
import json
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
from soulscript.core.runtime import RelationshipGraph
from soulscript.core.scheduler import SimulationScheduler
from soulscript.core.types import Decision
from soulscript.demo.soulscript_demo import build_demo_world

ROLE_OPTIONS = ["all", config.STREAMLIT_LOCAL_TAG, config.STREAMLIT_VISITOR_TAG]
NPC_SYMBOLS = {"female": "triangle-up", "male": "square", "nonbinary": "circle"}
TRAIT_COLOR_ORDER = [
    ("bravery", "#e76f51"),
    ("kindness", "#f4a261"),
    ("curiosity", "#2a9d8f"),
    ("intelligence", "#264653"),
    ("optimism", "#7bc8f6"),
    ("charisma", "#ffafcc"),
]
TABLE_RADIUS = 0.24  # 3x bigger tables
CHAIR_RADIUS = 0.05
NPC_SPEED = 0.0  # pop movement (no tweening)
DWELL_PROBABILITY = 0.65
MOVE_PROBABILITY = 0.35


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
        st.session_state.pair_source = ""
        st.session_state.pair_target = ""
        st.session_state.selected_npc: Optional[str] = None
        st.session_state.npc_positions: Dict[str, Dict[str, float]] = {}
        st.session_state.table_positions: List[Tuple[float, float]] = _generate_tables()
        st.session_state.position_rng = random.Random(config.RANDOM_SEED)
        st.session_state.tavern_conversations: List[Dict[str, str]] = []
        st.session_state.npc_table_map: Dict[str, int] = {}
        st.session_state.active_pairs: Dict[int, Tuple[str, str]] = {}
        st.session_state.active_conversations: Dict[int, Dict[str, object]] = {}
        st.session_state.past_conversations: List[Dict[str, object]] = []
        st.session_state.chairs: List[Dict[str, object]] = _generate_chairs(st.session_state.table_positions)


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


def _generate_chairs(table_positions: List[Tuple[float, float]]) -> List[Dict[str, object]]:
    """Create two chairs per table offset around the table."""

    chairs: List[Dict[str, object]] = []
    offsets = [(0.12, 0), (-0.12, 0)]  # left/right of table
    for table_idx, (tx, ty) in enumerate(table_positions):
        for chair_idx, (dx, dy) in enumerate(offsets):
            chairs.append(
                {
                    "table_id": table_idx,
                    "chair_idx": chair_idx,
                    "x": tx + dx,
                    "y": ty + dy,
                    "occupied_by": None,
                }
            )
    return chairs


def _reset_world() -> None:
    """Reset scheduler, db-backed state, and UI caches."""

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
                    db_path.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        # If cleanup fails, proceed with a new scheduler so in-memory state is fresh.
        pass
    scheduler = build_demo_world()
    st.session_state.scheduler = scheduler
    st.session_state.log_lines = []
    st.session_state.running = False
    st.session_state.last_tick = 0.0
    st.session_state.npc_positions = {}
    st.session_state.npc_table_map = {}
    st.session_state.active_pairs = {}
    st.session_state.active_conversations = {}
    st.session_state.past_conversations = []
    st.session_state.chairs = _generate_chairs(st.session_state.table_positions)
    st.session_state.selected_npc = None
    st.session_state.tavern_conversations = []
    st.session_state.table_positions = _generate_tables()


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
            _reset_world()
        speed = st.selectbox(
            "Speed",
            options=config.STREAMLIT_SPEED_OPTIONS,
            index=config.STREAMLIT_SPEED_OPTIONS.index(config.STREAMLIT_DEFAULT_SPEED),
        )
        st.session_state.speed = speed


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
    _update_positions_and_tables(scheduler)
    _update_active_pairs(scheduler)
    world_context = {
        "gathering_spot": "tavern_floor",
        "table_map": st.session_state.npc_table_map,
        "active_pairs": st.session_state.active_pairs,
    }
    decisions = scheduler.step(world_context)
    st.session_state.last_tick = now
    log_lines = st.session_state.log_lines
    for decision in decisions:
        log_lines.append(decision)
    while len(log_lines) > config.STREAMLIT_MAX_LOG_LINES:
        log_lines.pop(0)
    st.session_state.log_lines = log_lines
    _record_conversations(decisions)
    _cleanup_finished_conversations(scheduler)
    time.sleep(0.05)
    st.experimental_rerun()


def _pair_key(npc_a: str, npc_b: str) -> Tuple[str, str]:
    """Stable tuple ordering for pair lookups."""

    return tuple(sorted((npc_a, npc_b)))


def _finalize_pair(scheduler: SimulationScheduler, npc_a: str, npc_b: str) -> None:
    """Finalize a single pair's conversation, preserving others."""

    key = _pair_key(npc_a, npc_b)
    pending = scheduler.pending_effects
    keep: List[Dict[str, str]] = []
    pair_effects: List[Dict[str, str]] = []
    for effect in pending:
        effect_key = _pair_key(effect["source_id"], effect["target_id"])
        if effect_key == key:
            pair_effects.append(effect)
        else:
            keep.append(effect)
    if not pair_effects:
        return
    table_id = st.session_state.npc_table_map.get(npc_a)
    # Capture transcript for past log before clearing.
    convo = st.session_state.active_conversations.get(table_id or -1)
    if convo:
        st.session_state.past_conversations.append(
            {
                "table": table_id,
                "pair": convo.get("pair", (npc_a, npc_b)),
                "lines": list(convo.get("lines", [])),
            }
        )
    scheduler.pending_effects = pair_effects
    scheduler.finalize_conversation()
    scheduler.pending_effects = keep
    if table_id is not None:
        st.session_state.active_conversations.pop(table_id, None)


def _update_positions_and_tables(scheduler: SimulationScheduler) -> None:
    """Maintain soft roaming and occasionally shift NPCs to a new table."""

    rng: random.Random = st.session_state.position_rng
    table_positions = st.session_state.table_positions
    npc_positions = st.session_state.npc_positions
    table_map = st.session_state.npc_table_map
    active_pairs = st.session_state.active_pairs
    chairs = st.session_state.chairs
    # 1 Create anchors for new NPCs.
    def _find_free_chair(table_idx: int) -> Dict[str, object] | None:
        free = [c for c in chairs if c["table_id"] == table_idx and c.get("occupied_by") is None]
        return rng.choice(free) if free else None

    def _assign_chair(npc_id: str, chair: Dict[str, object]) -> None:
        chair["occupied_by"] = npc_id
        npc_positions[npc_id] = {
            "x": chair["x"],
            "y": chair["y"],
            "anchor_x": chair["x"],
            "anchor_y": chair["y"],
        }
        table_map[npc_id] = chair["table_id"]

    def _release_chair(npc_id: str) -> None:
        for chair in chairs:
            if chair.get("occupied_by") == npc_id:
                chair["occupied_by"] = None

    for npc_id in scheduler.npcs:
        if npc_id not in npc_positions:
            table_idx = rng.randrange(len(table_positions))
            chair = _find_free_chair(table_idx) or _find_free_chair(rng.randrange(len(table_positions)))
            if chair:
                _assign_chair(npc_id, chair)
    # 2 Occasionally move to a different table and finalize old conversations.
    engaged_pairs = set()
    for pair in active_pairs.values():
        engaged_pairs.update(pair)
    for npc_id, pos in npc_positions.items():
        current_table = table_map.get(npc_id, 0)
        if npc_id in engaged_pairs:
            # Locked in conversation; stay put.
            pos["x"] = pos["anchor_x"]
            pos["y"] = pos["anchor_y"]
            continue
        if rng.random() < MOVE_PROBABILITY and len(table_positions) > 1:
            new_table = rng.randrange(len(table_positions))
            while new_table == current_table and len(table_positions) > 1:
                new_table = rng.randrange(len(table_positions))
            # End the current conversation for this NPC before moving.
            table_pair = active_pairs.get(current_table)
            if table_pair and npc_id in table_pair:
                partner = table_pair[0] if table_pair[1] == npc_id else table_pair[1]
                _finalize_pair(scheduler, npc_id, partner)
                active_pairs.pop(current_table, None)
            _release_chair(npc_id)
            chair = _find_free_chair(new_table) or _find_free_chair(current_table)
            if chair:
                _assign_chair(npc_id, chair)
        else:
            # Snap to anchor (no tweening).
            pos["x"] = pos["anchor_x"]
            pos["y"] = pos["anchor_y"]
    st.session_state.npc_positions = npc_positions
    st.session_state.npc_table_map = table_map
    st.session_state.active_pairs = active_pairs
    st.session_state.chairs = chairs


def _update_active_pairs(scheduler: SimulationScheduler) -> None:
    """Maintain per-table conversation bubbles."""

    table_map = st.session_state.npc_table_map
    active_pairs = st.session_state.active_pairs
    active_convos = st.session_state.active_conversations
    occupancy: Dict[int, List[str]] = {}
    for npc_id, table_id in table_map.items():
        occupancy.setdefault(table_id, []).append(npc_id)
    # Drop stale pairs when someone leaves the table.
    for table_id, pair in list(active_pairs.items()):
        occupants = occupancy.get(table_id, [])
        if len(occupants) < 2 or pair[0] not in occupants or pair[1] not in occupants:
            _finalize_pair(scheduler, pair[0], pair[1])
            active_pairs.pop(table_id, None)
            active_convos.pop(table_id, None)
    # Add new pairs when two or more share a table.
    for table_id, occupants in occupancy.items():
        if len(occupants) < 2:
            continue
        if table_id in active_pairs:
            continue
        chosen = tuple(sorted(occupants)[:2])
        active_pairs[table_id] = chosen  # only two chat; others at the table stay idle.
        active_convos[table_id] = {"pair": chosen, "lines": []}
    st.session_state.active_pairs = active_pairs
    st.session_state.active_conversations = active_convos


def _ensure_positions_initialized(scheduler: SimulationScheduler) -> None:
    """Guarantee positions exist before first render."""

    if st.session_state.npc_positions:
        return
    _update_positions_and_tables(scheduler)
    _update_active_pairs(scheduler)


def _record_conversations(decisions: List[Decision]) -> None:
    """Store active conversation snippets per table."""

    table_map = st.session_state.npc_table_map
    active_pairs = st.session_state.active_pairs
    active_convos = st.session_state.active_conversations
    for decision in decisions:
        action = decision.selected_action
        if action.action_type != action.action_type.SPEAK or not action.target_id:
            continue
        speaker = decision.npc_id
        target = action.target_id
        table_id = table_map.get(speaker)
        if table_id is None:
            continue
        pair = active_pairs.get(table_id)
        if not pair or speaker not in pair or target not in pair:
            continue
        convo = active_convos.setdefault(table_id, {"pair": pair, "lines": []})
        line = decision.dialogue_line or "(silent glance)"
        convo["lines"].append(f"{speaker}: {line}")
        convo["pair"] = pair
    # Keep lines short for readability.
    for convo in active_convos.values():
        lines = convo.get("lines", [])
        if len(lines) > 12:
            convo["lines"] = lines[-12:]
    st.session_state.active_conversations = active_convos


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
    # Chairs
    chair_x = [chair["x"] for chair in st.session_state.chairs]
    chair_y = [chair["y"] for chair in st.session_state.chairs]
    chair_trace = go.Scatter(
        x=chair_x,
        y=chair_y,
        mode="markers",
        marker=dict(size=220 * CHAIR_RADIUS, color="#d8c3a5", opacity=0.9, symbol="square"),
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
    fig = go.Figure(data=[table_trace, chair_trace, npc_trace])
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#f2e9e4",
        plot_bgcolor="#f2e9e4",
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        clickmode="event+select",
    )
    selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="tavern-map")
    if selected_points:
        for point in selected_points:
            # npc_trace is the third trace (index 2)
            if point.get("curveNumber") != 2:
                continue
            custom = point.get("customdata")
            if custom:
                st.session_state.selected_npc = custom
                st.experimental_rerun()
                break
    return st.session_state.selected_npc


def _render_detail_panel(scheduler: SimulationScheduler, selected_npc: Optional[str]) -> None:
    """Show character stats and conversations in the side panel."""

    # 1 Handle missing selection with a friendly hint.                          # steps
    npc_ids = list(scheduler.npcs.keys())
    current_id = selected_npc if selected_npc in npc_ids else (npc_ids[0] if npc_ids else None)
    chosen = st.selectbox("Character", npc_ids, index=npc_ids.index(current_id) if current_id else 0)
    st.session_state.selected_npc = chosen
    if not chosen:
        st.markdown("Select a character to view their story.")
        return
    npc = scheduler.npcs[chosen]
    current_table = st.session_state.npc_table_map.get(chosen)
    # Build a compact agent state snapshot.
    truth = npc.truth_vector().as_dict()
    relations: Dict[str, Dict[str, int | str]] = {}
    for other_id in scheduler.npcs:
        if other_id == selected_npc:
            continue
        edge = scheduler.relationships.get_edge(selected_npc, other_id)
        relations[other_id] = {
            "trust": edge.trust,
            "affinity": edge.affinity,
            "summary": edge.summary or "",
        }
    latest = _find_latest_conversation(selected_npc)
    state = {
        "npc_id": npc.npc_id,
        "name": npc.profile.truth.name,
        "role": npc.role,
        "table": current_table,
        "mood": npc.mood,
        "backstory": npc.profile.truth.backstory,
        "motivation": npc.profile.truth.motivation,
        "traits_summary": npc.profile.truth.traits,
        "traits_truth": truth,
        "relationships": relations,
        "active_conversation": latest,
    }
    st.markdown(f"### {npc.profile.truth.name} — Character State")
    st.text_area(
        "State JSON",
        json.dumps(state, indent=2),
        height=320,
        key=f"state_json_{npc.npc_id}",
    )


def _find_latest_conversation(npc_id: str) -> Optional[Dict[str, str]]:
    """Locate the latest conversation entry involving the NPC."""

    active_convos = st.session_state.active_conversations
    for convo in active_convos.values():
        pair = convo.get("pair", ())
        if npc_id in pair:
            lines = convo.get("lines", [])
            if lines:
                last = lines[-1]
                parts = last.split(":", 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    text = parts[1].strip()
                    target = pair[0] if speaker == pair[1] else pair[1]
                    return {"speaker": speaker, "target": target, "line": text, "reason": ""}
    return None


def _render_log_section(log_lines: List[Decision]) -> None:
    """Show a filtered slice of the dialogue log."""

    st.write("Conversation log moved to the bubbles below.")


def _render_conversations_panel() -> None:
    """Show active conversation bubbles per table."""

    st.markdown("### Conversations")
    active_convos = st.session_state.active_conversations
    scheduler: SimulationScheduler = st.session_state.scheduler
    if not active_convos:
        st.write("No conversations yet. Characters will chat when sharing a table.")
        return
    for table_id, convo in active_convos.items():
        pair = convo.get("pair", ())
        lines = convo.get("lines", [])
        names = []
        for npc_id in pair:
            if npc_id in scheduler.npcs:
                names.append(scheduler.npcs[npc_id].profile.truth.name)
            else:
                names.append(npc_id)
        st.markdown(f"**Table {table_id}** — {' ↔ '.join(names)}")
        if lines:
            st.code("\n".join(lines[-8:]), language="text")
        else:
            st.write("Just having a drink...")


def _cleanup_finished_conversations(scheduler: SimulationScheduler) -> None:
    """Remove UI pairs whose sessions are done in the scheduler."""

    active_pairs = st.session_state.active_pairs
    active_convos = st.session_state.active_conversations
    live_sessions = set(scheduler.conversation_sessions.keys())
    for table_id, pair in list(active_pairs.items()):
        key = _pair_key(pair[0], pair[1])
        if key not in live_sessions:
            convo = active_convos.get(table_id)
            if convo:
                st.session_state.past_conversations.append(
                    {"table": table_id, "pair": convo.get("pair", pair), "lines": list(convo.get("lines", []))}
                )
            active_pairs.pop(table_id, None)
            active_convos.pop(table_id, None)
    st.session_state.active_pairs = active_pairs
    st.session_state.active_conversations = active_convos


def _render_relationship_matrix(scheduler: SimulationScheduler) -> None:
    """Display current affinity between all NPCs."""

    npc_ids = list(scheduler.npcs.keys())
    headers = ["NPC"] + [scheduler.npcs[npc_id].profile.truth.name for npc_id in npc_ids]
    rows: List[Dict[str, str | int]] = []
    for source_id in npc_ids:
        row: Dict[str, str | int] = {"NPC": scheduler.npcs[source_id].profile.truth.name}
        for target_id in npc_ids:
            header = scheduler.npcs[target_id].profile.truth.name
            if source_id == target_id:
                row[header] = "-"
            else:
                edge = scheduler.relationships.get_edge(source_id, target_id)
                scaled = int(round((edge.affinity * edge.trust) / 100))
                row[header] = scaled
        rows.append(row)
    frame = pd.DataFrame(rows, columns=headers)
    st.markdown("### Relationship Affinity Matrix")
    st.dataframe(frame, hide_index=True, use_container_width=True)


def _render_past_conversations() -> None:
    """Show completed conversations after the matrix."""

    past = st.session_state.past_conversations
    st.markdown("### Past Conversations")
    if not past:
        st.write("None yet.")
        return
    for convo in past[-8:]:
        pair = convo.get("pair", ())
        table = convo.get("table")
        lines = convo.get("lines", [])
        st.markdown(f"**Table {table}** — {' ↔ '.join(pair)}")
        if lines:
            st.code("\n".join(lines[-10:]), language="text")
        else:
            st.write("No transcript captured.")


def _render_main_layout() -> None:
    """Compose the map and detail panel layout."""

    scheduler: SimulationScheduler = st.session_state.scheduler
    _ensure_positions_initialized(scheduler)
    if not st.session_state.selected_npc and scheduler.npcs:
        st.session_state.selected_npc = next(iter(scheduler.npcs.keys()))
    col_map, col_detail = st.columns([3, 2])
    with col_map:
        st.markdown("### Tavern Floor")
        selected = _build_tavern_figure(scheduler)
        if selected:
            st.caption(f"Selected: {selected}")
    with col_detail:
        _render_detail_panel(scheduler, st.session_state.selected_npc)


def main() -> None:
    """Streamlit app entrypoint."""

    st.set_page_config(page_title="SoulScript Tavern", layout="wide")
    _init_session()
    _render_sidebar_controls()
    _render_main_layout()
    _render_conversations_panel()
    _render_relationship_matrix(st.session_state.scheduler)
    _render_past_conversations()
    _maybe_run_tick()


if __name__ == "__main__":
    main()



