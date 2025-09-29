########## Engine API ##########
# Lightweight FastAPI server intended for future integrations.

from __future__ import annotations

from typing import Dict, List

from fastapi import FastAPI

from ..core.scheduler import SimulationScheduler
from ..core.types import Decision
from ..demo.soulscript_demo import build_demo_world

app = FastAPI(title="SoulScript Engine API", version="0.2.0")
_scheduler: SimulationScheduler = build_demo_world()
_recent_decisions: List[Decision] = []


@app.post("/tick")
def tick() -> Dict[str, int]:
    """Advance one scheduler step and report how many decisions were made."""

    # 1 Run a single tick and store results.                                    # steps
    decisions = _scheduler.step({"gathering_spot": "tavern_floor"})
    _recent_decisions.extend(decisions)
    return {"decisions": len(decisions)}


@app.get("/npc_state")
def npc_state() -> List[Dict[str, str]]:
    """Expose NPC public state for other clients."""

    # 1 Return sanitized card view for each NPC.                                # steps
    return [npc.public_view() for npc in _scheduler.npcs.values()]


@app.get("/dialogue")
def dialogue(limit: int = 25) -> List[Dict[str, str]]:
    """Return the most recent dialogue entries."""

    # 1 Convert Decision objects into simple dicts.                             # steps
    selected = _recent_decisions[-limit:]
    payload: List[Dict[str, str]] = []
    for decision in selected:
        payload.append(
            {
                "npc": decision.npc_id,
                "action": decision.selected_action.action_type.value,
                "line": decision.dialogue_line or "",
                "reason": decision.reason,
            }
        )
    return payload
