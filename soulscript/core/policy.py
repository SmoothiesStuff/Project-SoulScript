########## Policy Layer ##########
# Describes node specific allowed actions and tool prefetch logic.

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .npc import NPC
from .tools import inventory_for, location_of, schedule_for
from .types import Action, ActionType

POLICY_CONFIG: Dict[str, Dict[str, Any]] = {
    "idle": {
        "actions": [ActionType.IDLE],
        "tools": ["location"],
        "requires_partner": False,
    },
    "speak": {
        "actions": [ActionType.SPEAK, ActionType.IDLE],
        "tools": ["inventory", "schedule"],
        "requires_partner": True,
    },
    "decide_join": {
        "actions": [ActionType.JOIN, ActionType.IDLE],
        "tools": ["location", "schedule"],
        "requires_partner": False,
    },
    "adjust_relationship": {
        "actions": [ActionType.ADJUST_RELATIONSHIP, ActionType.IDLE],
        "tools": [],
        "requires_partner": True,
    },
}


def apply_policy(npc: NPC, node_name: str, context: Dict[str, Any]) -> Tuple[List[Action], Dict[str, Any]]:
    """Return allowed actions plus tool outputs for the given node."""

    # 1 Pull configuration and pre compute tool values.                        # steps
    config = POLICY_CONFIG.get(node_name, POLICY_CONFIG["idle"])
    tool_outputs: Dict[str, Any] = {}
    for tool_name in config.get("tools", []):
        if tool_name == "inventory":
            tool_outputs["inventory"] = inventory_for(npc.profile)
        elif tool_name == "location":
            tool_outputs["location"] = location_of(npc.profile)
        elif tool_name == "schedule":
            tool_outputs["schedule"] = schedule_for(npc.profile)
    nearby = context.get("nearby_npcs", [])
    focus_target = context.get("focus_target")
    target_id = focus_target
    if target_id is None and nearby:
        target_id = nearby[0]
    allowed_actions: List[Action] = []
    requires_partner = config.get("requires_partner", False)
    if requires_partner and target_id is None:
        allowed_actions.append(Action(action_type=ActionType.IDLE))
        return allowed_actions, tool_outputs
    for action_type in config.get("actions", []):
        action = Action(action_type=action_type, target_id=target_id)
        allowed_actions.append(action)
    return allowed_actions, tool_outputs


# TODO: expose policy knobs via config for rapid tuning during playtests.
