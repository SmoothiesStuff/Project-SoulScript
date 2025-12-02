########## Demo Tools ##########
# Registry-backed utilities for LangGraph nodes.

from __future__ import annotations

from typing import Any, Callable, Dict, List

from .types import NPCProfile

ToolFunc = Callable[[NPCProfile], Any]


########## Tool Functions ##########
# Simple lookups exposed via the registry.


def inventory_for(profile: NPCProfile) -> List[str]:
    """Return the inventory list declared in the truth seed."""

    return list(profile.truth.inventory)


def location_of(profile: NPCProfile) -> str:
    """Return the default hangout spot for the NPC."""

    schedule = profile.truth.schedule
    if "morning" in schedule:
        return schedule["morning"]
    for _, location in schedule.items():
        return location
    return "tavern_floor"


def schedule_for(profile: NPCProfile) -> Dict[str, str]:
    """Return a simple schedule map for the NPC."""

    return dict(profile.truth.schedule)


########## Registry ##########
# Maps tool names to callables for predictable access.

TOOL_REGISTRY: Dict[str, ToolFunc] = {
    "inventory": inventory_for,
    "location": location_of,
    "schedule": schedule_for,
}


def fetch_tool_output(tool_name: str, profile: NPCProfile) -> Any:
    """Return tool output when the name is registered."""

    func = TOOL_REGISTRY.get(tool_name)
    if func is None:
        return None
    return func(profile)
