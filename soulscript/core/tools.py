########## Demo Tools ##########
# Utility lookups used by LangGraph nodes to ground decisions.

from __future__ import annotations

from typing import Dict, List

from .types import NPCProfile

FILTERED_WORDS = {"curse", "dang", "heck"}


def inventory_for(profile: NPCProfile) -> List[str]:
    """Return the inventory list declared in the truth seed."""

    # 1 Copy inventory so callers can mutate safely.                            # steps
    return list(profile.truth.inventory)


def location_of(profile: NPCProfile) -> str:
    """Return the default hangout spot for the NPC."""

    # 1 Use morning schedule slot first, otherwise fall back to tavern floor.  # steps
    schedule = profile.truth.schedule
    if "morning" in schedule:
        return schedule["morning"]
    for _, location in schedule.items():
        return location
    return "tavern_floor"


def schedule_for(profile: NPCProfile) -> Dict[str, str]:
    """Return a simple schedule map for the NPC."""

    # 1 Copy schedule to avoid accidental caller side mutation.                # steps
    return dict(profile.truth.schedule)


def style_and_lore_filter(text: str) -> str:
    """Scrub out words that break the demo style guidelines."""

    # 1 Split words, filter, then join again.                                   # steps
    tokens = text.split()
    cleaned: List[str] = []
    for token in tokens:
        lowered = token.lower().strip(".,!?")
        if lowered in FILTERED_WORDS:
            cleaned.append("...")
        else:
            cleaned.append(token)
    return " ".join(cleaned)


# TODO: consider loading filter list from a lore config file for modders.
