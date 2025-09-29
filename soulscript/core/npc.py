########## NPC Container ##########
# NPC wraps truth data, runtime self perception, and persistence hooks.

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from . import config
from .db import load_npc_state, upsert_npc_state
from .types import NPCProfile, TraitVector, TRAIT_AXES


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp helper for mood values."""

    # 1 Apply min/max in two steps for clarity.                                # steps
    if value < low:
        return low
    if value > high:
        return high
    return value


class NPC:
    """NPC binds developer truth with mutable self perception."""

    def __init__(self, profile: NPCProfile) -> None:
        # 1 Store the immutable profile truth.                                  # steps
        # 2 Hydrate runtime fields from the database when present.             # steps
        self.profile = profile
        self.npc_id = profile.truth.npc_id
        self.role = profile.truth.role
        self.mood: int = config.INITIAL_MOOD
        self.last_tick: datetime = datetime.utcnow()
        self.self_perception: TraitVector = profile.truth.traits_self_perception
        self._load_runtime_state()

    def _load_runtime_state(self) -> None:
        """Pull stored self perception and mood from sqlite."""

        # 1 Query the db and override defaults when data exists.               # steps
        state = load_npc_state(self.npc_id)
        if state is None:
            return
        if "mood" in state:
            self.mood = _clamp(int(state["mood"]), config.MOOD_MIN, config.MOOD_MAX)
        trait_payload: Dict[str, int] = {}
        for axis in TRAIT_AXES:
            trait_payload[axis] = int(state.get(axis, getattr(self.self_perception, axis)))
        self.self_perception = TraitVector(**trait_payload)
        last_tick_value = state.get("last_tick")
        if last_tick_value:
            self.last_tick = datetime.fromisoformat(last_tick_value)

    def tick(self, timestamp: datetime) -> None:
        """Record the passage of time for persistence."""

        # 1 Update timestamp and persist current state.                         # steps
        self.last_tick = timestamp
        upsert_npc_state(self.npc_id, self.mood, self.last_tick, self.self_perception.as_dict())

    def adjust_mood(self, delta: int) -> None:
        """Modify mood while respecting configured bounds."""

        # 1 Clamp adjustments so mood stays inside 0..100.                     # steps
        self.mood = _clamp(self.mood + delta, config.MOOD_MIN, config.MOOD_MAX)

    def adjust_self_perception(self, deltas: Dict[str, int]) -> None:
        """Apply trait deltas with the configured soft limit."""

        # 1 Trim deltas to the per event cap.                                   # steps
        trimmed: Dict[str, int] = {}
        for axis in TRAIT_AXES:
            raw_delta = deltas.get(axis, 0)
            if raw_delta > config.TRAIT_EVENT_DELTA:
                trimmed[axis] = config.TRAIT_EVENT_DELTA
            elif raw_delta < -config.TRAIT_EVENT_DELTA:
                trimmed[axis] = -config.TRAIT_EVENT_DELTA
            else:
                trimmed[axis] = raw_delta
        self.self_perception = self.self_perception.with_delta(trimmed)

    def truth_vector(self) -> TraitVector:
        """Return the immutable truth trait vector."""

        # 1 Expose the developer authored trait snapshot.                      # steps
        return self.profile.truth.traits_truth

    def truth_table(self) -> List[Dict[str, int]]:
        """Return rows for truth vs self comparison."""

        # 1 Build list of dictionaries for use in the UI table.                # steps
        rows: List[Dict[str, int]] = []
        truth = self.truth_vector()
        for axis in TRAIT_AXES:
            rows.append(
                {
                    "trait": axis,
                    "truth": getattr(truth, axis),
                    "self": getattr(self.self_perception, axis),
                }
            )
        return rows

    def public_view(self) -> Dict[str, str]:
        """Return a sanitized dict for UI consumption."""

        # 1 Expose key presentation fields for cards and filters.              # steps
        return {
            "npc_id": self.npc_id,
            "name": self.profile.truth.name,
            "mood": f"{self.mood}",
            "motivation": self.profile.truth.motivation,
            "role": self.role,
        }

    def serialize_self_traits(self) -> Dict[str, int]:
        """Expose the self perception dict for external modules."""

        # 1 Delegate to TraitVector utility but keep naming consistent.        # steps
        return self.self_perception.as_dict()


# TODO: extend NPC with helper for localized schedule/inventory once gameplay expands.
