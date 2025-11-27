########## Core Types ##########
# Pydantic models and enums that describe SoulScript tavern data.

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from . import config

TRAIT_AXES: List[str] = [
    "kindness",
    "bravery",
    "extraversion",
    "ego",
    "honesty",
    "curiosity",
    "patience",
    "optimism",
    "intelligence",
    "charisma",
]


class TraitVector(BaseModel):
    """Fixed set of bipolar trait values with validation and helpers."""

    kindness: int = Field(default=0)
    bravery: int = Field(default=0)
    extraversion: int = Field(default=0)
    ego: int = Field(default=0)
    honesty: int = Field(default=0)
    curiosity: int = Field(default=0)
    patience: int = Field(default=0)
    optimism: int = Field(default=0)
    intelligence: int = Field(default=0)
    charisma: int = Field(default=0)

    @model_validator(mode="after")
    def _clamp_values(self) -> "TraitVector":
        # 1 Clamp each axis into the configured hard range.                    # steps
        for axis in TRAIT_AXES:
            value = getattr(self, axis)
            if value < config.TRAIT_MIN:
                setattr(self, axis, config.TRAIT_MIN)
            if value > config.TRAIT_MAX:
                setattr(self, axis, config.TRAIT_MAX)
        return self

    def as_dict(self) -> Dict[str, int]:
        """Return a plain dictionary version."""

        # 1 Explicitly copy each axis to avoid surprises.                      # steps
        payload: Dict[str, int] = {}
        for axis in TRAIT_AXES:
            payload[axis] = getattr(self, axis)
        return payload

    def with_delta(self, deltas: Dict[str, int], soft_limit: int = config.TRAIT_SOFT_LIMIT) -> "TraitVector":
        """Return a new vector with deltas applied and soft limited."""

        # 1 Apply delta per axis, reduce magnitude if we cross the soft cap.    # steps
        updated: Dict[str, int] = {}
        for axis in TRAIT_AXES:
            base_value = getattr(self, axis)
            delta = deltas.get(axis, 0)
            candidate = base_value + delta
            if delta > 0 and base_value >= soft_limit:
                candidate = base_value + int(delta * 0.5)
            if delta < 0 and base_value <= -soft_limit:
                candidate = base_value + int(delta * 0.5)
            if candidate > config.TRAIT_MAX:
                candidate = config.TRAIT_MAX
            if candidate < config.TRAIT_MIN:
                candidate = config.TRAIT_MIN
            updated[axis] = candidate
        return TraitVector(**updated)

    def drift_toward(self, target: "TraitVector", weight: float) -> "TraitVector":
        """Move toward target by weight (0..1)."""

        # 1 Blend each axis toward the target using the supplied weight.        # steps
        updated: Dict[str, int] = {}
        for axis in TRAIT_AXES:
            base_value = getattr(self, axis)
            target_value = getattr(target, axis)
            delta = (target_value - base_value) * weight
            candidate = int(round(base_value + delta))
            if candidate > config.TRAIT_MAX:
                candidate = config.TRAIT_MAX
            if candidate < config.TRAIT_MIN:
                candidate = config.TRAIT_MIN
            updated[axis] = candidate
        return TraitVector(**updated)


class RelationshipSeed(BaseModel):
    """Seed data for locals who already know someone."""

    trust: int = 50
    affinity: int = 50
    traits_perception: TraitVector = Field(default_factory=TraitVector)
    summary: Optional[str] = None


class NPCTruth(BaseModel):
    """Developer-authored truth profile loaded from JSON."""

    npc_id: str = Field(alias="id")
    name: str
    age: int
    race: str
    sex: str
    sexual_orientation: str
    backstory: str
    traits: str
    motivation: str
    role: str = Field(default="local")
    trait_inputs: Dict[str, int] = Field(default_factory=dict)
    traits_truth: TraitVector
    traits_self_perception: TraitVector
    initial_relationships: Dict[str, RelationshipSeed] = Field(default_factory=dict)
    inventory: List[str] = Field(default_factory=list)
    schedule: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _clamp_trait_inputs(self) -> "NPCTruth":
        """Keep trait inputs in -100..100 so prompts stay sane."""

        clamped: Dict[str, int] = {}
        for key, value in self.trait_inputs.items():
            if value < -100:
                clamped[key] = -100
            elif value > 100:
                clamped[key] = 100
            else:
                clamped[key] = int(value)
        self.trait_inputs = clamped
        return self


class NPCProfile(BaseModel):
    """Base profile exposed to other modules."""

    truth: NPCTruth


class ActionType(str, Enum):
    """ActionType enumerates the high level behaviors NPCs may perform."""

    IDLE = "idle"
    SPEAK = "speak"
    JOIN = "decide_join"
    ADJUST_RELATIONSHIP = "adjust_relationship"
    OBSERVE = "observe"


class EventType(str, Enum):
    """EventType captures the kinds of happenings NPCs can observe."""

    PLAYER = "player_event"
    NPC = "npc_event"
    TIME = "time_event"
    SYSTEM = "system_event"


class Action(BaseModel):
    """Represents a single action that an NPC may execute."""

    action_type: ActionType
    target_id: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class Decision(BaseModel):
    """Decision package returned by the LLM policy layer."""

    npc_id: str
    selected_action: Action
    reason: str
    dialogue_line: Optional[str] = None
    confidence: float = 0.0


class ConversationItem(BaseModel):
    """Single dialogue line stored per NPC pair."""

    timestamp: datetime
    speaker_id: str
    text: str


class RelationshipEdge(BaseModel):
    """Stores affinity, trust, and trait perception between two NPCs."""

    source_id: str
    target_id: str
    trust: int = config.RELATIONSHIP_NEUTRAL
    affinity: int = config.RELATIONSHIP_NEUTRAL
    traits: TraitVector = Field(default_factory=TraitVector)
    summary: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SimulationEvent(BaseModel):
    """Raw event flowing through the simulation loop."""

    event_type: EventType
    source_id: str
    target_id: Optional[str] = None
    payload: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# TODO: add enums for visitor vs local roles to tighten validation once seeds stabilize.
