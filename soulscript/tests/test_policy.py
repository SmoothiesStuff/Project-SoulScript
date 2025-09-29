########## Policy Tests ##########
# Exercises policy configuration for allowed actions and tools.

from __future__ import annotations

from soulscript.core.npc import NPC
from soulscript.core.policy import apply_policy
from soulscript.core.types import ActionType, NPCProfile, NPCTruth, TraitVector


def _build_npc(npc_id: str = "npc_test", role: str = "local") -> NPC:
    """Helper to construct a minimal NPC for policy tests."""

    # 1 Create a profile and state for the given npc_id.                        # steps
    truth = NPCTruth(
        id=npc_id,
        name="Test NPC",
        age=30,
        race="human",
        sex="female",
        sexual_orientation="straight",
        backstory="",
        personality="",
        motivation="",
        role=role,
        traits_truth=TraitVector(),
        traits_self_perception=TraitVector(),
        initial_relationships={},
        inventory=["cup"],
        schedule={"morning": "tavern"},
    )
    profile = NPCProfile(truth=truth)
    return NPC(profile)


def test_speak_policy_selects_target_from_context() -> None:
    """Speak policy should populate action target when partner present."""

    # 1 Run policy with focus target context.                                   # steps
    npc = _build_npc()
    actions, _ = apply_policy(npc, "speak", {"focus_target": "npc_other", "nearby_npcs": ["npc_other"]})
    speak_actions = [action for action in actions if action.action_type == ActionType.SPEAK]
    assert speak_actions
    assert speak_actions[0].target_id == "npc_other"


def test_speak_policy_defaults_to_idle_without_partner() -> None:
    """Speak policy should degrade to idle when no partner is available."""

    # 1 Invoke speak policy with empty nearby list.                             # steps
    npc = _build_npc()
    actions, _ = apply_policy(npc, "speak", {"nearby_npcs": []})
    assert len(actions) == 1
    assert actions[0].action_type == ActionType.IDLE
