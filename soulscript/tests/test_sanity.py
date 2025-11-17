########## Sanity Test Suite ##########
# Quick checks for helpers, memory, policy filtering, and stub LLM behavior.

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from soulscript.core import config
from soulscript.core import llm as llm_module
from soulscript.core.agentic_helpers import parse_json_loose
from soulscript.core import db
from soulscript.core.memory import ConversationMemory
from soulscript.core.policy import apply_policy
from soulscript.core.types import Action, ActionType, Decision, NPCProfile, NPCTruth, TraitVector


def _make_profile(npc_id: str) -> NPCProfile:
    """Build a minimal NPC profile for tests."""

    truth = NPCTruth(
        id=npc_id,
        name=npc_id,
        age=20,
        race="human",
        sex="female",
        sexual_orientation="bi",
        backstory="",
        personality="calm",
        motivation="belonging",
        role="local",
        traits_truth=TraitVector(),
        traits_self_perception=TraitVector(),
        initial_relationships={},
        inventory=[],
        schedule={},
    )
    return NPCProfile(truth=truth)


def test_parse_json_loose_handles_think_block() -> None:
    """Should recover JSON wrapped inside <think> noise."""

    raw = "<think>internal</think> ```json\n{\"action\":\"speak\",\"reason\":\"hi\"}\n```"
    parsed = parse_json_loose(raw)
    assert parsed is not None
    assert parsed.get("action") == "speak"


def test_conversation_memory_bundle_builds_short_and_long_term(tmp_path) -> None:
    """Context bundle should include short_term lines and a long_term summary."""

    config.DB_FILE = str(tmp_path / "memory_test.sqlite")
    db._ENGINE = None  # reset cached engine for isolated test
    memory = ConversationMemory(keep=4)
    now = datetime.utcnow()
    memory.record("npc_a", "npc_b", "npc_a", "Hello", now)
    memory.record("npc_a", "npc_b", "npc_b", "Hey there", now)
    bundle = memory.context_bundle("npc_a", "npc_b", existing_summary=None)
    assert any(line.endswith("Hey there") for line in bundle["short_term"])
    assert isinstance(bundle["long_term"], str)
    assert config.GLOBAL_KNOWLEDGE[0] in bundle["global_facts"]


def test_policy_respects_action_set(monkeypatch) -> None:
    """Policy filtering should only allow actions present in ACTION_SET."""

    original_actions = list(config.ACTION_SET)
    config.ACTION_SET = ["speak"]
    try:
        npc = llm_module.NPC(_make_profile("npc_a"))  # type: ignore[attr-defined]
    except AttributeError:
        # NPC is in another module; import lazily to avoid circulars in type hints.
        from soulscript.core.npc import NPC  # noqa: WPS433

        npc = NPC(_make_profile("npc_a"))
    allowed, _ = apply_policy(npc, "speak", {"focus_target": "npc_b"})
    assert all(action.action_type == ActionType.SPEAK for action in allowed)
    config.ACTION_SET = original_actions


def test_stub_llm_speak_includes_memory_and_facts() -> None:
    """Stub client should return a speak line when allowed."""

    stub = llm_module.StubLLMClient()
    actions = [Action(action_type=ActionType.SPEAK, target_id="npc_b")]
    context: Dict[str, Any] = {
        "short_term": ["npc_b: Hello"],
        "global_facts": ["Fact A"],
    }
    decision: Decision = stub.select_action("npc_a", context, actions)
    assert decision.selected_action.action_type == ActionType.SPEAK
    assert "Hello" in (decision.dialogue_line or "")
    assert "Fact" in (decision.dialogue_line or "")


def test_llm_factory_falls_back_when_client_init_fails(monkeypatch) -> None:
    """LLMClient should return the stub when OpenAI client construction fails."""

    class FailingOpenAI:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("boom")

    original_openai = llm_module.OpenAI
    llm_module.OpenAI = FailingOpenAI  # type: ignore[assignment]
    try:
        client = llm_module.LLMClient()
        assert isinstance(client, llm_module.StubLLMClient)
    finally:
        llm_module.OpenAI = original_openai  # type: ignore[assignment]
