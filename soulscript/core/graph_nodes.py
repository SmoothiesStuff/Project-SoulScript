########## Graph Nodes ##########
# LangGraph style node functions coordinating policy, LLM, and effects.

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from . import config
from .db import log_event
from .llm import LLMClient
from .memory import ConversationMemory
from .npc import NPC
from .policy import apply_policy
from .relationships import RelationshipGraph
from .tools import style_and_lore_filter
from .types import ActionType, TraitVector


def node_idle(
    npc: NPC,
    conversation_memory: ConversationMemory,
    relationships: RelationshipGraph,
    llm_client: LLMClient,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Handle idle decisions and gentle state recovery."""

    # 1 Gather allowed actions and helper data.                                # steps
    allowed_actions, tool_outputs = apply_policy(npc, "idle", context)
    llm_context = _build_llm_context(npc, conversation_memory, context, tool_outputs, relationships)
    decision = llm_client.select_action(npc.npc_id, llm_context, allowed_actions)
    if decision.selected_action.action_type == ActionType.IDLE:
        npc.adjust_mood(1)
    return {"decision": decision, "tool_outputs": tool_outputs}


def node_speak(
    npc: NPC,
    conversation_memory: ConversationMemory,
    relationships: RelationshipGraph,
    llm_client: LLMClient,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Coordinate conversation actions and relationship nudges."""

    # 1 Pull nearby NPCs then ask the policy for allowed speak actions.        # steps
    allowed_actions, tool_outputs = apply_policy(npc, "speak", context)
    llm_context = _build_llm_context(npc, conversation_memory, context, tool_outputs, relationships)
    decision = llm_client.select_action(npc.npc_id, llm_context, allowed_actions)
    action = decision.selected_action
    if action.action_type == ActionType.SPEAK and action.target_id:
        line = decision.dialogue_line or style_and_lore_filter("I have been thinking.")
        timestamp = datetime.utcnow()
        items = conversation_memory.record(npc.npc_id, action.target_id, npc.npc_id, line, timestamp)
        summaries = conversation_memory.build_summaries(npc.npc_id, action.target_id, items)
        for (source_id, target_id), summary_text in summaries.items():
            relationships.update_summary(source_id, target_id, summary_text, timestamp)
        target_truth: TraitVector = context.get("target_truth", TraitVector())
        trait_deltas = _toward_truth_delta(relationships, npc.npc_id, action.target_id, target_truth)
        relationships.adjust_relation(
            npc.npc_id,
            action.target_id,
            trust_delta=2,
            affinity_delta=2,
            trait_deltas=trait_deltas,
            summary=summaries.get((npc.npc_id, action.target_id)),
            timestamp=timestamp,
        )
        log_event(
            npc_id=npc.npc_id,
            target_id=action.target_id,
            event_type="action",
            data_json=line,
            timestamp=timestamp,
        )
        npc.adjust_mood(2)
    return {"decision": decision, "tool_outputs": tool_outputs}


def node_decide_join(
    npc: NPC,
    conversation_memory: ConversationMemory,
    relationships: RelationshipGraph,
    llm_client: LLMClient,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Determine whether the NPC joins a highlighted gathering."""

    # 1 Evaluate current gathering info to set up decisions.                   # steps
    allowed_actions, tool_outputs = apply_policy(npc, "decide_join", context)
    llm_context = _build_llm_context(npc, conversation_memory, context, tool_outputs, relationships)
    decision = llm_client.select_action(npc.npc_id, llm_context, allowed_actions)
    if decision.selected_action.action_type == ActionType.JOIN:
        destination = context.get("gathering_spot", "tavern_floor")
        npc.adjust_mood(1)
        log_event(
            npc_id=npc.npc_id,
            target_id=None,
            event_type="action",
            data_json=f"join:{destination}",
            timestamp=datetime.utcnow(),
        )
    return {"decision": decision, "tool_outputs": tool_outputs}


def node_adjust_rel(
    npc: NPC,
    conversation_memory: ConversationMemory,
    relationships: RelationshipGraph,
    llm_client: LLMClient,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply subtle relationship tweaks based on recent context."""

    # 1 Use policy to request adjustment actions.                              # steps
    allowed_actions, tool_outputs = apply_policy(npc, "adjust_relationship", context)
    llm_context = _build_llm_context(npc, conversation_memory, context, tool_outputs, relationships)
    decision = llm_client.select_action(npc.npc_id, llm_context, allowed_actions)
    action = decision.selected_action
    if action.action_type == ActionType.ADJUST_RELATIONSHIP and action.target_id:
        sentiment = context.get("sentiment", 0.0)
        trust_delta = int(round(sentiment * 4))
        affinity_delta = int(round(sentiment * 3))
        relationships.adjust_relation(
            npc.npc_id,
            action.target_id,
            trust_delta=trust_delta,
            affinity_delta=affinity_delta,
            trait_deltas=None,
            summary=None,
            timestamp=datetime.utcnow(),
        )
    return {"decision": decision, "tool_outputs": tool_outputs}


def _build_llm_context(
    npc: NPC,
    conversation_memory: ConversationMemory,
    context: Dict[str, Any],
    tool_outputs: Dict[str, Any],
    relationships: RelationshipGraph,
) -> Dict[str, Any]:
    """Merge base context, tool data, memory bundles, and shared facts."""

    partner_id = context.get("focus_target")
    long_term_summary = ""
    if partner_id:
        edge = relationships.get_edge(npc.npc_id, partner_id)
        long_term_summary = edge.summary or ""
    bundle = conversation_memory.context_bundle(npc.npc_id, partner_id, long_term_summary) if partner_id else {
        "short_term": [],
        "long_term": "",
        "global_facts": list(config.GLOBAL_KNOWLEDGE),
    }
    llm_context: Dict[str, Any] = {
        "profile": {
            "name": npc.profile.truth.name,
            "backstory": npc.profile.truth.backstory,
            "motivation": npc.profile.truth.motivation,
        },
        "state": {
            "mood": npc.mood,
            "self_traits": npc.serialize_self_traits(),
        },
        "short_term": bundle["short_term"],
        "long_term": bundle["long_term"],
        "global_facts": bundle["global_facts"],
        "conversation_partner": partner_id,
    }
    llm_context.update(tool_outputs)
    llm_context.update(context)
    return llm_context


def _toward_truth_delta(
    relationships: RelationshipGraph,
    source_id: str,
    target_id: str,
    target_truth: TraitVector,
) -> Dict[str, int]:
    """Compute trait deltas nudging perception toward truth."""

    # 1 Compare current perception to truth and move a single step.            # steps
    edge = relationships.get_edge(source_id, target_id)
    deltas: Dict[str, int] = {}
    for axis in edge.traits.as_dict():
        current_value = getattr(edge.traits, axis)
        truth_value = getattr(target_truth, axis)
        if truth_value > current_value:
            deltas[axis] = 1
        elif truth_value < current_value:
            deltas[axis] = -1
        else:
            deltas[axis] = 0
    return deltas


# TODO: surface gossip nodes once faction rumors are implemented.
