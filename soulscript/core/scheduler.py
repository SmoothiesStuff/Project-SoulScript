########## Simulation Scheduler ##########
# Coordinates tick updates across NPCs and invokes graph nodes.

from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import config
from .db import log_event
from .llm import LLMClient
from .npc import NPC
from .runtime import (
    ConversationMemory,
    RelationshipGraph,
    compute_relationship_update,
    log_run_event,
    node_idle,
    node_speak,
)
from .types import ActionType, Decision, RelationshipEdge


########## Sentiment Helpers ##########
# Derive simple sentiment scores from mood for relationship updates.


def _mood_to_sentiment_value(mood: int) -> float:
    """Translate mood into a bounded sentiment score."""

    denom = abs(config.RELATIONSHIP_NEUTRAL) if config.RELATIONSHIP_NEUTRAL != 0 else 100
    score = (mood - config.RELATIONSHIP_NEUTRAL) / denom
    return max(-1.0, min(1.0, score))


class SimulationScheduler:
    """Ticks NPCs in order and records emitted decisions."""

    def __init__(self) -> None:
        # 1 Keep shared systems handy for nodes to touch.                       # steps
        # 2 Prepare storage for NPC roster and run logs.                        # steps
        self.conversation_memory = ConversationMemory()
        self.relationships = RelationshipGraph()
        self.npcs: Dict[str, NPC] = {}
        self.llm_client = LLMClient()
        self.decisions: List[Decision] = []
        self.random = random.Random(config.RANDOM_SEED)
        self.current_tick: int = 0
        self._seed_relationships: Dict[str, Dict[str, RelationshipEdge]] = {}
        self.conversation_sessions: Dict[tuple[str, str], Dict[str, int]] = {}  # tiny in-memory chat locks
        self.pending_effects: List[Dict[str, Any]] = []

    def register(self, npc: NPC) -> None:
        """Add an NPC to the scheduler roster."""

        # 1 Store NPC and collect seed relationships for later bootstrap.      # steps
        self.npcs[npc.npc_id] = npc
        edges: Dict[str, RelationshipEdge] = {}
        for target_id, seed in npc.profile.truth.initial_relationships.items():
            edges[target_id] = RelationshipEdge(
                source_id=npc.npc_id,
                target_id=target_id,
                trust=seed.trust,
                affinity=seed.affinity,
                traits=seed.traits_perception,
                summary=seed.summary,
                updated_at=datetime.utcnow(),
            )
        self._seed_relationships[npc.npc_id] = edges

    def initialize_relationships(self) -> None:
        """Bootstrap the relationship graph once all NPCs are registered."""

        # 1 Pass npc ids and seed edges into the relationship graph.           # steps
        npc_ids = list(self.npcs.keys())
        self.relationships.bootstrap(npc_ids, self._seed_relationships)

    def step(self, world_context: Optional[Dict[str, Any]] = None) -> List[Decision]:
        """Execute one tick across all NPCs and return their decisions."""

        # 1 Update decay timing for relationships.                              # steps
        # 2 Iterate through NPCs and run the node stack.                        # steps
        if world_context is None:
            world_context = {}
        now = datetime.utcnow()
        self.relationships.decay_all(now)
        allowed_ids = world_context.get("allowed_npcs")
        if allowed_ids:
            ordered_npcs = [self.npcs[npc_id] for npc_id in allowed_ids if npc_id in self.npcs]
        else:
            ordered_npcs = list(self.npcs.values())
        self.random.shuffle(ordered_npcs)
        tick_decisions: List[Decision] = []
        for npc in ordered_npcs:
            npc.tick(now)
            context = self._build_npc_context(npc, world_context)
            context["pending_effects"] = self.pending_effects
            partner_id, session = self._session_for_npc(npc.npc_id)
            has_session = session is not None
            if not has_session:
                result_idle = node_idle(
                    npc.profile,
                    npc.mood,
                    self.conversation_memory,
                    self.relationships,
                    self.llm_client,
                    context,
                )
                tick_decisions.append(result_idle["decision"])
            result_speak = node_speak(
                npc.profile,
                npc.mood,
                self.conversation_memory,
                self.relationships,
                self.llm_client,
                context,
            )
            tick_decisions.append(result_speak["decision"])
            action = result_speak["decision"].selected_action
            if action.action_type == ActionType.SPEAK and action.target_id:
                self._advance_session(npc.npc_id, action.target_id)
        self.decisions.extend(tick_decisions)
        self.current_tick += 1
        return tick_decisions

    def _build_npc_context(self, npc: NPC, world_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble per NPC context including nearby roster."""

        # 1 Determine nearby NPC ids for policy targeting.                      # steps
        context = dict(world_context)
        nearby: List[str] = []
        table_map: Dict[str, Any] | None = context.get("table_map")
        active_pairs: Dict[Any, tuple[str, str]] = context.get("active_pairs", {}) if context else {}
        bubble_pair: Optional[tuple[str, str]] = None
        npc_table = None
        if table_map:
            npc_table = table_map.get(npc.npc_id)
            bubble_pair = active_pairs.get(npc_table)
            if bubble_pair and npc.npc_id not in bubble_pair:
                candidate_ids: List[str] = []
            else:
                candidate_ids = [other_id for other_id, tbl in table_map.items() if tbl == npc_table and other_id != npc.npc_id]
        else:
            allowed_ids = context.get("allowed_npcs")
            candidate_ids = allowed_ids if allowed_ids else self.npcs.keys()
        for other_id in candidate_ids:
            if other_id == npc.npc_id:
                continue
            nearby.append(other_id)
        context["nearby_npcs"] = nearby
        if npc_table is not None:
            context["table_id"] = npc_table
        session_partner, session = self._session_for_npc(npc.npc_id)
        focus_target = session_partner or context.get("focus_target")
        if bubble_pair and npc.npc_id in bubble_pair:
            partner = bubble_pair[0] if bubble_pair[1] == npc.npc_id else bubble_pair[1]
            focus_target = partner
            context["defer_effects"] = True
        if focus_target not in nearby:
            focus_target = self._select_focus(npc, nearby)
        if focus_target and session is None:
            session = self._ensure_session(npc.npc_id, focus_target)
        context["focus_target"] = focus_target
        if focus_target:
            context["target_truth"] = self.npcs[focus_target].truth_vector()
            context["conversation_partner"] = focus_target
        if session:
            context["session_turns_left"] = session["turns_left"]
            context["defer_effects"] = True
        denom = abs(config.RELATIONSHIP_NEUTRAL) if config.RELATIONSHIP_NEUTRAL != 0 else 100
        sentiment = (npc.mood - config.RELATIONSHIP_NEUTRAL) / denom
        context["sentiment"] = max(-1.0, min(1.0, sentiment))
        context.setdefault("gathering_spot", "tavern_floor")
        return context

    def _select_focus(self, npc: NPC, nearby: List[str]) -> Optional[str]:
        """Choose a conversation target using trust-biased sampling."""

        # 1 Weight locals by current trust to keep conversations varied.        # steps
        if not nearby:
            return None
        weights: List[float] = []
        for target_id in nearby:
            edge = self.relationships.get_edge(npc.npc_id, target_id)
            weights.append(max(edge.trust, 1))
        total = sum(weights)
        pick = self.random.uniform(0, total)
        cumulative = 0.0
        for target_id, weight in zip(nearby, weights):
            cumulative += weight
            if pick <= cumulative:
                return target_id
        return nearby[0]

    def reset(self) -> None:
        """Clear run state while keeping NPC roster intact."""

        # 1 Reset tick counter and decision log.                                # steps
        self.decisions.clear()
        self.current_tick = 0
        self.conversation_sessions.clear()
        self.pending_effects.clear()

    def run(self, ticks: int, world_context: Optional[Dict[str, Any]] = None) -> List[Decision]:
        """Convenience helper to run multiple ticks."""

        # 1 Loop for the requested number of ticks.                              # steps
        collected: List[Decision] = []
        for _ in range(ticks):
            cycle_decisions = self.step(world_context)
            collected.extend(cycle_decisions)
        return collected

    def finalize_conversation(self) -> None:
        """Apply deferred relationship effects at conversation end."""

        now = datetime.utcnow()
        pair_effects: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for effect in list(self.pending_effects):
            key = (effect["source_id"], effect["target_id"])
            pair_effects.setdefault(key, []).append(effect)
        processed_pairs: set[tuple[str, str]] = set()
        for (source_id, target_id), effects in pair_effects.items():
            timestamp = effects[-1].get("timestamp", now)
            items = self.conversation_memory.history(source_id, target_id)
            lines = [f"{item.speaker_id}: {item.text}" for item in items]
            # Update summaries both directions.
            edge_st = self.relationships.get_edge(source_id, target_id)
            prev_summary_st = edge_st.summary or ""
            summary_st = self.llm_client.summarize_conversation(prev_summary_st, lines, source_id, target_id, edge_st.trust, edge_st.affinity) if lines else prev_summary_st
            edge_ts = self.relationships.get_edge(target_id, source_id)
            prev_summary_ts = edge_ts.summary or ""
            summary_ts = self.llm_client.summarize_conversation(prev_summary_ts, lines, target_id, source_id, edge_ts.trust, edge_ts.affinity) if lines else prev_summary_ts
            trait_deltas = effects[-1].get("trait_deltas")
            update_st = compute_relationship_update(
                self.npcs[source_id].profile.truth.traits_truth,
                self.npcs[target_id].profile.truth.traits_truth,
                summary_st or "",
                lines,
                sentiment=_mood_to_sentiment_value(self.npcs[source_id].mood),
                current_trust=edge_st.trust,
                current_affinity=edge_st.affinity,
            )
            update_ts = compute_relationship_update(
                self.npcs[target_id].profile.truth.traits_truth,
                self.npcs[source_id].profile.truth.traits_truth,
                summary_ts or "",
                lines,
                sentiment=_mood_to_sentiment_value(self.npcs[target_id].mood),
                current_trust=edge_ts.trust,
                current_affinity=edge_ts.affinity,
            )
            self.relationships.adjust_relation(
                source_id,
                target_id,
                trust_delta=update_st["trust_delta"],
                affinity_delta=update_st["affinity_delta"],
                trait_deltas=trait_deltas,
                summary=summary_st,
                timestamp=timestamp,
            )
            self.relationships.update_summary(source_id, target_id, summary_st or "", timestamp)
            self.relationships.adjust_relation(
                target_id,
                source_id,
                trust_delta=update_ts["trust_delta"],
                affinity_delta=update_ts["affinity_delta"],
                trait_deltas=None,
                summary=summary_ts,
                timestamp=timestamp,
            )
            self.relationships.update_summary(target_id, source_id, summary_ts or "", timestamp)
            log_run_event(
                f"summary {source_id}->{target_id}: '{summary_st}' trust_delta={update_st['trust_delta']} affinity_delta={update_st['affinity_delta']} traits={trait_deltas or {}}"
            )
            log_run_event(
                f"summary {target_id}->{source_id}: '{summary_ts}' trust_delta={update_ts['trust_delta']} affinity_delta={update_ts['affinity_delta']}"
            )
            log_event(
                npc_id=source_id,
                target_id=target_id,
                event_type="action",
                data_json=effects[-1]["line"],
                timestamp=timestamp,
            )
            self.npcs[source_id].adjust_mood(2)
            processed_pairs.add(self._pair_key(source_id, target_id))
        for key in processed_pairs:
            self.conversation_sessions.pop(key, None)
            # Reset conversation history so future chats start clean.
            self.conversation_memory.clear_history(key[0], key[1])
        # Clear transcript logs in memory too.
        self.decisions = []
        self.pending_effects.clear()

    def _pair_key(self, npc_a: str, npc_b: str) -> tuple[str, str]:
        """Stable pair key for session tracking."""

        return tuple(sorted((npc_a, npc_b)))

    def _ensure_session(self, npc_a: str, npc_b: str) -> Dict[str, int]:
        """Start or reuse a tiny chat session for the pair."""

        key = self._pair_key(npc_a, npc_b)
        if key not in self.conversation_sessions:
            self.conversation_sessions[key] = {"turns_left": config.CONVERSATION_LENGTH_TURNS}
        return self.conversation_sessions[key]

    def _session_for_npc(self, npc_id: str) -> tuple[Optional[str], Optional[Dict[str, int]]]:
        """Find an active session for this NPC, if any."""

        for (first, second), session in self.conversation_sessions.items():
            if npc_id == first:
                return second, session
            if npc_id == second:
                return first, session
        return None, None

    def _advance_session(self, npc_a: str, npc_b: str) -> None:
        """Count down the chat session and clear when done."""

        key = self._pair_key(npc_a, npc_b)
        session = self.conversation_sessions.get(key)
        if not session:
            return
        session["turns_left"] -= 1
        if session["turns_left"] <= 0:
            self.conversation_sessions.pop(key, None)
            # Apply deferred effects when a conversation naturally ends.
            if self.pending_effects:
                self.finalize_conversation()


# TODO: add concurrency once UI polling moves to async.
