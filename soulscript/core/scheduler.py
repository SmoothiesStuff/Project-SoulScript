########## Simulation Scheduler ##########
# Coordinates tick updates across NPCs and invokes graph nodes.

from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import config
from .graph_nodes import node_idle, node_speak
from .llm import LLMClient
from .memory import ConversationMemory
from .npc import NPC
from .relationships import RelationshipGraph
from .types import Decision, RelationshipEdge


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
        ordered_npcs = list(self.npcs.values())
        self.random.shuffle(ordered_npcs)
        tick_decisions: List[Decision] = []
        for npc in ordered_npcs:
            npc.tick(now)
            context = self._build_npc_context(npc, world_context)
            result_idle = node_idle(npc, self.conversation_memory, self.relationships, self.llm_client, context)
            tick_decisions.append(result_idle["decision"])
            result_speak = node_speak(npc, self.conversation_memory, self.relationships, self.llm_client, context)
            tick_decisions.append(result_speak["decision"])
        self.decisions.extend(tick_decisions)
        self.current_tick += 1
        return tick_decisions

    def _build_npc_context(self, npc: NPC, world_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble per NPC context including nearby roster."""

        # 1 Determine nearby NPC ids for policy targeting.                      # steps
        context = dict(world_context)
        nearby: List[str] = []
        for other_id in self.npcs:
            if other_id == npc.npc_id:
                continue
            nearby.append(other_id)
        context["nearby_npcs"] = nearby
        focus_target = context.get("focus_target")
        if focus_target not in nearby:
            focus_target = self._select_focus(npc, nearby)
            context["focus_target"] = focus_target
        if focus_target:
            target_truth = self.npcs[focus_target].truth_vector()
            context["target_truth"] = target_truth
            context["conversation_partner"] = focus_target
        sentiment = (npc.mood - config.RELATIONSHIP_NEUTRAL) / config.RELATIONSHIP_NEUTRAL
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

    def run(self, ticks: int, world_context: Optional[Dict[str, Any]] = None) -> List[Decision]:
        """Convenience helper to run multiple ticks."""

        # 1 Loop for the requested number of ticks.                              # steps
        collected: List[Decision] = []
        for _ in range(ticks):
            cycle_decisions = self.step(world_context)
            collected.extend(cycle_decisions)
        return collected


# TODO: add concurrency once UI polling moves to async.
