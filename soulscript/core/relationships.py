########## Relationship Graph ##########
# Tracks trust, affinity, and perceived traits with sqlite persistence.

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional

import networkx as nx

from . import config
from .db import ensure_schema, load_relationship, upsert_relationship
from .types import RelationshipEdge, TraitVector, TRAIT_AXES


def _soft_metric_update(base: int, delta: int) -> int:
    """Apply delta with soft limits near extremes."""

    # 1 Shrink deltas when we approach the soft clamp threshold.               # steps
    candidate = base + delta
    if base >= config.RELATIONSHIP_SOFT_LIMIT and delta > 0:
        candidate = base + int(delta * 0.5)
    lower_soft = config.RELATIONSHIP_TRUST_CLAMP[0] + (config.RELATIONSHIP_NEUTRAL - (config.RELATIONSHIP_SOFT_LIMIT - config.RELATIONSHIP_NEUTRAL))
    if base <= lower_soft and delta < 0:
        candidate = base + int(delta * 0.5)
    if candidate > config.RELATIONSHIP_TRUST_CLAMP[1]:
        return config.RELATIONSHIP_TRUST_CLAMP[1]
    if candidate < config.RELATIONSHIP_TRUST_CLAMP[0]:
        return config.RELATIONSHIP_TRUST_CLAMP[0]
    return candidate


class RelationshipGraph:
    """Graph helper that owns perceptions and syncs with sqlite."""

    def __init__(self) -> None:
        # 1 Initialize directed graph and ensure database schema exists.        # steps
        self.graph = nx.DiGraph()
        ensure_schema()

    def bootstrap(self, npc_ids: Iterable[str], seeds: Dict[str, Dict[str, RelationshipEdge]]) -> None:
        """Load existing edges from seeds and sqlite."""

        # 1 Seed graph with provided edges then ensure sqlite state mirrors it. # steps
        for npc_id in npc_ids:
            if not self.graph.has_node(npc_id):
                self.graph.add_node(npc_id)
        for source_id, entries in seeds.items():
            for target_id, edge in entries.items():
                self._write_edge(edge)
        # 2 Pull existing sqlite rows so repeat runs remain consistent.        # steps
        for source_id in npc_ids:
            for target_id in npc_ids:
                if source_id == target_id:
                    continue
                self._ensure_edge(source_id, target_id)

    def get_edge(self, source_id: str, target_id: str) -> RelationshipEdge:
        """Expose the current edge state for external modules."""

        # 1 Delegate to ensure edge so newly seen pairs get neutral defaults.   # steps
        return self._ensure_edge(source_id, target_id)

    def _ensure_edge(self, source_id: str, target_id: str) -> RelationshipEdge:
        """Fetch the edge data or initialize a neutral one."""

        # 1 Try the in-memory graph then fall back to sqlite and defaults.      # steps
        if self.graph.has_edge(source_id, target_id):
            data = self.graph[source_id][target_id]
            return RelationshipEdge(
                source_id=source_id,
                target_id=target_id,
                trust=data["trust"],
                affinity=data["affinity"],
                traits=data["traits"],
                summary=data.get("summary"),
                updated_at=data.get("updated_at", datetime.utcnow()),
            )
        stored = load_relationship(source_id, target_id)
        if stored:
            edge = RelationshipEdge(
                source_id=source_id,
                target_id=target_id,
                trust=stored.get("trust", config.RELATIONSHIP_NEUTRAL),
                affinity=stored.get("affinity", config.RELATIONSHIP_NEUTRAL),
                traits=TraitVector(**{axis: stored[axis] for axis in TRAIT_AXES}),
                summary=stored.get("summary"),
                updated_at=datetime.fromisoformat(stored.get("updated_at")) if stored.get("updated_at") else datetime.utcnow(),
            )
        else:
            edge = RelationshipEdge(
                source_id=source_id,
                target_id=target_id,
            )
        self._write_edge(edge)
        return edge

    def adjust_relation(
        self,
        source_id: str,
        target_id: str,
        trust_delta: int,
        affinity_delta: int,
        trait_deltas: Optional[Dict[str, int]] = None,
        summary: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> RelationshipEdge:
        """Apply interaction deltas and persist updated state."""

        # 1 Ensure edge exists before mutating values.                          # steps
        edge = self._ensure_edge(source_id, target_id)
        timestamp = timestamp or datetime.utcnow()
        trust_value = _soft_metric_update(edge.trust, trust_delta)
        affinity_value = _soft_metric_update(edge.affinity, affinity_delta)
        updated_traits = edge.traits
        if trait_deltas:
            updated_traits = edge.traits.with_delta(trait_deltas)
        edge = RelationshipEdge(
            source_id=source_id,
            target_id=target_id,
            trust=trust_value,
            affinity=affinity_value,
            traits=updated_traits,
            summary=summary if summary is not None else edge.summary,
            updated_at=timestamp,
        )
        self._write_edge(edge)
        return edge

    def update_summary(self, source_id: str, target_id: str, summary: str, timestamp: datetime) -> None:
        """Update directional summary while keeping other metrics intact."""

        # 1 Fetch the existing edge then overwrite the summary field.          # steps
        edge = self._ensure_edge(source_id, target_id)
        edge.summary = summary
        edge.updated_at = timestamp
        self._write_edge(edge)

    def decay_all(self, now: datetime) -> None:
        """Ease trust, affinity, and traits back toward neutral."""

        # 1 Walk each edge and apply small drift when idle.                    # steps
        for source_id, target_id, data in list(self.graph.edges(data=True)):
            last_update: datetime = data.get("updated_at", now)
            elapsed = now - last_update
            if elapsed < config.RELATIONSHIP_DECAY_TIMESTEP:
                continue
            trust_value = self._decay_metric(data["trust"])
            affinity_value = self._decay_metric(data["affinity"])
            traits: TraitVector = data["traits"].drift_toward(TraitVector(), weight=0.2)
            edge = RelationshipEdge(
                source_id=source_id,
                target_id=target_id,
                trust=trust_value,
                affinity=affinity_value,
                traits=traits,
                summary=data.get("summary"),
                updated_at=now,
            )
            self._write_edge(edge)

    def export_edges(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Return a serializable view of the graph for UI work."""

        # 1 Walk edges and build nested dict for JSON.                          # steps
        export: Dict[str, Dict[str, Dict[str, float]]] = {}
        for source_id, target_id, data in self.graph.edges(data=True):
            if source_id not in export:
                export[source_id] = {}
            export[source_id][target_id] = {
                "trust": float(data["trust"]),
                "affinity": float(data["affinity"]),
            }
        return export

    def _write_edge(self, edge: RelationshipEdge) -> None:
        """Persist edge to graph and sqlite."""

        # 1 Update in-memory graph attributes.                                 # steps
        if not self.graph.has_node(edge.source_id):
            self.graph.add_node(edge.source_id)
        if not self.graph.has_node(edge.target_id):
            self.graph.add_node(edge.target_id)
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            trust=edge.trust,
            affinity=edge.affinity,
            traits=edge.traits,
            summary=edge.summary,
            updated_at=edge.updated_at,
        )
        upsert_relationship(
            edge.source_id,
            edge.target_id,
            trust=edge.trust,
            affinity=edge.affinity,
            trait_perception=edge.traits.as_dict(),
            summary=edge.summary,
            updated_at=edge.updated_at,
        )

    def _decay_metric(self, value: int) -> int:
        """Move a metric toward neutral by a configured step."""

        # 1 Determine direction relative to neutral.                           # steps
        neutral = config.RELATIONSHIP_NEUTRAL
        if value > neutral:
            next_value = value - config.RELATIONSHIP_DECAY_PER_TICK
            if next_value < neutral:
                return neutral
            return int(next_value)
        if value < neutral:
            next_value = value + config.RELATIONSHIP_DECAY_PER_TICK
            if next_value > neutral:
                return neutral
            return int(next_value)
        return neutral


# TODO: extend relationship graph with reputation layers once factions arrive.
