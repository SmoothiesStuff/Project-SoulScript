########## Relationship Tests ##########
# Confirms bounded updates, perception decay, and stability.

from __future__ import annotations

from datetime import datetime, timedelta

from soulscript.core import config
from soulscript.core.relationships import RelationshipGraph
from soulscript.core.types import RelationshipEdge, TraitVector


def _mock_seed(source_id: str, target_id: str) -> RelationshipEdge:
    return RelationshipEdge(
        source_id=source_id,
        target_id=target_id,
        trust=60,
        affinity=55,
        traits=TraitVector(kindness=10),
        summary="",
        updated_at=datetime.utcnow(),
    )


def test_adjust_relation_clamps_and_soft_limits() -> None:
    """Large deltas should clamp and soften near extremes."""

    # 1 Apply a big delta to trigger soft clamp behavior.                       # steps
    graph = RelationshipGraph()
    graph.bootstrap(["a", "b"], {"a": {"b": _mock_seed("a", "b")}})
    edge = graph.adjust_relation("a", "b", trust_delta=20, affinity_delta=20, trait_deltas={"kindness": 10})
    assert edge.trust <= config.RELATIONSHIP_TRUST_CLAMP[1]
    assert edge.affinity <= config.RELATIONSHIP_AFFINITY_CLAMP[1]


def test_decay_all_moves_edges_toward_neutral_without_overshoot() -> None:
    """Decay should drift values toward neutral without crossing it."""

    # 1 Seed an edge with positive trust and affinity then decay twice.        # steps
    graph = RelationshipGraph()
    graph.bootstrap(["a", "b"], {"a": {"b": _mock_seed("a", "b")}})
    # Force earlier timestamp
    graph.adjust_relation("a", "b", trust_delta=20, affinity_delta=20, timestamp=datetime.utcnow() - config.RELATIONSHIP_DECAY_TIMESTEP - timedelta(seconds=1))
    first_now = datetime.utcnow()
    graph.decay_all(first_now)
    first_edge = graph.get_edge("a", "b")
    assert first_edge.trust <= 80
    second_now = first_now + config.RELATIONSHIP_DECAY_TIMESTEP
    graph.decay_all(second_now)
    second_edge = graph.get_edge("a", "b")
    assert config.RELATIONSHIP_NEUTRAL <= first_edge.trust
    assert config.RELATIONSHIP_NEUTRAL <= second_edge.trust <= first_edge.trust
