########## Trait Tests ##########
# Validates trait clamping and drift helpers.

from __future__ import annotations

from soulscript.core import config
from soulscript.core.types import TraitVector


def test_trait_vector_clamps_values() -> None:
    """TraitVector should clamp constructor inputs into configured bounds."""

    # 1 Instantiate with out-of-range values to ensure clamping.               # steps
    vector = TraitVector(kindness=200, bravery=-150)
    assert vector.kindness == config.TRAIT_MAX
    assert vector.bravery == config.TRAIT_MIN


def test_trait_vector_soft_limit_on_delta() -> None:
    """Applying delta past soft limit should dampen the change."""

    # 1 Start near the soft cap and apply a positive delta.                    # steps
    base = TraitVector(kindness=config.TRAIT_SOFT_LIMIT)
    updated = base.with_delta({"kindness": 10})
    assert updated.kindness == config.TRAIT_SOFT_LIMIT + 5


def test_trait_vector_drift_toward_truth() -> None:
    """Drift should move traits toward the target value by weight."""

    # 1 Blend halfway toward target and ensure movement is proportional.       # steps
    origin = TraitVector(kindness=0)
    target = TraitVector(kindness=80)
    blended = origin.drift_toward(target, weight=0.5)
    assert blended.kindness == 40
