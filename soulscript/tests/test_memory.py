########## Conversation Memory Tests ##########
# Validates last-five retention and summary generation.

from __future__ import annotations

from datetime import datetime, timedelta

from soulscript.core.memory import ConversationMemory


def test_conversation_memory_keeps_last_five_lines() -> None:
    """Recording more than five lines should trim oldest entries."""

    # 1 Insert six lines with increasing timestamps.                           # steps
    memory = ConversationMemory(keep=5)
    base = datetime.utcnow()
    for index in range(6):
        timestamp = base + timedelta(seconds=index)
        memory.record("npc_a", "npc_b", "npc_a" if index % 2 == 0 else "npc_b", f"line {index}", timestamp)
    items = memory.history("npc_a", "npc_b")
    assert len(items) == 5
    assert items[0].text == "line 1"
    assert items[-1].text == "line 5"


def test_conversation_summary_mentions_recent_exchange() -> None:
    """Summaries should reference the latest dialogue line."""

    # 1 Record lines and ensure summary references the last partner line.      # steps
    memory = ConversationMemory(keep=5)
    base = datetime.utcnow()
    memory.record("npc_a", "npc_b", "npc_a", "Hello", base)
    memory.record("npc_a", "npc_b", "npc_b", "I brought news", base + timedelta(seconds=1))
    items = memory.history("npc_a", "npc_b")
    summaries = memory.build_summaries("npc_a", "npc_b", items)
    summary = summaries[("npc_a", "npc_b")]
    assert "npc_b" in summary or "news" in summary.lower()
