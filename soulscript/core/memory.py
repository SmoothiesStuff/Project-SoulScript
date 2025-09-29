########## Conversation Memory ##########
# Stores pairwise conversation snippets and maintains summaries.

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

from . import config
from .db import get_conversation_lines, record_conversation_line
from .tools import style_and_lore_filter
from .types import ConversationItem


class ConversationMemory:
    """Handles per pair conversation history backed by sqlite."""

    def __init__(self, keep: int = config.CONVERSATION_KEEP) -> None:
        # 1 Store retention configuration for future inserts.                  # steps
        self.keep = keep

    def record(
        self,
        npc_a: str,
        npc_b: str,
        speaker_id: str,
        text_line: str,
        timestamp: datetime,
    ) -> List[ConversationItem]:
        """Persist a new line and return ordered history objects."""

        # 1 Insert via db utility which auto prunes to the keep count.         # steps
        rows = record_conversation_line(npc_a, npc_b, speaker_id, text_line, timestamp, self.keep)
        return self._rows_to_items(rows)

    def history(self, npc_a: str, npc_b: str) -> List[ConversationItem]:
        """Fetch existing conversation lines for the pair."""

        # 1 Convert db rows to strongly typed items.                            # steps
        rows = get_conversation_lines(npc_a, npc_b, self.keep)
        return self._rows_to_items(rows)

    def build_summaries(
        self,
        npc_a: str,
        npc_b: str,
        items: List[ConversationItem],
    ) -> Dict[Tuple[str, str], str]:
        """Derive first person summaries for both participants."""

        # 1 Create directional summaries (A about B, B about A).               # steps
        summaries: Dict[Tuple[str, str], str] = {}
        perspective_pairs = [(npc_a, npc_b), (npc_b, npc_a)]
        for perspective_id, partner_id in perspective_pairs:
            summary_text = self._summarize_pair(perspective_id, partner_id, items)
            summaries[(perspective_id, partner_id)] = summary_text
        return summaries

    def _summarize_pair(
        self,
        perspective_id: str,
        partner_id: str,
        items: List[ConversationItem],
    ) -> str:
        """Construct a concise first-person summary string."""

        # 1 Gather lines spoken by partner and by the perspective npc.         # steps
        partner_lines: List[str] = []
        perspective_lines: List[str] = []
        for item in items[-config.CONVERSATION_SUMMARY_LENGTH - 1 :]:
            if item.speaker_id == partner_id:
                partner_lines.append(item.text)
            elif item.speaker_id == perspective_id:
                perspective_lines.append(item.text)
        fragments: List[str] = [f"I caught up with {partner_id}."]
        if partner_lines:
            fragments.append(f"They mentioned '{partner_lines[-1]}'")
        if perspective_lines:
            fragments.append(f"I replied '{perspective_lines[-1]}'")
        summary_text = " ".join(fragments)
        return style_and_lore_filter(summary_text)

    def _rows_to_items(self, rows: List[Dict[str, str]]) -> List[ConversationItem]:
        """Convert raw sqlite rows to Pydantic objects."""

        # 1 Instantiate ConversationItem for each row in order.                # steps
        items: List[ConversationItem] = []
        for row in rows:
            items.append(
                ConversationItem(
                    timestamp=datetime.fromisoformat(row["ts"]),
                    speaker_id=row["speaker_id"],
                    text=row["text"],
                )
            )
        return items


# TODO: expand summarizer with sentiment cues once mood analysis lands.
