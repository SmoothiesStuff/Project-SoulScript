########## Runtime ##########
# Bundles LangGraph-style nodes, policy, tools, state, and logging.

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import config
from .db import (
    delete_conversation,
    ensure_schema,
    get_conversation_lines,
    load_relationship,
    log_event,
    record_conversation_line,
    upsert_relationship,
)
from .types import (
    Action,
    ActionType,
    ConversationItem,
    Decision,
    NPCProfile,
    RelationshipEdge,
    TraitVector,
    TRAIT_AXES,
)

########## Text Logging ##########
# Lightweight, human-readable log lines for demo runs.


def log_run_event(message: str) -> None:
    """Append a single readable line to the demo log file."""

    if not config.LOG_TEXT_ENABLED:  # fast skip when disabled               # intent
        return
    log_dir = Path(config.LOG_TEXT_DIR)
    if not log_dir.is_absolute():
        log_dir = Path(__file__).resolve().parents[2] / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / config.LOG_TEXT_FILENAME
    timestamp = datetime.utcnow().isoformat()
    line = f"[{timestamp}] {message}"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    _trim_log_file(log_path, config.LOG_TEXT_MAX_LINES)


def _trim_log_file(log_path: Path, max_lines: int) -> None:
    """Keep the log file short and readable."""

    if max_lines <= 0 or not log_path.exists():
        return
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= max_lines:
        return
    trimmed = "\n".join(lines[-max_lines:]) + "\n"
    log_path.write_text(trimmed, encoding="utf-8")


########## Demo Tools ##########
# Utility lookups used by LangGraph nodes to ground decisions.

FILTERED_WORDS = {"curse", "dang", "heck"}


def inventory_for(profile: NPCProfile) -> List[str]:
    """Return the inventory list declared in the truth seed."""

    return list(profile.truth.inventory)  # safe copy                           # intent


def location_of(profile: NPCProfile) -> str:
    """Return the default hangout spot for the NPC."""

    schedule = profile.truth.schedule
    if "morning" in schedule:
        return schedule["morning"]
    for _, location in schedule.items():
        return location
    return "tavern_floor"


def schedule_for(profile: NPCProfile) -> Dict[str, str]:
    """Return a simple schedule map for the NPC."""

    return dict(profile.truth.schedule)  # avoid caller mutation                 # intent


def style_and_lore_filter(text: str) -> str:
    """Scrub out words that break the demo style guidelines."""

    tokens = text.split()
    cleaned: List[str] = []
    for token in tokens:
        lowered = token.lower().strip(".,!?")
        if lowered in FILTERED_WORDS:
            cleaned.append("...")
        else:
            cleaned.append(token)
    return " ".join(cleaned)


########## Conversation Memory ##########
# Stores pairwise conversation snippets and maintains summaries.


class ConversationMemory:
    """Handles per pair conversation history backed by sqlite."""

    def __init__(self, keep: int = config.INTERACTION_LINES_KEEP) -> None:
        self.keep = keep  # retention window                                      # intent

    def record(
        self,
        npc_a: str,
        npc_b: str,
        speaker_id: str,
        text_line: str,
        timestamp: datetime,
    ) -> List[ConversationItem]:
        """Persist a new line and return ordered history objects."""

        rows = record_conversation_line(npc_a, npc_b, speaker_id, text_line, timestamp, self.keep)
        return self._rows_to_items(rows)

    def history(self, npc_a: str, npc_b: str) -> List[ConversationItem]:
        """Fetch existing conversation lines for the pair."""

        rows = get_conversation_lines(npc_a, npc_b, self.keep)
        return self._rows_to_items(rows)

    def build_summaries(
        self,
        npc_a: str,
        npc_b: str,
        items: List[ConversationItem],
    ) -> Dict[Tuple[str, str], str]:
        """Derive first person summaries for both participants."""

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

    def context_bundle(
        self,
        npc_a: str,
        npc_b: str,
        existing_summary: str | None,
    ) -> Dict[str, List[str] | str]:
        """Return short term lines, summary, and shared knowledge for prompts."""

        items = self.history(npc_a, npc_b)
        short_term: List[str] = []
        for item in items[-4:]:  # keep the prompt lean to reduce repetition
            short_term.append(f"{item.speaker_id}: {item.text}")
        summary = existing_summary or ""
        if len(items) >= config.LONG_TERM_SUMMARY_TRIGGER:
            summaries = self.build_summaries(npc_a, npc_b, items)
            summary = summaries.get((npc_a, npc_b), summary)
        return {
            "short_term": short_term,
            "long_term": summary,
            "global_facts": list(config.GLOBAL_KNOWLEDGE),
        }

    def _rows_to_items(self, rows: List[Dict[str, str]]) -> List[ConversationItem]:
        """Convert raw sqlite rows to Pydantic objects."""

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

    def clear_history(self, npc_a: str, npc_b: str) -> None:
        """Erase stored conversation lines for a pair."""

        delete_conversation(npc_a, npc_b)


########## Relationship Graph ##########
# Tracks trust, affinity, and perceived traits with sqlite persistence.


def _soft_metric_update(base: int, delta: int) -> int:
    """Apply delta with soft limits near extremes."""

    candidate = base + delta
    if base >= config.RELATIONSHIP_SOFT_LIMIT and delta > 0:
        candidate = base + int(delta * 0.5)
    if base <= -config.RELATIONSHIP_SOFT_LIMIT and delta < 0:
        candidate = base + int(delta * 0.5)
    if candidate > config.RELATIONSHIP_TRUST_CLAMP[1]:
        return config.RELATIONSHIP_TRUST_CLAMP[1]
    if candidate < config.RELATIONSHIP_TRUST_CLAMP[0]:
        return config.RELATIONSHIP_TRUST_CLAMP[0]
    return candidate


class RelationshipGraph:
    """Graph helper that owns perceptions and syncs with sqlite."""

    def __init__(self) -> None:
        self.graph = None  # lazy import to avoid hard dep when unused           # intent
        ensure_schema()
        self._ensure_graph()

    def _ensure_graph(self) -> None:
        if self.graph is None:
            import networkx as nx

            self.graph = nx.DiGraph()

    def bootstrap(self, npc_ids: Iterable[str], seeds: Dict[str, Dict[str, RelationshipEdge]]) -> None:
        """Load existing edges from seeds and sqlite."""

        self._ensure_graph()
        for npc_id in npc_ids:
            if not self.graph.has_node(npc_id):
                self.graph.add_node(npc_id)
        for source_id, entries in seeds.items():
            for target_id, edge in entries.items():
                self._write_edge(edge)
        for source_id in npc_ids:
            for target_id in npc_ids:
                if source_id == target_id:
                    continue
                self._ensure_edge(source_id, target_id)

    def get_edge(self, source_id: str, target_id: str) -> RelationshipEdge:
        """Expose the current edge state for external modules."""

        return self._ensure_edge(source_id, target_id)

    def _ensure_edge(self, source_id: str, target_id: str) -> RelationshipEdge:
        """Fetch the edge data or initialize a neutral one."""

        self._ensure_graph()
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

        edge = self._ensure_edge(source_id, target_id)
        edge.summary = summary
        edge.updated_at = timestamp
        self._write_edge(edge)

    def decay_all(self, now: datetime) -> None:
        """Ease trust, affinity, and traits back toward neutral."""

        self._ensure_graph()
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

        self._ensure_graph()
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

        self._ensure_graph()
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
        """Ease a metric toward neutral using configured drift."""

        if value > config.RELATIONSHIP_NEUTRAL:
            value = max(config.RELATIONSHIP_NEUTRAL, value - int(config.RELATIONSHIP_DECAY_PER_TICK))
        elif value < config.RELATIONSHIP_NEUTRAL:
            value = min(config.RELATIONSHIP_NEUTRAL, value + int(config.RELATIONSHIP_DECAY_PER_TICK))
        return value


########## Relationship Update Agent ##########
# Computes small, context-aware trust/affinity deltas.


def _avg_trait_gap(a: TraitVector, b: TraitVector) -> float:
    total = 0
    for axis in TRAIT_AXES:
        total += abs(getattr(a, axis) - getattr(b, axis))
    return total / len(TRAIT_AXES)


def _compatibility_factor(source_truth: TraitVector, target_truth: TraitVector) -> float:
    gap = _avg_trait_gap(source_truth, target_truth)
    return max(-1.0, min(1.0, (50.0 - gap) / 50.0))


def _conversation_energy(lines: List[str]) -> float:
    if not lines:
        return 0.0
    return min(1.0, len(lines) / 8.0)


def compute_relationship_update(
    source_truth: TraitVector,
    target_truth: TraitVector,
    summary: str,
    lines: List[str],
    sentiment: float = 0.0,
    current_trust: int = 0,
    current_affinity: int = 0,
) -> Dict[str, int]:
    """Produce modest, context-aware deltas for trust and affinity."""

    similarity = _compatibility_factor(source_truth, target_truth)
    energy = _conversation_energy(lines)
    summary_bias = 0.3 if summary else 0.0
    trust_inertia = -0.1 if current_trust > 60 else 0.1 if current_trust < -40 else 0.0
    affinity_inertia = -0.1 if current_affinity > 60 else 0.1 if current_affinity < -40 else 0.0
    trust_score = (similarity * 0.9) + (energy * 0.8) + (sentiment * 0.6) + summary_bias + trust_inertia
    affinity_score = (similarity * 1.0) + (energy * 0.9) + (sentiment * 0.5) + summary_bias * 0.5 + affinity_inertia
    llm_trust_delta = int(round(max(-3.0, min(3.0, trust_score * 2.0))))
    llm_affinity_delta = int(round(max(-3.0, min(3.0, affinity_score * 2.0))))
    trust_delta = llm_trust_delta + config.RELATIONSHIP_TRUST_BASE_INCREMENT
    trust_delta = max(-3, min(3, trust_delta))
    affinity_delta = max(-3, min(3, llm_affinity_delta))
    return {"trust_delta": trust_delta, "affinity_delta": affinity_delta}


########## Policy Layer ##########
# Describes node specific allowed actions and tool prefetch logic.

POLICY_CONFIG: Dict[str, Dict[str, Any]] = {
    "idle": {
        "actions": [ActionType.IDLE],
        "tools": ["location"],
        "requires_partner": False,
    },
    "speak": {
        "actions": [ActionType.SPEAK, ActionType.IDLE],
        "tools": ["inventory", "schedule"],
        "requires_partner": True,
    },
    "decide_join": {
        "actions": [ActionType.JOIN, ActionType.IDLE],
        "tools": ["location", "schedule"],
        "requires_partner": False,
    },
    "adjust_relationship": {
        "actions": [ActionType.ADJUST_RELATIONSHIP, ActionType.IDLE],
        "tools": [],
        "requires_partner": True,
    },
}


def apply_policy(npc_profile: NPCProfile, node_name: str, context: Dict[str, Any]) -> Tuple[List[Action], Dict[str, Any]]:
    """Return allowed actions plus tool outputs for the given node."""

    config_entry = POLICY_CONFIG.get(node_name, POLICY_CONFIG["idle"])
    tool_outputs: Dict[str, Any] = {}
    for tool_name in config_entry.get("tools", []):
        if tool_name == "inventory":
            tool_outputs["inventory"] = inventory_for(npc_profile)
        elif tool_name == "location":
            tool_outputs["location"] = location_of(npc_profile)
        elif tool_name == "schedule":
            tool_outputs["schedule"] = schedule_for(npc_profile)
    nearby = context.get("nearby_npcs", [])
    focus_target = context.get("focus_target")
    target_id = focus_target
    if target_id is None and nearby:
        target_id = nearby[0]
    allowed_actions: List[Action] = []
    requires_partner = config_entry.get("requires_partner", False)
    turns_left = context.get("session_turns_left", 0)
    if requires_partner and target_id is None:
        allowed_actions.append(Action(action_type=ActionType.IDLE))
        return allowed_actions, tool_outputs
    filtered_actions = _filtered_actions(config_entry.get("actions", []))
    force_speak = turns_left > 0
    for action_type in filtered_actions:
        if force_speak and action_type != ActionType.SPEAK:
            continue
        action = Action(action_type=action_type, target_id=target_id)
        allowed_actions.append(action)
    if not allowed_actions:
        allowed_actions.append(Action(action_type=ActionType.IDLE, target_id=target_id))
    return allowed_actions, tool_outputs


def _filtered_actions(actions: List[ActionType]) -> List[ActionType]:
    """Limit actions to the configurable set for predictability."""

    if not config.ACTION_SET:
        return actions
    selected: List[ActionType] = []
    allowed = {name.lower() for name in config.ACTION_SET}
    for action_type in actions:
        if action_type.value in allowed:
            selected.append(action_type)
    return selected


########## LangGraph Nodes ##########
# Coordinates policy, LLM calls, logging, and relationship effects.


def node_idle(
    npc_profile: NPCProfile,
    npc_mood: int,
    conversation_memory: ConversationMemory,
    relationships: RelationshipGraph,
    llm_client: Any,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Handle idle decisions and gentle state recovery."""

    allowed_actions, tool_outputs = apply_policy(npc_profile, "idle", context)
    llm_context = _build_llm_context(npc_profile, npc_mood, conversation_memory, context, tool_outputs, relationships)
    log_run_event(
        f"llm_in npc={npc_profile.truth.npc_id} node=idle partner={context.get('focus_target')} allowed={[a.action_type.value for a in allowed_actions]} short={len(llm_context.get('short_term', []))}"
    )
    decision = llm_client.select_action(npc_profile.truth.npc_id, llm_context, allowed_actions)
    log_run_event(
        f"llm_out npc={npc_profile.truth.npc_id} node=idle action={decision.selected_action.action_type.value} target={decision.selected_action.target_id or '-'} reason='{decision.reason[:80]}'"
    )
    return {"decision": decision, "tool_outputs": tool_outputs}


def node_speak(
    npc_profile: NPCProfile,
    npc_mood: int,
    conversation_memory: ConversationMemory,
    relationships: RelationshipGraph,
    llm_client: Any,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Coordinate conversation actions and relationship nudges."""

    allowed_actions, tool_outputs = apply_policy(npc_profile, "speak", context)
    llm_context = _build_llm_context(npc_profile, npc_mood, conversation_memory, context, tool_outputs, relationships)
    log_run_event(
        f"llm_in npc={npc_profile.truth.npc_id} node=speak partner={context.get('focus_target')} allowed={[a.action_type.value for a in allowed_actions]} short={len(llm_context.get('short_term', []))}"
    )
    decision = _validated_decision(llm_client, npc_profile.truth.npc_id, llm_context, allowed_actions, context)
    action = decision.selected_action
    log_run_event(
        f"llm_out npc={npc_profile.truth.npc_id} node=speak action={action.action_type.value} target={action.target_id or '-'} line='{(decision.dialogue_line or '')[:120]}'"
    )
    defer_effects = context.get("defer_effects", False)
    pending_effects: List[Dict[str, Any]] | None = context.get("pending_effects")
    if action.action_type == ActionType.SPEAK and action.target_id:
        line = decision.dialogue_line or style_and_lore_filter("I have been thinking.")
        timestamp = datetime.utcnow()
        conversation_memory.record(npc_profile.truth.npc_id, action.target_id, npc_profile.truth.npc_id, line, timestamp)
        target_truth: TraitVector = context.get("target_truth", TraitVector())
        trait_deltas = _toward_truth_delta(relationships, npc_profile.truth.npc_id, action.target_id, target_truth)
        payload = {
            "source_id": npc_profile.truth.npc_id,
            "target_id": action.target_id,
            "line": line,
            "timestamp": timestamp,
            "trait_deltas": trait_deltas,
        }
        if pending_effects is not None:
            pending_effects.append(payload)
        else:
            relationships.update_summary(npc_profile.truth.npc_id, action.target_id, "", timestamp)
            log_event(
                npc_id=npc_profile.truth.npc_id,
                target_id=action.target_id,
                event_type="action",
                data_json=line,
                timestamp=timestamp,
            )
        log_run_event(
            f"dialogue {npc_profile.truth.npc_id}->{action.target_id}: '{line}' (defer_effects={defer_effects})"
        )
    return {"decision": decision, "tool_outputs": tool_outputs}


def node_decide_join(
    npc_profile: NPCProfile,
    npc_mood: int,
    conversation_memory: ConversationMemory,
    relationships: RelationshipGraph,
    llm_client: Any,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Determine whether the NPC joins a highlighted gathering."""

    allowed_actions, tool_outputs = apply_policy(npc_profile, "decide_join", context)
    llm_context = _build_llm_context(npc_profile, npc_mood, conversation_memory, context, tool_outputs, relationships)
    decision = llm_client.select_action(npc_profile.truth.npc_id, llm_context, allowed_actions)
    if decision.selected_action.action_type == ActionType.JOIN:
        destination = context.get("gathering_spot", "tavern_floor")
        log_event(
            npc_id=npc_profile.truth.npc_id,
            target_id=None,
            event_type="action",
            data_json=f"join:{destination}",
            timestamp=datetime.utcnow(),
        )
        log_run_event(f"join npc={npc_profile.truth.npc_id} destination={destination}")
    return {"decision": decision, "tool_outputs": tool_outputs}


def node_adjust_rel(
    npc_profile: NPCProfile,
    npc_mood: int,
    conversation_memory: ConversationMemory,
    relationships: RelationshipGraph,
    llm_client: Any,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply subtle relationship tweaks based on recent context."""

    allowed_actions, tool_outputs = apply_policy(npc_profile, "adjust_relationship", context)
    llm_context = _build_llm_context(npc_profile, npc_mood, conversation_memory, context, tool_outputs, relationships)
    decision = llm_client.select_action(npc_profile.truth.npc_id, llm_context, allowed_actions)
    action = decision.selected_action
    if action.action_type == ActionType.ADJUST_RELATIONSHIP and action.target_id:
        sentiment = context.get("sentiment", 0.0)
        trust_delta = int(round(sentiment * 4))
        affinity_delta = int(round(sentiment * 3))
        relationships.adjust_relation(
            npc_profile.truth.npc_id,
            action.target_id,
            trust_delta=trust_delta,
            affinity_delta=affinity_delta,
            trait_deltas=None,
            summary=None,
            timestamp=datetime.utcnow(),
        )
        log_run_event(
            f"adjust_rel npc={npc_profile.truth.npc_id}->{action.target_id} trust_delta={trust_delta} affinity_delta={affinity_delta}"
        )
    return {"decision": decision, "tool_outputs": tool_outputs}


########## LLM Context Helpers ##########
# Builds the prompt-ready context and validates model decisions.


def _build_llm_context(
    npc_profile: NPCProfile,
    npc_mood: int,
    conversation_memory: ConversationMemory,
    context: Dict[str, Any],
    tool_outputs: Dict[str, Any],
    relationships: RelationshipGraph,
) -> Dict[str, Any]:
    """Merge base context, tool data, memory bundles, and shared facts."""

    partner_id = context.get("focus_target")
    long_term_summary = ""
    if partner_id:
        edge = relationships.get_edge(npc_profile.truth.npc_id, partner_id)
        long_term_summary = edge.summary or ""
    bundle = (
        conversation_memory.context_bundle(npc_profile.truth.npc_id, partner_id, long_term_summary)
        if partner_id
        else {
            "short_term": [],
            "long_term": long_term_summary,
            "global_facts": list(config.GLOBAL_KNOWLEDGE),
        }
    )
    short_term = bundle["short_term"]
    last_partner_line = ""
    if partner_id:
        for line in reversed(short_term):
            if line.startswith(f"{partner_id}:"):
                last_partner_line = line.split(":", 1)[1].strip()
                break
    interaction_tone = "You two are strangers; keep it warm and brief." if not short_term else "Stay light and conversational."
    trait_scale = {
        "kindness": "-100=cruel, 100=kind",
        "bravery": "-100=cowardly, 100=brave",
        "extraversion": "-100=withdrawn, 100=gregarious",
        "ego": "-100=selfless, 100=proud",
        "honesty": "-100=dishonest, 100=honest",
        "curiosity": "-100=apathetic, 100=seeking",
        "patience": "-100=impulsive, 100=patient",
        "optimism": "-100=cynical, 100=hopeful",
        "intelligence": "-100=dim, 100=brilliant",
        "charisma": "-100=off-putting, 100=magnetic",
    }
    llm_context: Dict[str, Any] = {
        "profile": {
            "name": npc_profile.truth.name,
            "backstory": npc_profile.truth.backstory,
            "motivation": npc_profile.truth.motivation,
            "traits_summary": npc_profile.truth.traits,
            "trait_scale": trait_scale,
        },
        "state": {
            "mood": npc_mood,
            "traits_truth": npc_profile.truth.traits_truth.as_dict(),
        },
        "short_term": short_term,
        "recent_thread": short_term[-4:],
        "last_partner_line": last_partner_line,
        "long_term": bundle["long_term"],
        "global_facts": bundle["global_facts"],
        "conversation_partner": partner_id,
        "interaction_tone": interaction_tone,
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


def _validated_decision(
    llm_client: Any,
    npc_id: str,
    llm_context: Dict[str, Any],
    allowed_actions: List[Action],
    context: Dict[str, Any],
) -> Decision:
    """Ask the LLM, validate, and retry once if the output is junk."""

    max_attempts = 2  # initial + one retry to avoid loops
    require_speak = bool(context.get("session_turns_left") or context.get("defer_effects"))
    fallback_action = _first_speak_action(allowed_actions) or (allowed_actions[0] if allowed_actions else Action(action_type=ActionType.IDLE))
    last_decision: Optional[Decision] = None
    for _ in range(max_attempts):
        decision = llm_client.select_action(npc_id, llm_context, allowed_actions)
        last_decision = decision
        if _decision_ok(decision, allowed_actions, require_speak):
            return decision
    partner_last = llm_context.get("last_partner_line", "")
    fallback_line = style_and_lore_filter(f"Got it about '{partner_last[:40]}'." if partner_last else "Sure, tell me more.")
    if last_decision and last_decision.dialogue_line and not _looks_like_junk(last_decision.dialogue_line):
        fallback_line = last_decision.dialogue_line
    return Decision(
        npc_id=npc_id,
        selected_action=fallback_action,
        reason=last_decision.reason if last_decision else "Validator fallback.",
        dialogue_line=fallback_line,
        confidence=0.5,
    )


def _decision_ok(decision: Decision, allowed_actions: List[Action], require_speak: bool) -> bool:
    """Heuristic validator for LLM decisions."""

    action = decision.selected_action
    if require_speak and action.action_type != ActionType.SPEAK:
        return False
    if action not in allowed_actions:
        return False
    if action.action_type == ActionType.SPEAK:
        line = decision.dialogue_line or ""
        if not line.strip():
            return False
        if _looks_like_junk(line):
            return False
        if len(line) > 240:
            return False
    return True


def _looks_like_junk(line: str) -> bool:
    """Filter prompty or meta content."""

    lowered = line.lower()
    bad_markers = ["###", "instruction", "your task", "iambic", "essay", "problem:", "question:", "task:"]
    return any(marker in lowered for marker in bad_markers)


def _first_speak_action(allowed_actions: List[Action]) -> Optional[Action]:
    """Pick the first speak action in the list, if any."""

    for action in allowed_actions:
        if action.action_type == ActionType.SPEAK:
            return action
    return None


# TODO: surface gossip nodes once faction rumors are implemented.
