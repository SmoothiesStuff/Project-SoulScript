########## LLM Interface ##########
# Coordinates local Ollama calls with strict JSON decoding for actions.

from __future__ import annotations

import json
import random
from typing import Any, Dict, List

from openai import APIError, OpenAI

from . import config
from .agentic_helpers import parse_json_loose, strip_think
from .types import Action, ActionType, Decision


class BaseLLMClient:
    """Shared interface for concrete LLM clients."""

    def select_action(self, npc_id: str, context: Dict[str, Any], allowed_actions: List[Action]) -> Decision:
        raise NotImplementedError


class OllamaLLMClient(BaseLLMClient):
    """Talks to a local Ollama endpoint using the OpenAI compatible client."""

    def __init__(self) -> None:
        # 1 Pull settings from config so students can tweak them easily.       # steps
        base_url = config.LLM_BASE_URL
        api_key = config.LLM_API_KEY
        self.model = config.LLM_MODEL_NAME
        self.temperature = config.LLM_TEMPERATURE
        self.top_p = config.LLM_TOP_P
        self.max_tokens = config.LLM_MAX_TOKENS
        self.num_gpu = config.LLM_NUM_GPU
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def select_action(self, npc_id: str, context: Dict[str, Any], allowed_actions: List[Action]) -> Decision:
        """Request a decision from Ollama and translate it into a Decision."""

        if not allowed_actions:
            action = Action(action_type=ActionType.IDLE)
            return Decision(
                npc_id=npc_id,
                selected_action=action,
                reason="No actions available.",
                dialogue_line=None,
                confidence=0.3,
            )
        messages = self._build_messages(npc_id, context, allowed_actions)
        try:
            extra_body: Dict[str, Any] = {}
            if self.num_gpu >= 0:
                extra_body["options"] = {"num_gpu": self.num_gpu}
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body=extra_body,
            )
            content = response.choices[0].message.content if response.choices else ""
        except APIError as error:
            raise RuntimeError(f"Ollama request failed: {error}") from error
        except Exception as error:
            raise RuntimeError(f"Unexpected Ollama failure: {error}") from error
        parsed = parse_json_loose(content) or {}
        return self._decision_from_payload(npc_id, allowed_actions, parsed, raw=content)

    def _build_messages(self, npc_id: str, context: Dict[str, Any], allowed_actions: List[Action]) -> List[Dict[str, str]]:
        """Craft chat messages that describe the state and choices."""

        action_lines: List[str] = []
        for index, action in enumerate(allowed_actions, start=1):
            label = action.action_type.value
            target = action.target_id or "none"
            action_lines.append(f"{index}. type={label} target={target}")

        filtered_context: Dict[str, Any] = {}
        keys_to_surface = {
            "profile",
            "state",
            "short_term",
            "long_term",
            "global_facts",
            "sentiment",
            "gathering_spot",
        }
        for key, value in context.items():
            if key in keys_to_surface:
                filtered_context[key] = value

        context_block = json.dumps(filtered_context, ensure_ascii=False, indent=2)
        instructions = (
            "You decide NPC actions in a cozy tavern sim. "
            "Return JSON with keys action, reason, and optional line. "
            "Use one allowed action only."
        )
        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": (
                    f"NPC: {npc_id}\n"
                    "Allowed actions:\n"
                    + "\n".join(action_lines)
                    + "\nContext:\n"
                    + context_block
                ),
            },
        ]
        return messages

    def _decision_from_payload(
        self,
        npc_id: str,
        allowed_actions: List[Action],
        payload: Dict[str, Any],
        raw: str,
    ) -> Decision:
        """Translate parsed JSON into a typed Decision."""

        action_field = payload.get("action")
        selected = self._pick_action(allowed_actions, action_field)
        reason = str(payload.get("reason", ""))
        dialogue_line = payload.get("line")
        if dialogue_line:
            dialogue_line = strip_think(str(dialogue_line))
        confidence = 0.6 if selected else 0.2
        action = selected or Action(action_type=ActionType.IDLE)
        if selected is None and config.DEBUG_VERBOSE:
            print(f"[LLM fallback] payload={payload} raw={raw}")
        return Decision(
            npc_id=npc_id,
            selected_action=action,
            reason=reason or "Defaulted to idle.",
            dialogue_line=dialogue_line,
            confidence=confidence,
        )

    def _pick_action(self, allowed_actions: List[Action], action_field: Any) -> Action | None:
        """Match action field to allowed action list."""

        if action_field is None:
            return None
        if isinstance(action_field, int) and 1 <= action_field <= len(allowed_actions):
            return allowed_actions[action_field - 1]
        if isinstance(action_field, str):
            normalized = action_field.strip().lower()
            for action in allowed_actions:
                if action.action_type.value == normalized:
                    return action
        return None


class StubLLMClient(BaseLLMClient):
    """Deterministic fallback used when Ollama is unavailable."""

    def __init__(self) -> None:
        self.random = random.Random(config.RANDOM_SEED)

    def select_action(self, npc_id: str, context: Dict[str, Any], allowed_actions: List[Action]) -> Decision:
        if not allowed_actions:
            action = Action(action_type=ActionType.IDLE)
            return Decision(npc_id=npc_id, selected_action=action, reason="No actions.", dialogue_line=None, confidence=0.5)

        choice = self.random.choice(allowed_actions)
        dialogue_line = None
        if choice.action_type == ActionType.SPEAK:
            short_term = context.get("short_term", [])
            global_facts = context.get("global_facts", [])
            line = short_term[-1] if short_term else "Nice to meet you."
            info = global_facts[0] if global_facts else ""
            dialogue_line = f"{line} {info}".strip()
        return Decision(
            npc_id=npc_id,
            selected_action=choice,
            reason="Stubbed decision.",
            dialogue_line=dialogue_line,
            confidence=0.4,
        )


def LLMClient() -> BaseLLMClient:  # factory mirrors prior usage
    """Return a working LLM client, falling back to the stub when needed."""

    try:
        return OllamaLLMClient()
    except Exception:
        return StubLLMClient()
