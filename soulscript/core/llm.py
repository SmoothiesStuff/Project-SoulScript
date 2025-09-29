########## LLM Interface ##########
# Coordinates Ollama backed chat completions with a safe fallback stub.

"""Ollama setup quickstart.
1 Install Ollama: https://ollama.ai/download
2 Pull the model: ollama pull llama3:8b
3 Optional interactive test: ollama run llama3:8b
Model cache lives at ~/.ollama/models/ (Linux/macOS) or %LOCALAPPDATA%\Ollama\models\ (Windows).
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

from openai import APIError, OpenAI

from . import config
from .tools import style_and_lore_filter
from .types import Action, ActionType, Decision


class BaseLLMClient:
    """Shared interface for concrete LLM clients."""

    def select_action(self, npc_id: str, context: Dict[str, Any], allowed_actions: List[Action]) -> Decision:
        raise NotImplementedError


class OllamaLLMClient(BaseLLMClient):
    """Talks to a local Ollama endpoint using the OpenAI compatible client."""

    def __init__(self) -> None:
        # 1 Read configuration from environment with sensible defaults.         # steps
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("LLM_API_KEY", "ollama")
        self.model = os.getenv("LLM_MODEL", "llama3:8b")
        self.temperature = 0.7
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def select_action(self, npc_id: str, context: Dict[str, Any], allowed_actions: List[Action]) -> Decision:
        """Request a decision from Ollama and translate it into a Decision."""

        # 1 Fall back immediately when no actions exist.                        # steps
        if not allowed_actions:
            action = Action(action_type=ActionType.IDLE)
            return Decision(npc_id=npc_id, selected_action=action, reason="No actions available.", dialogue_line=None, confidence=0.3)
        payload = self._build_messages(npc_id, context, allowed_actions)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=self.temperature,
            )
        except APIError as error:
            raise RuntimeError("Ollama request failed") from error
        except Exception as error:
            raise RuntimeError("Unexpected Ollama failure") from error
        content = response.choices[0].message.content if response.choices else ""
        parsed = self._parse_response(content)
        return self._decision_from_payload(npc_id, allowed_actions, parsed)

    def _build_messages(self, npc_id: str, context: Dict[str, Any], allowed_actions: List[Action]) -> List[Dict[str, str]]:
        """Craft chat messages that describe the state and choices."""

        # 1 Summarize allowed actions in a numbered list.                        # steps
        action_lines: List[str] = []
        for index, action in enumerate(allowed_actions, start=1):
            label = action.action_type.value
            target = action.target_id or "none"
            action_lines.append(f"{index}. type={label} target={target}")
        # 2 Serialize contextual data for the LLM input.                        # steps
        filtered_context: Dict[str, Any] = {}
        for key, value in context.items():
            if key in {"memory_recent", "conversation_recent"}:
                filtered_context[key] = value
            elif key in {"conversation_summary", "profile", "state", "sentiment", "gathering_spot"}:
                filtered_context[key] = value
        context_block = json.dumps(filtered_context, ensure_ascii=False, indent=2)
        instructions = (
            "You decide NPC actions in a cozy tavern sim."
            " Respond with JSON: {\"action\": action index or name, \"reason\": short text, \"line\": optional dialogue}."
            " Choose one allowed action only."
        )
        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": (
                    f"NPC: {npc_id}\n" +
                    "Allowed actions:\n" + "\n".join(action_lines) +
                    "\nContext:\n" + context_block
                ),
            },
        ]
        return messages

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON output and handle loose text gracefully."""

        # 1 Try to parse strict JSON first.                                     # steps
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        # 2 Look for a JSON snippet inside the text.                            # steps
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = content[start:end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {}

    def _decision_from_payload(self, npc_id: str, allowed_actions: List[Action], payload: Dict[str, Any]) -> Decision:
        """Map the parsed payload back to an Action."""

        # 1 Default to first allowed action when parsing fails.                 # steps
        selected = allowed_actions[0]
        raw_choice = payload.get("action") if isinstance(payload, dict) else None
        if isinstance(raw_choice, int):
            index = raw_choice - 1
            if 0 <= index < len(allowed_actions):
                selected = allowed_actions[index]
        elif isinstance(raw_choice, str):
            for action in allowed_actions:
                if action.action_type.value == raw_choice:
                    selected = action
                    break
        reason = payload.get("reason") if isinstance(payload, dict) else None
        if not reason:
            reason = "Following instinct based on context."
        line = payload.get("line") if isinstance(payload, dict) else None
        if line:
            line = style_and_lore_filter(line)
        return Decision(
            npc_id=npc_id,
            selected_action=selected,
            reason=reason,
            dialogue_line=line,
            confidence=0.75,
        )


class StubLLMClient(BaseLLMClient):
    """Heuristic driven stand in for offline runs."""

    def __init__(self, seed: int) -> None:
        # 1 Initialize deterministic random generator so demos are stable.      # steps
        self._random = random.Random(seed)

    def select_action(self, npc_id: str, context: Dict[str, Any], allowed_actions: List[Action]) -> Decision:
        """Pick an action using simple heuristics that favor conversation."""

        # 1 Prefer speak when there is a partner and available history.         # steps
        speak_actions = [action for action in allowed_actions if action.action_type == ActionType.SPEAK]
        idle_actions = [action for action in allowed_actions if action.action_type == ActionType.IDLE]
        adjust_actions = [action for action in allowed_actions if action.action_type == ActionType.ADJUST_RELATIONSHIP]
        join_actions = [action for action in allowed_actions if action.action_type == ActionType.JOIN]
        partner_id = context.get("conversation_partner")
        if speak_actions and partner_id:
            selected = self._random.choice(speak_actions)
            line = self._craft_line(npc_id, partner_id, context)
            reason = f"Checking in with {partner_id}."
            return Decision(
                npc_id=npc_id,
                selected_action=selected,
                reason=reason,
                dialogue_line=line,
                confidence=0.7,
            )
        if adjust_actions and partner_id:
            selected = self._random.choice(adjust_actions)
            reason = f"Reflecting on how {partner_id} behaved earlier."
            return Decision(
                npc_id=npc_id,
                selected_action=selected,
                reason=reason,
                dialogue_line=None,
                confidence=0.55,
            )
        if join_actions:
            selected = self._random.choice(join_actions)
            reason = "Considering whether to mingle."
            return Decision(
                npc_id=npc_id,
                selected_action=selected,
                reason=reason,
                dialogue_line=None,
                confidence=0.5,
            )
        if idle_actions:
            selected = idle_actions[0]
            return Decision(
                npc_id=npc_id,
                selected_action=selected,
                reason="No strong pull to act right now.",
                dialogue_line=None,
                confidence=0.4,
            )
        fallback_action = Action(action_type=ActionType.IDLE)
        return Decision(
            npc_id=npc_id,
            selected_action=fallback_action,
            reason="Defaulting to idle.",
            dialogue_line=None,
            confidence=0.3,
        )

    def _craft_line(self, npc_id: str, partner_id: str, context: Dict[str, Any]) -> str:
        """Build a short lore-friendly response."""

        # 1 Pull last conversation bits and summary for grounding.              # steps
        history: List[str] = context.get("conversation_recent", [])
        summary: str = context.get("conversation_summary", "")
        if history:
            sample = self._random.choice(history)
            line = f"About earlier, {sample.lower()}"
        elif summary:
            line = f"I keep thinking about how {summary.lower()}"
        else:
            moods = [
                "Weather feels gentle tonight",
                "This tavern hums softly",
                "Good to see you",
            ]
            line = self._random.choice(moods)
        line = style_and_lore_filter(line)
        if len(line) > 120:
            line = line[:117] + "..."
        return line


class LLMClient:
    """Factory that prefers Ollama but falls back to the deterministic stub."""

    def __init__(self) -> None:
        # 1 Instantiate the stub for use during outages.                        # steps
        self._stub = StubLLMClient(config.RANDOM_SEED)
        try:
            self._primary: BaseLLMClient = OllamaLLMClient()
        except Exception as error:
            print(f"[SoulScript] Ollama unavailable ({error}). Using stub.")
            self._primary = self._stub

    def select_action(self, npc_id: str, context: Dict[str, Any], allowed_actions: List[Action]) -> Decision:
        """Call Ollama when ready, otherwise revert to the stub."""

        # 1 Try the primary client and revert to stub on failure.               # steps
        if self._primary is self._stub:
            return self._stub.select_action(npc_id, context, allowed_actions)
        try:
            return self._primary.select_action(npc_id, context, allowed_actions)
        except Exception as error:
            print(f"[SoulScript] Ollama call failed ({error}). Falling back to stub.")
            return self._stub.select_action(npc_id, context, allowed_actions)


# TODO: explore gossip and faction-aware prompts once those systems exist.

