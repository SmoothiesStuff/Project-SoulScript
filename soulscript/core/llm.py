########## LLM Interface ##########
# Coordinates local Ollama calls with strict JSON decoding for actions.

from __future__ import annotations

import json
import os
import random
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

from openai import APIError, OpenAI

from . import config
from .agentic_helpers import parse_json_loose, strip_think
from .types import Action, ActionType, Decision

DEBUG_LOG: list[str] = []  # shared buffer for UI to read verbose exchanges


def _force_ollama_cpu_mode() -> None:
    """Hard-set environment so Ollama runners stay on CPU even on Windows."""

    # Ollama runners on Windows may ignore shell env; set before client init.
    os.environ["OLLAMA_NUM_GPU"] = "0"
    os.environ["OLLAMA_LOAD_GPU"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
    # Keep only one runner/model to reduce accidental GPU allocations.
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = os.environ.get("OLLAMA_MAX_LOADED_MODELS", "1")
    os.environ["OLLAMA_RUNNERS_NUM_PARALLEL"] = os.environ.get("OLLAMA_RUNNERS_NUM_PARALLEL", "1")


_OLLAMA_BOOTSTRAPPED = False
_OLLAMA_RESET_DONE = False


def _ollama_reachable(base_url: str, timeout: float = 1.0) -> bool:
    """Check if the Ollama endpoint responds."""

    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def _ensure_ollama_server(base_url: str) -> None:
    """Attempt to start Ollama serve with CPU env if it's not reachable."""

    global _OLLAMA_BOOTSTRAPPED
    if _OLLAMA_BOOTSTRAPPED:
        return
    _force_ollama_cpu_mode()
    _kill_ollama_processes()
    _stop_ollama_runners()
    if _ollama_reachable(base_url):
        _OLLAMA_BOOTSTRAPPED = True
        return
    # Best-effort start; if it fails we surface a clear error later.
    env = dict(os.environ)
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        time.sleep(1.0)
    except Exception:
        pass
    _OLLAMA_BOOTSTRAPPED = True


def _reset_ollama_runners() -> None:
    """Stop active runners and relaunch serve in CPU-only mode."""

    global _OLLAMA_RESET_DONE
    if _OLLAMA_RESET_DONE:
        return
    _force_ollama_cpu_mode()
    _kill_ollama_processes()
    _stop_ollama_runners()
    env = dict(os.environ)
    try:
        subprocess.run(["ollama", "stop", "all"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, timeout=5)
    except Exception:
        pass
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        time.sleep(1.0)
    except Exception:
        pass
    _OLLAMA_RESET_DONE = True


def _kill_ollama_processes() -> None:
    """Terminate existing Ollama processes so stale GPU runners are not reused."""

    # Windows: taskkill; POSIX: pkill (best effort).
    try:
        subprocess.run(["taskkill", "/IM", "ollama.exe", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        pass
    try:
        subprocess.run(["pkill", "-f", "ollama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        pass


def _stop_ollama_runners() -> None:
    """Use ollama CLI to stop running models."""

    env = dict(os.environ)
    try:
        subprocess.run(["ollama", "stop", "all"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, timeout=5)
    except Exception:
        pass


class BaseLLMClient:
    """Shared interface for concrete LLM clients."""

    def select_action(self, npc_id: str, context: Dict[str, Any], allowed_actions: List[Action]) -> Decision:
        raise NotImplementedError

    def summarize_conversation(self, prior_summary: str, lines: List[str], source_id: str, target_id: str, trust: int, affinity: int) -> str:
        """Summarize conversation updates from source perspective."""

        raise NotImplementedError


class OllamaLLMClient(BaseLLMClient):
    """Talks to a local Ollama endpoint using the OpenAI compatible client."""

    def __init__(self) -> None:
        # 1 Pull settings from config so students can tweak them easily.       # steps
        if os.getenv("SOULSCRIPT_LLM_STUB", "").lower() in {"1", "true", "yes"}:
            raise RuntimeError("Stub requested via SOULSCRIPT_LLM_STUB")
        _force_ollama_cpu_mode()
        _reset_ollama_runners()
        base_url = config.LLM_BASE_URL
        _ensure_ollama_server(base_url)
        if not _ollama_reachable(base_url):
            raise RuntimeError(
                "Ollama is not reachable. Ensure ollama is installed and run with CPU mode: "
                "'OLLAMA_NUM_GPU=0 OLLAMA_LOAD_GPU=0 ollama serve'"
            )
        api_key = config.LLM_API_KEY
        self.model = config.LLM_MODEL_NAME
        self.temperature = config.ACTION_TEMPERATURE
        self.top_p = config.ACTION_TOP_P
        self.max_tokens = config.ACTION_MAX_TOKENS
        self.num_gpu = 0  # force CPU-only to avoid Windows CUDA crashes
        self.ollama_options = dict(getattr(config, "OLLAMA_OPTIONS", {"num_gpu": 0, "gpu_layers": 0}))
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=15)

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
        attempts = 2
        content = ""
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                content = self._run_completion(messages, self.num_gpu)
                if config.DEBUG_VERBOSE:
                    DEBUG_LOG.append(f"{npc_id} raw: {content}")
                break
            except Exception as error:
                last_error = error
                print(f"[LLM] Completion attempt failed for {npc_id}: {error}")
                message = str(error)
                if "cuda" in message.lower() or "exit status 2" in message.lower():
                    _reset_ollama_runners()
                    continue
                fallback_to_cpu = self._should_fallback_to_cpu(message)
                if fallback_to_cpu:
                    content = self._run_completion(messages, 0)
                    self.num_gpu = 0
                    if config.DEBUG_VERBOSE:
                        DEBUG_LOG.append(f"{npc_id} cpu-retry raw: {content}")
                    break
                else:
                    continue
        if not content and last_error:
            raise RuntimeError(f"Ollama request failed: {last_error}") from last_error
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
            "short_term_summary",
            "recent_thread",
            "last_partner_line",
            "long_term",
            "global_facts",
            "sentiment",
            "gathering_spot",
            "interaction_tone",
            "trait_scale",
        }
        for key, value in context.items():
            if key in keys_to_surface:
                filtered_context[key] = value

        context_block = json.dumps(filtered_context, ensure_ascii=False, indent=2)
        instructions = config.ACTION_SYSTEM_PROMPT
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
        if selected is None and len(allowed_actions) == 1:
            # Default to the sole allowed action to keep the flow moving.
            selected = allowed_actions[0]
        confidence = 0.6 if selected else 0.2
        action = selected or Action(action_type=ActionType.IDLE)
        if config.DEBUG_VERBOSE:
            DEBUG_LOG.append(f"{npc_id} parsed: {payload}")
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
            # Handle loose formats like "type=speak" or "action: speak".
            for token in ("=", ":", " "):
                if token in normalized:
                    normalized = normalized.split(token)[-1].strip()
            for action in allowed_actions:
                if action.action_type.value == normalized:
                    return action
            if "speak" in normalized:
                for action in allowed_actions:
                    if action.action_type == ActionType.SPEAK:
                        return action
            if "idle" in normalized:
                for action in allowed_actions:
                    if action.action_type == ActionType.IDLE:
                        return action
        return None

    def _resolve_num_gpu(self) -> int:
        """Return configured GPU count; defaults to CPU-only."""

        # Force CPU mode regardless of config to avoid Windows CUDA crashes.
        return 0

    def _run_completion(self, messages: List[Dict[str, str]], num_gpu: int | None) -> str:
        """Call the chat completion endpoint with the requested GPU setting."""

        # 1 Build extra_body when num_gpu is provided.                         # steps
        extra_body: Dict[str, Any] = {}
        # Always force CPU-only execution by passing num_gpu=0.
        extra_body["options"] = dict(self.ollama_options)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body=extra_body,
            timeout=15,
        )
        return response.choices[0].message.content if response.choices else ""

    def _should_fallback_to_cpu(self, message: str) -> bool:
        """Detect common CUDA OOM strings so we can retry on CPU automatically."""

        # 1 Scan for cudaMalloc or generic out of memory markers.              # steps
        lowered = message.lower()
        if "cuda" in lowered:
            return True
        if "cudamalloc failed" in lowered:
            return True
        if "gpu" in lowered and "buffer" in lowered:
            return True
        return False

    def summarize_conversation(self, prior_summary: str, lines: List[str], source_id: str, target_id: str, trust: int, affinity: int) -> str:
        """Use the chat model to refine the long-term summary."""

        prompt_lines = "\n".join(lines[-12:])
        messages = [
            {
                "role": "system",
                "content": config.SUMMARY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"NPC (you): {source_id}\n"
                    f"Partner: {target_id}\n"
                    f"Previous summary: {prior_summary or 'None'}\n"
                    f"Current trust: {trust}\n"
                    f"Current affinity: {affinity}\n"
                    f"Recent dialogue (verbatim, use only these facts):\n{prompt_lines}\n"
                    "Rewrite the summary as first-person about the partner, 1-2 sentences. "
                    "Include only details mentioned in prior summary or these lines, and only about the partner. "
                    "No inventions, no speculation. Keep it concise, natural, and grounded."
                ),
            },
        ]
        try:
            content = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=config.SUMMARY_TEMPERATURE,
                top_p=config.SUMMARY_TOP_P,
                max_tokens=config.SUMMARY_MAX_TOKENS,
                extra_body={"options": dict(self.ollama_options)},
                timeout=15,
            ).choices[0].message.content
            return content.strip()
        except Exception:
            return prior_summary or ""


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

    def summarize_conversation(self, prior_summary: str, lines: List[str], source_id: str, target_id: str, trust: int, affinity: int) -> str:
        last_line = lines[-1] if lines else ""
        base = prior_summary or f"I caught up with {target_id}."
        if last_line:
            return f"{base} Recently we discussed '{last_line[:60]}'."
        return base


class OpenRouterLLMClient(BaseLLMClient):
    """Talks to OpenRouter using OpenAI-compatible SDK."""

    def __init__(self) -> None:
        api_key = config.LLM_OPENROUTER_API_KEY
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.model = config.LLM_OPENROUTER_MODEL
        self.temperature = config.ACTION_TEMPERATURE
        self.top_p = config.ACTION_TOP_P
        self.max_tokens = config.ACTION_MAX_TOKENS
        self.extra_body_options = {"options": dict(getattr(config, "OLLAMA_OPTIONS", {}))}
        self.client = OpenAI(base_url=config.LLM_OPENROUTER_BASE_URL, api_key=api_key, timeout=15)

    def select_action(self, npc_id: str, context: Dict[str, Any], allowed_actions: List[Action]) -> Decision:
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
        content = self._run_completion(messages)
        parsed = parse_json_loose(content) or {}
        return self._decision_from_payload(npc_id, allowed_actions, parsed, raw=content)

    def summarize_conversation(self, prior_summary: str, lines: List[str], source_id: str, target_id: str, trust: int, affinity: int) -> str:
        prompt_lines = "\n".join(lines[-12:])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are updating a first-person relationship summary about the partner NPC. "
                    "Keep it to one or two short sentences. "
                    "Only mention details about the partner, not yourself. "
                    "Stay friendly and grounded."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"NPC (you): {source_id}\n"
                    f"Partner: {target_id}\n"
                    f"Previous summary: {prior_summary or 'None'}\n"
                    f"Current trust: {trust}\n"
                    f"Current affinity: {affinity}\n"
                    f"Recent dialogue (verbatim, use only these facts):\n{prompt_lines}\n"
                    "Rewrite the summary as first-person about the partner, 1-2 sentences. "
                    "Include only details mentioned in prior summary or these lines, and only about the partner. "
                    "No inventions, no speculation. Keep it concise, natural, and grounded."
                ),
            },
        ]
        try:
            return self._run_completion(messages).strip()
        except Exception:
            return prior_summary or ""

    def _build_messages(self, npc_id: str, context: Dict[str, Any], allowed_actions: List[Action]) -> List[Dict[str, str]]:
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
            "short_term_summary",
            "recent_thread",
            "last_partner_line",
            "long_term",
            "global_facts",
            "sentiment",
            "gathering_spot",
            "interaction_tone",
            "trait_scale",
        }
        for key, value in context.items():
            if key in keys_to_surface:
                filtered_context[key] = value

        context_block = json.dumps(filtered_context, ensure_ascii=False, indent=2)
        instructions = config.ACTION_SYSTEM_PROMPT
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
        action_field = payload.get("action")
        selected = self._pick_action(allowed_actions, action_field)
        reason = str(payload.get("reason", ""))
        dialogue_line = payload.get("line")
        if dialogue_line:
            dialogue_line = strip_think(str(dialogue_line))
        if selected is None and len(allowed_actions) == 1:
            selected = allowed_actions[0]
        confidence = 0.6 if selected else 0.2
        action = selected or Action(action_type=ActionType.IDLE)
        if config.DEBUG_VERBOSE:
            DEBUG_LOG.append(f"{npc_id} parsed: {payload}")
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
        if action_field is None:
            return None
        if isinstance(action_field, int) and 1 <= action_field <= len(allowed_actions):
            return allowed_actions[action_field - 1]
        if isinstance(action_field, str):
            normalized = action_field.strip().lower()
            for token in ("=", ":", " "):
                if token in normalized:
                    normalized = normalized.split(token)[-1].strip()
            for action in allowed_actions:
                if action.action_type.value == normalized:
                    return action
            if "speak" in normalized:
                for action in allowed_actions:
                    if action.action_type == ActionType.SPEAK:
                        return action
            if "idle" in normalized:
                for action in allowed_actions:
                    if action.action_type == ActionType.IDLE:
                        return action
        return None

    def _run_completion(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=config.SUMMARY_TEMPERATURE,
            top_p=config.SUMMARY_TOP_P,
            max_tokens=config.SUMMARY_MAX_TOKENS,
            extra_body=self.extra_body_options if self.extra_body_options.get("options") else None,
            timeout=15,
        )
        return response.choices[0].message.content if response.choices else ""


def LLMClient() -> BaseLLMClient:  # factory mirrors prior usage
    """Return a working LLM client, falling back to the stub when needed."""

    if os.getenv("SOULSCRIPT_LLM_STUB", "").lower() in {"1", "true", "yes"}:
        print("[LLM] SOULSCRIPT_LLM_STUB=1 -> using stub client")
        return StubLLMClient()
    provider = config.LLM_PROVIDER.lower()
    try:
        if provider == "openrouter":
            return OpenRouterLLMClient()
        return OllamaLLMClient()
    except Exception as error:
        print(
            "[LLM] Failed to initialize LLM client.\n"
            f"      Provider: {provider}\n"
            "      For ollama: ensure `ollama serve` is running and `LLM_BASE_URL` is reachable.\n"
            "      For openrouter: ensure OPENROUTER_API_KEY is set and network access is available.\n"
            f"      Error: {error}"
        )
        raise
