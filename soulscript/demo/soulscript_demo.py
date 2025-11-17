########## Demo Runner ##########
# Builds the tavern world, ticks the scheduler, and exports run logs.

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..core import config
from ..core.db import fetch_events
from ..core.npc import NPC
from ..core.scheduler import SimulationScheduler
from ..core.types import Decision, NPCProfile, NPCTruth

SEED_DIR = Path(__file__).resolve().parent / "seeds"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


########## Env Loader ##########
# Reads a simple .env file so local services are configured.

def _load_env_file() -> None:
    """Load .env key value pairs if present."""

    # 1 Return quickly when the .env file does not exist.                    # steps
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    # 2 Read each line and apply missing entries to os.environ.               # steps
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_env_file()


def load_seed_profiles() -> List[NPCProfile]:
    """Load NPC profiles from seed JSON files."""

    # 1 Walk seed directory and parse JSON.                                     # steps
    profiles: List[NPCProfile] = []
    for path in sorted(SEED_DIR.glob("npc_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        truth = NPCTruth(**raw)
        if config.ACTIVE_NPCS and truth.npc_id not in config.ACTIVE_NPCS:
            continue
        profiles.append(NPCProfile(truth=truth))
    return profiles


def build_demo_world() -> SimulationScheduler:
    """Create scheduler, load NPCs, and prep global systems."""

    # 1 Instantiate scheduler and register NPCs from seeds.                     # steps
    scheduler = SimulationScheduler()
    for profile in load_seed_profiles():
        npc = NPC(profile)
        scheduler.register(npc)
    scheduler.initialize_relationships()
    return scheduler


def export_run_log(decisions: List[Decision]) -> Path:
    """Persist decisions to JSONL for quick inspection."""

    # 1 Ensure export directory exists.                                        # steps
    export_dir = Path(config.DEFAULT_DIALOGUE_EXPORT)
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = export_dir / config.DEFAULT_DIALOGUE_FILENAME_TEMPLATE.format(timestamp=timestamp)
    with file_path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            payload = {
                "npc_id": decision.npc_id,
                "action": decision.selected_action.action_type.value,
                "target_id": decision.selected_action.target_id,
                "dialogue": decision.dialogue_line,
                "reason": decision.reason,
                "confidence": decision.confidence,
            }
            handle.write(json.dumps(payload) + "\n")
    return file_path


def export_event_log() -> Path:
    """Dump the sqlite event log to CSV for analysts."""

    # 1 Fetch events and write a simple CSV file.                               # steps
    export_dir = Path(config.DEFAULT_DIALOGUE_EXPORT)
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = export_dir / Path(config.DEFAULT_EVENT_LOG_EXPORT.format(timestamp=timestamp))
    events = fetch_events()
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write("npc_id,target_id,type,data,ts\n")
        for event in events:
            row = [
                event.get("npc_id", ""),
                event.get("target_id", ""),
                event.get("type", ""),
                event.get("data", ""),
                event.get("ts", ""),
            ]
            safe = [str(value).replace(",", ";") for value in row]
            handle.write(",".join(safe) + "\n")
    return file_path


def run_demo(ticks: int = 30) -> List[Decision]:
    """Run the demo scheduler for a handful of ticks."""

    # 1 Build the world and simulate requested ticks.                           # steps
    scheduler = build_demo_world()
    world_context: Dict[str, Any] = {"gathering_spot": "tavern_floor"}
    decisions = scheduler.run(ticks, world_context)
    export_run_log(decisions)
    export_event_log()
    return decisions


def main() -> None:
    """Entry point when running the demo script directly."""

    # 1 Kick off a small run and report where logs went.                        # steps
    decisions = run_demo(20)
    print(f"Ran {len(decisions)} decisions. Logs saved to {config.DEFAULT_DIALOGUE_EXPORT}.")


if __name__ == "__main__":
    main()
