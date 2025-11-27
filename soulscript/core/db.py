########## Database Utilities ##########
# Manages SQLite persistence for runtime perceptions, relationships, and logs.

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy import Engine, create_engine, text

from . import config
from .types import TRAIT_AXES

_ENGINE: Optional[Engine] = None


def _db_path() -> Path:
    """Return the configured sqlite path and ensure its directory exists."""

    # 1 Resolve the configured path under the project workspace.               # steps
    # 2 Create parent directories when needed.                                 # steps
    path = Path(config.DB_FILE).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_engine() -> Engine:
    """Create or reuse the SQLAlchemy engine."""

    # 1 Cache the engine so future calls reuse the same connection pool.       # steps
    global _ENGINE
    if _ENGINE is None:
        path = _db_path()
        _ENGINE = create_engine(f"sqlite:///{path}", echo=config.DB_ECHO, future=True)
    return _ENGINE


def ensure_schema() -> None:
    """Create tables when they do not exist."""

    # 1 Execute CREATE TABLE statements with IF NOT EXISTS.                    # steps
    engine = get_engine()
    statements = _schema_statements()
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
        _ensure_new_columns(connection)


def _ensure_new_columns(connection) -> None:
    """Add newly introduced trait columns if they do not exist."""

    def _has_column(table: str, column: str) -> bool:
        info = connection.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
        return any(row[1] == column for row in info)

    # relationships: add intelligence, charisma if missing
    for col in ["intelligence", "charisma"]:
        if not _has_column("relationships", col):
            connection.exec_driver_sql(f"ALTER TABLE relationships ADD COLUMN {col} INTEGER DEFAULT 0")

    # npc_state: add self_intelligence, self_charisma if missing
    for col in ["self_intelligence", "self_charisma"]:
        if not _has_column("npc_state", col):
            connection.exec_driver_sql(f"ALTER TABLE npc_state ADD COLUMN {col} INTEGER")


def _schema_statements() -> List[str]:
    """Provide the schema definitions for idempotent creation."""

    # 1 Build SQL strings explicitly to keep visibility high.                  # steps
    npc_state = """
    CREATE TABLE IF NOT EXISTS npc_state (
        npc_id TEXT PRIMARY KEY,
        mood INTEGER DEFAULT 50,
        last_tick TEXT,
        self_kindness INTEGER,
        self_bravery INTEGER,
        self_extraversion INTEGER,
        self_ego INTEGER,
        self_honesty INTEGER,
        self_curiosity INTEGER,
        self_patience INTEGER,
        self_optimism INTEGER,
        self_intelligence INTEGER,
        self_charisma INTEGER
    )
    """
    relationships = """
    CREATE TABLE IF NOT EXISTS relationships (
        source_id TEXT,
        target_id TEXT,
        trust INTEGER DEFAULT 50,
        affinity INTEGER DEFAULT 50,
        kindness INTEGER,
        bravery INTEGER,
        extraversion INTEGER,
        ego INTEGER,
        honesty INTEGER,
        curiosity INTEGER,
        patience INTEGER,
        optimism INTEGER,
        intelligence INTEGER,
        charisma INTEGER,
        summary TEXT DEFAULT NULL,
        updated_at TEXT,
        PRIMARY KEY (source_id, target_id)
    )
    """
    conversations = """
    CREATE TABLE IF NOT EXISTS conversations (
        convo_id INTEGER PRIMARY KEY AUTOINCREMENT,
        a_id TEXT,
        b_id TEXT,
        ts TEXT,
        speaker_id TEXT,
        text TEXT
    )
    """
    event_log = """
    CREATE TABLE IF NOT EXISTS event_log (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        npc_id TEXT,
        target_id TEXT,
        type TEXT,
        data TEXT,
        ts TEXT
    )
    """
    return [npc_state, relationships, conversations, event_log]


def load_npc_state(npc_id: str) -> Optional[Dict[str, int]]:
    """Fetch stored self perception values for an NPC."""

    # 1 Query the row and map trait columns into a dictionary.                 # steps
    ensure_schema()
    engine = get_engine()
    statement = text(
        """
        SELECT * FROM npc_state
        WHERE npc_id = :npc_id
        """
    )
    with engine.begin() as connection:
        result = connection.execute(statement, {"npc_id": npc_id}).mappings().first()
    if result is None:
        return None
    payload: Dict[str, int] = {}
    for axis in TRAIT_AXES:
        key = f"self_{axis}"
        payload[axis] = int(result.get(key, 0))
    payload["mood"] = int(result.get("mood", 50))
    payload["last_tick"] = result.get("last_tick")
    return payload


def upsert_npc_state(npc_id: str, mood: int, last_tick: datetime, self_traits: Dict[str, int]) -> None:
    """Insert or update the npc_state row."""

    # 1 Prepare column assignments with explicit trait loops.                  # steps
    ensure_schema()
    engine = get_engine()
    assignments: List[str] = []
    parameters: Dict[str, int | str] = {"npc_id": npc_id, "mood": mood, "last_tick": last_tick.isoformat()}
    for axis in TRAIT_AXES:
        column = f"self_{axis}"
        value = int(self_traits.get(axis, 0))
        assignments.append(f"{column} = :{column}")
        parameters[column] = value
    set_clause = ", ".join(["mood = :mood", "last_tick = :last_tick", *assignments])
    statement = text(
        f"""
        INSERT INTO npc_state (npc_id, mood, last_tick, {', '.join(f'self_{axis}' for axis in TRAIT_AXES)})
        VALUES (:npc_id, :mood, :last_tick, {', '.join(f':self_{axis}' for axis in TRAIT_AXES)})
        ON CONFLICT(npc_id)
        DO UPDATE SET {set_clause}
        """
    )
    with engine.begin() as connection:
        connection.execute(statement, parameters)


def load_relationship(source_id: str, target_id: str) -> Optional[Dict[str, int]]:
    """Return the stored perception row for source->target."""

    # 1 Fetch relational data and transform into trait dict plus metrics.      # steps
    ensure_schema()
    engine = get_engine()
    statement = text(
        """
        SELECT * FROM relationships
        WHERE source_id = :source_id AND target_id = :target_id
        """
    )
    with engine.begin() as connection:
        row = connection.execute(statement, {"source_id": source_id, "target_id": target_id}).mappings().first()
    if row is None:
        return None
    payload: Dict[str, int] = {
        "trust": int(row.get("trust", 50)),
        "affinity": int(row.get("affinity", 50)),
    }
    for axis in TRAIT_AXES:
        payload[axis] = int(row.get(axis, 0))
    payload["summary"] = row.get("summary")
    payload["updated_at"] = row.get("updated_at")
    return payload


def upsert_relationship(
    source_id: str,
    target_id: str,
    trust: int,
    affinity: int,
    trait_perception: Dict[str, int],
    summary: Optional[str],
    updated_at: datetime,
) -> None:
    """Insert or update the perception record for source about target."""

    # 1 Construct the insert with ON CONFLICT to keep code explicit.           # steps
    ensure_schema()
    engine = get_engine()
    column_names = ["source_id", "target_id", "trust", "affinity", "summary", "updated_at"]
    trait_columns = list(TRAIT_AXES)
    all_columns = column_names + trait_columns
    placeholders = [f":{name}" for name in column_names] + [f":{axis}" for axis in trait_columns]
    parameters: Dict[str, int | str | None] = {
        "source_id": source_id,
        "target_id": target_id,
        "trust": trust,
        "affinity": affinity,
        "summary": summary,
        "updated_at": updated_at.isoformat(),
    }
    for axis in trait_columns:
        parameters[axis] = int(trait_perception.get(axis, 0))
    set_segments: List[str] = []
    for name in ["trust", "affinity", "summary", "updated_at"]:
        set_segments.append(f"{name} = :{name}")
    for axis in trait_columns:
        set_segments.append(f"{axis} = :{axis}")
    statement = text(
        f"""
        INSERT INTO relationships ({', '.join(all_columns)})
        VALUES ({', '.join(placeholders)})
        ON CONFLICT(source_id, target_id)
        DO UPDATE SET {', '.join(set_segments)}
        """
    )
    with engine.begin() as connection:
        connection.execute(statement, parameters)


def list_relationships(source_id: str) -> List[Dict[str, int]]:
    """Return all perceptions the source holds."""

    # 1 Query rows for the source and sort by updated timestamp.               # steps
    ensure_schema()
    engine = get_engine()
    statement = text(
        """
        SELECT * FROM relationships
        WHERE source_id = :source_id
        ORDER BY updated_at DESC
        """
    )
    with engine.begin() as connection:
        rows = connection.execute(statement, {"source_id": source_id}).mappings().all()
    payloads: List[Dict[str, int]] = []
    for row in rows:
        payload = {
            "target_id": row["target_id"],
            "trust": int(row.get("trust", 50)),
            "affinity": int(row.get("affinity", 50)),
            "summary": row.get("summary"),
            "updated_at": row.get("updated_at"),
        }
        for axis in TRAIT_AXES:
            payload[axis] = int(row.get(axis, 0))
        payloads.append(payload)
    return payloads


def record_conversation_line(
    npc_a: str,
    npc_b: str,
    speaker_id: str,
    text_line: str,
    timestamp: datetime,
    keep: int = 5,
) -> List[Dict[str, str]]:
    """Insert a conversation line and prune to the configured keep count."""

    # 1 Normalize the pair order so we treat conversations as unordered.       # steps
    # 2 Insert the new line then remove oldest rows past the limit.            # steps
    ensure_schema()
    engine = get_engine()
    first_id, second_id = _normalize_pair(npc_a, npc_b)
    insert_statement = text(
        """
        INSERT INTO conversations (a_id, b_id, ts, speaker_id, text)
        VALUES (:a_id, :b_id, :ts, :speaker_id, :text)
        """
    )
    insert_parameters = {
        "a_id": first_id,
        "b_id": second_id,
        "ts": timestamp.isoformat(),
        "speaker_id": speaker_id,
        "text": text_line,
    }
    with engine.begin() as connection:
        connection.execute(insert_statement, insert_parameters)
        id_statement = text(
            """
            SELECT convo_id FROM conversations
            WHERE a_id = :a_id AND b_id = :b_id
            ORDER BY ts DESC
            """
        )
        rows = connection.execute(id_statement, {"a_id": first_id, "b_id": second_id}).fetchall()
        if len(rows) > keep:
            for row in rows[keep:]:
                convo_id = int(row[0])
                connection.execute(
                    text("DELETE FROM conversations WHERE convo_id = :convo_id"),
                    {"convo_id": convo_id},
                )
    return get_conversation_lines(npc_a, npc_b, keep)


def get_conversation_lines(npc_a: str, npc_b: str, limit: int = 5) -> List[Dict[str, str]]:
    """Return the most recent lines between two NPCs."""

    # 1 Normalize ids, fetch ordered lines, and truncate to limit.             # steps
    ensure_schema()
    engine = get_engine()
    first_id, second_id = _normalize_pair(npc_a, npc_b)
    statement = text(
        """
        SELECT ts, speaker_id, text
        FROM conversations
        WHERE a_id = :a_id AND b_id = :b_id
        ORDER BY ts DESC
        LIMIT :limit
        """
    )
    with engine.begin() as connection:
        rows = connection.execute(statement, {"a_id": first_id, "b_id": second_id, "limit": limit}).mappings().all()
    payloads: List[Dict[str, str]] = []
    for row in rows:
        payloads.append(
            {
                "ts": row["ts"],
                "speaker_id": row["speaker_id"],
                "text": row["text"],
            }
        )
    payloads.reverse()
    return payloads


def log_event(npc_id: str, target_id: Optional[str], event_type: str, data_json: str, timestamp: datetime) -> None:
    """Persist an event to the event_log table."""

    # 1 Insert a row with explicit parameters.                                # steps
    ensure_schema()
    engine = get_engine()
    statement = text(
        """
        INSERT INTO event_log (npc_id, target_id, type, data, ts)
        VALUES (:npc_id, :target_id, :type, :data, :ts)
        """
    )
    parameters = {
        "npc_id": npc_id,
        "target_id": target_id,
        "type": event_type,
        "data": data_json,
        "ts": timestamp.isoformat(),
    }
    with engine.begin() as connection:
        connection.execute(statement, parameters)


def fetch_events(limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Return recent events for export."""

    # 1 Query event table in descending time order.                            # steps
    ensure_schema()
    engine = get_engine()
    query = "SELECT npc_id, target_id, type, data, ts FROM event_log ORDER BY ts DESC"
    if limit is not None:
        query += " LIMIT :limit"
    statement = text(query)
    params = {"limit": limit} if limit is not None else {}
    with engine.begin() as connection:
        rows = connection.execute(statement, params).mappings().all()
    payloads: List[Dict[str, str]] = []
    for row in rows:
        payloads.append(
            {
                "npc_id": row["npc_id"],
                "target_id": row["target_id"],
                "type": row["type"],
                "data": row["data"],
                "ts": row["ts"],
            }
        )
    payloads.reverse()
    return payloads


def delete_conversation(npc_a: str, npc_b: str) -> None:
    """Remove all stored conversation lines for a pair."""

    ensure_schema()
    engine = get_engine()
    first_id, second_id = _normalize_pair(npc_a, npc_b)
    statement = text(
        """
        DELETE FROM conversations
        WHERE a_id = :a_id AND b_id = :b_id
        """
    )
    with engine.begin() as connection:
        connection.execute(statement, {"a_id": first_id, "b_id": second_id})


def _normalize_pair(npc_a: str, npc_b: str) -> Tuple[str, str]:
    """Ensure conversation pairs use a consistent alphabetical order."""

    # 1 Sort ids so the same pair maps to one row set.                         # steps
    if npc_a <= npc_b:
        return npc_a, npc_b
    return npc_b, npc_a


# TODO: add migrations once new tables are introduced beyond MVP scope.
