#!/usr/bin/env python3
"""
Poll OpenRouter cost snapshots from Redis and persist them into OpenWebUI-Monitor’s Postgres tables so the UI dashboards remain accurate.
Repository: https://github.com/rbb-dev/OpenWebUI-Monitor
This script enables OpenWebUI-Monitor to run without the openwebui_monitor.py filter.

Behavior overview:
    * Connects to Redis using REDIS_URL and scans for keys matching REDIS_COST_PATTERN
    * For each payload, upserts the user (mirroring app/api/v1/inlet), debits balance,
      and inserts a user_usage_records row (mirroring app/api/v1/outlet)
    * Deletes the Redis key only after a successful Postgres transaction
    * On failures, extends the key TTL so data can be retried later
Note: this is a working example, not production ready
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional, Tuple

import psycopg
import redis


LOGGER = logging.getLogger("usage_ingest")

def load_env_file(path: Optional[str]) -> None:
    if not path:
        return

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        LOGGER.debug("Env file %s not found; skipping", path)
    except OSError as exc:
        LOGGER.debug("Failed to read env file %s: %s", path, exc)


# ── Redis configuration ──────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_SCAN_PATTERN = os.getenv("REDIS_COST_PATTERN", "costs:openrouter_responses_api_pipe:*")
REDIS_SCAN_COUNT = int(os.getenv("REDIS_SCAN_COUNT", "200"))
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "3"))
FAILED_KEY_TTL_SECONDS = int(os.getenv("FAILED_KEY_TTL_SECONDS", "3600"))
INIT_BALANCE = Decimal(os.getenv("INIT_BALANCE", "0")).quantize(
    Decimal("0.0001"), rounding=ROUND_HALF_UP
)
POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA")
POSTGRES_CONNECT_TIMEOUT = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10"))


# ── Postgres configuration (placeholders for future use) ─────────────────────
@dataclass(frozen=True)
class PostgresConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "database")
    user: str = os.getenv("POSTGRES_USER", "username")
    password: str = os.getenv("POSTGRES_PASSWORD", "password")
    ssl_mode: str = os.getenv("POSTGRES_SSL_MODE", "require")
    ssl_root_cert: Optional[str] = os.getenv("POSTGRES_SSL_ROOT_CERT")


POSTGRES = PostgresConfig()

_PG_CONN: Optional[psycopg.Connection] = None


def _format_decimal(value: Any, places: str = "0.0001") -> Decimal:
    try:
        dec_value = Decimal(str(value))
    except (ValueError, TypeError, ArithmeticError):
        return Decimal("0").quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return dec_value.quantize(Decimal(places), rounding=ROUND_HALF_UP)


def get_postgres_connection() -> psycopg.Connection:
    global _PG_CONN
    if _PG_CONN and not _PG_CONN.closed:
        return _PG_CONN

    conn_kwargs = {
        "host": POSTGRES.host,
        "port": POSTGRES.port,
        "dbname": POSTGRES.database,
        "user": POSTGRES.user,
        "password": POSTGRES.password,
        "sslmode": POSTGRES.ssl_mode,
        "connect_timeout": POSTGRES_CONNECT_TIMEOUT,
    }

    if POSTGRES.ssl_root_cert:
        conn_kwargs["sslrootcert"] = POSTGRES.ssl_root_cert

    _PG_CONN = psycopg.connect(**conn_kwargs)
    _PG_CONN.autocommit = False

    if POSTGRES_SCHEMA:
        with _PG_CONN.cursor() as cur:
            cur.execute("SET search_path TO %s", (POSTGRES_SCHEMA,))

    return _PG_CONN


def reset_postgres_connection() -> None:
    global _PG_CONN
    if _PG_CONN and not _PG_CONN.closed:
        try:
            _PG_CONN.close()
        except Exception:
            LOGGER.exception("Error closing Postgres connection")
    _PG_CONN = None


@contextmanager
def postgres_cursor():
    conn = get_postgres_connection()
    try:
        with conn.cursor() as cur:
            yield conn, cur
    except psycopg.Error:
        reset_postgres_connection()
        raise


def extract_identity(snapshot: Dict[str, Any]) -> Tuple[str, str, str]:
    user_id = snapshot.get("guid")
    if not user_id:
        raise ValueError("Snapshot missing required 'guid' field for user id")
    email = snapshot.get("email") or f"{user_id}@unknown.local"
    name = snapshot.get("name") or email
    return str(user_id), str(email), str(name)


def parse_tokens(usage: Dict[str, Any]) -> Tuple[int, int]:
    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens")

    if input_tokens is None and usage.get("total_tokens") is not None and output_tokens is not None:
        input_tokens = _as_int(usage.get("total_tokens")) - _as_int(output_tokens)

    return _as_int(input_tokens), _as_int(output_tokens)


def parse_use_time(raw_ts: Any) -> datetime:
    if raw_ts is None:
        return datetime.now(timezone.utc)
    try:
        if isinstance(raw_ts, (int, float)):
            return datetime.fromtimestamp(raw_ts, tz=timezone.utc)
        raw_str = str(raw_ts).strip()
        if raw_str.isdigit():
            return datetime.fromtimestamp(int(raw_str), tz=timezone.utc)
        if raw_str.replace(".", "", 1).isdigit():
            return datetime.fromtimestamp(float(raw_str), tz=timezone.utc)
        if raw_str.endswith("Z"):
            raw_str = raw_str[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        LOGGER.warning("Failed to parse timestamp %s; defaulting to now", raw_ts)
        return datetime.now(timezone.utc)


def upsert_user(cur: psycopg.Cursor, user_id: str, email: str, name: str) -> None:
    cur.execute(
        """
        INSERT INTO users (id, email, name, role, balance)
        VALUES (%s, %s, %s, 'user', %s)
        ON CONFLICT (id) DO UPDATE
        SET email = EXCLUDED.email,
            name = EXCLUDED.name
        """,
        (user_id, email, name, INIT_BALANCE),
    )


def debit_user_balance(cur: psycopg.Cursor, user_id: str, cost: Decimal) -> Decimal:
    cur.execute(
        """
        UPDATE users
        SET balance = LEAST(
            balance - CAST(%s AS DECIMAL(16,4)),
            999999.9999
        )
        WHERE id = %s
        RETURNING balance
        """,
        (cost, user_id),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"User {user_id} does not exist for balance update")
    return Decimal(row[0])


def insert_usage_record(
    cur: psycopg.Cursor,
    *,
    user_id: str,
    nickname: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cost: Decimal,
    balance_after: Decimal,
    use_time: datetime,
) -> None:
    cur.execute(
        """
        INSERT INTO user_usage_records (
            user_id,
            nickname,
            use_time,
            model_name,
            input_tokens,
            output_tokens,
            cost,
            balance_after
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            user_id,
            nickname,
            use_time,
            model_name,
            input_tokens,
            output_tokens,
            cost,
            balance_after,
        ),
    )


def configure_logging(debug: bool = False) -> None:
    level_name = os.getenv("LOG_LEVEL", "ERROR").upper()
    if debug:
        level_name = "DEBUG"

    level = getattr(logging, level_name, logging.ERROR)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    LOGGER.setLevel(level)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drain OpenRouter cost snapshots into Postgres")
    parser.add_argument(
        "--env-file",
        help="Optional path to a .env-style file to preload into the process environment.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging output for troubleshooting",
    )
    return parser.parse_args()


# ── Redis helpers ────────────────────────────────────────────────────────────
def get_redis_client() -> redis.Redis:
    """Return a Redis client using the configured URL."""
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


# ── Postgres placeholders ────────────────────────────────────────────────────
def describe_postgres_target() -> str:
    """Return a human-readable description of the configured Postgres target."""
    ssl_bits = f"sslmode={POSTGRES.ssl_mode}"
    if POSTGRES.ssl_root_cert:
        ssl_bits += f", sslrootcert={POSTGRES.ssl_root_cert}"
    return (
        f"{POSTGRES.user}@{POSTGRES.host}:{POSTGRES.port}/{POSTGRES.database} "
        f"({ssl_bits})"
    )


def process_snapshot(snapshot: Dict[str, Any], redis_key: str) -> None:
    """Persist a single snapshot inside a Postgres transaction."""
    usage = snapshot.get("usage") or {}
    user_id, email, nickname = extract_identity(snapshot)

    input_tokens, output_tokens = parse_tokens(usage)
    total_cost = _format_decimal(usage.get("cost", 0))
    if total_cost < Decimal("0"):
        total_cost = Decimal("0.0000")

    model_name = snapshot.get("model") or "unknown"
    use_time = parse_use_time(snapshot.get("ts"))

    with postgres_cursor() as (conn, cur):
        try:
            upsert_user(cur, user_id, email, nickname)
            new_balance = debit_user_balance(cur, user_id, total_cost)
            insert_usage_record(
                cur,
                user_id=user_id,
                nickname=nickname,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=total_cost,
                balance_after=new_balance,
                use_time=use_time,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    LOGGER.info(
        "Persisted %s | user=%s cost=%s in/out=%s/%s balance=%s",
        redis_key,
        user_id,
        total_cost,
        input_tokens,
        output_tokens,
        new_balance,
    )


def drain_cost_snapshots(client: redis.Redis) -> int:
    """Scan Redis for cost records, emit them, and delete the processed keys.

    Returns:
        int: Number of snapshots handled during this pass.
    """
    cursor = 0
    processed = 0
    while True:
        cursor, keys = client.scan(
            cursor=cursor,
            match=REDIS_SCAN_PATTERN,
            count=REDIS_SCAN_COUNT,
        )
        for key in keys:
            raw_payload = client.get(key)
            if raw_payload is None:
                continue
            try:
                snapshot = json.loads(raw_payload)
            except json.JSONDecodeError:
                LOGGER.warning("Discarding malformed payload for key %s", key)
                client.delete(key)
                continue

            try:
                process_snapshot(snapshot, key)
            except Exception:
                LOGGER.exception("Failed to persist snapshot for key %s", key)
                if FAILED_KEY_TTL_SECONDS > 0:
                    client.expire(key, FAILED_KEY_TTL_SECONDS)
                continue

            client.delete(key)
            processed += 1

        if cursor == 0:
            break
    return processed


def main() -> None:
    args = parse_args()
    configure_logging(debug=args.debug)
    load_env_file(getattr(args, "env_file", None))

    LOGGER.info("Starting usage drain. Postgres target: %s", describe_postgres_target())
    client = get_redis_client()

    try:
        while True:
            handled = drain_cost_snapshots(client)
            if handled == 0:
                LOGGER.debug(
                    "No snapshots found (pattern=%s). Sleeping for %ss.",
                    REDIS_SCAN_PATTERN,
                    POLL_INTERVAL_SECONDS,
                )
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        LOGGER.info("Shutting down.")
        sys.exit(0)
    finally:
        reset_postgres_connection()


if __name__ == "__main__":
    main()
