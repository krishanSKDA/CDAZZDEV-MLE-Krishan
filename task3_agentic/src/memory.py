"""
Task 3C - Memory and Observability

- observed_tool: decorator that logs every tool call (name, args, output
  truncated to 200 chars, wall-clock duration) to agent_trace.jsonl.
- ShortTermMemory: an in-session cache keyed by (tool_name, args) so a
  follow-up question can be answered from a previous tool result
  without re-calling the tool.
- PersistentCache: saves/loads a finished research brief to/from a JSON
  file keyed by ticker + date, so a second run on the same day for the
  same ticker skips re-running all tools.
"""
from __future__ import annotations

import functools
import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable

TRACE_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "agent_trace.jsonl")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "cache")


def _append_trace(record: dict) -> None:
    os.makedirs(os.path.dirname(TRACE_LOG_PATH), exist_ok=True)
    with open(TRACE_LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def observed_tool(func: Callable) -> Callable:
    """Decorator: logs every call of `func` to agent_trace.jsonl."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            error = None
        except Exception as exc:  # tools are expected to fail soft, but belt & braces
            result = {"error": str(exc)}
            error = str(exc)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        output_str = json.dumps(result, default=str)
        record = {
            "timestamp": time.time(),
            "tool": func.__name__,
            "args": args,
            "kwargs": kwargs,
            "output_truncated": output_str[:200],
            "duration_ms": duration_ms,
            "error": error,
        }
        _append_trace(record)
        return result

    return wrapper


class ShortTermMemory:
    """Session-scoped cache: avoids re-calling a tool for a question already answered."""

    def __init__(self):
        self._store: dict[str, Any] = {}

    def _key(self, tool_name: str, *args, **kwargs) -> str:
        return f"{tool_name}:{args}:{sorted(kwargs.items())}"

    def get_or_call(self, tool_fn: Callable, *args, **kwargs) -> tuple[Any, bool]:
        """Returns (result, was_cache_hit)."""
        key = self._key(tool_fn.__name__, *args, **kwargs)
        if key in self._store:
            return self._store[key], True
        result = tool_fn(*args, **kwargs)
        self._store[key] = result
        return result, False


class PersistentCache:
    """Keyed by ticker + date; survives across process runs on disk."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str, as_of: str | None = None) -> Path:
        as_of = as_of or date.today().isoformat()
        return self.cache_dir / f"{ticker}_{as_of}.json"

    def load(self, ticker: str, as_of: str | None = None) -> dict | None:
        path = self._path(ticker, as_of)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def save(self, ticker: str, brief: dict, as_of: str | None = None) -> str:
        path = self._path(ticker, as_of)
        with open(path, "w") as f:
            json.dump(brief, f, indent=2, default=str)
        return str(path)
