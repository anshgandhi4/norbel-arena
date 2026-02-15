"""Environment loading helpers for agent providers."""

from __future__ import annotations

import os
from pathlib import Path

_DOTENV_LOADED = False


def load_dotenv(path: str | Path = ".env") -> None:
    """Load environment variables from a .env file without overriding existing values."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    dotenv_path = Path(path)
    if not dotenv_path.exists():
        _DOTENV_LOADED = True
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)

    _DOTENV_LOADED = True


def getenv_any(*names: str, default: str | None = None) -> str | None:
    """Return first defined env var from a list of candidate names."""
    load_dotenv()
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def require_env_any(*names: str) -> str:
    """Return first defined env var value or raise a readable error."""
    value = getenv_any(*names)
    if value is not None:
        return value
    joined = ", ".join(names)
    raise ValueError(f"Missing required environment variable. Set one of: {joined}")
