"""Personal dictionary — loads custom vocabulary from a TOML config file.

Terms are used to:
1. Inject into the LLM cleanup prompt ("Always spell these terms exactly: ...")
2. Feed to Whisper's initial_prompt to bias ASR toward correct recognition.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

log = logging.getLogger(__name__)

_EMPTY: dict = {"terms": [], "prompt_hint": "", "whisper_hint": ""}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PATH = _PROJECT_ROOT / "config" / "dictionary.toml"


def load_dictionary(path: Path | None = None) -> dict:
    """Load custom vocabulary terms from a TOML config file.

    Parameters
    ----------
    path:
        Path to the dictionary TOML file.  Defaults to
        ``<project_root>/config/dictionary.toml``.

    Returns
    -------
    dict
        ``{"terms": [...], "prompt_hint": "...", "whisper_hint": "..."}``
        On any error the returned dict contains empty values.
    """
    path = path or _DEFAULT_PATH

    if not path.exists():
        log.debug("Dictionary file not found: %s — returning empty dictionary", path)
        return dict(_EMPTY)

    try:
        raw = path.read_bytes()
        if not raw.strip():
            log.debug("Dictionary file is empty: %s", path)
            return dict(_EMPTY)

        data = tomllib.loads(raw.decode())
    except Exception:
        log.exception("Failed to load dictionary from %s", path)
        return dict(_EMPTY)

    names: list[str] = data.get("names", [])
    terms: list[str] = data.get("terms", [])
    all_terms: list[str] = names + terms

    if not all_terms:
        return dict(_EMPTY)

    whisper_hint = ", ".join(all_terms)
    prompt_hint = f"Custom vocabulary: {whisper_hint}"

    return {
        "terms": all_terms,
        "prompt_hint": prompt_hint,
        "whisper_hint": whisper_hint,
    }
