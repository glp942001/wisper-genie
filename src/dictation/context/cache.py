"""Short-lived cache for focused screen context."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from dictation.context.screen import FocusedTextDetails, get_focused_text_details


@dataclass(frozen=True)
class CachedScreenContext:
    details: FocusedTextDetails
    fetched_at: float

    @property
    def age_ms(self) -> float:
        return (time.monotonic() - self.fetched_at) * 1000


class ScreenContextCache:
    """Caches focused-field metadata briefly to reduce AX churn on the hot path."""

    def __init__(self, ttl_ms: int = 750) -> None:
        self._ttl_s = ttl_ms / 1000.0
        self._lock = threading.Lock()
        self._cached_light: CachedScreenContext | None = None
        self._cached_full: CachedScreenContext | None = None

    def prefetch(self, *, include_full_text: bool = False) -> FocusedTextDetails:
        return self.get(include_full_text=include_full_text, force_refresh=True)

    def get(
        self,
        *,
        include_full_text: bool = False,
        force_refresh: bool = False,
    ) -> FocusedTextDetails:
        with self._lock:
            cached = self._cached_full if include_full_text else self._cached_light
            if not force_refresh and cached is not None and (time.monotonic() - cached.fetched_at) <= self._ttl_s:
                return cached.details

        details = get_focused_text_details(include_full_text=include_full_text)
        snapshot = CachedScreenContext(details=details, fetched_at=time.monotonic())
        with self._lock:
            if include_full_text:
                self._cached_full = snapshot
                self._cached_light = CachedScreenContext(
                    details=FocusedTextDetails(
                        app_name=details.app_name,
                        full_text="",
                        field_text=details.field_text,
                        selected_text=details.selected_text,
                        selected_range=details.selected_range,
                        focused_element=details.focused_element,
                        app_pid=details.app_pid,
                    ),
                    fetched_at=snapshot.fetched_at,
                )
            else:
                self._cached_light = snapshot
        return details

    def clear(self) -> None:
        with self._lock:
            self._cached_light = None
            self._cached_full = None
