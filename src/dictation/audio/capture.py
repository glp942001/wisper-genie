"""Microphone audio capture via sounddevice."""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd


def _log_mic(msg: str) -> None:
    """Log mic selection — only visible with --verbose."""
    import sys
    if "--verbose" in sys.argv or "--debug" in sys.argv or "-v" in sys.argv:
        print(f"[MicCapture] {msg}")


def auto_select_mic() -> int | None:
    """Pick the best input device, preferring external mics over built-in."""
    devices = sd.query_devices()
    skip_keywords = {"macbook", "built-in", "iphone"}

    external: list[tuple[int, str]] = []
    bluetooth: list[tuple[int, str]] = []

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue
        name_lower = dev["name"].lower()
        if any(keyword in name_lower for keyword in skip_keywords):
            continue
        if any(keyword in name_lower for keyword in ("airpod", "beats", "bluetooth", "headphone")):
            bluetooth.append((idx, dev["name"]))
        else:
            external.append((idx, dev["name"]))

    if bluetooth:
        idx, name = bluetooth[0]
        _log_mic(f"Auto-selected mic: {name} (device {idx})")
        return idx
    if external:
        idx, name = external[0]
        _log_mic(f"Auto-selected mic: {name} (device {idx})")
        return idx

    default = sd.query_devices(kind="input")
    _log_mic(f"Using default mic: {default['name']}")
    return None


class MicCapture:
    """Captures audio from the microphone and feeds frames to a ring buffer."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_duration_ms: int = 30,
        device: int | str | None = None,
        dtype: str = "int16",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        if device == "auto":
            self.device = auto_select_mic()
        elif device == "default" or device is None:
            self.device = None
        else:
            self.device = device

        self._max_buffered_frames = 100
        self._frames: deque[np.ndarray] = deque(maxlen=self._max_buffered_frames)
        self._stream: sd.InputStream | None = None
        self._running = False
        self._lock = threading.Lock()
        self._buffer_cv = threading.Condition(self._lock)

        self._frames_dropped = 0
        self._device_errors = 0
        self._last_device_error: str | None = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            self._device_errors += 1
            self._last_device_error = str(status)
            if self._device_errors <= 5 or self._device_errors % 50 == 0:
                print(f"[MicCapture] Device warning: {status} (count: {self._device_errors})")
            if status.input_overflow:
                return

        with self._buffer_cv:
            if len(self._frames) == self._max_buffered_frames:
                self._frames.popleft()
                self._frames_dropped += 1
                if self._frames_dropped == 1 or self._frames_dropped % 100 == 0:
                    print(f"[MicCapture] WARNING: Dropped {self._frames_dropped} stale frames total")
            self._frames.append(indata.copy())
            self._buffer_cv.notify()

    def start(self) -> None:
        """Open the microphone stream."""
        with self._lock:
            if self._running:
                return
            self._frames.clear()
            self._frames_dropped = 0
            self._device_errors = 0
            self._last_device_error = None
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    blocksize=self.frame_size,
                    device=self.device,
                    callback=self._audio_callback,
                )
                self._stream.start()
            except Exception as exc:
                self._stream = None
                raise RuntimeError(
                    f"Failed to open microphone: {exc}. "
                    "Check that the mic is connected and not in use by another app."
                ) from exc
            self._running = True
            self._buffer_cv.notify_all()

    def stop(self) -> None:
        """Close the microphone stream."""
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._running = False
            self._frames.clear()
            self._buffer_cv.notify_all()

    def restart(self) -> None:
        """Restart the input stream after repeated read stalls or device errors."""
        self.stop()
        time.sleep(0.1)
        self.start()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Read the next audio frame. Returns None on timeout."""
        deadline = time.monotonic() + timeout
        with self._buffer_cv:
            while not self._frames:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._buffer_cv.wait(timeout=remaining)
            return self._frames.popleft()

    def read_all(self) -> list[np.ndarray]:
        """Drain all available frames from the buffer without blocking."""
        with self._buffer_cv:
            frames = list(self._frames)
            self._frames.clear()
            return frames

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        """Return capture health stats."""
        with self._lock:
            return {
                "buffered_frames": len(self._frames),
                "frames_dropped": self._frames_dropped,
                "device_errors": self._device_errors,
                "last_device_error": self._last_device_error,
            }

    @staticmethod
    def list_devices() -> str:
        """Return a string listing available audio devices."""
        return str(sd.query_devices())
