"""Microphone audio capture via sounddevice."""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from collections.abc import Callable


def auto_select_mic() -> int | None:
    """Pick the best input device, preferring external mics over built-in.

    Priority order:
      1. AirPods / Bluetooth headsets (name contains "airpod", "beats", "bluetooth")
      2. Other external mics (USB, headset — any input device that isn't built-in)
      3. System default input (None)

    Returns the device index, or None for the system default.
    """
    devices = sd.query_devices()
    # Skip built-in mics and iPhone Continuity mics (high latency, unreliable)
    skip_keywords = {"macbook", "built-in", "iphone"}

    external: list[tuple[int, str]] = []
    bluetooth: list[tuple[int, str]] = []

    for i, dev in enumerate(devices):
        # Skip output-only devices
        if dev["max_input_channels"] < 1:
            continue
        name_lower = dev["name"].lower()
        if any(kw in name_lower for kw in skip_keywords):
            continue
        # Classify
        if any(kw in name_lower for kw in ("airpod", "beats", "bluetooth", "headphone")):
            bluetooth.append((i, dev["name"]))
        else:
            external.append((i, dev["name"]))

    if bluetooth:
        idx, name = bluetooth[0]
        print(f"[MicCapture] Auto-selected mic: {name} (device {idx})")
        return idx
    if external:
        idx, name = external[0]
        print(f"[MicCapture] Auto-selected mic: {name} (device {idx})")
        return idx

    # Fall back to system default
    default = sd.query_devices(kind="input")
    print(f"[MicCapture] Using default mic: {default['name']}")
    return None


class MicCapture:
    """Captures audio from the microphone and feeds frames to a queue.

    Thread-safe: start/stop are protected by an internal lock.
    Logs dropped frames and device errors for observability.
    """

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
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1000)
        self._stream: sd.InputStream | None = None
        self._running = False
        self._lock = threading.Lock()

        # Observability
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
                return  # Don't queue corrupted frames
        try:
            self._queue.put_nowait(indata.copy())
        except queue.Full:
            self._frames_dropped += 1
            if self._frames_dropped == 1 or self._frames_dropped % 100 == 0:
                print(f"[MicCapture] WARNING: Dropped {self._frames_dropped} frames total")

    def start(self) -> None:
        """Open the microphone stream."""
        with self._lock:
            if self._running:
                return
            self._frames_dropped = 0
            self._device_errors = 0
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

    def stop(self) -> None:
        """Close the microphone stream."""
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._running = False
        # Drain queue outside lock to avoid deadlock
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Read the next audio frame. Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_all(self) -> list[np.ndarray]:
        """Drain all available frames from the queue without blocking."""
        frames = []
        while True:
            try:
                frames.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return frames

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        """Return capture health stats."""
        return {
            "frames_dropped": self._frames_dropped,
            "device_errors": self._device_errors,
            "last_device_error": self._last_device_error,
        }

    @staticmethod
    def list_devices() -> str:
        """Return a string listing available audio devices."""
        return str(sd.query_devices())
