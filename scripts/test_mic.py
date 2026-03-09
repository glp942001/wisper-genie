#!/usr/bin/env python3
"""Quick mic sanity check — records 3 seconds and prints stats."""

import sys
import time

import numpy as np
import sounddevice as sd


def main() -> None:
    print("Available audio devices:")
    print(sd.query_devices())
    print()

    sample_rate = 16000
    duration = 3.0
    print(f"Recording {duration}s at {sample_rate}Hz... Speak now!")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    audio_flat = audio.flatten()
    rms = np.sqrt(np.mean(audio_flat.astype(np.float32) ** 2))
    peak = np.max(np.abs(audio_flat))

    print(f"Recorded {len(audio_flat)} samples ({duration}s)")
    print(f"RMS level: {rms:.1f}")
    print(f"Peak level: {peak}")
    print(f"Silence detected: {'yes (check mic)' if rms < 50 else 'no (audio captured)'}")

    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        from scipy.io import wavfile
        path = "samples/test_audio.wav"
        wavfile.write(path, sample_rate, audio_flat)
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
