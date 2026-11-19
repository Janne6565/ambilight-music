"""Loudness and beats from the capture card's audio input.

Reads raw PCM from `arecord` (no Python audio bindings needed, and it releases the device as
soon as the process ends), and turns each ~21 ms block into two numbers:

* `level`: loudness relative to the recent peak (auto gain), so quiet and loud recordings
  both use the full range; fast attack, slow release.
* `beat`: set to 1 when the bass energy jumps well above its running average, then decays.
"""
from __future__ import annotations

import logging
import math
import subprocess
import threading
import time

import numpy as np

from .renderer import AudioState

log = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS = 2
BLOCK_FRAMES = 1024

BASS_LOW_HZ = 30.0
BASS_HIGH_HZ = 150.0

SILENCE_RMS = 0.002          # normalised; below this the room is treated as silent
PEAK_DECAY_PER_BLOCK = 0.9985  # auto gain forgets a loud peak over ~15 s
LEVEL_ATTACK = 0.5
LEVEL_RELEASE = 0.06

BEAT_RATIO = 1.45            # bass this much above its running average counts as a beat
BASS_AVERAGE_ALPHA = 0.03    # running average over roughly the last second
MIN_BEAT_INTERVAL_S = 0.22   # nothing faster than ~270 BPM
BEAT_DECAY_S = 0.15


class AudioAnalyzer:
    """Pure analysis, fed block by block. Kept separate from capture so it can be tested."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.state = AudioState()
        freqs = np.fft.rfftfreq(BLOCK_FRAMES, 1.0 / sample_rate)
        self._bass = (freqs >= BASS_LOW_HZ) & (freqs <= BASS_HIGH_HZ)
        self._window = np.hanning(BLOCK_FRAMES)
        self._peak = SILENCE_RMS * 4
        self._bass_avg = 0.0
        self._last_beat = -math.inf
        self._beat_at = -math.inf

    def feed(self, mono: np.ndarray, now: float) -> AudioState:
        """`mono` is one block of float samples in -1..1."""
        rms = float(np.sqrt(np.mean(mono * mono)))
        spectrum = np.abs(np.fft.rfft(mono * self._window))
        bass = float(np.sqrt(np.mean(spectrum[self._bass] ** 2)))

        if rms < SILENCE_RMS:
            target = 0.0
        else:
            self._peak = max(rms, self._peak * PEAK_DECAY_PER_BLOCK)
            target = min(1.0, rms / self._peak)
        level = self.state.level
        level += (target - level) * (LEVEL_ATTACK if target > level else LEVEL_RELEASE)

        if (
            rms >= SILENCE_RMS
            and self._bass_avg > 0
            and bass > BEAT_RATIO * self._bass_avg
            and now - self._last_beat >= MIN_BEAT_INTERVAL_S
        ):
            self._last_beat = now
            self._beat_at = now
        self._bass_avg += (bass - self._bass_avg) * BASS_AVERAGE_ALPHA

        self.state = AudioState(level=level, beat=self.beat_at(now))
        return self.state

    def beat_at(self, now: float) -> float:
        """Beat pulse decayed to `now`, so the renderer sees a smooth falloff between blocks."""
        return math.exp(-(now - self._beat_at) / BEAT_DECAY_S) if now >= self._beat_at else 0.0


class AudioCapture:
    """Runs `arecord` in a thread while started; `state()` is safe to call from anywhere."""

    def __init__(self, device: str):
        self.device = device
        self._analyzer = AudioAnalyzer()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="audio", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            proc = self._proc
        if proc is not None:
            proc.terminate()

    def state(self, now: float) -> AudioState:
        with self._lock:
            s = self._analyzer.state
            return AudioState(level=s.level, beat=self._analyzer.beat_at(now))

    def _loop(self) -> None:
        block_bytes = BLOCK_FRAMES * CHANNELS * 2
        while self._running:
            cmd = ["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                   "-c", str(CHANNELS), "-t", "raw", "--buffer-time=50000"]
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except OSError as e:
                log.error("Cannot start arecord: %s", e)
                time.sleep(5)
                continue
            with self._lock:
                self._proc = proc
            log.info("Audio capture started on %s", self.device)
            while self._running:
                data = proc.stdout.read(block_bytes)
                if len(data) < block_bytes:
                    break
                samples = np.frombuffer(data, dtype="<i2").reshape(-1, CHANNELS).astype(np.float64)
                mono = samples.mean(axis=1) / 32768.0
                now = time.monotonic()
                with self._lock:
                    self._analyzer.feed(mono, now)
            proc.terminate()
            err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
            proc.wait()
            with self._lock:
                self._proc = None
                self._analyzer.state = AudioState()
            if self._running:
                # Device busy (HyperHDR's own music effects hold it) or unplugged: retry calmly.
                log.warning("Audio capture ended%s, retrying in 5 s", f": {err}" if err else "")
                time.sleep(5)
        log.info("Audio capture stopped")
