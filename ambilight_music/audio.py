"""Beats and dynamics from the capture card's audio input.

Reads raw PCM from `arecord` (no Python audio bindings needed, and it releases the device as
soon as the process ends) and analyses every ~21 ms block.

Loudness alone is useless for modern, heavily compressed music: it sits at its ceiling the
whole song. So the analysis looks for *onsets* instead, sudden rises of energy per band, each
judged against that band's own recent behaviour (spectral flux with an adaptive threshold):

* kick:  bass band, the main pulse of the light
* snare: mid band, a secondary accent
* hats:  high band, a continuous "shimmer" amount

Loudness is only used slowly and relatively (this moment vs. the last ~10 s), so a quiet verse
and a loud chorus still look different even when both are compressed to the ceiling.
"""
from __future__ import annotations

import logging
import math
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .renderer import AudioState

log = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS = 2
BLOCK_FRAMES = 1024
BLOCK_S = BLOCK_FRAMES / SAMPLE_RATE

SILENCE_RMS = 0.002  # normalised; below this the input is treated as silent


@dataclass(frozen=True)
class Band:
    name: str
    low_hz: float
    high_hz: float
    sensitivity: float     # threshold = mean + sensitivity * std of the recent flux
    min_flux: float        # absolute floor, in log10 energy units (0.1 = 1 dB)
    refractory_s: float    # no second onset in this band before this much time passed
    decay_s: float         # how long the pulse takes to fade


KICK = Band("kick", 35, 150, sensitivity=1.4, min_flux=0.12, refractory_s=0.24, decay_s=0.18)
SNARE = Band("snare", 300, 3000, sensitivity=1.6, min_flux=0.10, refractory_s=0.14, decay_s=0.12)
HATS = Band("hats", 5000, 14000, sensitivity=1.0, min_flux=0.05, refractory_s=0.0, decay_s=0.0)

FLUX_HISTORY_S = 1.0
LEVEL_HISTORY_S = 10.0
LEVEL_RANGE_DB = 12.0      # +-6 dB around the 10 s average maps onto 0..1
LEVEL_SMOOTHING = 0.05
SHIMMER_SMOOTHING = 0.25


class _BandTracker:
    def __init__(self, band: Band, freqs: np.ndarray):
        self.band = band
        self.mask = (freqs >= band.low_hz) & (freqs <= band.high_hz)
        self.prev_energy: float | None = None
        self.history: deque[float] = deque(maxlen=round(FLUX_HISTORY_S / BLOCK_S))
        self.last_onset = -math.inf

    def feed(self, power: np.ndarray, now: float) -> tuple[bool, float]:
        """Returns (onset?, flux relative to the threshold)."""
        energy = math.log10(1e-12 + float(power[self.mask].mean()))
        flux = 0.0 if self.prev_energy is None else max(0.0, energy - self.prev_energy)
        self.prev_energy = energy

        onset = False
        ratio = 0.0
        if len(self.history) >= self.history.maxlen // 2:
            h = np.fromiter(self.history, float)
            threshold = max(self.band.min_flux, h.mean() + self.band.sensitivity * h.std())
            ratio = flux / threshold
            if flux > threshold and now - self.last_onset >= self.band.refractory_s:
                onset = True
                self.last_onset = now
        self.history.append(flux)
        return onset, ratio

    def pulse(self, now: float) -> float:
        if self.band.decay_s <= 0 or now < self.last_onset:
            return 0.0
        return math.exp(-(now - self.last_onset) / self.band.decay_s)

    def reset(self) -> None:
        self.prev_energy = None
        self.history.clear()
        self.last_onset = -math.inf


class AudioAnalyzer:
    """Pure analysis, fed block by block. Kept separate from capture so it can be tested."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        freqs = np.fft.rfftfreq(BLOCK_FRAMES, 1.0 / sample_rate)
        self._window = np.hanning(BLOCK_FRAMES)
        self.kick = _BandTracker(KICK, freqs)
        self.snare = _BandTracker(SNARE, freqs)
        self.hats = _BandTracker(HATS, freqs)
        self._db_history: deque[float] = deque(maxlen=round(LEVEL_HISTORY_S / BLOCK_S))
        self._level = 0.0
        self._shimmer = 0.0
        self.state = AudioState()

    def feed(self, mono: np.ndarray, now: float) -> AudioState:
        """`mono` is one block of float samples in -1..1."""
        rms = float(np.sqrt(np.mean(mono * mono)))
        if rms < SILENCE_RMS:
            for tracker in (self.kick, self.snare, self.hats):
                tracker.reset()
            self._level += (0.0 - self._level) * LEVEL_SMOOTHING
            self._shimmer = 0.0
            self.state = AudioState(level=self._level)
            return self.state

        power = np.abs(np.fft.rfft(mono * self._window)) ** 2
        self.kick.feed(power, now)
        self.snare.feed(power, now)
        _, hats_ratio = self.hats.feed(power, now)

        db = 20 * math.log10(rms)
        self._db_history.append(db)
        relative = (db - float(np.mean(self._db_history))) / LEVEL_RANGE_DB + 0.5
        self._level += (min(1.0, max(0.0, relative)) - self._level) * LEVEL_SMOOTHING
        self._shimmer += (min(1.0, hats_ratio) - self._shimmer) * SHIMMER_SMOOTHING

        self.state = self.at(now)
        return self.state

    def at(self, now: float) -> AudioState:
        """State with pulses decayed to `now`, so the renderer sees smooth falloffs between blocks."""
        return AudioState(level=self._level, beat=self.kick.pulse(now), snare=self.snare.pulse(now),
                          shimmer=self._shimmer)


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
            return self._analyzer.at(now)

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
                mono = to_mono(data)
                now = time.monotonic()
                with self._lock:
                    self._analyzer.feed(mono, now)
            proc.terminate()
            err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
            proc.wait()
            with self._lock:
                self._proc = None
                self._analyzer = AudioAnalyzer()
            if self._running:
                # Device busy (HyperHDR's own music effects hold it) or unplugged: retry calmly.
                log.warning("Audio capture ended%s, retrying in 5 s", f": {err}" if err else "")
                time.sleep(5)
        log.info("Audio capture stopped")


def to_mono(data: bytes) -> np.ndarray:
    samples = np.frombuffer(data, dtype="<i2").reshape(-1, CHANNELS).astype(np.float64)
    return samples.mean(axis=1) / 32768.0
