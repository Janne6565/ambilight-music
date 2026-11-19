"""Beats and dynamics from the capture card's audio input.

Reads raw PCM from `arecord` (no Python audio bindings needed, and it releases the device as
soon as the process ends) and analyses a 1024-sample window every 512 samples (~10.7 ms).

Loudness alone is useless for modern, heavily compressed music: it sits at its ceiling the
whole song. So the analysis looks for *onsets* instead, sudden rises of energy per band, each
judged against that band's own recent behaviour (spectral flux with an adaptive threshold):

* kick:  bass band, the main pulse of the light
* snare: mid band, a secondary accent
* hats:  high band, a continuous "shimmer" amount

The kick pulse goes through a `BeatClock`: light always arrives a little late (capture card,
buffers, HyperHDR smoothing, the LED controller), so once the kicks are steady the clock fires
each pulse *ahead* of the expected kick by a configurable lead, and the real kick only
corrects the phase. Breaks and irregular rhythms fall back to plain reaction.

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
from typing import Callable

import numpy as np

from .renderer import AudioState

log = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS = 2
WINDOW_FRAMES = 1024
HOP_FRAMES = 512
HOP_S = HOP_FRAMES / SAMPLE_RATE
# Kept for callers that feed whole windows (tests, tools).
BLOCK_FRAMES = WINDOW_FRAMES

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


KICK = Band("kick", 35, 150, sensitivity=1.4, min_flux=0.12, refractory_s=0.24, decay_s=0.24)
SNARE = Band("snare", 300, 3000, sensitivity=1.8, min_flux=0.10, refractory_s=0.18, decay_s=0.14)
HATS = Band("hats", 5000, 14000, sensitivity=1.0, min_flux=0.05, refractory_s=0.0, decay_s=0.0)

FLUX_HISTORY_S = 1.0
LEVEL_HISTORY_S = 10.0
LEVEL_RANGE_DB = 12.0      # +-6 dB around the 10 s average maps onto 0..1
LEVEL_SMOOTHING = 0.03
SHIMMER_SMOOTHING = 0.12

# Beat prediction
DEFAULT_LEAD_S = 0.10
MAX_LEAD_S = 0.40
LOCK_ONSETS = 5            # this many kicks in a row with steady spacing lock the tempo
LOCK_TOLERANCE = 0.05      # interval spread (std / median) to lock; real music plus ~11 ms jitter stays below
MIN_PERIOD_S = 0.27        # ~220 BPM
MAX_PERIOD_S = 1.5         # ~40 BPM
MATCH_WINDOW = 0.25        # a kick within this share of a period of the expected one confirms it
PHASE_CORRECTION = 0.5     # how far a confirming kick pulls the phase towards itself
MAX_MISSES = 2             # predicted pulses without a real kick before the clock gives up


class _BandTracker:
    def __init__(self, band: Band, freqs: np.ndarray, hop_s: float):
        self.band = band
        self.mask = (freqs >= band.low_hz) & (freqs <= band.high_hz)
        self.prev_energy: float | None = None
        self.history: deque[float] = deque(maxlen=max(8, round(FLUX_HISTORY_S / hop_s)))
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


class BeatClock:
    """Fires the kick pulse, ahead of time once the tempo is steady.

    `onset(t)` reports a detected kick, `update(now)` lets predicted pulses fire. Both return
    True when a pulse fired, so the caller can render right away.
    """

    def __init__(self, decay_s: float = KICK.decay_s, lead_s: float = DEFAULT_LEAD_S):
        self.decay_s = decay_s
        self.lead_s = lead_s
        self.onsets: deque[float] = deque(maxlen=LOCK_ONSETS + 3)
        self.period: float | None = None   # set while locked
        self.anchor = 0.0                  # time of the last (corrected) beat while locked
        self.misses = 0
        self.fired_at = -math.inf
        self._fired_for: float | None = None  # expected beat time the last early pulse was for

    @property
    def locked(self) -> bool:
        return self.period is not None

    def pulse(self, now: float) -> float:
        if now < self.fired_at:
            return 0.0
        return math.exp(-(now - self.fired_at) / self.decay_s)

    def onset(self, t: float) -> bool:
        self.onsets.append(t)
        if self.locked:
            expected = self._nearest_expected(t)
            if abs(t - expected) <= MATCH_WINDOW * self.period:
                self.anchor = expected + PHASE_CORRECTION * (t - expected)
                self.misses = 0
                self._relock()
                if self._fired_for == expected:
                    return False  # the early pulse already covered this kick
                return self._fire(t, expected)
            # Off-grid kick (fill, syncopation): react to it, but keep the clock.
            return self._fire(t, None)
        self._relock()
        return self._fire(t, None)

    def update(self, now: float) -> bool:
        if not self.locked:
            return False
        expected = self.anchor + self.period
        if now > expected + MATCH_WINDOW * self.period:
            # The kick never came: freewheel for a moment, then give up.
            self.misses += 1
            self.anchor = expected
            if self.misses >= MAX_MISSES:
                log.debug("Beat clock lost the tempo")
                self.period = None
                self.onsets.clear()
            return False
        if now >= expected - self.lead_s and self._fired_for != expected and self.misses < MAX_MISSES:
            return self._fire(now, expected)
        return False

    def reset(self) -> None:
        self.onsets.clear()
        self.period = None
        self.misses = 0

    def _fire(self, now: float, expected: float | None) -> bool:
        self.fired_at = now
        self._fired_for = expected
        return True

    def _nearest_expected(self, t: float) -> float:
        n = round((t - self.anchor) / self.period)
        return self.anchor + max(1, n) * self.period

    def _relock(self) -> None:
        if len(self.onsets) < LOCK_ONSETS:
            return
        recent = np.fromiter(self.onsets, float)[-LOCK_ONSETS:]
        intervals = np.diff(recent)
        median = float(np.median(intervals))
        spread = float(intervals.std()) / median
        in_range = MIN_PERIOD_S <= median <= MAX_PERIOD_S
        if self.locked and (not in_range or spread > 2 * LOCK_TOLERANCE):
            # The rhythm changed or was never really steady: back to plain reaction.
            log.debug("Beat clock released (spread %.2f)", spread)
            self.period = None
            return
        if in_range and spread <= LOCK_TOLERANCE:
            if not self.locked:
                log.debug("Beat clock locked at %.0f BPM", 60 / median)
                self.anchor = float(recent[-1])
                self.misses = 0
            self.period = median


class AudioAnalyzer:
    """Pure analysis, fed hop by hop. Kept separate from capture so it can be tested."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, lead_s: float = DEFAULT_LEAD_S):
        freqs = np.fft.rfftfreq(WINDOW_FRAMES, 1.0 / sample_rate)
        self.sample_rate = sample_rate
        self._window = np.hanning(WINDOW_FRAMES)
        self._buffer = np.zeros(WINDOW_FRAMES)
        self.kick = _BandTracker(KICK, freqs, HOP_S)
        self.snare = _BandTracker(SNARE, freqs, HOP_S)
        self.hats = _BandTracker(HATS, freqs, HOP_S)
        self.clock = BeatClock(lead_s=lead_s)
        self._db_history: deque[float] = deque(maxlen=round(LEVEL_HISTORY_S / HOP_S))
        self._level = 0.0
        self._shimmer = 0.0
        self.state = AudioState()

    def feed(self, chunk: np.ndarray, now: float) -> AudioState:
        """`chunk` is the newest mono samples (-1..1), typically one hop; the analysis window
        slides over the last WINDOW_FRAMES of them."""
        n = min(len(chunk), WINDOW_FRAMES)
        self._buffer = np.concatenate([self._buffer[n:], chunk[-n:]])
        mono = self._buffer

        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms < SILENCE_RMS:
            for tracker in (self.kick, self.snare, self.hats):
                tracker.reset()
            self.clock.reset()
            self._level += (0.0 - self._level) * LEVEL_SMOOTHING
            self._shimmer = 0.0
            self.state = self.at(now)
            return self.state

        power = np.abs(np.fft.rfft(mono * self._window)) ** 2
        kick, _ = self.kick.feed(power, now)
        if kick:
            self.clock.onset(now)
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
        """State at `now`: lets predicted beats fire and decays the pulses smoothly between hops."""
        self.clock.update(now)
        return AudioState(level=self._level, beat=self.clock.pulse(now), snare=self.snare.pulse(now),
                          shimmer=self._shimmer)


class AudioCapture:
    """Runs `arecord` in a thread while started; `state()` is safe to call from anywhere."""

    def __init__(self, device: str, lead_s: float = DEFAULT_LEAD_S, on_kick: Callable[[], None] | None = None):
        self.device = device
        self.lead_s = lead_s
        self.on_kick = on_kick
        self._analyzer = AudioAnalyzer(lead_s=lead_s)
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

    def set_lead(self, lead_s: float) -> None:
        with self._lock:
            self.lead_s = lead_s
            self._analyzer.clock.lead_s = lead_s

    def state(self, now: float) -> AudioState:
        with self._lock:
            return self._analyzer.at(now)

    def _loop(self) -> None:
        hop_bytes = HOP_FRAMES * CHANNELS * 2
        while self._running:
            # Small ALSA buffer: every millisecond here is a millisecond the light is late.
            cmd = ["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                   "-c", str(CHANNELS), "-t", "raw", "--buffer-time=20000", "--period-time=5000"]
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            except OSError as e:
                log.error("Cannot start arecord: %s", e)
                time.sleep(5)
                continue
            with self._lock:
                self._proc = proc
            log.info("Audio capture started on %s", self.device)
            while self._running:
                data = _read_exactly(proc.stdout, hop_bytes)
                if data is None:
                    break
                chunk = to_mono(data)
                now = time.monotonic()
                with self._lock:
                    before = self._analyzer.clock.fired_at
                    self._analyzer.feed(chunk, now)
                    fired = self._analyzer.clock.fired_at != before
                if fired and self.on_kick is not None:
                    self.on_kick()
            proc.terminate()
            err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
            proc.wait()
            with self._lock:
                self._proc = None
                self._analyzer = AudioAnalyzer(lead_s=self.lead_s)
            if self._running:
                # Device busy (HyperHDR's own music effects hold it) or unplugged: retry calmly.
                log.warning("Audio capture ended%s, retrying in 5 s", f": {err}" if err else "")
                time.sleep(5)
        log.info("Audio capture stopped")


def _read_exactly(stream, size: int) -> bytes | None:
    parts, remaining = [], size
    while remaining:
        part = stream.read(remaining)
        if not part:
            return None
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def to_mono(data: bytes) -> np.ndarray:
    samples = np.frombuffer(data, dtype="<i2").reshape(-1, CHANNELS).astype(np.float64)
    return samples.mean(axis=1) / 32768.0
