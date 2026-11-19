"""Turns a palette plus the current audio state into one frame for HyperHDR.

The frame is a small "fake screen". HyperHDR maps its edges onto the LED layout exactly as it
would a video frame, so nothing here needs to know where the LEDs are. Every pixel only gets a
position around the frame (0 = top centre, clockwise), and the palette colors live on that ring
as soft blobs:

* the blobs drift slowly on their own (lava lamp), so it is never static, even in silence;
* the louder the music, the brighter the frame and the wider the secondary colors spread;
* every detected beat adds a short brightness pulse that decays within a few hundred ms;
* a new palette crossfades in instead of jumping.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .palette import Palette

FRAME_WIDTH = 64
FRAME_HEIGHT = 36

ROTATION_PERIOD_S = 150.0   # one full slow turn of the whole pattern
DRIFT_AMPLITUDE = 0.07      # how far each blob wanders from its home, in turns
BASE_WIDTH = 0.06
WEIGHT_WIDTH = 0.22         # dominant colors get wider blobs
LOUDNESS_WIDTH = 0.10       # extra width for secondary colors when it gets loud

QUIET_BRIGHTNESS = 0.55
LOUDNESS_BRIGHTNESS = 0.30
BEAT_BRIGHTNESS = 0.30

CROSSFADE_S = 1.5


@dataclass
class AudioState:
    level: float = 0.0  # smoothed loudness, 0..1
    beat: float = 0.0   # beat pulse, 1 right after a beat, decaying to 0


def _ring_positions(width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    angle = np.arctan2(yy - height / 2 + 0.5, xx - width / 2 + 0.5)
    return (angle / (2 * math.pi) + 0.25) % 1.0


class Renderer:
    def __init__(self, intensity: float = 1.0, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self.intensity = intensity
        self._u = _ring_positions(width, height)
        self._palette: Palette | None = None
        self._previous: Palette | None = None
        self._changed_at = 0.0
        # Fixed per-slot phases and periods so the blobs do not move in lockstep.
        rng = np.random.default_rng(7)
        self._phases = rng.uniform(0, 2 * math.pi, 8)
        self._periods = rng.uniform(18.0, 42.0, 8)

    def set_palette(self, palette: Palette, now: float) -> None:
        if self._palette is not None:
            self._previous = self._palette
            self._changed_at = now
        self._palette = palette

    def render(self, now: float, audio: AudioState) -> np.ndarray:
        """Returns an (height, width, 3) uint8 RGB frame."""
        palette = self._palette
        if palette is None:
            raise ValueError("no palette set")

        frame = self._ring(palette, now, audio)
        fade = (now - self._changed_at) / CROSSFADE_S
        if self._previous is not None and fade < 1.0:
            # Smoothstep so the crossfade starts and ends gently.
            f = fade * fade * (3 - 2 * fade)
            frame = self._ring(self._previous, now, audio) * (1 - f) + frame * f
        elif self._previous is not None:
            self._previous = None

        level = min(1.0, max(0.0, audio.level))
        beat = min(1.0, max(0.0, audio.beat))
        brightness = QUIET_BRIGHTNESS + self.intensity * (LOUDNESS_BRIGHTNESS * level + BEAT_BRIGHTNESS * beat)
        frame = frame * min(1.0, brightness)
        return np.clip(frame * 255.0, 0, 255).astype(np.uint8)

    def _ring(self, palette: Palette, now: float, audio: AudioState) -> np.ndarray:
        n = len(palette.weights)
        if n == 1:
            return np.broadcast_to(palette.colors[0], self._u.shape + (3,)).copy()

        # Home positions: the dominant color centred at the top, the others after it clockwise,
        # each taking a slice proportional to its weight.
        w = palette.weights
        homes = np.concatenate([[0.0], np.cumsum((w[:-1] + w[1:]) / 2)])

        rotation = now / ROTATION_PERIOD_S
        slots = np.arange(n) % len(self._phases)
        drift = DRIFT_AMPLITUDE * np.sin(2 * math.pi * now / self._periods[slots] + self._phases[slots])
        centres = (homes + rotation + drift) % 1.0

        loud = min(1.0, max(0.0, audio.level)) * self.intensity
        widths = BASE_WIDTH + WEIGHT_WIDTH * w
        widths[1:] += LOUDNESS_WIDTH * loud

        # Circular distance from every pixel to every blob centre, Gaussian falloff.
        d = np.abs(self._u[..., None] - centres)
        d = np.minimum(d, 1.0 - d)
        k = (w + 0.15) * np.exp(-0.5 * (d / widths) ** 2) + 1e-6
        return (k @ palette.colors) / k.sum(-1, keepdims=True)
