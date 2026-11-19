"""Turns a palette plus the current audio state into one frame for HyperHDR.

The frame is a small "fake screen". HyperHDR maps its edges onto the LED layout exactly as it
would a video frame, so nothing here needs to know where the LEDs are. Every pixel only gets a
position around the frame (0 = top centre, clockwise), and the palette colors live on that ring
as soft blobs.

What the music does to it, scaled by `intensity` (0 = no reaction, 1 = default, 2 = maximum):

* **kick**: the main pulse. Between kicks the light sits below full brightness (deeper with
  more intensity) and every kick lifts it to full; it also gives the whole pattern a shove
  forward, so the motion itself follows the beat.
* **snare**: briefly widens the secondary colors, so they flash further into the frame.
* **shimmer** (hi-hats): a fine ripple of brightness along the ring.
* **level** (this moment vs. the last ~10 s): a quiet verse sits a little darker than the chorus.

Without music the blobs still drift slowly (lava lamp), and a new palette crossfades in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .palette import Palette

FRAME_WIDTH = 64
FRAME_HEIGHT = 36

BASE_SPEED = 1 / 150.0      # turns per second without music
KICK_PUSH = 0.30            # extra turns per second at a full kick pulse (x intensity)
DRIFT_AMPLITUDE = 0.07      # how far each blob wanders from its home, in turns
BASE_WIDTH = 0.06
WEIGHT_WIDTH = 0.22         # dominant colors get wider blobs
SNARE_WIDTH = 0.12          # extra width for secondary colors on a snare (x intensity)

PULSE_DEPTH = 0.55          # how far below full brightness the light sits between kicks at intensity 1
LEVEL_DEPTH = 0.15          # extra dimming in quiet passages
STATIC_BRIGHTNESS = 0.9     # intensity 0
SHIMMER_DEPTH = 0.12
SHIMMER_SPATIAL = 23.0      # ripples around the ring

CROSSFADE_S = 1.5


@dataclass
class AudioState:
    level: float = 0.5    # loudness relative to the last ~10 s, 0..1 (0.5 = average)
    beat: float = 0.0     # kick pulse, 1 right after a kick, decaying to 0
    snare: float = 0.0    # snare pulse, same shape
    shimmer: float = 0.0  # hi-hat activity, 0..1


def brightness(audio: AudioState, intensity: float) -> float:
    """Overall brightness 0..1 for this audio state."""
    amount = min(1.0, max(0.0, intensity))
    if amount == 0.0:
        return STATIC_BRIGHTNESS
    # Above 1 the pulse is not deeper (it would just get dark) but snappier: a smaller kick
    # already reaches full brightness.
    gain = max(1.0, intensity)
    floor = 1.0 - PULSE_DEPTH * amount - LEVEL_DEPTH * amount * (1.0 - _clip(audio.level))
    floor = STATIC_BRIGHTNESS * (1 - amount) + floor * amount
    return floor + (1.0 - floor) * min(1.0, _clip(audio.beat) * gain)


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


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
        self._phase = 0.0
        self._last_now: float | None = None
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

        dt = 0.0 if self._last_now is None else min(0.2, max(0.0, now - self._last_now))
        self._last_now = now
        amount = max(0.0, self.intensity)
        self._phase = (self._phase + dt * (BASE_SPEED + KICK_PUSH * amount * _clip(audio.beat))) % 1.0

        frame = self._ring(palette, now, audio)
        fade = (now - self._changed_at) / CROSSFADE_S
        if self._previous is not None and fade < 1.0:
            f = fade * fade * (3 - 2 * fade)  # smoothstep
            frame = self._ring(self._previous, now, audio) * (1 - f) + frame * f
        elif self._previous is not None:
            self._previous = None

        light = brightness(audio, self.intensity)
        if audio.shimmer > 0 and amount > 0:
            ripple = np.sin(2 * math.pi * (self._u * SHIMMER_SPATIAL - now * 3.0))
            light = light * (1 - SHIMMER_DEPTH * min(1.0, amount) * _clip(audio.shimmer) * (0.5 + 0.5 * ripple))
        frame = frame * (light[..., None] if isinstance(light, np.ndarray) else light)
        return np.clip(frame * 255.0, 0, 255).astype(np.uint8)

    def _ring(self, palette: Palette, now: float, audio: AudioState) -> np.ndarray:
        n = len(palette.weights)
        if n == 1:
            return np.broadcast_to(palette.colors[0], self._u.shape + (3,)).copy()

        # Home positions: the dominant color centred at the top, the others after it clockwise,
        # each taking a slice proportional to its weight.
        w = palette.weights
        homes = np.concatenate([[0.0], np.cumsum((w[:-1] + w[1:]) / 2)])

        slots = np.arange(n) % len(self._phases)
        drift = DRIFT_AMPLITUDE * np.sin(2 * math.pi * now / self._periods[slots] + self._phases[slots])
        centres = (homes + self._phase + drift) % 1.0

        widths = BASE_WIDTH + WEIGHT_WIDTH * w
        widths[1:] += SNARE_WIDTH * max(0.0, self.intensity) * _clip(audio.snare)

        # Circular distance from every pixel to every blob centre, Gaussian falloff.
        d = np.abs(self._u[..., None] - centres)
        d = np.minimum(d, 1.0 - d)
        k = (w + 0.15) * np.exp(-0.5 * (d / widths) ** 2) + 1e-6
        return (k @ palette.colors) / k.sum(-1, keepdims=True)
