"""Dominant colors of an album cover.

k-means on a downscaled copy, then drop clusters that are background, borders or text
(too dark, too grey) and push the rest a little, because LEDs wash colors out.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np
from PIL import Image

SAMPLE_SIZE = 64
CLUSTERS = 5
ITERATIONS = 20

MIN_BRIGHTNESS = 0.15
MIN_SATURATION = 0.20
MIN_WEIGHT = 0.05  # smaller clusters are mostly blur at edges between two colors

SATURATION_GAIN = 1.35
SATURATION_OFFSET = 0.10
BRIGHTNESS_GAIN = 1.20
MIN_OUTPUT_BRIGHTNESS = 0.55


@dataclass(frozen=True)
class Palette:
    colors: np.ndarray   # (k, 3) floats in 0..1, dominant first
    weights: np.ndarray  # (k,), sums to 1

    def describe(self) -> str:
        return ", ".join(
            "#%02x%02x%02x %d%%" % (*(int(round(v * 255)) for v in c), round(w * 100))
            for c, w in zip(self.colors, self.weights)
        )


FALLBACK = Palette(colors=np.array([[1.0, 0.62, 0.32]]), weights=np.array([1.0]))
"""Warm white-ish amber, used before any cover has been seen."""


def extract(image: Image.Image) -> Palette:
    px = np.asarray(image.convert("RGB").resize((SAMPLE_SIZE, SAMPLE_SIZE)), dtype=np.float64).reshape(-1, 3)

    # Deterministic init at luminance quantiles: the same cover always gives the same palette.
    lum = px @ np.array([0.299, 0.587, 0.114])
    centroids = px[np.argsort(lum)[np.linspace(0, len(px) - 1, CLUSTERS).astype(int)]].copy()
    for _ in range(ITERATIONS):
        labels = np.argmin(((px[:, None, :] - centroids[None, :, :]) ** 2).sum(-1), axis=1)
        for k in range(CLUSTERS):
            members = px[labels == k]
            if len(members):
                centroids[k] = members.mean(0)
    sizes = np.bincount(labels, minlength=CLUSTERS) / len(px)

    candidates = []
    for centroid, size in zip(centroids, sizes):
        if size == 0:
            continue
        h, s, v = colorsys.rgb_to_hsv(*(centroid / 255.0))
        candidates.append((size, h, s, v))

    kept = [c for c in candidates if c[0] >= MIN_WEIGHT and c[3] > MIN_BRIGHTNESS and c[2] > MIN_SATURATION]
    if not kept:
        # Black-and-white covers: take the most colorful thing there is instead of going dark.
        kept = [max(candidates, key=lambda c: c[2] * c[3])]
    kept.sort(key=lambda c: -c[0])

    colors = np.array([_boost(h, s, v) for _, h, s, v in kept])
    weights = np.array([c[0] for c in kept])
    return Palette(colors=colors, weights=weights / weights.sum())


def _boost(h: float, s: float, v: float) -> tuple[float, float, float]:
    s = min(1.0, s * SATURATION_GAIN + SATURATION_OFFSET)
    v = max(MIN_OUTPUT_BRIGHTNESS, min(1.0, v * BRIGHTNESS_GAIN))
    return colorsys.hsv_to_rgb(h, s, v)
