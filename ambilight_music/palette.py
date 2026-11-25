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

    kept += _accents(px, kept)

    colors = np.array([_boost(h, s, v) for _, h, s, v in kept])
    weights = np.array([c[0] for c in kept])
    return Palette(colors=colors, weights=weights / weights.sum())


ACCENT_MIN_SATURATION = 0.45
ACCENT_MIN_BRIGHTNESS = 0.25
ACCENT_MIN_SHARE = 0.015   # of all pixels; smaller specks are noise
ACCENT_HUE_GAP = 0.10      # an accent must be this far (in turns, 36 deg) from every main color
ACCENT_HUE_WINDOW = 0.06   # pixels within this hue distance of the peak belong to the accent
ACCENT_MIN_WEIGHT = 0.12   # guaranteed share on the strip, or a small accent vanishes again
MAX_ACCENTS = 2


def _hue_distance(a, b):
    d = np.abs(a - b) % 1.0
    return np.minimum(d, 1.0 - d)


def _accents(px: np.ndarray, kept: list) -> list:
    """Small but vivid colors that k-means swallows.

    k-means splits a cover by area, so a red tree that covers 3 % of a dark blue cover ends up
    averaged into a blue cluster and never shows. Look for strongly saturated hues that are
    clearly different from every main color and add them with a guaranteed minimum share.
    """
    rgb = px / 255.0
    mx, mn = rgb.max(1), rgb.min(1)
    delta = mx - mn
    sat = np.where(mx > 0, delta / np.maximum(mx, 1e-9), 0.0)
    r, g, b = rgb.T
    safe = np.maximum(delta, 1e-9)
    hue = np.select(
        [mx == r, mx == g],
        [((g - b) / safe) % 6, (b - r) / safe + 2],
        (r - g) / safe + 4,
    ) / 6.0

    vivid = (sat > ACCENT_MIN_SATURATION) & (mx > ACCENT_MIN_BRIGHTNESS)
    main_hues = np.array([h for _, h, s, _ in kept if s > MIN_SATURATION])
    if len(main_hues):
        far = np.all(_hue_distance(hue[:, None], main_hues[None, :]) > ACCENT_HUE_GAP, axis=1)
        vivid &= far

    accents = []
    for _ in range(MAX_ACCENTS):
        if vivid.sum() < ACCENT_MIN_SHARE * len(px):
            break
        bins = np.bincount((hue[vivid] * 24).astype(int) % 24, minlength=24)
        peak = (np.argmax(bins) + 0.5) / 24
        members = vivid & (_hue_distance(hue, peak) <= ACCENT_HUE_WINDOW)
        share = members.mean()
        if share < ACCENT_MIN_SHARE:
            break
        h, s, v = colorsys.rgb_to_hsv(*rgb[members].mean(0))
        accents.append((max(ACCENT_MIN_WEIGHT, share), h, s, v))
        # Neither this accent nor anything near it can be picked again.
        vivid &= _hue_distance(hue, peak) > ACCENT_HUE_GAP
    return accents


NEUTRAL_SATURATION = 0.20  # below this the offset fades out, so grey stays grey


def _boost(h: float, s: float, v: float) -> tuple[float, float, float]:
    # A grey has no real hue (colorsys reports 0 = red), so a flat saturation offset tinted
    # black-and-white covers pink. Fade the offset in with the color that is actually there.
    s = min(1.0, s * SATURATION_GAIN + SATURATION_OFFSET * min(1.0, s / NEUTRAL_SATURATION))
    v = max(MIN_OUTPUT_BRIGHTNESS, min(1.0, v * BRIGHTNESS_GAIN))
    return colorsys.hsv_to_rgb(h, s, v)
