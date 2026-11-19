import math

import numpy as np
from PIL import Image, ImageDraw

from ambilight_music import palette
from ambilight_music.audio import BLOCK_FRAMES, AudioAnalyzer
from ambilight_music.hyperhdr import MAX_ORIGIN_LENGTH, HyperHdr
from ambilight_music.main import ORIGIN
from ambilight_music.renderer import FRAME_HEIGHT, FRAME_WIDTH, AudioState, Renderer


def _cover() -> Image.Image:
    img = Image.new("RGB", (300, 300), "black")
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, 270, 180], fill=(220, 40, 60))   # red, biggest
    d.rectangle([30, 180, 270, 270], fill=(30, 90, 220))  # blue
    d.rectangle([100, 100, 200, 115], fill="white")       # title text
    return img


def test_palette_keeps_colors_drops_black_and_white():
    p = palette.extract(_cover())
    assert len(p.colors) == 2
    red, blue = p.colors
    assert red[0] > red[2] and blue[2] > blue[0]
    assert p.weights[0] > p.weights[1]
    assert math.isclose(p.weights.sum(), 1.0)


def test_greyscale_cover_still_gives_a_color():
    assert len(palette.extract(Image.new("RGB", (50, 50), (60, 60, 60))).colors) == 1


def test_renderer_frame_shape_and_beat_brightens():
    r = Renderer()
    r.set_palette(palette.extract(_cover()), 0.0)
    quiet = r.render(10.0, AudioState(level=0.0, beat=0.0))
    beat = r.render(10.0, AudioState(level=1.0, beat=1.0))
    assert quiet.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3) and quiet.dtype == np.uint8
    assert beat.mean() > quiet.mean() * 1.3


def test_renderer_crossfades_to_new_palette():
    r = Renderer()
    red = palette.Palette(np.array([[1.0, 0, 0]]), np.array([1.0]))
    blue = palette.Palette(np.array([[0, 0, 1.0]]), np.array([1.0]))
    r.set_palette(red, 0.0)
    r.set_palette(blue, 5.0)
    mid = r.render(5.75, AudioState()).mean(axis=(0, 1))
    end = r.render(7.0, AudioState()).mean(axis=(0, 1))
    assert mid[0] > 0 and mid[2] > 0          # both colors during the fade
    assert end[0] == 0 and end[2] > 0         # only the new one afterwards


def test_analyzer_detects_kicks_and_stays_quiet_in_silence():
    a = AudioAnalyzer()
    t = np.arange(BLOCK_FRAMES) / 48000
    kick = 0.6 * np.sin(2 * np.pi * 60 * t)
    hum = 0.05 * np.sin(2 * np.pi * 1000 * t)
    beats = 0
    now = 0.0
    for i in range(200):  # ~4.3 s, a kick every 24 blocks (~0.5 s, 120 BPM)
        block = hum + (kick if i % 24 == 0 else 0)
        before = a.clock.fired_at
        a.feed(block, now)
        if a.clock.fired_at != before:
            beats += 1
        now += BLOCK_FRAMES / 48000
    assert 6 <= beats <= 9
    assert a.state.level > 0.1

    silent = AudioAnalyzer()
    for i in range(50):
        state = silent.feed(np.zeros(BLOCK_FRAMES), i * 0.02)
    assert state.level == 0.0 and state.beat == 0.0


def test_origin_fits_hyperhdr_schema():
    assert 4 <= len(ORIGIN) <= MAX_ORIGIN_LENGTH
    HyperHdr("127.0.0.1", 19444, 100, ORIGIN)  # constructor validates too


def test_compressed_music_still_pulses():
    """Loudness pinned at the ceiling the whole time (modern mastering): kicks must still show."""
    from ambilight_music.renderer import brightness

    rng = np.random.default_rng(1)
    a = AudioAnalyzer()
    t = np.arange(BLOCK_FRAMES) / 48000
    kick = 0.5 * np.sin(2 * np.pi * 55 * t) * np.exp(-t / 0.01)
    lights, kicks, now = [], 0, 0.0
    for i in range(400):  # ~8.5 s at ~120 BPM
        bed = 0.25 * rng.standard_normal(BLOCK_FRAMES)  # dense, loud, always there
        before = a.kick.last_onset
        state = a.feed(bed + (kick if i % 23 == 0 else 0), now)
        kicks += a.kick.last_onset != before
        lights.append(brightness(state, 1.0))
        now += BLOCK_FRAMES / 48000
    assert 14 <= kicks <= 19
    assert min(lights[100:]) < 0.75 and max(lights[100:]) > 0.95


def test_full_intensity_does_not_pin_brightness():
    from ambilight_music.renderer import brightness

    assert 0.59 <= brightness(AudioState(level=0.0, beat=0.0), 2.0) and brightness(AudioState(level=1.0, beat=0.0), 2.0) <= 0.71
    assert brightness(AudioState(level=1.0, beat=0.6), 2.0) > 0.99   # snappier above 1
    assert brightness(AudioState(level=0.5, beat=0.0), 0.0) == brightness(AudioState(level=0.5, beat=1.0), 0.0)


def test_pulse_ramps_up_and_scales_with_kick_strength():
    from ambilight_music.audio import ATTACK_S, MIN_PULSE, BeatClock

    weak, strong = BeatClock(), BeatClock()
    weak.onset(1.0, strength=1.05)
    strong.onset(1.0, strength=3.0)
    assert weak.pulse(1.0) == 0.0                      # no instant jump
    assert 0 < strong.pulse(1.0 + ATTACK_S / 2) < 1.0  # still rising
    assert abs(strong.pulse(1.0 + ATTACK_S) - 1.0) < 1e-6  # strong kick: full pulse
    assert abs(weak.pulse(1.0 + ATTACK_S) - MIN_PULSE) < 0.05  # weak kick: small pulse


def test_overlapping_pulse_never_dips():
    from ambilight_music.audio import ATTACK_S, BeatClock

    c = BeatClock()
    c.onset(1.0, strength=3.0)
    level = c.pulse(1.1)
    c.onset(1.1, strength=1.2)  # a second, weaker kick while the first still glows
    assert c.pulse(1.1) >= level - 1e-9
    assert c.pulse(1.1 + ATTACK_S / 2) >= level - 1e-9
