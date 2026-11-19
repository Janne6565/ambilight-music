"""Run the audio analysis over a raw recording and print what the light would do.

Record a sample first (the service must not hold the device meanwhile):

    arecord -D <device> -f S16_LE -r 48000 -c 2 -t raw -d 12 /tmp/sample.raw
    python tools/analyze.py /tmp/sample.raw

Prints onsets per band, the tempo the kicks imply, and a timeline with one row per ~0.25 s:
P = pulse fired, K = kick detected, s = snare, the bar is the brightness the renderer would use at intensity 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ambilight_music.audio import CHANNELS, HOP_FRAMES, HOP_S, AudioAnalyzer, to_mono  # noqa: E402
from ambilight_music.renderer import brightness  # noqa: E402


def main() -> None:
    data = Path(sys.argv[1]).read_bytes()
    block_bytes = HOP_FRAMES * CHANNELS * 2
    analyzer = AudioAnalyzer()
    kicks, snares, pulses, rows = [], [], [], []
    row = {"k": False, "s": False, "p": False, "b": []}
    locked_hops = 0
    for i in range(len(data) // block_bytes):
        now = i * HOP_S
        before_k, before_s, before_p = analyzer.kick.last_onset, analyzer.snare.last_onset, analyzer.clock.fired_at
        state = analyzer.feed(to_mono(data[i * block_bytes:(i + 1) * block_bytes]), now)
        if analyzer.kick.last_onset != before_k:
            kicks.append(now)
            row["k"] = True
        if analyzer.snare.last_onset != before_s:
            snares.append(now)
            row["s"] = True
        if analyzer.clock.fired_at != before_p:
            pulses.append((now, analyzer.clock.locked))
            row["p"] = True
        locked_hops += analyzer.clock.locked
        row["b"].append(brightness(state, 1.0))
        if len(row["b"]) == 24:
            rows.append((now, row))
            row = {"k": False, "s": False, "p": False, "b": []}

    hops = len(data) // block_bytes
    duration = hops * HOP_S
    print(f"{duration:.1f} s, kicks {len(kicks)}, snares {len(snares)}, pulses {len(pulses)}, "
          f"tempo locked {100 * locked_hops / max(1, hops):.0f}% of the time")
    # How early the predicted pulses came compared to the kick they were meant for.
    early = [k - p for p, locked in pulses if locked for k in kicks if 0 <= k - p < 0.25]
    if early:
        print(f"locked pulses before their kick: median {1000 * float(np.median(early)):.0f} ms")
    if len(kicks) > 2:
        interval = float(np.median(np.diff(kicks)))
        print(f"median kick interval {interval:.3f} s = {60 / interval:.0f} BPM")
    for t, r in rows:
        peak = max(r["b"])
        low = min(r["b"])
        print(f"{t:5.2f} {'P' if r['p'] else ' '}{'K' if r['k'] else ' '}{'s' if r['s'] else ' '} "
              f"{'#' * round(low * 30):<30}|{'#' * round((peak - low) * 30)}")


if __name__ == "__main__":
    main()
