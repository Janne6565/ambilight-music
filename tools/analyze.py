"""Run the audio analysis over a raw recording and print what the light would do.

Record a sample first (the service must not hold the device meanwhile):

    arecord -D <device> -f S16_LE -r 48000 -c 2 -t raw -d 12 /tmp/sample.raw
    python tools/analyze.py /tmp/sample.raw

Prints onsets per band, the tempo the kicks imply, and a timeline with one row per ~0.25 s:
K = kick, s = snare, the bar is the brightness the renderer would use at intensity 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ambilight_music.audio import BLOCK_FRAMES, BLOCK_S, CHANNELS, AudioAnalyzer, to_mono  # noqa: E402
from ambilight_music.renderer import brightness  # noqa: E402


def main() -> None:
    data = Path(sys.argv[1]).read_bytes()
    block_bytes = BLOCK_FRAMES * CHANNELS * 2
    analyzer = AudioAnalyzer()
    kicks, snares, rows = [], [], []
    row = {"k": False, "s": False, "b": []}
    for i in range(len(data) // block_bytes):
        now = i * BLOCK_S
        before_k, before_s = analyzer.kick.last_onset, analyzer.snare.last_onset
        state = analyzer.feed(to_mono(data[i * block_bytes:(i + 1) * block_bytes]), now)
        if analyzer.kick.last_onset != before_k:
            kicks.append(now)
            row["k"] = True
        if analyzer.snare.last_onset != before_s:
            snares.append(now)
            row["s"] = True
        row["b"].append(brightness(state, 1.0))
        if len(row["b"]) == 12:
            rows.append((now, row))
            row = {"k": False, "s": False, "b": []}

    duration = len(data) / block_bytes * BLOCK_S
    print(f"{duration:.1f} s, kicks {len(kicks)}, snares {len(snares)}, level now {analyzer.state.level:.2f}")
    if len(kicks) > 2:
        interval = float(np.median(np.diff(kicks)))
        print(f"median kick interval {interval:.3f} s = {60 / interval:.0f} BPM")
    for t, r in rows:
        peak = max(r["b"])
        low = min(r["b"])
        print(f"{t:5.2f} {'K' if r['k'] else ' '}{'s' if r['s'] else ' '} "
              f"{'#' * round(low * 30):<30}|{'#' * round((peak - low) * 30)}")


if __name__ == "__main__":
    main()
