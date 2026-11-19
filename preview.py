"""Look at the animation on the real strip without audio or the service.

Plays a cover's palette for a while with a synthetic beat (a kick at the given BPM and a slowly
swelling loudness), then clears its priority so the strip returns to whatever was showing.

    python preview.py --host <hyperhdr-host> --cover <image-url> [--seconds 60] [--bpm 120]
"""
from __future__ import annotations

import argparse
import io
import logging
import math
import time
import urllib.request

from PIL import Image

from ambilight_music import palette as palette_mod
from ambilight_music.hyperhdr import HyperHdr
from ambilight_music.renderer import AudioState, Renderer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=19444)
    parser.add_argument("--cover", required=True, help="image URL or local path")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--bpm", type=float, default=120)
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=50)
    parser.add_argument("--priority", type=int, default=99)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cover.startswith(("http://", "https://")):
        image = Image.open(io.BytesIO(urllib.request.urlopen(args.cover, timeout=10).read()))
    else:
        image = Image.open(args.cover)
    palette = palette_mod.extract(image)
    logging.info("Palette: %s", palette.describe())

    renderer = Renderer(intensity=args.intensity)
    hyper = HyperHdr(args.host, args.port, args.priority, "AmbilightPreview")
    start = time.monotonic()
    renderer.set_palette(palette, 0.0)
    beat_interval = 60.0 / args.bpm
    try:
        while (now := time.monotonic() - start) < args.seconds:
            since_beat = now % beat_interval
            audio = AudioState(
                level=0.5 + 0.5 * math.sin(2 * math.pi * now / 20.0),  # swells every 20 s
                beat=math.exp(-since_beat / 0.15),
            )
            hyper.show(renderer.render(now, audio), duration_ms=1000)
            time.sleep(max(0.0, 1.0 / args.fps - ((time.monotonic() - start) - now)))
    finally:
        hyper.clear()
        hyper.close()


if __name__ == "__main__":
    main()
