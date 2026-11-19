"""Follows the current album cover on the hub's `/audio` websocket.

The hub replays the last value of every key on connect, so after a (re)connect the current
cover arrives immediately. Messages look like
`{"type": "albumCover", "value": "<url>", "source": "...", "updated": ...}`.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import urllib.request
from typing import Callable

import websockets
from PIL import Image

from . import palette as palette_mod

log = logging.getLogger(__name__)

COVER_KEY = "albumCover"
RECONNECT_S = 5


class CoverFollower:
    def __init__(self, ws_url: str, on_palette: Callable[[palette_mod.Palette, str], None]):
        self.ws_url = ws_url
        self.on_palette = on_palette
        self._last_url: str | None = None

    def start(self) -> None:
        threading.Thread(target=lambda: asyncio.run(self._run()), name="cover", daemon=True).start()

    async def _run(self) -> None:
        while True:
            try:
                async with websockets.connect(self.ws_url, open_timeout=10, ping_interval=30) as ws:
                    log.info("Connected to %s", self.ws_url)
                    async for message in ws:
                        self._handle(message)
            except Exception as e:  # network errors, hub restarts: just reconnect
                log.warning("Cover websocket: %s, reconnecting in %d s", e, RECONNECT_S)
            await asyncio.sleep(RECONNECT_S)

    def _handle(self, message: str | bytes) -> None:
        try:
            data = json.loads(message)
        except ValueError:
            return
        if data.get("type") != COVER_KEY:
            return
        url = data.get("value") or ""
        if not url.startswith(("http://", "https://")) or url == self._last_url:
            # Empty or cleared cover (Shazam found nothing): keep the current palette.
            return
        threading.Thread(target=self._load, args=(url,), name="cover-load", daemon=True).start()

    def _load(self, url: str) -> None:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                image = Image.open(io.BytesIO(response.read()))
                image.load()
            palette = palette_mod.extract(image)
        except Exception as e:
            log.warning("Could not build palette from %s: %s", url, e)
            return
        self._last_url = url
        log.info("New palette %s for %s", palette.describe(), url)
        self.on_palette(palette, url)
