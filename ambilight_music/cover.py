"""Follows the current album cover on the hub's `/audio` websocket.

The hub publishes two cover keys, each as
`{"type": "<key>", "value": "<url>", "source": "...", "updated": ...}`:

* `localAlbumCover`: the artwork the playing device actually sent (the AV receiver's own copy
  of the AirPlay/streaming art) or the cover of a Shazam match. This is the real one.
* `albumCover`: the hub's own guess from an iTunes search by title/artist. It can be plainly
  wrong (The Cure's "Friday I'm In Love" came back as Lady Gaga's single "The Cure").

So `localAlbumCover` wins whenever it is set, and `albumCover` is only the fallback. Both keys
usually arrive within a few milliseconds of each other on a song change, so the choice waits a
moment for the pair before loading anything. The hub replays the last value of every key on
connect, so after a (re)connect the current cover arrives immediately.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import time
import urllib.request
from typing import Callable

import websockets
from PIL import Image

from . import palette as palette_mod

log = logging.getLogger(__name__)

PREFERRED_KEYS = ("localAlbumCover", "albumCover")
SETTLE_S = 0.4          # wait this long after a cover message for its partner
RECONNECT_S = 5
DOWNLOAD_ATTEMPTS = 3
USER_AGENT = "ambilight-music/1 (+https://github.com/Janne6565/ambilight-music)"


class CoverFollower:
    def __init__(self, ws_url: str, on_palette: Callable[[palette_mod.Palette, str], None]):
        self.ws_url = ws_url
        self.on_palette = on_palette
        self._latest: dict[str, str] = {}
        self._last_url: str | None = None
        self._pending: asyncio.TimerHandle | None = None

    def start(self) -> None:
        threading.Thread(target=lambda: asyncio.run(self._run()), name="cover", daemon=True).start()

    def choose(self) -> str | None:
        """The cover to show for what is currently known: the first usable preferred key."""
        for key in PREFERRED_KEYS:
            url = self._latest.get(key, "")
            if url.startswith(("http://", "https://")):
                return url
        return None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                async with websockets.connect(self.ws_url, open_timeout=10, ping_interval=30) as ws:
                    log.info("Connected to %s", self.ws_url)
                    async for message in ws:
                        if self._handle(message):
                            if self._pending is not None:
                                self._pending.cancel()
                            self._pending = loop.call_later(SETTLE_S, self._settled)
            except Exception as e:  # network errors, hub restarts: just reconnect
                log.warning("Cover websocket: %s, reconnecting in %d s", e, RECONNECT_S)
            await asyncio.sleep(RECONNECT_S)

    def _handle(self, message: str | bytes) -> bool:
        """Records a cover message; True when it was one."""
        try:
            data = json.loads(message)
        except ValueError:
            return False
        key = data.get("type")
        if key not in PREFERRED_KEYS:
            return False
        self._latest[key] = data.get("value") or ""
        return True

    def _settled(self) -> None:
        url = self.choose()
        if url is None or url == self._last_url:
            # Nothing usable (Shazam found nothing, cover cleared): keep the current palette.
            return
        self._last_url = url
        threading.Thread(target=self._load, args=(url,), name="cover-load", daemon=True).start()

    def _load(self, url: str) -> None:
        try:
            image = download(url)
            palette = palette_mod.extract(image)
        except Exception as e:
            log.warning("Could not build palette from %s: %s", url, e)
            if self._last_url == url:
                self._last_url = None  # let the next message for the same cover try again
            return
        if self._last_url != url:
            # A newer cover was chosen while this one downloaded (a slow internet fetch can
            # finish after a fast local one): applying it now would show the previous song.
            log.info("Dropping stale palette for %s", url)
            return
        log.info("New palette %s for %s", palette.describe(), url)
        self.on_palette(palette, url)


def download(url: str) -> Image.Image:
    """Fetches an image; retries because Apple's artwork CDN now and then resets connections."""
    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=10) as response:
                image = Image.open(io.BytesIO(response.read()))
                image.load()
                return image
        except Exception as e:
            last_error = e
            time.sleep(0.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]
