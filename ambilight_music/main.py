"""ambilight-music: album cover colors that move with the music, rendered into HyperHDR.

Configuration comes from environment variables (see config.example.env). Home Assistant (or
anything else) switches it with `/on`, `/off`, `/status` and `/intensity?value=0..2` on the
control port. The on/off state survives restarts.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import palette as palette_mod
from .audio import AudioCapture
from .cover import CoverFollower
from .hyperhdr import HyperHdr
from .renderer import Renderer

log = logging.getLogger("ambilight_music")

ORIGIN = "AmbilightMusic"
FRAME_DURATION_MS = 1000  # if this process dies, HyperHDR falls back to the grabber within 1 s


class Config:
    def __init__(self, env: dict[str, str]):
        def need(name: str) -> str:
            value = env.get(name, "").strip()
            if not value:
                raise SystemExit(f"{name} must be set")
            return value

        self.hub_ws_url = need("HUB_WS_URL")
        self.control_token = need("CONTROL_TOKEN")
        self.audio_device = env.get("AUDIO_DEVICE", "default")
        self.hyperhdr_host = env.get("HYPERHDR_HOST", "127.0.0.1")
        self.hyperhdr_port = int(env.get("HYPERHDR_PORT", "19444"))
        self.priority = int(env.get("HYPERHDR_PRIORITY", "100"))
        self.control_host = env.get("CONTROL_HOST", "0.0.0.0")
        self.control_port = int(env.get("CONTROL_PORT", "8096"))
        self.fps = float(env.get("FPS", "50"))
        self.intensity = float(env.get("INTENSITY", "1.0"))
        self.state_file = Path(env.get("STATE_FILE", str(Path.home() / ".local/state/ambilight-music/state.json")))


class MusicMode:
    def __init__(self, config: Config):
        self.config = config
        self.renderer = Renderer(intensity=config.intensity)
        self.renderer.set_palette(palette_mod.FALLBACK, time.monotonic())
        self.hyper = HyperHdr(config.hyperhdr_host, config.hyperhdr_port, config.priority, ORIGIN)
        self.audio = AudioCapture(config.audio_device)
        self.covers = CoverFollower(config.hub_ws_url, self._on_palette)
        self._enabled = threading.Event()
        self._stop = threading.Event()
        self._palette_lock = threading.Lock()

    # --- state -----------------------------------------------------------------------------

    def load_state(self) -> None:
        try:
            state = json.loads(self.config.state_file.read_text())
        except (OSError, ValueError):
            return
        self.renderer.intensity = float(state.get("intensity", self.renderer.intensity))
        if state.get("enabled"):
            self.set_enabled(True, persist=False)

    def _save_state(self) -> None:
        try:
            self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.config.state_file.write_text(json.dumps(
                {"enabled": self._enabled.is_set(), "intensity": self.renderer.intensity}))
        except OSError as e:
            log.warning("Could not save state: %s", e)

    @property
    def enabled(self) -> bool:
        return self._enabled.is_set()

    def set_enabled(self, value: bool, persist: bool = True) -> None:
        if value == self._enabled.is_set():
            return
        if value:
            self.audio.start()
            self._enabled.set()
        else:
            self._enabled.clear()
            self.audio.stop()
            try:
                self.hyper.clear()
            except ConnectionError as e:
                log.warning("%s", e)
        log.info("Music mode %s", "on" if value else "off")
        if persist:
            self._save_state()

    def set_intensity(self, value: float) -> None:
        self.renderer.intensity = max(0.0, min(2.0, value))
        self._save_state()

    def _on_palette(self, palette: palette_mod.Palette, _url: str) -> None:
        with self._palette_lock:
            self.renderer.set_palette(palette, time.monotonic())

    # --- loops -----------------------------------------------------------------------------

    def run(self) -> None:
        self.covers.start()
        interval = 1.0 / self.config.fps
        failing = False
        while not self._stop.is_set():
            if not self._enabled.wait(timeout=0.5):
                continue
            started = time.monotonic()
            try:
                with self._palette_lock:
                    frame = self.renderer.render(started, self.audio.state(started))
                self.hyper.show(frame, FRAME_DURATION_MS)
                if failing:
                    log.info("HyperHDR reachable again")
                failing = False
            except ConnectionError as e:
                if not failing:
                    log.warning("%s", e)
                failing = True
                time.sleep(2)
            time.sleep(max(0.0, interval - (time.monotonic() - started)))

    def shutdown(self) -> None:
        self._stop.set()
        self.audio.stop()
        if self._enabled.is_set():
            try:
                self.hyper.clear()
            except ConnectionError:
                pass
        self.hyper.close()


def make_handler(mode: MusicMode, token: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._handle()

        def do_POST(self) -> None:
            self._handle()

        def _handle(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            supplied = query.get("token", [""])[0]
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                supplied = auth[len("Bearer "):]
            if not hmac.compare_digest(supplied.encode(), token.encode()):
                return self._reply(401, "unauthorized")

            if url.path == "/on":
                mode.set_enabled(True)
            elif url.path == "/off":
                mode.set_enabled(False)
            elif url.path == "/intensity":
                # 0 = colors only drift, no reaction to the music; 1 = default; 2 = maximum.
                # Without ?value= it just reports the current setting.
                if "value" in query:
                    try:
                        mode.set_intensity(float(query["value"][0]))
                    except ValueError:
                        return self._reply(400, "value must be a number")
                return self._reply(200, f"{mode.renderer.intensity:g}")
            elif url.path != "/status":
                return self._reply(404, "not found")
            self._reply(200, "1" if mode.enabled else "0")

        def _reply(self, status: int, body: str) -> None:
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args) -> None:  # keep tokens out of the journal
            log.debug("%s %s", self.command, urlparse(self.path).path)

    return Handler


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")
    config = Config(dict(os.environ))
    mode = MusicMode(config)
    server = ThreadingHTTPServer((config.control_host, config.control_port), make_handler(mode, config.control_token))
    threading.Thread(target=server.serve_forever, name="control", daemon=True).start()
    log.info("Control API on %s:%d", config.control_host, config.control_port)

    def stop(*_):
        mode.shutdown()
        server.shutdown()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    mode.load_state()
    mode.run()


if __name__ == "__main__":
    main()
