import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from ambilight_music.main import Config, MusicMode, make_handler

TOKEN = "test-token"


@pytest.fixture
def api(tmp_path):
    config = Config({"HUB_WS_URL": "ws://example.invalid/audio", "CONTROL_TOKEN": TOKEN,
                     "STATE_FILE": str(tmp_path / "state.json")})
    mode = MusicMode(config)  # stays disabled, so nothing touches audio or HyperHDR
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(mode, TOKEN))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, token=TOKEN):
        req = urllib.request.Request(base + path, method="POST", headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    yield call, mode, config
    server.shutdown()


def test_intensity_slider(api):
    call, mode, _ = api
    assert call("/intensity") == (200, "1")
    assert call("/intensity?value=0.3") == (200, "0.3")
    assert call("/intensity?value=5") == (200, "2")   # clamped
    assert call("/intensity?value=-1") == (200, "0")
    assert call("/intensity?value=abc")[0] == 400
    assert call("/intensity") == (200, "0")           # read without changing


def test_intensity_survives_restart(api):
    call, _, config = api
    call("/intensity?value=0.4")
    restarted = MusicMode(config)
    restarted.load_state()
    assert restarted.renderer.intensity == pytest.approx(0.4)


def test_wrong_token_is_rejected(api):
    call, mode, _ = api
    assert call("/intensity?value=0", token="nope")[0] == 401
    assert mode.renderer.intensity == 1.0


def test_lead_setting(api):
    call, mode, config = api
    assert call("/lead") == (200, "100")
    assert call("/lead?value=150") == (200, "150")
    assert call("/lead?value=9999") == (200, "400")   # clamped
    restarted = MusicMode(config)
    restarted.load_state()
    assert round(restarted.audio.lead_s * 1000) == 400
