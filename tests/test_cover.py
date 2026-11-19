import json

from ambilight_music.cover import CoverFollower


def _msg(key, value):
    return json.dumps({"type": key, "value": value, "source": "yamaha", "updated": 1.0})


def _follower():
    return CoverFollower("ws://example.invalid/audio", lambda p, u: None)


def test_device_artwork_beats_the_itunes_guess():
    f = _follower()
    f._handle(_msg("albumCover", "https://itunes.example/lady-gaga-the-cure.jpg"))
    f._handle(_msg("localAlbumCover", "http://amp.example/AlbumART5368.jpg"))
    assert f.choose() == "http://amp.example/AlbumART5368.jpg"


def test_itunes_guess_is_the_fallback():
    f = _follower()
    f._handle(_msg("localAlbumCover", ""))
    f._handle(_msg("albumCover", "https://itunes.example/cover.jpg"))
    assert f.choose() == "https://itunes.example/cover.jpg"


def test_other_messages_are_ignored():
    f = _follower()
    assert f._handle(_msg("volume", "61")) is False
    assert f._handle("not json") is False
    assert f.choose() is None


def test_slow_download_of_an_older_cover_does_not_win(monkeypatch):
    import threading

    from PIL import Image

    from ambilight_music import cover as cover_mod

    applied = []
    f = CoverFollower("ws://example.invalid/audio", lambda p, u: applied.append(u))
    release_slow = threading.Event()

    def fake_download(url):
        if "slow" in url:
            release_slow.wait(5)
        return Image.new("RGB", (40, 40), (200, 30, 40) if "slow" in url else (30, 60, 200))

    monkeypatch.setattr(cover_mod, "download", fake_download)
    f._last_url = "https://cdn.example/slow-previous-song.jpg"
    slow = threading.Thread(target=f._load, args=("https://cdn.example/slow-previous-song.jpg",))
    slow.start()
    f._last_url = "http://amp.example/current.jpg"
    f._load("http://amp.example/current.jpg")
    release_slow.set()
    slow.join(5)
    assert applied == ["http://amp.example/current.jpg"]
