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
