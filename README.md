# ambilight-music

Turns a [HyperHDR](https://github.com/awawa-dev/HyperHDR) ambilight into a music light: the
dominant colors of the current album cover drift slowly around the screen, and the music
itself (loudness and beats) drives brightness and spread.

It is meant for a setup where the projector/TV ambilight would otherwise sit dark while music
plays: a home automation switches it on when the AV receiver is on a non-video input, and off
again for movies, where the HDMI grabber takes over.

## How it works

```
AV receiver ── HDMI OUT (audio on) ── capture card ─┬─ video → HyperHDR grabber (priority 240)
                                                    └─ audio → ambilight-music
cover websocket ── albumCover URL ────────────────────────────→ ambilight-music
                                                                 │ ~50 frames/s
                                                                 └→ HyperHDR JSON TCP (priority 100)
```

* **Palette**: k-means on the cover; clusters that are too dark, too grey or tiny are dropped
  (borders, text), the rest is boosted a little for LEDs.
* **Frame**: a 64x36 image where each palette color is a soft blob on a ring around the frame,
  the dominant one at the top. HyperHDR maps the frame onto the LEDs exactly like a video
  frame, so this needs no knowledge of the LED layout.
* **Motion**: blobs drift and the ring turns slowly (lava lamp); loud passages widen the
  secondary colors and raise brightness; detected bass beats add short pulses; a new cover
  crossfades in over 1.5 s.
* **Audio**: raw PCM from `arecord`, so the capture device is only held while music mode is on.
* **Safety**: every frame expires after 1 s, so if the process dies the grabber is back
  immediately. Off clears the priority.

The cover source is any websocket that sends
`{"type": "albumCover", "value": "<image url>"}` messages and replays the last one on connect.

## Install (Debian/Raspberry Pi OS, as a user service)

```sh
git clone https://github.com/Janne6565/ambilight-music ~/ambilight-music
cd ~/ambilight-music
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.example.env ~/.config/ambilight-music.env   # then fill it in
mkdir -p ~/.config/systemd/user && cp ambilight-music.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now ambilight-music
```

## Control API

All routes take GET or POST and need the token, as `?token=...` or `Authorization: Bearer ...`.

| Route | Effect | Answer |
|---|---|---|
| `/on` | start music mode | `1` |
| `/off` | stop, clear the HyperHDR priority | `0` |
| `/status` | | `1` or `0` |
| `/intensity?value=0..2` | how strongly it reacts (1 = default) | the new value |

State (on/off, intensity) is kept across restarts.

## Preview without audio

```sh
python preview.py --host <hyperhdr-host> --cover <image url> --seconds 60 --bpm 120
```

Plays a cover's palette with a synthetic beat at priority 99 and clears it afterwards.

## Tests

```sh
pip install pytest && python -m pytest tests
```
