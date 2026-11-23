"""HyperHDR client over the raw JSON TCP server (default port 19444).

One persistent socket instead of an HTTP request per frame. HyperHDR answers every command
with a line of JSON; a reader thread drains those answers so the socket buffer never fills up
and stalls the stream, and reports rejected commands to the log.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import socket
import threading

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

MAX_ORIGIN_LENGTH = 20
"""HyperHDR rejects longer origins with nothing but "Errors during specific message validation"."""


class HyperHdr:
    SEND_TIMEOUT_S = 3.0

    def __init__(self, host: str, port: int, priority: int, origin: str):
        if not 4 <= len(origin) <= MAX_ORIGIN_LENGTH:
            raise ValueError(f"HyperHDR origin must be 4..{MAX_ORIGIN_LENGTH} characters, got {origin!r}")
        self.host, self.port, self.priority, self.origin = host, port, priority, origin
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def show(self, frame: np.ndarray, duration_ms: int) -> None:
        buf = io.BytesIO()
        Image.fromarray(frame, "RGB").save(buf, "PNG")
        self._send({
            "command": "image",
            "priority": self.priority,
            "origin": self.origin,
            "format": "auto",
            "imagedata": base64.b64encode(buf.getvalue()).decode("ascii"),
            "duration": duration_ms,
        })

    def clear(self) -> None:
        self._send({"command": "clear", "priority": self.priority})

    def close(self) -> None:
        with self._lock:
            self._drop()

    def _send(self, command: dict) -> None:
        data = (json.dumps(command) + "\n").encode()
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._sock is None:
                        self._connect()
                    self._sock.sendall(data)
                    return
                except OSError as e:
                    self._drop()
                    if attempt == 2:
                        raise ConnectionError(f"HyperHDR {self.host}:{self.port} unreachable: {e}") from e

    def _connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.SEND_TIMEOUT_S)
        sock.settimeout(self.SEND_TIMEOUT_S)
        self._sock = sock
        threading.Thread(target=self._drain, args=(sock,), name="hyperhdr-reader", daemon=True).start()
        log.info("Connected to HyperHDR at %s:%d", self.host, self.port)

    def _drop(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _drain(self, sock: socket.socket) -> None:
        # Never touch the socket's timeout here: it is shared with the sender. This used to call
        # settimeout(None), which made sendall() block forever as soon as HyperHDR stopped
        # reading, while holding the lock every /on and /off needs, so the whole control API
        # hung with its port still open. Read with the same 3 s timeout and just keep waiting.
        buffer = b""
        while True:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                try:
                    reply = json.loads(line)
                except ValueError:
                    continue
                if reply.get("success") is False:
                    log.warning("HyperHDR rejected %s: %s", reply.get("command"), reply.get("error"))
