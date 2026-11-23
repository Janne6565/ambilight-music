import socket
import threading
import time

import numpy as np
import pytest

from ambilight_music.hyperhdr import HyperHdr


def test_stalled_hyperhdr_raises_instead_of_hanging_forever():
    """A HyperHDR that accepts but never reads used to wedge the sender (and with it the whole
    control API) for good. It must now give up within the send timeout."""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    held = []
    threading.Thread(target=lambda: [held.append(server.accept()) for _ in range(4)], daemon=True).start()

    client = HyperHdr("127.0.0.1", server.getsockname()[1], 100, "TestOrigin")
    client.SEND_TIMEOUT_S = 0.3
    noise = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)  # big, incompressible PNG

    started = time.monotonic()
    with pytest.raises(ConnectionError):
        for _ in range(200):
            client.show(noise, 1000)
    assert time.monotonic() - started < 10
    client.close()
    server.close()
