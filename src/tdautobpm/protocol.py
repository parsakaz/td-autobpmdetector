"""Wire format between TouchDesigner and the sidecar process.

Deliberately tiny and dependency-free: this module is imported by the TouchDesigner
side, which runs in TD's own interpreter and must not need torch or numpy.

Every message is ``magic | type | length | payload``::

    b"TDBP"   4 bytes
    type      1 byte   (see MSG_* below)
    length    4 bytes  unsigned, little-endian
    payload   `length` bytes

Audio payloads are raw little-endian float32 mono samples. Everything else is UTF-8
JSON. Framing is explicit because a stream socket gives no message boundaries, and
the natural alternative - newline-delimited JSON - would force base64 on the audio.
"""

from __future__ import annotations

import json
import struct
from typing import Optional, Tuple

MAGIC = b"TDBP"
HEADER = struct.Struct("<4sBI")
HEADER_SIZE = HEADER.size  # 9

# client -> server
MSG_AUDIO = 1
MSG_CONFIG = 2
MSG_RESET = 3
MSG_PING = 4
MSG_SHUTDOWN = 5

# server -> client
MSG_RESULT = 16
MSG_ERROR = 17
MSG_READY = 18
MSG_PONG = 19

#: Refuse absurd frames rather than trying to allocate them.
MAX_PAYLOAD = 64 * 1024 * 1024

NAMES = {
    MSG_AUDIO: "AUDIO", MSG_CONFIG: "CONFIG", MSG_RESET: "RESET", MSG_PING: "PING",
    MSG_SHUTDOWN: "SHUTDOWN", MSG_RESULT: "RESULT", MSG_ERROR: "ERROR",
    MSG_READY: "READY", MSG_PONG: "PONG",
}


class ProtocolError(Exception):
    """Raised on a malformed or oversized frame."""


def encode(msg_type: int, payload: bytes = b"") -> bytes:
    """Frame a payload for transmission."""
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"payload too large: {len(payload)} bytes")
    return HEADER.pack(MAGIC, msg_type, len(payload)) + payload


def encode_json(msg_type: int, obj) -> bytes:
    return encode(msg_type, json.dumps(obj).encode("utf-8"))


def encode_audio(samples) -> bytes:
    """Frame mono audio. Accepts a numpy array, a torch tensor, or a float sequence."""
    buf = getattr(samples, "tobytes", None)
    if buf is not None:  # numpy
        arr = samples
        if getattr(arr, "dtype", None) is not None and arr.dtype.name != "float32":
            arr = arr.astype("float32")
        return encode(MSG_AUDIO, arr.tobytes())
    if hasattr(samples, "numpy"):  # torch
        return encode_audio(samples.detach().cpu().float().numpy())
    data = struct.pack("<%df" % len(samples), *samples)
    return encode(MSG_AUDIO, data)


def decode_audio(payload: bytes) -> list:
    """Decode an audio payload to a list of floats (numpy-free fallback path)."""
    n = len(payload) // 4
    return list(struct.unpack("<%df" % n, payload[: n * 4]))


class FrameReader:
    """Incremental frame parser for a non-blocking stream socket.

    Feed it whatever bytes arrive; pull whole frames out as they complete.
    """

    def __init__(self):
        self._buf = bytearray()

    def feed(self, data: bytes) -> None:
        if data:
            self._buf.extend(data)

    def __len__(self) -> int:
        return len(self._buf)

    def next_frame(self) -> Optional[Tuple[int, bytes]]:
        """Return the next complete ``(msg_type, payload)``, or None if incomplete."""
        if len(self._buf) < HEADER_SIZE:
            return None

        magic, msg_type, length = HEADER.unpack_from(self._buf, 0)
        if magic != MAGIC:
            raise ProtocolError(f"bad magic {bytes(magic)!r}; stream is out of sync")
        if length > MAX_PAYLOAD:
            raise ProtocolError(f"frame claims {length} bytes")

        total = HEADER_SIZE + length
        if len(self._buf) < total:
            return None

        payload = bytes(self._buf[HEADER_SIZE:total])
        del self._buf[:total]
        return msg_type, payload

    def frames(self):
        """Yield every complete frame currently buffered."""
        while True:
            frame = self.next_frame()
            if frame is None:
                return
            yield frame


def recv_exactly(sock, n: int) -> Optional[bytes]:
    """Blocking read of exactly ``n`` bytes; None if the peer closed first."""
    chunks, got = [], 0
    while got < n:
        b = sock.recv(n - got)
        if not b:
            return None
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def recv_frame(sock) -> Optional[Tuple[int, bytes]]:
    """Blocking read of one whole frame; None if the peer closed."""
    head = recv_exactly(sock, HEADER_SIZE)
    if head is None:
        return None
    magic, msg_type, length = HEADER.unpack(head)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r}; stream is out of sync")
    if length > MAX_PAYLOAD:
        raise ProtocolError(f"frame claims {length} bytes")
    payload = recv_exactly(sock, length) if length else b""
    if payload is None:
        return None
    return msg_type, payload
