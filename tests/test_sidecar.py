"""Protocol framing and the sidecar round trip."""

from __future__ import annotations

import struct
import sys

import numpy as np
import pytest

from tdautobpm import protocol as P
from tdautobpm.client import SidecarClient

from test_engine import click_track, metrical_error  # noqa: E402


class TestProtocol:
    def test_roundtrip(self):
        frame = P.encode_json(P.MSG_RESULT, {"bpm": 128.0})
        r = P.FrameReader()
        r.feed(frame)
        msg_type, payload = r.next_frame()
        assert msg_type == P.MSG_RESULT
        assert b"128" in payload

    def test_incomplete_frame_yields_nothing(self):
        frame = P.encode_json(P.MSG_RESULT, {"bpm": 128.0})
        r = P.FrameReader()
        r.feed(frame[:-3])
        assert r.next_frame() is None
        r.feed(frame[-3:])
        assert r.next_frame() is not None

    def test_frames_split_across_arbitrary_chunks(self):
        """A stream socket gives no message boundaries; framing must not assume any."""
        blob = b"".join(P.encode_json(P.MSG_RESULT, {"i": i}) for i in range(20))
        r = P.FrameReader()
        got = []
        for i in range(0, len(blob), 7):
            r.feed(blob[i : i + 7])
            got.extend(r.frames())
        assert len(got) == 20

    def test_multiple_frames_in_one_chunk(self):
        blob = P.encode(P.MSG_RESET) + P.encode(P.MSG_PING)
        r = P.FrameReader()
        r.feed(blob)
        assert [t for t, _ in r.frames()] == [P.MSG_RESET, P.MSG_PING]

    def test_audio_roundtrip_preserves_samples(self):
        x = np.linspace(-1, 1, 512, dtype=np.float32)
        frame = P.encode_audio(x)
        r = P.FrameReader()
        r.feed(frame)
        msg_type, payload = r.next_frame()
        assert msg_type == P.MSG_AUDIO
        assert np.allclose(np.frombuffer(payload, dtype="<f4"), x)

    def test_audio_accepts_float64_input(self):
        x = np.linspace(-1, 1, 64, dtype=np.float64)
        _, payload = _one(P.encode_audio(x))
        assert len(payload) == 64 * 4

    def test_audio_accepts_plain_list(self):
        _, payload = _one(P.encode_audio([0.1, 0.2, 0.3]))
        assert np.allclose(np.frombuffer(payload, dtype="<f4"), [0.1, 0.2, 0.3], atol=1e-6)

    def test_bad_magic_is_rejected(self):
        r = P.FrameReader()
        r.feed(b"XXXX" + struct.pack("<BI", P.MSG_PING, 0))
        with pytest.raises(P.ProtocolError):
            r.next_frame()

    def test_oversized_frame_is_rejected(self):
        r = P.FrameReader()
        r.feed(P.MAGIC + struct.pack("<BI", P.MSG_AUDIO, P.MAX_PAYLOAD + 1))
        with pytest.raises(P.ProtocolError):
            r.next_frame()


def _one(frame):
    r = P.FrameReader()
    r.feed(frame)
    return r.next_frame()


# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    c = SidecarClient(
        sys.executable, sample_rate=44100, update_hz=8.0, device="cpu",
        reset_seconds=0, autorestart=False,
    )
    c.start()
    yield c
    c.stop()


def _drain(c, timeout=30.0):
    import time

    out, idle, t0 = [], 0.0, time.time()
    while c.pending_bytes or idle < 0.5:
        r = c.poll()
        if r:
            out.extend(r)
            idle = 0.0
        else:
            time.sleep(0.02)
            idle += 0.02
        if time.time() - t0 > timeout:
            break
    return out


class TestSidecar:
    def test_reports_ready_with_status(self, client):
        assert client.status["ready"] is True
        assert client.status["model_sample_rate"] == 22050
        assert client.status["input_sample_rate"] == 44100

    def test_detects_tempo_over_the_socket(self, client):
        x = click_track(140, duration=30.0)
        for i in range(0, len(x), 2048):
            client.send_audio(x[i : i + 2048])
            client.poll()
        results = _drain(client)
        assert results, "no results returned"
        assert metrical_error(results[-1]["bpm"], 140) < 1.5
        assert not client.last_error

    def test_backpressure_drops_audio_instead_of_growing(self, client):
        """A host outrunning the model must not build an unbounded backlog."""
        x = click_track(128, duration=60.0)
        for i in range(0, len(x), 4096):
            client.send_audio(x[i : i + 4096])
        assert client.pending_bytes <= client.max_queue_bytes + 4096 * 4
        assert client.dropped_bytes > 0
        # Dropping must not corrupt the stream: the connection stays healthy.
        assert _drain(client)
        assert not client.last_error

    def test_reset_is_accepted(self, client):
        x = click_track(128, duration=10.0)
        for i in range(0, len(x), 4096):
            client.send_audio(x[i : i + 4096])
        _drain(client)
        client.reset()
        assert not client.last_error

    def test_stop_terminates_the_child(self, client):
        assert client.alive
        client.stop()
        assert not client.alive
