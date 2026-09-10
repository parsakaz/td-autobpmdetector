"""BPM detection server.

Runs in its own process so the environment holding torch never has to be ABI-compatible
with the host, and so a crash in native code cannot take TouchDesigner down with it.

    python -m tdautobpm.sidecar --socket /tmp/tdautobpm.sock

Accepts one client at a time (reconnection is fine). The client streams audio frames
and receives a JSON result whenever a new estimate is available.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from typing import Optional

from . import protocol as P

log = logging.getLogger("tdautobpm.sidecar")


class BpmServer:
    """Serves tempo estimates over a stream socket."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        sample_rate: int = 44100,
        update_hz: float = 4.0,
        estimate: str = "local_mean",
        reset_seconds: Optional[float] = 5.0,
        lock_confidence: float = 0.0,
        smooth_alpha: float = 0.90,
    ):
        self.settings = dict(
            checkpoint=checkpoint,
            device=device,
            sample_rate=sample_rate,
            update_hz=update_hz,
            estimate=estimate,
            reset_seconds=reset_seconds,
            lock_confidence=lock_confidence,
            smooth_alpha=smooth_alpha,
        )
        self.predictor = None
        self._build()

    def _build(self) -> None:
        # Imported lazily so `--help` and startup errors don't require torch.
        from .engine import make_predictor

        s = self.settings
        log.info(
            "loading model (device=%s, input=%d Hz, update=%.1f Hz)",
            s["device"], s["sample_rate"], s["update_hz"],
        )
        t0 = time.time()
        self.predictor = make_predictor(
            s["checkpoint"],
            device=s["device"],
            input_sample_rate=s["sample_rate"],
            update_hz=s["update_hz"],
            estimate=s["estimate"],
            reset_seconds=s["reset_seconds"],
            lock_confidence=s["lock_confidence"],
            smooth_alpha=s["smooth_alpha"],
        )
        log.info("model ready in %.2fs", time.time() - t0)

    # -- message handling -------------------------------------------------

    def reconfigure(self, cfg: dict) -> None:
        """Apply a config message, rebuilding the model only when it must be."""
        rebuild_keys = {"checkpoint", "device", "sample_rate"}
        needs_rebuild = any(
            k in cfg and cfg[k] != self.settings.get(k) for k in rebuild_keys
        )
        self.settings.update({k: v for k, v in cfg.items() if k in self.settings})

        if needs_rebuild:
            self._build()
            return

        p = self.predictor
        if "update_hz" in cfg and cfg["update_hz"]:
            p.update_hz = float(cfg["update_hz"])
            p.update_n = max(1, int(p.cfg.sample_rate / p.update_hz))
        if "estimate" in cfg:
            p.estimate = str(cfg["estimate"])
        if "smooth_alpha" in cfg:
            p.smooth_alpha = float(cfg["smooth_alpha"])
        if "lock_confidence" in cfg:
            p.lock_confidence = float(cfg["lock_confidence"])
        if "reset_seconds" in cfg:
            rs = cfg["reset_seconds"]
            p.reset_seconds = None if not rs or float(rs) <= 0 else float(rs)
            p._reset_n = (
                max(1, int(p.reset_seconds * p.cfg.sample_rate))
                if p.reset_seconds is not None
                else None
            )

    def handle(self, msg_type: int, payload: bytes) -> Optional[bytes]:
        """Process one frame, returning a reply frame or None."""
        if msg_type == P.MSG_AUDIO:
            import numpy as np

            samples = np.frombuffer(payload, dtype="<f4")
            if samples.size == 0:
                return None
            out = self.predictor.push_audio(samples)
            if out is None:
                return None
            bpm, conf, win_s = out
            return P.encode_json(
                P.MSG_RESULT,
                {"bpm": bpm, "confidence": conf, "window_s": win_s, "t": time.time()},
            )

        if msg_type == P.MSG_CONFIG:
            self.reconfigure(json.loads(payload.decode("utf-8")))
            return P.encode_json(P.MSG_READY, self._status())

        if msg_type == P.MSG_RESET:
            self.predictor.reset()
            return P.encode_json(P.MSG_READY, self._status())

        if msg_type == P.MSG_PING:
            return P.encode_json(P.MSG_PONG, {"t": time.time()})

        if msg_type == P.MSG_SHUTDOWN:
            raise SystemExit(0)

        return P.encode_json(P.MSG_ERROR, {"error": f"unknown message type {msg_type}"})

    def _status(self) -> dict:
        p = self.predictor
        return {
            "ready": True,
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "model_sample_rate": p.cfg.sample_rate,
            "input_sample_rate": p.input_sample_rate,
            "bpm_min": p.bpm_min,
            "bpm_max": p.bpm_max,
            "settings": {k: v for k, v in self.settings.items()},
        }

    # -- serving ----------------------------------------------------------

    def serve(self, sock: socket.socket) -> None:
        """Serve clients until interrupted."""
        while True:
            conn, _ = sock.accept()
            log.info("client connected")
            try:
                self._serve_one(conn)
            except (ConnectionResetError, BrokenPipeError):
                log.info("client disconnected abruptly")
            except P.ProtocolError as exc:
                log.error("protocol error: %s", exc)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
            log.info("client disconnected")

    def _serve_one(self, conn: socket.socket) -> None:
        conn.sendall(P.encode_json(P.MSG_READY, self._status()))
        while True:
            frame = P.recv_frame(conn)
            if frame is None:
                return
            try:
                reply = self.handle(*frame)
            except SystemExit:
                raise
            except Exception as exc:  # keep the server alive for the next frame
                log.exception("error handling %s", P.NAMES.get(frame[0], frame[0]))
                reply = P.encode_json(P.MSG_ERROR, {"error": str(exc)})
            if reply:
                conn.sendall(reply)


def bind(socket_path: Optional[str] = None, port: Optional[int] = None) -> socket.socket:
    """Create and bind the listening socket (Unix domain by default)."""
    if port:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(port)))
        sock.listen(1)
        log.info("listening on 127.0.0.1:%d", sock.getsockname()[1])
        return sock

    path = os.path.expanduser(socket_path or default_socket_path())
    if os.path.exists(path):
        os.unlink(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.listen(1)
    log.info("listening on %s", path)
    return sock


def default_socket_path() -> str:
    base = os.environ.get("TMPDIR") or "/tmp"
    return os.path.join(base, "tdautobpm-%d.sock" % os.getuid())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tdautobpm.sidecar", description=__doc__)
    ap.add_argument("--socket", help="unix socket path")
    ap.add_argument("--port", type=int, help="listen on TCP loopback instead")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--update-hz", type=float, default=4.0)
    ap.add_argument("--estimate", default="local_mean",
                    choices=["local_mean", "mode", "mean", "median"])
    ap.add_argument("--reset-seconds", type=float, default=5.0,
                    help="0 accumulates evidence indefinitely")
    ap.add_argument("--lock-confidence", type=float, default=0.0)
    ap.add_argument("--smooth-alpha", type=float, default=0.90)
    ap.add_argument("--announce", action="store_true",
                    help="print the socket path on stdout once listening")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    sock = bind(args.socket, args.port)

    server = BpmServer(
        checkpoint=args.checkpoint,
        device=args.device,
        sample_rate=args.sample_rate,
        update_hz=args.update_hz,
        estimate=args.estimate,
        reset_seconds=args.reset_seconds,
        lock_confidence=args.lock_confidence,
        smooth_alpha=args.smooth_alpha,
    )

    if args.announce:
        endpoint = (
            f"127.0.0.1:{sock.getsockname()[1]}" if args.port else sock.getsockname()
        )
        print(endpoint, flush=True)

    try:
        server.serve(sock)
    except KeyboardInterrupt:
        log.info("interrupted")
    except SystemExit:
        pass
    finally:
        sock.close()
        if not args.port:
            try:
                os.unlink(sock.getsockname())
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
