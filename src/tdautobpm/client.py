"""Client for the BPM sidecar.

Stdlib-only on purpose: this runs inside TouchDesigner's interpreter, which has no
torch and must never be made to need it.

:class:`SidecarClient` owns the child process and the socket, and is safe to poll from
a cook callback - :meth:`poll` never blocks.
"""

from __future__ import annotations

import collections
import errno
import json
import logging
import os
import socket
import subprocess
import time
from typing import List, Optional

from . import protocol as P

log = logging.getLogger("tdautobpm.client")


class SidecarClient:
    """Spawn and talk to a ``tdautobpm.sidecar`` process."""

    def __init__(
        self,
        python: str,
        socket_path: Optional[str] = None,
        port: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        startup_timeout: float = 120.0,
        autorestart: bool = True,
        max_queue_bytes: int = 2 * 1024 * 1024,
        **options,
    ):
        self.python = python
        self.socket_path = socket_path or _default_socket_path()
        self.port = port
        self.cwd = cwd
        self.env = env
        self.startup_timeout = float(startup_timeout)
        self.autorestart = bool(autorestart)
        #: Cap on unsent audio. If the sidecar cannot keep up we drop the *oldest*
        #: audio rather than growing without bound - for live tempo detection stale
        #: audio is worthless, and an unbounded queue would leak and lag forever.
        self.max_queue_bytes = int(max_queue_bytes)
        self.dropped_bytes = 0
        self.options = options

        self.proc: Optional[subprocess.Popen] = None
        self.sock: Optional[socket.socket] = None
        self.status: dict = {}
        self.last_error: str = ""
        self._reader = P.FrameReader()
        # A deque of whole frames plus an offset into the head frame. Using a single
        # bytearray and deleting the sent prefix is O(n) per send, which turns into
        # gigabytes of memmove once the queue reaches a few megabytes.
        self._sendq: "collections.deque" = collections.deque()
        self._head = 0
        self._queued = 0
        self._restarts = 0
        self._last_restart = 0.0

    # -- lifecycle --------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self.sock is not None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _argv(self) -> List[str]:
        # --once: this process is ours alone, so it exits with our connection rather
        # than outliving the host.
        argv = [self.python, "-m", "tdautobpm.sidecar", "--once"]
        if self.port:
            argv += ["--port", str(self.port)]
        else:
            argv += ["--socket", self.socket_path]
        for key, flag in (
            ("device", "--device"),
            ("sample_rate", "--sample-rate"),
            ("update_hz", "--update-hz"),
            ("estimate", "--estimate"),
            ("reset_seconds", "--reset-seconds"),
            ("lock_confidence", "--lock-confidence"),
            ("smooth_alpha", "--smooth-alpha"),
            ("range_min", "--range-min"),
            ("range_max", "--range-max"),
            ("checkpoint", "--checkpoint"),
        ):
            val = self.options.get(key)
            if val is not None:
                argv += [flag, str(val)]
        return argv

    def start(self) -> None:
        """Launch the sidecar and connect. Raises RuntimeError on failure."""
        self.stop()

        env = dict(os.environ if self.env is None else self.env)
        # The sidecar must import `tdautobpm` from this checkout, not from whatever
        # happens to be installed in the target environment.
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            pkg_parent + (os.pathsep + existing if existing else "")
        )
        # An inherited venv/conda activation would shadow the interpreter we chose.
        env.pop("PYTHONHOME", None)
        env.pop("VIRTUAL_ENV", None)

        if not self.port and os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        log.info("starting sidecar: %s", " ".join(self._argv()))
        self.proc = subprocess.Popen(
            self._argv(),
            cwd=self.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            self._connect()
        except Exception:
            self.stop()
            raise

    def _connect(self) -> None:
        deadline = time.time() + self.startup_timeout
        last: Optional[Exception] = None

        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    "sidecar exited during startup (code %s)\n%s"
                    % (self.proc.returncode, self._drain_stderr())
                )
            try:
                if self.port:
                    s = socket.create_connection(("127.0.0.1", self.port), timeout=5.0)
                else:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.settimeout(5.0)
                    s.connect(self.socket_path)
                self.sock = s
                break
            except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                last = exc
                time.sleep(0.05)
        else:
            raise RuntimeError(f"could not connect to sidecar: {last}")

        # Wait for READY, which only arrives once the model is loaded.
        self.sock.settimeout(max(1.0, deadline - time.time()))
        frame = P.recv_frame(self.sock)
        if frame is None:
            raise RuntimeError("sidecar closed before signalling ready\n" + self._drain_stderr())
        msg_type, payload = frame
        if msg_type == P.MSG_ERROR:
            raise RuntimeError(json.loads(payload.decode()).get("error", "unknown error"))
        if msg_type != P.MSG_READY:
            raise RuntimeError(f"unexpected first message: {P.NAMES.get(msg_type, msg_type)}")

        self.status = json.loads(payload.decode("utf-8"))
        self.sock.setblocking(False)
        self.last_error = ""
        log.info("sidecar ready (pid %s, python %s)",
                 self.status.get("pid"), self.status.get("python"))

    def stop(self) -> None:
        """Close the socket and terminate the child."""
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    try:
                        self.proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            self.proc = None

        self._reader = P.FrameReader()
        self._sendq.clear()
        self._head = 0
        self._queued = 0

    def _drain_stderr(self) -> str:
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            os.set_blocking(self.proc.stderr.fileno(), False)
            data = self.proc.stderr.read() or b""
        except Exception:
            return ""
        return data.decode("utf-8", "replace").strip()

    # -- messaging --------------------------------------------------------

    @property
    def pending_bytes(self) -> int:
        """Unsent bytes still queued. Sustained growth means the sidecar is behind."""
        return self._queued - self._head

    def _send(self, frame: bytes) -> None:
        """Queue a frame and flush as much as the socket will take."""
        if self.sock is None:
            return
        self._sendq.append(frame)
        self._queued += len(frame)
        self._trim()
        self._flush()

    def _trim(self) -> None:
        """Drop the oldest queued frames once past the cap.

        The frame at the head may be half-transmitted; the peer has already received
        its header and part of its payload, so discarding the remainder would leave
        the stream desynchronised and the connection is torn down on the next frame.
        That frame is therefore protected and the drop starts one behind it.
        """
        while self.pending_bytes > self.max_queue_bytes:
            protect = 1 if self._head > 0 else 0
            if len(self._sendq) <= protect + 1:
                break  # keep the in-flight frame and the freshest one
            if protect:
                in_flight = self._sendq.popleft()
                dropped = self._sendq.popleft()
                self._sendq.appendleft(in_flight)
            else:
                dropped = self._sendq.popleft()
            self._queued -= len(dropped)
            self.dropped_bytes += len(dropped)

    def _flush(self) -> None:
        while self._sendq and self.sock is not None:
            head = self._sendq[0]
            try:
                sent = self.sock.send(memoryview(head)[self._head:])
            except BlockingIOError:
                return  # kernel buffer full; try again next cook
            except OSError as exc:
                self._fail(f"send failed: {exc}")
                return
            if sent <= 0:
                return
            self._head += sent
            if self._head >= len(head):
                self._sendq.popleft()
                self._queued -= len(head)
                self._head = 0

    def send_audio(self, samples) -> None:
        self._send(P.encode_audio(samples))

    def configure(self, **cfg) -> None:
        self.options.update({k: v for k, v in cfg.items() if v is not None})
        self._send(P.encode_json(P.MSG_CONFIG, cfg))

    def reset(self) -> None:
        self._send(P.encode(P.MSG_RESET))

    def ping(self) -> None:
        self._send(P.encode(P.MSG_PING))

    def poll(self) -> List[dict]:
        """Return any results that have arrived. Never blocks.

        Call once per cook. If the sidecar has died and ``autorestart`` is set, this
        brings it back up rather than raising.
        """
        if self.sock is None:
            self._maybe_restart()
            return []

        self._flush()

        while True:
            try:
                data = self.sock.recv(65536)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                self._fail(f"recv failed: {exc}")
                return []
            if not data:
                self._fail("sidecar closed the connection\n" + self._drain_stderr())
                return []
            self._reader.feed(data)

        results = []
        try:
            for msg_type, payload in self._reader.frames():
                if msg_type == P.MSG_RESULT:
                    results.append(json.loads(payload.decode("utf-8")))
                elif msg_type == P.MSG_READY:
                    self.status = json.loads(payload.decode("utf-8"))
                elif msg_type == P.MSG_ERROR:
                    self.last_error = json.loads(payload.decode("utf-8")).get("error", "")
                    log.error("sidecar: %s", self.last_error)
        except P.ProtocolError as exc:
            self._fail(str(exc))

        if self.proc is not None and self.proc.poll() is not None:
            self._fail("sidecar exited (code %s)\n%s"
                       % (self.proc.returncode, self._drain_stderr()))

        return results

    def _fail(self, message: str) -> None:
        self.last_error = message
        log.error("%s", message)
        self.stop()
        self._maybe_restart()

    def _maybe_restart(self) -> None:
        if not self.autorestart:
            return
        # Back off so a reliably-crashing sidecar doesn't spin.
        delay = min(30.0, 1.0 * (2 ** min(self._restarts, 5)))
        if time.time() - self._last_restart < delay:
            return
        self._last_restart = time.time()
        self._restarts += 1
        log.info("restarting sidecar (attempt %d)", self._restarts)
        try:
            self.start()
            self._restarts = 0
        except Exception as exc:
            self.last_error = str(exc)

    # -- context manager --------------------------------------------------

    def __enter__(self) -> "SidecarClient":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _default_socket_path() -> str:
    base = os.environ.get("TMPDIR") or "/tmp"
    return os.path.join(base, "tdautobpm-%d-%d.sock" % (os.getuid(), os.getpid()))
