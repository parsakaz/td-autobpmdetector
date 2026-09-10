"""TouchDesigner extension for the AutoBpm component.

Runs inside TouchDesigner's own interpreter. It must not import torch, numpy or
anything else from the detection environment at module scope - in sidecar mode those
packages live in a completely different interpreter, and in in-process mode they are
only reachable after ``sys.path`` has been extended.

Two runtimes, one interface
---------------------------
``sidecar``     - a separate process holds torch; audio goes over a local socket.
                  Works with any Python 3.9+ environment regardless of what TD ships,
                  and a native crash cannot take TouchDesigner down.
``inprocess``   - the environment's ``site-packages`` is appended to ``sys.path`` and
                  inference runs on a worker thread. Lower latency, but the
                  environment must be the same CPython minor version and architecture
                  as TouchDesigner, because compiled extensions are not portable
                  across either.

Switch with the Runtime parameter. ``Envpath`` may be blank (auto-detect), an
interpreter path, a venv or conda env directory, or a bare conda environment name.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback

# The repository root, so `src/tdautobpm` is importable from TouchDesigner.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tdautobpm import envresolve as E  # noqa: E402  (stdlib-only, safe here)
from tdautobpm.client import SidecarClient  # noqa: E402  (stdlib-only)

#: CHOP channels this component outputs.
CHANNELS = ("bpm", "confidence", "beat", "phase")


class AutoBpm:
    """Extension class for the AutoBpm COMP."""

    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.detector = None
        self.env = None
        self.status = "idle"
        self.error = ""

        self.bpm = 0.0
        self.confidence = 0.0
        self.phase = 0.0
        self._beat = 0.0
        self._last_cook = None

        self._sample_rate = 0

    # -- parameters -------------------------------------------------------

    def _par(self, name, default=None):
        p = getattr(self.ownerComp.par, name, None)
        return default if p is None else p.eval()

    def _settings(self) -> dict:
        return dict(
            device=self._par("Torchdevice", "cpu"),
            update_hz=float(self._par("Updaterate", 4.0)),
            estimate=self._par("Estimate", "local_mean"),
            smooth_alpha=float(self._par("Smoothing", 0.90)),
            reset_seconds=float(self._par("Resetseconds", 5.0)),
            lock_confidence=float(self._par("Lockconfidence", 0.0)),
        )

    # -- lifecycle --------------------------------------------------------

    def Start(self, sample_rate: int = 44100):
        """Resolve an environment and bring the detector up."""
        self.Stop()
        self.error = ""
        self._sample_rate = int(sample_rate)

        runtime = self._par("Runtime", "sidecar")
        spec = (self._par("Envpath", "") or "").strip() or None

        try:
            self.env = E.resolve(spec, project_root=_REPO)
        except E.EnvError as exc:
            self._fail(str(exc))
            return

        try:
            if runtime == "inprocess":
                self._start_inprocess()
            else:
                self._start_sidecar()
        except Exception as exc:
            self._fail(f"{exc}\n{traceback.format_exc()}")

    def _start_sidecar(self):
        why = E.sidecar_incompatibility(self.env)
        if why:
            raise RuntimeError(f"environment cannot host the sidecar: {why}")

        self.detector = _SidecarDetector(
            self.env.python, sample_rate=self._sample_rate, cwd=_REPO, **self._settings()
        )
        self.detector.start()
        self.status = "running (sidecar, %s)" % self.env.version

    def _start_inprocess(self):
        why = E.inprocess_incompatibility(self.env)
        if why:
            raise RuntimeError(
                "environment cannot be imported into TouchDesigner: %s.\n"
                "Switch Runtime to 'sidecar', which has no such restriction." % why
            )

        for path in self.env.site_packages():
            if path not in sys.path:
                sys.path.append(path)

        self.detector = _InProcessDetector(sample_rate=self._sample_rate, **self._settings())
        self.detector.start()
        self.status = "running (in-process, %s)" % self.env.version

    def Stop(self):
        if self.detector is not None:
            try:
                self.detector.stop()
            except Exception:
                pass
            self.detector = None
        self.status = "stopped"

    def Restart(self):
        self.Start(self._sample_rate or 44100)

    def Reset(self):
        """Clear accumulated evidence and restart the phase."""
        self.bpm = 0.0
        self.confidence = 0.0
        self.phase = 0.0
        if self.detector is not None:
            self.detector.reset()

    def _fail(self, message: str):
        self.error = message
        self.status = "error"
        self.detector = None
        print("[AutoBpm] " + message)

    # -- per-cook ---------------------------------------------------------

    def Cook(self, scriptOp):
        """Drive one cook of the Script CHOP. Called from the CHOP's callback."""
        scriptOp.clear()
        source = scriptOp.inputs[0] if scriptOp.inputs else None

        rate = int(source.rate) if source is not None and source.rate else 44100
        active = bool(self._par("Active", True))

        if active and (self.detector is None or rate != self._sample_rate):
            self.Start(rate)

        if active and self.detector is not None and source is not None:
            try:
                self._pump(source)
            except Exception as exc:
                self._fail(f"{exc}\n{traceback.format_exc()}")

        self._advance_phase()
        self._write(scriptOp)

    def _pump(self, source):
        """Send this cook's samples and collect any new estimate."""
        if len(source.chans()) == 0 or source.numSamples <= 0:
            return

        # Mix to mono. CHOP channels come back as sample sequences.
        chans = source.chans()
        if len(chans) == 1:
            block = list(chans[0].vals)
        else:
            n = source.numSamples
            inv = 1.0 / len(chans)
            block = [sum(c[i] for c in chans) * inv for i in range(n)]

        self.detector.push(block)

        latest = self.detector.poll()
        if latest is not None:
            self.bpm = latest["bpm"]
            self.confidence = latest["confidence"]

        err = self.detector.error()
        if err:
            self.error = err
            self.status = "error"

    def _advance_phase(self):
        """Free-running beat phase at the detected tempo.

        The model estimates *tempo*, not beat position - it has no notion of where a
        downbeat falls. So this phase runs freely at the detected rate and is only
        aligned by an explicit Reset. Treat `beat` as a metronome locked to the right
        speed, not as an onset detector.
        """
        now = time.time()
        dt = 0.0 if self._last_cook is None else max(0.0, now - self._last_cook)
        self._last_cook = now

        prev = self.phase
        if self.bpm > 0:
            self.phase = (self.phase + dt * self.bpm / 60.0) % 1.0
        self._beat = 1.0 if self.phase < prev else 0.0

    def _write(self, scriptOp):
        for name in CHANNELS:
            scriptOp.appendChan(name)
        scriptOp.numSamples = 1
        scriptOp["bpm"][0] = self.bpm
        scriptOp["confidence"][0] = self.confidence
        scriptOp["beat"][0] = self._beat
        scriptOp["phase"][0] = self.phase

        p = getattr(self.ownerComp.par, "Status", None)
        if p is not None:
            p.val = self.error.splitlines()[0] if self.error else self.status

    # -- tempo sync -------------------------------------------------------

    def SyncTempo(self) -> bool:
        """Write the detected BPM to the project timeline tempo."""
        if self.bpm <= 0:
            return False
        return set_project_tempo(self.bpm)

    def OnAutosync(self):
        if self._par("Autosync", False) and self.bpm > 0:
            minimum = float(self._par("Autosyncconfidence", 0.0))
            if self.confidence >= minimum:
                self.SyncTempo()


def set_project_tempo(bpm: float) -> bool:
    """Set the timeline tempo, trying the access paths TD has used across versions."""
    # Qualified through `td` rather than relying on the globals TouchDesigner injects
    # into DAT modules, so this works however the extension gets imported.
    import td

    attempts = []

    for path in ("/local/time", "/time"):
        try:
            comp = td.op(path)
            if comp is not None and hasattr(comp.par, "tempo"):
                comp.par.tempo = float(bpm)
                return True
        except Exception as exc:
            attempts.append(f"{path}: {exc}")

    try:
        td.root.time.tempo = float(bpm)
        return True
    except Exception as exc:
        attempts.append(f"root.time.tempo: {exc}")

    print("[AutoBpm] could not set project tempo: " + "; ".join(attempts))
    return False


# ---------------------------------------------------------------------------
# detector adapters
# ---------------------------------------------------------------------------


class _SidecarDetector:
    """Runs the model in a child process."""

    def __init__(self, python, sample_rate, cwd=None, **options):
        self.client = SidecarClient(
            python, sample_rate=sample_rate, cwd=cwd, autorestart=True, **options
        )

    def start(self):
        self.client.start()

    def stop(self):
        self.client.stop()

    def reset(self):
        self.client.reset()

    def push(self, block):
        self.client.send_audio(block)

    def poll(self):
        results = self.client.poll()
        return results[-1] if results else None

    def error(self):
        return self.client.last_error


class _InProcessDetector:
    """Runs the model on a worker thread inside TouchDesigner.

    Inference is kept off the cook thread; the cook only hands over samples and picks
    up whatever estimate is ready.
    """

    def __init__(self, sample_rate, **options):
        self.sample_rate = int(sample_rate)
        self.options = options
        self._in: "queue.Queue" = queue.Queue(maxsize=256)
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._error = ""
        self._reset = threading.Event()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tdautobpm", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def reset(self):
        self._reset.set()

    def push(self, block):
        try:
            self._in.put_nowait(block)
        except queue.Full:
            # Drop the oldest block: stale audio is worthless for live tempo.
            try:
                self._in.get_nowait()
                self._in.put_nowait(block)
            except queue.Empty:
                pass

    def poll(self):
        with self._lock:
            latest, self._latest = self._latest, None
        return latest

    def error(self):
        return self._error

    def _run(self):
        try:
            import numpy as np

            from tdautobpm.engine import make_predictor

            predictor = make_predictor(
                input_sample_rate=self.sample_rate, **self.options
            )
        except Exception as exc:
            self._error = f"{exc}\n{traceback.format_exc()}"
            return

        while not self._stop.is_set():
            if self._reset.is_set():
                self._reset.clear()
                predictor.reset()
            try:
                block = self._in.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                out = predictor.push_audio(np.asarray(block, dtype="float32"))
            except Exception as exc:
                self._error = f"{exc}\n{traceback.format_exc()}"
                continue
            if out is not None:
                bpm, conf, win = out
                with self._lock:
                    self._latest = {"bpm": bpm, "confidence": conf, "window_s": win}
