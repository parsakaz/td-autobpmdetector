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

#: CHOP channels this component outputs.
CHANNELS = ("bpm", "confidence", "beat", "phase")

#: Marker that identifies a directory as the repository root.
_MARKER = os.path.join("src", "tdautobpm", "engine.py")


def find_repo(hint=None, owner=None):
    """Locate the td-autobpmdetector checkout.

    This module cannot use ``__file__``: it is stored as a Text DAT inside the .tox
    and has no file on disk, so ``__file__`` is either undefined or points wherever
    the importing namespace happened to point - in TouchDesigner, into the application
    bundle. Candidates are therefore always confirmed by looking for `src/tdautobpm`.
    """
    candidates = []

    if hint:
        candidates.append(os.path.abspath(os.path.expanduser(os.path.expandvars(hint))))
    if os.environ.get("TDAUTOBPM_REPO"):
        candidates.append(
            os.path.abspath(os.path.expanduser(os.environ["TDAUTOBPM_REPO"]))
        )

    # Walk up from the saved project and from the cwd.
    starts = [os.getcwd()]
    try:
        import td

        if td.project.folder:
            starts.append(td.project.folder)
    except Exception:
        pass
    if owner is not None:
        try:
            ext_dat = owner.op("AutoBpmExt")
            if ext_dat is not None and ext_dat.par.file.eval():
                starts.append(os.path.dirname(os.path.abspath(ext_dat.par.file.eval())))
        except Exception:
            pass

    for start in starts:
        directory = os.path.abspath(start)
        for _ in range(6):
            candidates.append(directory)
            parent = os.path.dirname(directory)
            if parent == directory:
                break
            directory = parent

    for directory in candidates:
        if os.path.isfile(os.path.join(directory, _MARKER)):
            return directory

    raise RuntimeError(
        "Could not find the td-autobpmdetector checkout (no src/tdautobpm/engine.py "
        "under any candidate). Set the component's Repo Path parameter to the "
        "repository root, or the TDAUTOBPM_REPO environment variable."
    )


def load_support(repo):
    """Put `<repo>/src` on sys.path and return the stdlib-only helper modules.

    Deliberately not done at module scope: the repository location comes from a
    parameter on the owning component, which does not exist until the extension is
    constructed.
    """
    src = os.path.join(repo, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from tdautobpm import envresolve
    from tdautobpm.client import SidecarClient

    return envresolve, SidecarClient


class AutoBpm:
    """Extension class for the AutoBpm COMP."""

    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.detector = None
        self.env = None
        self.status = "idle"
        self.error = ""

        self.repo = None
        self._E = None
        self._SidecarClient = None
        try:
            self.repo = find_repo(self._par("Repopath", ""), owner=ownerComp)
            self._E, self._SidecarClient = load_support(self.repo)
        except Exception as exc:
            self.error = str(exc)
            self.status = "error"
            print("[AutoBpm] " + self.error)

        self.bpm = 0.0
        self.confidence = 0.0
        self.phase = 0.0
        self._beat = 0.0
        self._last_cook = None
        self._retries = 0
        self._retry_at = 0.0
        self._last_status = ""
        self._cooks = 0
        self._seen = None

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

        if self._E is None:
            # Repo never resolved; the message from __init__ still stands.
            return

        runtime = self._par("Runtime", "sidecar")
        spec = (self._par("Envpath", "") or "").strip() or None

        try:
            self.env = self._E.resolve(spec, project_root=self.repo)
        except self._E.EnvError as exc:
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
        why = self._E.sidecar_incompatibility(self.env)
        if why:
            raise RuntimeError(f"environment cannot host the sidecar: {why}")

        self.detector = _SidecarDetector(
            self._SidecarClient, self.env.python, sample_rate=self._sample_rate,
            cwd=self.repo, **self._settings()
        )
        self.detector.start()
        self.status = "running (sidecar, %s)" % self.env.version

    def _start_inprocess(self):
        why = self._E.inprocess_incompatibility(self.env)
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
        self._retries = 0
        self._retry_at = 0.0
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
        self._cooks += 1
        scriptOp.clear()
        source = scriptOp.inputs[0] if scriptOp.inputs else None

        active = bool(self._par("Active", True))

        # Only start once there is real audio. Without this the detector would be
        # built for the input's nominal rate - 60 Hz when nothing is connected, since
        # an unconnected In CHOP reports the frame rate - and a sidecar would be
        # spawned to resample 60 Hz "audio".
        chans = source.chans() if source is not None else []

        # Cook() and Diagnose() were disagreeing about the same input, so record what
        # this callback actually sees rather than inferring it from the outcome.
        self._seen = {
            "n_inputs": len(scriptOp.inputs),
            "source": source.path if source is not None else None,
            "chans": len(chans),
            "rate": (source.rate if source is not None else None),
            "samples": (source.numSamples if source is not None else None),
        }

        if not chans or not source.rate or source.rate < 1000:
            if not self.error:
                self.status = (
                    "waiting for audio input - wire an Audio Device In CHOP into this "
                    "component"
                )
            self._advance_phase(None)
            self._write(scriptOp)
            return

        rate = int(source.rate)

        if active and (self.detector is None or rate != self._sample_rate):
            # Backoff matters: Start() sets detector to None when it fails, so without
            # it a failing start would relaunch a sidecar process every single frame.
            if time.time() >= self._retry_at:
                self._retries += 1
                self.Start(rate)
                if self.detector is None:
                    self._retry_at = time.time() + min(30.0, 2.0 ** min(self._retries, 4))
                else:
                    self._retries = 0

        dt = None
        if active and self.detector is not None and source is not None:
            try:
                self._pump(source)
                if source.numSamples > 0 and rate:
                    dt = source.numSamples / float(rate)
            except Exception as exc:
                self._fail(f"{exc}\n{traceback.format_exc()}")

        self._advance_phase(dt)
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

    def _advance_phase(self, dt=None):
        """Free-running beat phase at the detected tempo.

        The model estimates *tempo*, not beat position - it has no notion of where a
        downbeat falls. So this phase runs freely at the detected rate and is only
        aligned by an explicit Reset. Treat `beat` as a metronome locked to the right
        speed, not as an onset detector.

        `dt` comes from the audio time slice when there is input, which keeps the
        phase locked to the audio clock rather than jittering with frame times.
        """
        if dt is None:
            now = time.time()
            dt = 0.0 if self._last_cook is None else max(0.0, now - self._last_cook)
            self._last_cook = now

        prev = self.phase
        if self.bpm > 0:
            self.phase = (self.phase + dt * self.bpm / 60.0) % 1.0
        self._beat = 1.0 if self.phase < prev else 0.0

    def _write(self, scriptOp):
        """Fill the current time slice with the latest estimate.

        Time Slice mode stays on and `numSamples` is left alone. Setting it warns
        ("Editing numSamples is not supported in Time Slice mode"), and turning Time
        Slice off would be worse: a time-sliced CHOP is guaranteed to cook every
        frame, which is what keeps the audio stream unbroken. A non-time-sliced one
        cooks only on demand, so any frame that did not cook would punch a hole in the
        audio going to the detector.
        """
        for name in CHANNELS:
            scriptOp.appendChan(name)

        n = max(1, scriptOp.numSamples)
        scriptOp["bpm"].vals = [self.bpm] * n
        scriptOp["confidence"].vals = [self.confidence] * n
        scriptOp["phase"].vals = [self.phase] * n

        # `beat` is an impulse, so it marks one sample rather than the whole slice.
        beat = [0.0] * n
        if self._beat:
            beat[0] = 1.0
        scriptOp["beat"].vals = beat

        text = self.error.splitlines()[0] if self.error else self.status
        par = getattr(self.ownerComp.par, "Status", None)
        if par is not None:
            par.val = text
        # Echo to the textport as well, once per change: the Status parameter is
        # read-only and easy to miss, and a silent component is hard to debug.
        if text != self._last_status:
            self._last_status = text
            print("[AutoBpm] " + text)

    def Tick(self):
        """Force the detector to run for this frame.

        Time Slice mode guarantees a CHOP *receives* a time slice when it cooks, but
        it does not make it cook: a CHOP cooks only when something pulls on it. With
        nothing connected downstream and no viewer open, the Script CHOP never ran at
        all, so no audio ever reached the detector. An Execute DAT calls this from
        onFrameStart so ingestion does not depend on anyone consuming the output.
        """
        # Cook the *end* of the chain, not the middle. Forcing `detect` directly
        # cooks it without necessarily having pulled `audio_in` for this frame, which
        # can hand the callback an input with no channels yet. Cooking the output CHOP
        # pulls detect, which pulls audio_in, in the normal order.
        target = self.ownerComp.op("bpm_out") or self.ownerComp.op("detect")
        if target is not None:
            source = self.ownerComp.op("audio_in")
            if source is not None:
                source.cook(force=True)
            target.cook(force=True)

    def Diagnose(self):
        """Print what the component can see. Call from the textport:

            op('/project1/AutoBpm').Diagnose()

        Forces a cook first, so the reported status reflects the network as it is now
        rather than whenever the component last happened to cook.
        """
        before = self._cooks
        try:
            self.Tick()
        except Exception as exc:
            print("[AutoBpm] Tick() raised: %s" % exc)
        print("[AutoBpm] cooks:      %d total, %d from this forced Tick"
              % (self._cooks, self._cooks - before))
        if self._cooks == 0:
            print("[AutoBpm] Cook() has NEVER run - the Script CHOP is not cooking.")

        print("[AutoBpm] Cook sees:  %s" % (self._seen,))
        print("[AutoBpm] repo:       %s" % self.repo)
        print("[AutoBpm] status:     %s" % self.status)
        print("[AutoBpm] error:      %s" % (self.error or "(none)"))
        print("[AutoBpm] runtime:    %s" % self._par("Runtime", "?"))
        print("[AutoBpm] env par:    %r" % (self._par("Envpath", ""),))
        if self.env is not None:
            print("[AutoBpm] env:        %s" % self.env.python)
            print("[AutoBpm]             %s %s, missing=%s"
                  % (self.env.version, self.env.machine, self.env.missing or "nothing"))
        print("[AutoBpm] detector:   %r" % (self.detector,))
        print("[AutoBpm] bpm:        %.2f  confidence %.3f" % (self.bpm, self.confidence))
        print("[AutoBpm] input rate: %s" % self._sample_rate)

        source = None
        script = self.ownerComp.op("detect")
        if script is not None and script.inputs:
            source = script.inputs[0]
        if source is None:
            print("[AutoBpm] NO AUDIO INPUT. Wire an Audio Device In CHOP into this "
                  "component's input.")
        else:
            chans = source.chans()
            print("[AutoBpm] source:     %s  %d chan, %d samples @ %s Hz"
                  % (source.path, len(chans), source.numSamples, source.rate))
            if not chans:
                print("[AutoBpm] NOTHING CONNECTED to this component's input. Wire an "
                      "Audio Device In CHOP into it.")
            elif not source.rate or source.rate < 1000:
                print("[AutoBpm] input rate is %s Hz, which is a frame rate, not an "
                      "audio rate - the input is not audio." % source.rate)
            elif source.numSamples:
                peak = max(abs(v) for v in chans[0].vals)
                print("[AutoBpm] peak level: %.4f%s"
                      % (peak, "  (SILENT - check the device)" if peak < 1e-6 else ""))

        # Internal wiring: a component imported from an older .tox will be missing
        # pieces that a later build added.
        children = sorted(c.name for c in self.ownerComp.children)
        print("[AutoBpm] children:   %s" % ", ".join(children))
        for name in ("detect", "frame_exec", "par_exec", "AutoBpmExt"):
            if self.ownerComp.op(name) is None:
                print("[AutoBpm] MISSING %s - this component predates the current "
                      "build_component.py. Rebuild and re-import the .tox." % name)

        fe = self.ownerComp.op("frame_exec")
        if fe is not None:
            pars = {p.name: p.eval() for p in fe.pars("*")}
            interesting = {k: v for k, v in pars.items()
                           if k in ("active", "framestart", "onframestart", "file")}
            print("[AutoBpm] frame_exec: %s" % interesting)
            if not (pars.get("framestart") or pars.get("onframestart")):
                print("[AutoBpm] frame_exec is not set to fire on frame start; that is "
                      "why nothing cooks. Available pars: %s"
                      % ", ".join(sorted(pars)))

        detector = self.detector
        client = getattr(detector, "client", None)
        if client is not None:
            print("[AutoBpm] sidecar:    alive=%s connected=%s pending=%sB dropped=%sB"
                  % (client.alive, client.connected, client.pending_bytes,
                     client.dropped_bytes))
            print("[AutoBpm] last error: %s" % (client.last_error or "(none)"))

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

    def __init__(self, client_cls, python, sample_rate, cwd=None, **options):
        self.client = client_cls(
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
