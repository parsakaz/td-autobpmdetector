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

#: Taps further apart than this start a new tap sequence (i.e. below 30 BPM).
TAP_TIMEOUT = 2.0

#: How many recent taps the tapped tempo is averaged over.
TAP_HISTORY = 8

#: Taps closer together than this are one tap: a key held down auto-repeats, and a
#: bouncing button can fire twice.
TAP_DEBOUNCE = 0.1

#: How far the Half and Double buttons can take the multiplier.
MULTIPLIER_RANGE = (0.25, 4.0)


def tap_tempo(times):
    """Tempo and consistency from a sequence of tap times in seconds.

    Returns ``(bpm, confidence)``, or None with fewer than two taps. The median interval
    keeps one fumbled tap from dragging the tempo; the intervals near it are then
    averaged for precision. Confidence is how evenly the taps were spaced, scaled down
    until there are enough of them to mean anything.
    """
    intervals = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not intervals:
        return None
    median = sorted(intervals)[len(intervals) // 2]
    near = [i for i in intervals if abs(i - median) <= 0.25 * median]
    mean = sum(near) / len(near)
    spread = (sum((i - mean) ** 2 for i in intervals) / len(intervals)) ** 0.5 / mean
    confidence = max(0.0, 1.0 - 4.0 * spread) * min(1.0, len(intervals) / 3.0)
    return 60.0 / mean, confidence


#: What the panel offers when the presets file is missing or has nothing usable.
FALLBACK_PRESETS = [("Any", 60.0, 240.0)]


def read_presets(text):
    """Parse the presets file into ``([(name, lowest, highest), ...], problems)``.

    The file is meant to be edited by people who have never seen a CSV spec, in
    whatever they have to hand, so this accepts what spreadsheet apps and text
    editors actually save: commas, semicolons (Excel in much of Europe) or tabs;
    decimal commas; a byte-order mark; a header row; blank lines and ``#`` notes;
    and the two tempos in either order. Anything else is reported by line number
    rather than silently dropped.
    """
    import csv

    rows = [
        (number, line)
        for number, line in enumerate(text.lstrip("\ufeff").splitlines(), 1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    try:
        dialect = csv.Sniffer().sniff("\n".join(l for _, l in rows[:10]), ",;\t")
    except csv.Error:
        dialect = csv.excel

    presets, problems = [], []
    for index, (number, line) in enumerate(rows):
        cells = next(csv.reader([line], dialect))
        if len(cells) < 3:  # a hand-typed line with a different separator
            for delimiter in ";\t,":
                split = next(csv.reader([line], delimiter=delimiter))
                if len(split) >= 3:
                    cells = split
                    break
        cells = [c.strip() for c in cells] + [""] * 3
        name = cells[0]
        try:
            low, high = (float(c.replace(",", ".")) for c in cells[1:3])
        except ValueError:
            if index > 0:  # the first row may be a header
                problems.append("line %d: expected a name and two tempos, got %r"
                                % (number, line.strip()))
            continue
        low, high = min(low, high), max(low, high)
        if not name:
            problems.append("line %d: the preset has no name" % number)
        elif low <= 0 or low == high:
            problems.append("line %d: %s needs two different tempos above 0"
                            % (number, name))
        else:
            presets.append((name, low, high))
    return presets, problems


class _Value:
    """Stand-in for `tdu.Dependency` outside TouchDesigner: just holds `.val`."""

    def __init__(self, val):
        self.val = val


def _dependency(val):
    """A value that UI expressions can depend on.

    `tdu.Dependency` makes an expression that reads `.val` re-evaluate when it
    changes, which a plain attribute would not. Imported here rather than at module
    scope so the module still loads without TouchDesigner.
    """
    try:
        import tdu

        return tdu.Dependency(val)
    except ImportError:
        return _Value(val)


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


#: The stdlib-only modules the TouchDesigner side needs, in dependency order. They
#: are embedded in the component as Text DATs, so a downloaded .tox works on its own.
SUPPORT_MODULES = ("protocol", "client", "envresolve", "envsetup")

#: Package name the embedded copies are registered under. Not `tdautobpm`, which
#: belongs to a real installation, if there is one.
EMBEDDED_PACKAGE = "tdautobpm_embedded"


def load_support(repo=None, owner=None):
    """Return the helper modules: ``(envresolve, SidecarClient, envsetup)``.

    From `<repo>/src` when there is a checkout, so that editing the source and
    rebuilding picks the changes up; otherwise from the copies inside the component.

    Deliberately not done at module scope: where to load from depends on parameters
    of the owning component, which do not exist until the extension is constructed.
    """
    if repo:
        src = os.path.join(repo, "src")
        if src not in sys.path:
            sys.path.insert(0, src)

        from tdautobpm import envresolve, envsetup
        from tdautobpm.client import SidecarClient

        return envresolve, SidecarClient, envsetup

    return load_embedded(owner)


def load_embedded(owner):
    """Build the helper modules from the Text DATs inside the component.

    Executed into a package of their own so that `client`'s ``from . import
    protocol`` resolves to the embedded copy rather than anything installed.
    """
    import types

    if owner is None:
        raise RuntimeError("no component to load the embedded modules from")

    package = sys.modules.get(EMBEDDED_PACKAGE)
    if package is None:
        package = types.ModuleType(EMBEDDED_PACKAGE)
        package.__path__ = []  # a package with nothing on disk
        sys.modules[EMBEDDED_PACKAGE] = package

    for name in SUPPORT_MODULES:
        full = EMBEDDED_PACKAGE + "." + name
        if full in sys.modules:
            continue
        dat = owner.op("lib/" + name)
        if dat is None:
            raise RuntimeError(
                "this component has no lib/%s; it predates the self-contained build. "
                "Set Repo Path to a checkout, or import a newer .tox." % name
            )
        module = types.ModuleType(full)
        module.__package__ = EMBEDDED_PACKAGE
        sys.modules[full] = module
        try:
            exec(compile(dat.text, "<%s>" % dat.path, "exec"), module.__dict__)
        except Exception:
            del sys.modules[full]
            raise
        setattr(package, name, module)

    return (sys.modules[EMBEDDED_PACKAGE + ".envresolve"],
            sys.modules[EMBEDDED_PACKAGE + ".client"].SidecarClient,
            sys.modules[EMBEDDED_PACKAGE + ".envsetup"])


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
        self._S = None
        self._SidecarClient = None
        self._resolve_repo()

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

        # Tap tempo. `_source` records where the current bpm came from, so a tapped
        # tempo can be told apart from one held over from detection.
        self._taps = []
        self._tap_count = 0
        self._force_beat = False
        self._source = "auto"

        # For the control panel: which state to show, and beats counted since the
        # last Reset or tap, for the beat-in-bar display. Not CHOP channels, because
        # neither is an output of the detector.
        self.Mode = _dependency("stopped")
        self.BeatCount = _dependency(0)

        # Hold: below the confidence threshold, keep reporting the last tempo that
        # was above it. `RawBpm` is the detector's latest estimate regardless, so the
        # panel can show what it is hearing while the output holds.
        self.RawBpm = _dependency(0.0)
        self.holding = False
        self._trusted = False  # whether there is a confident tempo to hold

        # Environment setup, for a component that arrived without one.
        self.setup = None
        self.needs_env = False
        self.env_error = ""

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
            range_min=float(self._par("Rangemin", 0.0)),
            range_max=float(self._par("Rangemax", 0.0)),
        )

    @property
    def OutputBpm(self) -> float:
        """The tempo this component outputs: the estimate times the multiplier."""
        return self.bpm * float(self._par("Multiplier", 1.0) or 1.0)

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
            self.needs_env = False
        except self._E.EnvError as exc:
            # Nothing on this machine can run the detector yet. That is a job for
            # SetupEnv, not an error to stare at, so the panel says so plainly and
            # the detail goes where someone looking for it will find it.
            self.needs_env = True
            self.env_error = str(exc)
            print("[AutoBpm] " + self.env_error)
            self._fail("no Python environment yet - press Install in the panel")
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

    def Configure(self):
        """Send the current settings to a running detector without restarting it.

        Everything but the runtime, environment and torch device applies live, so the
        range can be dragged while the music plays and the evidence gathered so far
        is kept.
        """
        if self.detector is not None:
            self.detector.configure(**self._settings())
        # A tempo held from before a range change may not fit the new range; let
        # the next estimate through rather than holding it indefinitely.
        low, high = self._par("Rangemin", 0.0), self._par("Rangemax", 0.0)
        if self._trusted and low and high and not low <= self.bpm <= high:
            self._trusted = False

    def Half(self):
        """Halve the output tempo, for when the detector counts double time."""
        self._scale_multiplier(0.5)

    def Double(self):
        """Double the output tempo, for when the detector counts half time."""
        self._scale_multiplier(2.0)

    def _scale_multiplier(self, factor):
        par = getattr(self.ownerComp.par, "Multiplier", None)
        if par is not None:
            low, high = MULTIPLIER_RANGE
            par.val = min(high, max(low, par.eval() * factor))

    def _resolve_repo(self):
        """Load the helper modules, from a checkout if there is one, and point the
        panel's presets at the right file."""
        self.repo = None
        self._E = None
        self._S = None
        self._SidecarClient = None
        try:
            self.repo = find_repo(self._par("Repopath", ""), owner=self.ownerComp)
        except RuntimeError:
            pass  # no checkout: the component carries what this side needs
        try:
            self._E, self._SidecarClient, self._S = load_support(
                self.repo, self.ownerComp)
        except Exception as exc:
            self.error = str(exc)
            self.status = "error"
            print("[AutoBpm] " + self.error)
        self.PointPresets()
        self._check_env()

    @property
    def PresetsPath(self) -> str:
        """The Presets File parameter, else presets.csv in the checkout."""
        path = (self._par("Presetsfile", "") or "").strip()
        if path:
            return os.path.abspath(os.path.expanduser(path))
        if self.repo:
            return os.path.join(self.repo, "touchdesigner", "presets.csv")
        return ""

    def PointPresets(self):
        """Load the panel's preset menu from PresetsPath."""
        dat = self.ownerComp.op("ui/presets_file")
        if dat is not None and dat.par.file.eval() != self.PresetsPath:
            dat.par.file = self.PresetsPath

    def Restart(self):
        """Start over: find the checkout again, then a fresh detector."""
        self._retries = 0
        self._retry_at = 0.0
        self.error = ""
        self._resolve_repo()
        self.Start(self._sample_rate or 44100)

    def Reset(self):
        """Clear accumulated evidence and restart the phase."""
        self.bpm = 0.0
        self.confidence = 0.0
        self.phase = 0.0
        self._taps = []
        self._tap_count = 0
        self._source = "auto"
        self.BeatCount.val = 0
        self.RawBpm.val = 0.0
        self.holding = False
        self._trusted = False
        if self.detector is not None:
            self.detector.reset()

    def Tap(self):
        """Tap tempo. Every tap puts the beat on the tap; two or more set the tempo.

        A single tap only realigns the phase, which is also how to line `beat` up with
        the music while detection runs: the model knows how fast, not where the beat
        falls. Once taps give a tempo, detection is switched off (Active), so the
        tapped tempo holds instead of being overwritten by the next estimate. Switch
        Active back on to resume detection.
        """
        now = time.time()
        if self._taps and now - self._taps[-1] < TAP_DEBOUNCE:
            return
        if self._taps and now - self._taps[-1] > TAP_TIMEOUT:
            self._taps = []
            self._tap_count = 0
        self._taps = (self._taps + [now])[-TAP_HISTORY:]
        self._tap_count += 1

        self.phase = 0.0
        self._last_cook = now
        self._force_beat = True
        self.BeatCount.val = self._tap_count - 1

        estimate = tap_tempo(self._taps)
        if estimate is not None:
            self.bpm, self.confidence = estimate
            self._source = "tap"
            # A tapped tempo is one to hold: after START it stays until the detector
            # is confident of its own.
            self._trusted = True
            self.holding = False
            # What was tapped is the tempo wanted out, so no multiplier applies to it.
            par = getattr(self.ownerComp.par, "Multiplier", None)
            if par is not None and par.eval() != 1.0:
                par.val = 1.0
            if self._par("Active", False):
                self.ownerComp.par.Active = False

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

        # Switching Active off tears the detector down rather than merely starving it,
        # so "stop" actually stops the sidecar process instead of leaving it running
        # on silence.
        if not active:
            if self.detector is not None:
                self.Stop()
                self._retries = 0
                self._retry_at = 0.0
            self._advance_phase(None)
            self._write(scriptOp)
            return

        # Only start once there is real audio. Without this the detector would be
        # built for the input's nominal rate - 60 Hz when nothing is connected, since
        # an unconnected In CHOP reports the frame rate - and a sidecar would be
        # spawned to resample 60 Hz "audio".
        chans = source.chans() if source is not None else []

        # Record what this callback actually sees, for Diagnose(), rather than
        # inferring it from the outcome.
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

        if self.detector is None or rate != self._sample_rate:
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
        if self.detector is not None:
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
            self._take_estimate(latest["bpm"], latest["confidence"])

        err = self.detector.error()
        if err:
            self.error = err
            self.status = "error"

    def _take_estimate(self, bpm, confidence):
        """Accept a new estimate from the detector, or hold the last confident one.

        With Hold on, an estimate below the confidence threshold does not replace one
        that was above it: in a breakdown or a mix the detector loses the beat, and a
        tempo drifting around is worse than the last one it was sure of. Until there
        is a confident tempo there is nothing to hold, so estimates pass straight
        through.
        """
        self.RawBpm.val = bpm
        self.confidence = confidence
        confident = confidence >= float(self._par("Autosyncconfidence", 0.0))
        if confident or not self._par("Hold", False) or not self._trusted:
            self.bpm = bpm
            self._source = "auto"
            self._trusted = self._trusted or confident
            self.holding = False
        else:
            self.holding = True

    def _advance_phase(self, dt=None):
        """Free-running beat phase at the detected tempo.

        The model estimates *tempo*, not beat position - it has no notion of where a
        downbeat falls. So this phase runs freely at the detected rate and is only
        aligned by an explicit Reset. Treat `beat` as a metronome locked to the right
        speed, not as an onset detector.

        `dt` comes from the audio time slice when there is input, which keeps the
        phase locked to the audio clock rather than jittering with frame times.
        """
        # The clock is stamped on every cook, audio-timed or not. Otherwise the first
        # wall-clock cook after a stretch on audio time - e.g. right after a tap
        # switches detection off - would advance by that whole stretch at once.
        now = time.time()
        if dt is None:
            dt = 0.0 if self._last_cook is None else max(0.0, now - self._last_cook)
        self._last_cook = now

        prev = self.phase
        bpm = self.OutputBpm
        if bpm > 0:
            self.phase = (self.phase + dt * bpm / 60.0) % 1.0
        wrapped = self.phase < prev
        if wrapped:
            self.BeatCount.val += 1
        # A tap puts the beat on the tap itself; it has already counted that beat.
        self._beat = 1.0 if (wrapped or self._force_beat) else 0.0
        self._force_beat = False

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
        scriptOp["bpm"].vals = [self.OutputBpm] * n
        scriptOp["confidence"].vals = [self.confidence] * n
        scriptOp["phase"].vals = [self.phase] * n

        # `beat` is an impulse, so it marks one sample rather than the whole slice.
        beat = [0.0] * n
        if self._beat:
            beat[0] = 1.0
        scriptOp["beat"].vals = beat

        self._publish_status()

    def _publish_status(self):
        """Show the current status and mode, without churning dependents.

        Only on change: writing a parameter on every cook makes the component's
        Parameter Execute DAT re-evaluate inside the same cook pass, which
        TouchDesigner reports as "Cook dependency loop detected".
        """
        text = self.error.splitlines()[0] if self.error else self.status
        if text != self._last_status:
            self._last_status = text
            par = getattr(self.ownerComp.par, "Status", None)
            if par is not None:
                par.val = text
            # Echoed to the textport too: Status is read-only and easy to miss, and a
            # silent component is hard to debug.
            print("[AutoBpm] " + text)

        mode = self._mode()
        if mode != self.Mode.val:  # only on change, so dependents do not recook
            self.Mode.val = mode

    def _mode(self) -> str:
        """What the panel should show: auto, hold, waiting, error, tap or stopped."""
        if not self._par("Active", True):
            return "tap" if self._source == "tap" and self.bpm > 0 else "stopped"
        if self.setup is not None:
            return "setup"
        if self.needs_env:
            return "noenv"
        if self.error or self.status == "error":
            return "error"
        if self.status.startswith("waiting"):
            return "waiting"
        return "hold" if self.holding else "auto"

    def Tick(self):
        """Force the detector to run for this frame.

        Time Slice mode guarantees a CHOP *receives* a time slice when it cooks, but
        it does not make it cook: a CHOP cooks only when something pulls on it, and
        with nothing connected downstream and no viewer open, nothing does. An
        Execute DAT calls this from onFrameStart, so audio reaches the detector
        whether or not anyone consumes the output.
        """
        # Cook the *end* of the chain, not the middle. Forcing `detect` directly
        # cooks it without necessarily having pulled `audio_in` for this frame, which
        # can hand the callback an input with no channels yet. Cooking the output CHOP
        # pulls detect, which pulls audio_in, in the normal order.
        self._poll_setup()

        target = self.ownerComp.op("bpm_out") or self.ownerComp.op("detect")
        if target is not None:
            source = self.ownerComp.op("audio_in")
            if source is not None:
                source.cook(force=True)
            target.cook(force=True)

    # -- environment setup -------------------------------------------------

    @property
    def EnvFolder(self) -> str:
        """Where SetupEnv builds an environment."""
        folder = (self._par("Envfolder", "") or "").strip()
        if folder:
            return os.path.abspath(os.path.expanduser(folder))
        return self._S.default_env_dir() if self._S else ""

    def _check_env(self):
        """Note whether anything here can run the detector.

        Done up front, so a component dropped into a project says "set up" straight
        away rather than waiting for audio to be wired before finding out.
        """
        if self._E is None:
            return
        spec = (self._par("Envpath", "") or "").strip() or None
        try:
            self._E.resolve(spec, project_root=self.repo)
            self.needs_env = False
        except Exception:
            self.needs_env = True
            self.status = "no Python environment yet - press Install in the panel"

    def SetupEnv(self):
        """Build a Python environment for the detector and install it there.

        Everything runs in the background: the download is hundreds of megabytes,
        and TouchDesigner has frames to draw. Progress lands in Status, the whole
        transcript in the setup_log DAT.
        """
        if self.setup is not None and not self.setup.done:
            return
        if self._S is None:
            self._fail("the component's setup helper is missing; set Repo Path")
            return

        package = (self._par("Package", "") or "").strip()
        if not package:
            self._fail("no package to install: fill in the Package parameter")
            return

        chosen = (self._par("Basepython", "") or "").strip()
        if chosen:
            found = [(chosen, "", False)]
        else:
            found = self._S.find_base_interpreters()
        if not found:
            self._fail(
                "no Python to build an environment on. Install Python 3.11 or 3.12 "
                "from python.org, or point Base Python at one."
            )
            return

        base, version, is_host = found[0]
        folder = self.EnvFolder
        self._log("[setup] " + self._S.describe_plan(folder, base, version, is_host))
        self.Stop()
        self.setup = _Setup(self._S.create_steps(folder, base, package),
                            on_line=self._log)
        self.setup.is_host = is_host
        self.setup.env_dir = folder
        self.setup.start()
        self.status = "setting up: " + self.setup.label

    def CancelSetup(self):
        if self.setup is not None:
            self.setup.cancel()

    def _poll_setup(self):
        """Follow a running setup; called every frame from Tick."""
        setup = self.setup
        if setup is None:
            return
        setup.poll()
        if not setup.done:
            self.status = "setting up: %s %s" % (setup.label, setup.progress)
            self._publish_status()
            return

        self.setup = None
        if setup.failed:
            self.needs_env = True
            self._fail("setup failed: " + (setup.error or "see the setup_log DAT"))
            return

        # Point the component at what was just built. An environment on
        # TouchDesigner's own Python can only be used from inside TouchDesigner, so
        # it has to run in-process.
        par = getattr(self.ownerComp.par, "Envpath", None)
        if par is not None:
            par.val = setup.env_dir
        if setup.is_host:
            runtime = getattr(self.ownerComp.par, "Runtime", None)
            if runtime is not None:
                runtime.val = "inprocess"
        self._log("[setup] done: " + setup.env_dir)
        self.needs_env = False
        self.Restart()

    def _log(self, line):
        """Append to the setup transcript, and echo it to the textport."""
        print("[AutoBpm] " + line)
        dat = self.ownerComp.op("setup_log")
        if dat is not None:
            dat.text = (dat.text + line + "\n")[-20000:]

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
        if self.needs_env:
            print("[AutoBpm] environment: %s" % (self.env_error or "none found"))
        print("[AutoBpm] runtime:    %s" % self._par("Runtime", "?"))
        print("[AutoBpm] env par:    %r" % (self._par("Envpath", ""),))
        if self.env is not None:
            print("[AutoBpm] env:        %s" % self.env.python)
            print("[AutoBpm]             %s %s, missing=%s"
                  % (self.env.version, self.env.machine, self.env.missing or "nothing"))
        print("[AutoBpm] detector:   %r" % (self.detector,))
        print("[AutoBpm] bpm:        %.2f  confidence %.3f  (x%g)"
              % (self.bpm, self.confidence, self._par("Multiplier", 1.0)))
        print("[AutoBpm] range:      %s-%s" % (self._par("Rangemin"), self._par("Rangemax")))
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
        return set_project_tempo(self.OutputBpm)

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


class _Setup:
    """Runs the setup commands one after another, off the cook thread.

    TouchDesigner must keep drawing while pip downloads torch, so each command runs
    as a child process with a thread draining its output. `poll` is called every
    frame and never blocks.

    The reading thread only collects lines. TouchDesigner objects may be touched from
    the main thread alone, so whatever the caller does with them - writing the log
    DAT, printing to the textport - happens in `poll`.
    """

    def __init__(self, steps, on_line=None):
        self.steps = list(steps)
        self.on_line = on_line or (lambda line: None)
        self._pending = []
        self._lock = threading.Lock()
        self.index = 0
        self.proc = None
        self.thread = None
        self.done = False
        self.failed = False
        self.error = ""
        self.progress = ""
        self.last_line = ""
        self.env_dir = ""
        self.is_host = False
        self._started = 0.0
        self._cancelled = False

    @property
    def label(self) -> str:
        if self.index < len(self.steps):
            return self.steps[self.index][0]
        return "finishing"

    def start(self):
        self._spawn()

    def _spawn(self):
        import subprocess
        import threading

        label, argv = self.steps[self.index]
        self.on_line("[setup] " + label)
        self.on_line("[setup] $ " + " ".join(argv))
        self.progress = ""
        self._started = time.time()
        try:
            self.proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except OSError as exc:
            self._fail("could not run %s: %s" % (argv[0], exc))
            return

        def drain(stream):
            for line in stream:
                line = line.rstrip()
                if line:
                    with self._lock:
                        self._pending.append(line)

        self.thread = threading.Thread(
            target=drain, args=(self.proc.stdout,), daemon=True)
        self.thread.start()

    def _flush(self):
        """Hand the reader's lines to the caller, on the calling (main) thread."""
        with self._lock:
            lines, self._pending = self._pending, []
        for line in lines:
            self.last_line = line
            self.on_line(line)

    def poll(self):
        """Move the run along. Returns once there is nothing to do this frame."""
        if self.done or self.proc is None:
            return
        self._flush()
        code = self.proc.poll()
        if code is None:
            # Downloading torch takes minutes and pip says little while it does, so
            # show the clock as well as whatever it last said.
            elapsed = int(time.time() - self._started)
            self.progress = ("%ds  %s" % (elapsed, self.last_line[-40:])
                             if self.last_line else "%ds" % elapsed)
            return
        if self._cancelled:
            self._fail("cancelled")
            return
        if code != 0:
            self._fail("%s exited with code %d" % (self.steps[self.index][0], code))
            return

        self.index += 1
        if self.index >= len(self.steps):
            self.done = True
            return
        self._spawn()

    def cancel(self):
        self._cancelled = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        else:
            self._fail("cancelled")

    def _fail(self, message):
        self._flush()
        self.error = message
        self.failed = True
        self.done = True
        self.on_line("[setup] " + message)


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

    def configure(self, **settings):
        self.client.configure(**settings)

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
        self._config = None  # settings waiting for the worker thread to apply

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

    def configure(self, **settings):
        self.options.update(settings)
        with self._lock:
            self._config = dict(settings)

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
            with self._lock:
                config, self._config = self._config, None
            if config:
                predictor.configure(**config)
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
