"""Build the AutoBpm component and save it as AutoBpm.tox.

Run this *inside TouchDesigner*, from the textport. `run()` takes a string of Python
*code*, not a file path, so use exec::

    p = "/full/path/to/td-autobpmdetector/touchdesigner/build_component.py"
    exec(open(p).read(), {"__file__": p})

Passing ``__file__`` matters: without it this script cannot tell where it lives.
TouchDesigner's textport namespace already defines ``__file__``, pointing into the
application bundle, so a bare ``exec(open(p).read())`` looks for AutoBpmExt.py inside
``TouchDesigner.app`` and fails. If you would rather not pass it, set the environment
variable ``TDAUTOBPM_REPO`` to the repository root instead.

By default the component is built at ``/AutoBpm``. To build it inside an existing
network instead - handy while iterating, since it avoids re-importing the .tox - pass
``PARENT``::

    exec(open(p).read(), {"__file__": p, "PARENT": "/project1"})

Alternatively, drag the file into a Text DAT and run that::

    op('/path/to/thisDAT').run()

It creates ``/AutoBpm`` (deleting any previous one), wires it up, and saves
``touchdesigner/AutoBpm.tox`` next to this file. Import that .tox into any project.

The component is built by script rather than shipped as a hand-made binary so the
network is reviewable and diffable in git: .toe/.tox files are opaque blobs.

Outputs, as CHOP channels:
    bpm         detected tempo, 0 until the first estimate
    confidence  posterior mass near the estimate, 0..1
    beat        1 on the cook where the phase wraps, else 0
    phase       free-running 0..1 beat phase at the detected tempo
"""

import os

try:
    import td
except ImportError:  # pragma: no cover - guard for accidental non-TD execution
    raise SystemExit("build_component.py must be run inside TouchDesigner")


def _td(name):
    """Fetch a TouchDesigner type or function from the `td` module.

    Everything is resolved through `td` rather than relying on the names
    TouchDesigner injects into the executing namespace, because `exec`ing this file
    with an explicit globals dict - the documented way to pass `__file__` - would not
    carry those injected names.
    """
    obj = getattr(td, name, None)
    if obj is None:
        raise RuntimeError(
            f"td.{name} is unavailable; this TouchDesigner build may be too old"
        )
    return obj


def _append(page, kind, name, label, default, **attrs):
    """Append a custom parameter and give it a value as well as a default.

    `append*` creates the parameter already holding a zero/empty value, so setting
    only `.default` afterwards leaves the *current* value untouched. That is how
    Active ended up switched off and Update Rate at 0.0 on a freshly built component:
    the defaults read correctly in the UI while the live values were all zero.
    """
    par = getattr(page, "append" + kind)(name, label=label)[0]
    read_only = attrs.pop("readOnly", None)
    for key, value in attrs.items():  # menu names must exist before a value is set
        setattr(par, key, value)
    par.default = default
    par.val = default
    if read_only is not None:  # applied last, so it cannot block the assignment above
        par.readOnly = read_only
    return par


def _set_par(owner, candidates, value, required=True):
    """Set the first parameter on `owner` whose name matches one of `candidates`.

    Parameter names on built-in operators are not stable across TouchDesigner
    versions, and guessing wrong aborts the whole build. So try the known spellings,
    and when none match, report what the operator actually has instead of raising an
    AttributeError with no context.
    """
    for name in candidates:
        par = getattr(owner.par, name, None)
        if par is not None:
            par.val = value
            return name

    available = sorted(p.name for p in owner.pars("*"))
    message = (
        f"{owner.path}: none of {candidates} exist. Available: {', '.join(available)}"
    )
    if required:
        raise RuntimeError(message)
    print("[AutoBpm] " + message)
    return None


def _locate_here():
    """Find the directory holding this script and AutoBpmExt.py.

    `__file__` is unreliable here: under `exec(open(path).read())` it is whatever the
    calling namespace already had, which in the textport is a path inside the
    TouchDesigner application bundle. So every candidate is confirmed by checking that
    AutoBpmExt.py actually sits next to it.
    """
    candidates = []

    env = os.environ.get("TDAUTOBPM_REPO")
    if env:
        env = os.path.abspath(os.path.expanduser(env))
        candidates.append(("$TDAUTOBPM_REPO", os.path.join(env, "touchdesigner")))
        candidates.append(("$TDAUTOBPM_REPO", env))

    try:
        candidates.append(("__file__", os.path.dirname(os.path.abspath(__file__))))
    except NameError:
        pass

    candidates.append(("cwd", os.path.abspath("touchdesigner")))
    candidates.append(("cwd", os.getcwd()))

    for _, directory in candidates:
        if os.path.isfile(os.path.join(directory, "AutoBpmExt.py")):
            return directory

    tried = "\n".join(f"    {origin}: {d}" for origin, d in candidates)
    raise RuntimeError(
        "Could not find AutoBpmExt.py. Looked in:\n" + tried + "\n\n"
        "Run this script so it knows where it lives:\n"
        '    p = "/path/to/td-autobpmdetector/touchdesigner/build_component.py"\n'
        '    exec(open(p).read(), {"__file__": p})\n'
        "or set the TDAUTOBPM_REPO environment variable to the repository root."
    )


HERE = _locate_here()
REPO = os.path.dirname(HERE)

COMP_NAME = "AutoBpm"
TOX_PATH = os.path.join(HERE, "AutoBpm.tox")


def build(parent_path="/"):
    parent_op = _td("op")(parent_path)
    if parent_op is None:
        raise RuntimeError(f"no such parent: {parent_path}")

    existing = parent_op.op(COMP_NAME)
    if existing is not None:
        existing.destroy()

    comp = parent_op.create(_td("baseCOMP"), COMP_NAME)
    comp.nodeX, comp.nodeY = 0, 0

    # -- extension ---------------------------------------------------------
    ext = comp.create(_td("textDAT"), "AutoBpmExt")
    with open(os.path.join(HERE, "AutoBpmExt.py")) as f:
        ext.text = f.read()
    ext.nodeX, ext.nodeY = -400, 200

    # Wiring the extension is deliberately left until the end of build(): the
    # extension's __init__ reads custom parameters, so instantiating it before those
    # exist makes it fail on a half-built component.

    # -- parameters --------------------------------------------------------
    page = comp.appendCustomPage("Auto BPM")

    _append(page, "Toggle", "Active", "Active", True)

    _append(page, "Menu", "Runtime", "Runtime", "sidecar",
            menuNames=["sidecar", "inprocess"],
            menuLabels=["Sidecar process", "In-process"])

    # Where the checkout lives. Defaults to wherever this was built from; change it
    # if the .tox is imported on another machine or the repo moves.
    _append(page, "Str", "Repopath", "Repo Path", REPO)

    # Blank auto-detects: $TDAUTOBPM_PYTHON, $CONDA_PREFIX, $VIRTUAL_ENV, then .venv.
    _append(page, "Str", "Envpath", "Python Env", "")

    _append(page, "Menu", "Torchdevice", "Torch Device", "cpu",
            menuNames=["cpu", "mps", "cuda"],
            menuLabels=["CPU", "MPS (Apple)", "CUDA"])

    _append(page, "Float", "Updaterate", "Update Rate (Hz)", 4.0,
            normMin=0.5, normMax=20.0)

    _append(page, "Menu", "Estimate", "Estimator", "local_mean",
            menuNames=["local_mean", "mode", "mean", "median"],
            menuLabels=["Local mean", "Mode", "Mean (biased)", "Median"])

    _append(page, "Float", "Smoothing", "Smoothing", 0.90, normMin=0.0, normMax=0.999)
    _append(page, "Float", "Resetseconds", "Reset Every (s)", 5.0,
            normMin=0.0, normMax=60.0)
    _append(page, "Float", "Lockconfidence", "Lock At Confidence", 0.0,
            normMin=0.0, normMax=1.0)

    page.appendPulse("Reset", label="Reset")
    page.appendPulse("Restartdetector", label="Restart Detector")
    page.appendPulse("Synctempo", label="Sync Tempo")

    _append(page, "Toggle", "Autosync", "Autosync", False)
    _append(page, "Float", "Autosyncconfidence", "Autosync Min Confidence", 0.5,
            normMin=0.0, normMax=1.0)

    _append(page, "Str", "Status", "Status", "idle", readOnly=True)

    # -- network -----------------------------------------------------------
    in_chop = comp.create(_td("inCHOP"), "audio_in")
    in_chop.nodeX, in_chop.nodeY = -400, 0

    script = comp.create(_td("scriptCHOP"), "detect")
    script.nodeX, script.nodeY = -100, 0
    script.inputConnectors[0].connect(in_chop)

    # Creating a Script CHOP auto-creates its own `<name>_callbacks` DAT. Creating
    # another with the same name gets it renamed to `detect_callbacks1`, leaving a
    # stray default DAT in the network, so reuse whatever is already there.
    callbacks = comp.op("detect_callbacks")
    if callbacks is None:
        callbacks = comp.create(_td("textDAT"), "detect_callbacks")
    callbacks.nodeX, callbacks.nodeY = -100, 200
    callbacks.text = CALLBACKS
    script.par.callbacks = callbacks

    stray = comp.op("detect_callbacks1")
    if stray is not None:
        stray.destroy()

    out_chop = comp.create(_td("outCHOP"), "bpm_out")
    out_chop.nodeX, out_chop.nodeY = 200, 0
    out_chop.inputConnectors[0].connect(script)

    # A CHOP cooks only when something pulls on it, so with nothing connected
    # downstream the detector never ran. This drives it every frame instead.
    frame_exec = comp.create(_td("executeDAT"), "frame_exec")
    frame_exec.nodeX, frame_exec.nodeY = -100, 400
    frame_exec.text = FRAME_EXEC
    _set_par(frame_exec, ["framestart", "onframestart"], True)

    par_exec = comp.create(_td("parameterexecuteDAT"), "par_exec")
    par_exec.nodeX, par_exec.nodeY = 200, 200
    _set_par(par_exec, ["op", "ops"], ".")
    _set_par(par_exec, ["pars", "parameters"],
             "Reset Restartdetector Synctempo Runtime Envpath Torchdevice Repopath")
    _set_par(par_exec, ["custom"], True, required=False)
    _set_par(par_exec, ["builtin"], False, required=False)
    _set_par(par_exec, ["valuechange", "onvaluechange"], True)
    _set_par(par_exec, ["onpulse", "pulse", "parpulse"], True)
    par_exec.text = PAR_EXEC

    # -- extension, last, now that its parameters exist --------------------
    # `me.op(...)` rather than a bare `op(...)`: the extension expression is evaluated
    # with `me` bound to this component, so this resolves unambiguously to the child.
    comp.par.extension1 = "me.op('AutoBpmExt').module.AutoBpm(me)"
    comp.par.promoteextension1 = True
    comp.par.reinitextensions.pulse()

    return comp


FRAME_EXEC = '''# Frame callbacks for AutoBpm.
#
# An Execute DAT can fire on many events - onStart, onCreate, onExit, onFrameStart,
# onFrameEnd, onPlayStateChange and so on - each gated by its own toggle on the DAT.
# Only "Frame Start" is enabled here, and only onFrameStart is defined; the others
# would never run even if they were written, so they are left out.
#
# It exists because a CHOP cooks only when something requests it. With nothing
# connected to the component's output and no viewer open, the detect CHOP never
# cooked, so no audio ever reached the detector. Driving it from the frame makes
# ingestion independent of whether anything consumes the output.

def onFrameStart(frame):
    parent().Tick()
    return
'''


CALLBACKS = '''# Script CHOP callbacks for AutoBpm.

def onCook(scriptOp):
    parent().Cook(scriptOp)
    parent().OnAutosync()
    return
'''


PAR_EXEC = '''# Parameter callbacks for AutoBpm.

def onPulse(par):
    comp = par.owner
    if par.name == 'Reset':
        comp.Reset()
    elif par.name == 'Restartdetector':
        comp.Restart()
    elif par.name == 'Synctempo':
        comp.SyncTempo()
    return


def onValueChange(par, prev):
    # Anything that changes which interpreter or device is used needs a fresh start.
    if par.name in ('Runtime', 'Envpath', 'Torchdevice', 'Repopath'):
        par.owner.Restart()
    return
'''


def main():
    print(f"[AutoBpm] repo: {REPO}")
    # Build wherever the caller asked. While iterating it is easier to rebuild
    # straight into the network you are working in than to re-import the .tox:
    #     exec(open(p).read(), {"__file__": p, "PARENT": "/project1"})
    parent_path = globals().get("PARENT") or os.environ.get("TDAUTOBPM_PARENT") or "/"
    comp = build(parent_path)
    comp.save(TOX_PATH)
    print(f"[AutoBpm] built {comp.path}")
    print(f"[AutoBpm] saved {TOX_PATH}")
    print("[AutoBpm] connect an Audio Device In CHOP to this component's input,")
    print("[AutoBpm] then: op('%s').Diagnose()" % comp.path)
    return comp


main()
