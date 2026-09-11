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
import sys

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

    `append*` creates the parameter already holding a zero/empty value, and setting
    only `.default` leaves that value untouched: the UI would show the right default
    while a freshly built component ran with Active off and an Update Rate of 0.
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

#: Parameters the detector applies without restarting (see AutoBpm.Configure).
LIVE_SETTINGS = ("Updaterate", "Estimate", "Smoothing", "Resetseconds",
                 "Lockconfidence", "Rangemin", "Rangemax")


def build(parent_path="/", save_to=None):
    parent_op = _td("op")(parent_path)
    if parent_op is None:
        raise RuntimeError(f"no such parent: {parent_path}")

    existing = parent_op.op(COMP_NAME)
    carried = None
    if existing is not None:
        carried = _carry_over(existing)
        # Destroying the component does not stop its detector, and would leave the
        # sidecar process running with nothing to talk to.
        try:
            existing.Stop()
        except Exception:
            pass
        existing.destroy()

    # The extension imports tdautobpm's helpers into TouchDesigner's interpreter,
    # which keeps them for the whole session. Forget them, so a rebuild after
    # editing src/ runs the new code rather than the version first imported.
    for name in [m for m in sys.modules if m.split(".")[0] == "tdautobpm"]:
        del sys.modules[name]

    comp = parent_op.create(_td("baseCOMP"), COMP_NAME)
    comp.nodeX, comp.nodeY = 0, 0
    # The panel reaches the component as `parent.AutoBpm`, whatever it is renamed to.
    comp.par.parentshortcut = "AutoBpm"

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

    # Where the checkout lives. Blank finds it by itself when the project is inside
    # the checkout; set it when the .tox is used from anywhere else. This build knows
    # where it is, so it fills it in - but not in the saved .tox (see below).
    _append(page, "Str", "Repopath", "Repo Path", "")
    comp.par.Repopath = REPO

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

    # The tempo range the music is in, folded into by octaves: 160-180 turns an
    # 87 BPM reading of drum and bass into 174. The default is the full scale.
    _append(page, "Float", "Rangemin", "Range Min (BPM)", float(STRIP_BPM[0]),
            normMin=STRIP_BPM[0], normMax=STRIP_BPM[1])
    _append(page, "Float", "Rangemax", "Range Max (BPM)", float(STRIP_BPM[1]),
            normMin=STRIP_BPM[0], normMax=STRIP_BPM[1])

    # Scales the output tempo: 2 when the detector counts half time, 0.5 for double.
    _append(page, "Float", "Multiplier", "Tempo Multiplier", 1.0,
            normMin=0.25, normMax=4.0)

    page.appendPulse("Reset", label="Reset")
    page.appendPulse("Tap", label="Tap Tempo")
    page.appendPulse("Halftempo", label="Half Tempo")
    page.appendPulse("Doubletempo", label="Double Tempo")
    page.appendPulse("Restartdetector", label="Restart Detector")
    page.appendPulse("Synctempo", label="Sync Tempo")

    _append(page, "Toggle", "Autosync", "Autosync", False)
    # One threshold for both: Autosync only follows tempos above it, and Hold keeps
    # the last tempo above it while confidence is below. The name predates Hold.
    _append(page, "Float", "Autosyncconfidence", "Confidence Threshold", 0.5,
            normMin=0.0, normMax=1.0)
    _append(page, "Toggle", "Hold", "Hold Below Threshold", True)

    # Blank means presets.csv beside this script in the checkout.
    _append(page, "File", "Presetsfile", "Presets File", "")

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

    # A CHOP cooks only when something pulls on it. This drives the detector every
    # frame, whether or not anything downstream reads its output.
    frame_exec = comp.create(_td("executeDAT"), "frame_exec")
    frame_exec.nodeX, frame_exec.nodeY = -100, 400
    frame_exec.text = FRAME_EXEC
    _set_par(frame_exec, ["framestart", "onframestart"], True)

    par_exec = comp.create(_td("parameterexecuteDAT"), "par_exec")
    par_exec.nodeX, par_exec.nodeY = 200, 200
    # "..", not ".": an OP parameter resolves relative to its own operator, so "."
    # would be this DAT itself.
    _set_par(par_exec, ["op", "ops"], "..")
    _set_par(par_exec, ["pars", "parameters"],
             "Reset Tap Halftempo Doubletempo Restartdetector Synctempo "
             "Runtime Envpath Torchdevice Repopath Presetsfile "
             + " ".join(LIVE_SETTINGS))
    _set_par(par_exec, ["custom"], True, required=False)
    _set_par(par_exec, ["builtin"], False, required=False)
    _set_par(par_exec, ["valuechange", "onvaluechange"], True)
    _set_par(par_exec, ["onpulse", "pulse", "parpulse"], True)
    par_exec.text = PAR_EXEC

    build_ui(comp)

    # -- extension, last, now that its parameters exist --------------------
    # `me.op(...)` rather than a bare `op(...)`: the extension expression is evaluated
    # with `me` bound to this component, so this resolves unambiguously to the child.
    comp.par.extension1 = "me.op('AutoBpmExt').module.AutoBpm(me)"
    comp.par.promoteextension1 = True
    comp.par.reinitextensions.pulse()

    # Saved before the rebuilt component gets back the settings it carried over, so
    # the .tox always ships with defaults rather than whatever the builder had set,
    # and without this machine's checkout path. Clearing and restoring Repo Path in
    # one go does not trigger par_exec: it only sees the value at the end of the frame.
    if save_to:
        comp.par.Repopath = ""
        comp.save(save_to)
        comp.par.Repopath = REPO

    # Also after the extension, since changing some of these restarts the detector.
    if carried is not None:
        _restore(comp, carried)

    return comp


def _carry_over(comp):
    """What a rebuild in place keeps: position, wiring and custom parameter values.

    Rebuilding into a live network would otherwise disconnect the audio input and
    revert every setting to its default.
    """
    values = {}
    for par in comp.customPars:
        if par.isPulse or par.readOnly:
            continue
        values[par.name] = par.eval()
    sources = []
    if comp.inputConnectors:
        sources = [conn.owner for conn in comp.inputConnectors[0].connections]
    targets = []
    if comp.outputConnectors:
        targets = [(conn.owner, conn.index)
                   for conn in comp.outputConnectors[0].connections]
    return dict(pos=(comp.nodeX, comp.nodeY), values=values,
                sources=sources, targets=targets)


def _restore(comp, carried):
    comp.nodeX, comp.nodeY = carried["pos"]
    for name, value in carried["values"].items():
        par = getattr(comp.par, name, None)
        if par is not None and par.eval() != value:
            par.val = value
    for source in carried["sources"]:
        comp.inputConnectors[0].connect(source)
    for target, index in carried["targets"]:
        comp.outputConnectors[0].connect(target.inputConnectors[index])


FRAME_EXEC = '''# Frame callbacks for AutoBpm.
#
# An Execute DAT can fire on many events - onStart, onCreate, onExit, onFrameStart,
# onFrameEnd, onPlayStateChange and so on - each gated by its own toggle on the DAT.
# Only "Frame Start" is enabled here, and only onFrameStart is defined; the others
# would never run even if they were written, so they are left out.
#
# It exists because a CHOP cooks only when something requests it. Driving the
# detector from the frame keeps audio flowing into it whether or not anything
# consumes the component's output.

def onFrameStart(frame):
    # Tick comes from the extension, which is attached last while the component is
    # being built; skip the frames before that. (Not getattr: TouchDesigner logs an
    # AttributeError for a missing COMP attribute even when a default is given.)
    comp = parent()
    if any(comp.extensions):
        comp.Tick()
    return
'''


#: Logical size of the control panel, in panel units. Its graphics are rendered at
#: UI_SCALE times this, so text stays sharp on high-density displays.
UI_W, UI_H = 480, 390
UI_SCALE = 2

#: Colour and header label for each state the extension reports as `Mode`. Kept in a
#: Table DAT inside the panel ("palette"), so it can be restyled without editing the
#: shader or any expression.
PALETTE = (
    ("mode", "label", "r", "g", "b"),
    ("auto", "AUTO", 1.0, 0.773, 0.498),
    ("waiting", "NO INPUT", 0.62, 0.58, 0.52),
    ("error", "ERROR", 1.0, 0.36, 0.34),
    ("tap", "TAP", 0.45, 0.76, 1.0),
    ("stopped", "STOPPED", 0.52, 0.53, 0.56),
    ("hold", "HOLD", 0.98, 0.86, 0.42),
)

#: The range strip's scale, and the default range. Log-scaled, so an octave is
#: always half its width - which is the distance the detector's usual mistake jumps.
#: The presets themselves are in presets.csv, for anyone to edit.
STRIP_BPM = (60, 240)

# Helvetica Neue ships as a .ttc that the Text TOP fails to load on macOS.
LABEL_FONT = "Arial"
NUMBER_FONT = "Arial Black"
GREY = (0.55, 0.56, 0.59)
DIM = (0.40, 0.41, 0.44)
LIGHT = (0.86, 0.87, 0.89)
INK = (0.07, 0.072, 0.078)  # the panel background, for text on the mode pill

#: Expression fragments shared by the panel's parameters.
_COMP = "parent.AutoBpm"
_BPM = _COMP + ".op('bpm_out')['bpm'][0]"
_CONF = "op('conf_smooth')[0].eval()"
_MODE = "op('palette')[parent.AutoBpm.Mode.val, '%s'].val"
_MULT = _COMP + ".par.Multiplier.eval()"
_LOGIC = "op('ui_logic').module"

#: Rows of buttons, bottom up: (y, height, buttons). Momentary buttons act through
#: ui_exec; toggles are bound straight to a parameter. `lit` keeps a button in the
#: accent colour while its expression is true; `weight` is its share of the row.
#: Labels outside ASCII are written as escapes inside the expression, because
#: TouchDesigner reads this file as ASCII.
BUTTON_ROWS = (
    (12, 44, (
        dict(name="btn_active", kind="toggledown", bound="Active",
             label_expr="'STOP' if parent.AutoBpm.par.Active else 'START'"),
        dict(name="btn_tap", label="TAP"),
        dict(name="btn_reset", label="RESET"),
        dict(name="btn_sync", label="AUTOSYNC", kind="toggledown", bound="Autosync"),
    )),
    (64, 36, (
        dict(name="btn_half", label_expr="'\\u00f7 2'", weight=0.6,
             lit=_MULT + " < 1"),
        dict(name="btn_double", label_expr="'\\u00d7 2'", weight=0.6,
             lit=_MULT + " > 1"),
        dict(name="btn_hold", label="HOLD", kind="toggledown", bound="Hold",
             weight=0.9),
        dict(name="btn_preset", weight=2.4,
             label_expr="'PRESET: ' + %s.preset_name()" % _LOGIC),
    )),
)


def build_ui(comp):
    """The control panel, shown as the component's viewer.

    Top to bottom: the BPM readout beside a beat ring (the arc is the phase, the
    centre pulses on each beat, the dots count beats in the bar); the confidence
    line, whose knob sets the Autosync threshold; the detection range strip, whose
    handles set the range; then two rows of buttons. Space taps while the panel has
    keyboard focus.

    The readout is one GLSL TOP with Text TOPs composited over it, used as the
    container's background, so the whole display is a single TOP (`display`) that
    can be sent anywhere else too. The interaction logic lives in `ui_logic`.
    """
    ui = comp.create(_td("containerCOMP"), "ui")
    ui.nodeX, ui.nodeY = 500, 0
    _set_par(ui, ["w"], UI_W)
    _set_par(ui, ["h"], UI_H)

    palette = ui.create(_td("tableDAT"), "palette")
    palette.nodeX, palette.nodeY = -800, 300
    palette.clear()
    for row in PALETTE:
        palette.appendRow(row)

    # The presets file, kept in sync with the disk: saving it in a spreadsheet app
    # updates the menu without touching TouchDesigner. Read as plain text and parsed
    # by read_presets, which is forgiving about what spreadsheet apps save.
    # Its path is set by the extension (AutoBpm.PointPresets), which knows where the
    # checkout is even when Repo Path is blank.
    presets = ui.create(_td("textDAT"), "presets_file")
    presets.nodeX, presets.nodeY = -800, 150
    _set_par(presets, ["syncfile"], True, required=False)
    _set_par(presets, ["loadonstart"], True, required=False)

    logic = ui.create(_td("textDAT"), "ui_logic")
    logic.nodeX, logic.nodeY = 150, -650
    logic.text = UI_LOGIC

    # Confidence arrives at the detector's update rate, a few steps a second; a lag
    # turns those steps into movement.
    conf_in = ui.create(_td("selectCHOP"), "conf_in")
    conf_in.nodeX, conf_in.nodeY = -800, 0
    _set_par(conf_in, ["chops", "chop"], "../bpm_out")
    _set_par(conf_in, ["channames"], "confidence")
    conf = ui.create(_td("lagCHOP"), "conf_smooth")
    conf.nodeX, conf.nodeY = -650, 0
    conf.inputConnectors[0].connect(conf_in)
    _set_par(conf, ["lag1"], 0.25)
    _set_par(conf, ["lag2"], 0.25)

    meter = ui.create(_td("glslTOP"), "meter")
    meter.nodeX, meter.nodeY = -400, 0
    # A GLSL TOP creates its own docked `<name>_pixel` DAT; reuse it rather than
    # leaving a stray default shader beside a second one.
    pixel = ui.op("meter_pixel") or ui.create(_td("textDAT"), "meter_pixel")
    pixel.text = METER_GLSL
    meter.par.pixeldat = pixel
    _size(meter)
    meter.seq.vec.numBlocks = 3
    meter.seq.color.numBlocks = 1
    uniforms = (
        ("vec0", "uState", (
            _COMP + ".op('bpm_out')['phase'][0]",
            _CONF,
            _COMP + ".BeatCount.val % 4",
            "float(%s > 0)" % _BPM)),
        ("vec1", "uAux", (
            _COMP + ".par.Autosyncconfidence.eval()",
            "float(%s.par.Autosync.eval())" % _COMP,
            str(float(UI_SCALE)),
            "0")),
        ("vec2", "uRange", (
            _COMP + ".par.Rangemin.eval()",
            _COMP + ".par.Rangemax.eval()",
            _COMP + ".RawBpm.val",  # what the detector hears, before hold or multiplier
            "0")),
    )
    for block, name, exprs in uniforms:
        _set_par(meter, [block + "name"], name)
        for axis, expr in zip("xyzw", exprs):
            getattr(meter.par, "%svalue%s" % (block, axis)).expr = expr
    _set_par(meter, ["color0name"], "uAccent")
    for channel in "rgb":
        getattr(meter.par, "color0rgb" + channel).expr = "float(%s)" % (_MODE % channel)

    status = "(lambda s: s if len(s) <= 44 else s[:41] + '...')(%s)" % (
        _COMP + ".par.Status.eval()")
    multiplier = ("(lambda m: '' if m == 1 else ('\\u00d7%%g' %% m if m > 1 "
                  "else '\\u00f7%%g' %% (1 / m)))(%s)" % _MULT)
    range_text = ("'%%d \\u2013 %%d BPM' %% (%s.par.Rangemin.eval(), "
                  "%s.par.Rangemax.eval())" % (_COMP, _COMP))
    texts = (
        _text(ui, "txt_title", 18, 373, 12, text="AUTO BPM", bold=True),
        _text(ui, "txt_mode", 418, 373, 11, expr=_MODE % "label", align="center",
              colour=INK, bold=True),
        _text(ui, "txt_bpm", 250, 280, 64, align="right", font=NUMBER_FONT,
              expr="('%%.1f' %% %s) if %s > 0 else '---'" % (_BPM, _BPM),
              colour=_MODE),
        _text(ui, "txt_mult", 256, 298, 14, expr=multiplier, colour=_MODE, bold=True),
        _text(ui, "txt_unit", 256, 264, 14, text="BPM", bold=True),
        _text(ui, "txt_conf_label", 18, 220, 11, text="CONFIDENCE", bold=True),
        _text(ui, "txt_conf", 270, 220, 11, align="right", colour=LIGHT, bold=True,
              expr="'%%d%%%%' %% round(100 * %s)" % _CONF),
        _text(ui, "txt_status", 18, 178, 11, expr=status, colour=DIM),
        _text(ui, "txt_range_label", 18, 146, 11, text="DETECTION RANGE", bold=True),
        _text(ui, "txt_range", 462, 146, 11, align="right", colour=LIGHT, bold=True,
              expr=range_text),
        _text(ui, "txt_scale_lo", 18, 110, 9, text=str(STRIP_BPM[0]), colour=DIM),
        _text(ui, "txt_scale_mid", 240, 110, 9, align="center", colour=DIM,
              text=str(STRIP_BPM[0] * 2)),
        _text(ui, "txt_scale_hi", 462, 110, 9, align="right", text=str(STRIP_BPM[1]),
              colour=DIM),
    )
    for index, top in enumerate(texts):
        top.nodeX, top.nodeY = -400, -150 - 120 * index

    display = ui.create(_td("compositeTOP"), "display")
    display.nodeX, display.nodeY = -150, 0
    _set_par(display, ["operand"], "over")
    for index, top in enumerate(texts + (meter,)):
        display.inputConnectors[index].connect(top)

    ui.par.top.expr = "me.op('display')"
    _set_par(ui, ["topfill"], "fill", required=False)

    _build_buttons(ui)

    # Invisible areas over the confidence line and the range strip, to drag on.
    # Each spans exactly the drawn line, so the panel's `u` is the position on it.
    for name, x, y, w, h in (("conf_drag", 18, 190, 252, 26),
                             ("range_drag", 18, 112, 444, 28)):
        area = ui.create(_td("containerCOMP"), name)
        area.nodeX, area.nodeY = 150 + (450 if name == "range_drag" else 0), -800
        for par_name, value in (("x", x), ("y", y), ("w", w), ("h", h)):
            _set_par(area, [par_name], value)
        _set_par(area, ["bgalpha"], 0, required=False)
        _set_par(area, ["cursor"], "arrowLeftRight", required=False)

    drag_exec = ui.create(_td("panelexecuteDAT"), "drag_exec")
    drag_exec.nodeX, drag_exec.nodeY = 150, -950
    _set_par(drag_exec, ["panels"], "conf_drag range_drag")
    _set_par(drag_exec, ["panelvalue"], "lselect u")
    _set_par(drag_exec, ["offtoon"], True)
    _set_par(drag_exec, ["valuechange"], True)
    drag_exec.text = DRAG_EXEC

    ui_exec = ui.create(_td("parameterexecuteDAT"), "ui_exec")
    ui_exec.nodeX, ui_exec.nodeY = 150, -500
    _set_par(ui_exec, ["op", "ops"],
             "btn_tap btn_reset btn_half btn_double btn_preset")
    _set_par(ui_exec, ["pars", "parameters"], "value0")
    _set_par(ui_exec, ["custom"], True, required=False)
    _set_par(ui_exec, ["builtin"], True, required=False)
    _set_par(ui_exec, ["valuechange", "onvaluechange"], True)
    ui_exec.text = UI_EXEC

    # Space taps, but only while this panel has keyboard focus (click it first), so
    # it cannot fire while typing elsewhere or fight TouchDesigner's own shortcuts.
    keys = ui.create(_td("keyboardinDAT"), "keys")
    keys.nodeX, keys.nodeY = 450, -500
    _set_par(keys, ["keys"], "space")
    _set_par(keys, ["panels"], "..")
    key_callbacks = ui.op("keys_callbacks") or ui.create(_td("textDAT"), "keys_callbacks")
    key_callbacks.nodeX, key_callbacks.nodeY = 450, -650
    key_callbacks.text = KEY_CALLBACKS
    keys.par.callbacks = key_callbacks

    # Show the panel as the component's viewer. An expression rather than a path,
    # so it still points at this panel after the .tox is imported somewhere else.
    comp.par.opviewer.expr = "me.op('ui')"
    comp.viewer = True
    return ui


def _build_buttons(ui):
    gap, margin = 8, 12
    for row_index, (y, height, buttons) in enumerate(BUTTON_ROWS):
        weights = [spec.get("weight", 1.0) for spec in buttons]
        unit = (UI_W - 2 * margin - gap * (len(buttons) - 1)) / sum(weights)
        x = margin
        for index, spec in enumerate(buttons):
            name = spec["name"]
            width = unit * weights[index]
            button = ui.create(_td("buttonCOMP"), name)
            button.name = name  # TD 2025 appends a digit to new Button COMPs' names
            button.nodeX, button.nodeY = 150 + index * 150, -300 + row_index * 150
            for par_name, value in (("x", x), ("y", y), ("w", width), ("h", height)):
                _set_par(button, [par_name], value)
            x += width + gap
            _set_par(button, ["buttontype"], spec.get("kind", "momentary"))
            _set_par(button, ["fontsize"], 13, required=False)

            # The Button COMP shades off, on and rollover from one colour, darkening
            # it to 0.4x when off and 0.8x when on. Swap that colour with the state:
            # neutral when off, the auto accent while on or held. A lit button is
            # off as far as the Button COMP knows, so it gets the full accent to
            # show through the 0.4x (going brighter would clip and shift the hue).
            lit = spec.get("lit") or "False"
            for channel, value in zip("rgb", PALETTE[1][2:]):
                par = getattr(button.par, "color" + channel, None)
                if par is not None:
                    par.expr = "%.3f if me.par.value0 else (%.3f if (%s) else 0.45)" % (
                        0.7 * value, value, lit)

            if spec.get("label_expr"):
                button.par.label.expr = spec["label_expr"]
            else:
                _set_par(button, ["label"], spec["label"])
            if spec.get("bound"):
                # Bound, so the parameter stays the single source of truth:
                # flipping Active from anywhere else flips the button too.
                # ParMode is not in `td` (it lives in tdutils), so take it from
                # the mode.
                value = button.par.value0
                value.bindExpr = "parent.AutoBpm.par." + spec["bound"]
                value.mode = type(value.mode).BIND


def _size(top):
    _set_par(top, ["outputresolution"], "custom")
    _set_par(top, ["resolutionw"], UI_W * UI_SCALE)
    _set_par(top, ["resolutionh"], UI_H * UI_SCALE)


def _text(ui, name, x, y, size, text=None, expr=None, align="left",
          font=LABEL_FONT, bold=False, colour=GREY):
    """A panel-sized Text TOP holding one line, placed at (x, y) in panel units.

    `colour` is an RGB tuple, or an expression pattern with one `%s` for the channel.
    """
    top = ui.create(_td("textTOP"), name)
    _size(top)
    if expr is not None:
        top.par.text.expr = expr
    else:
        top.par.text = text
    _set_par(top, ["font"], font)
    # `bold` is legacy font selection; the current Text TOP picks a typeface.
    if bold:
        _set_par(top, ["typeface"], "Bold", required=False)
    _set_par(top, ["fontsizexunit"], "pixels", required=False)  # default is points
    _set_par(top, ["fontsizex"], size * UI_SCALE)
    _set_par(top, ["alignx"], align)
    _set_par(top, ["aligny"], "center")
    # Position is an offset from whichever edge, or the centre, the text aligns to.
    anchor = {"left": 0, "center": UI_W / 2, "right": UI_W}[align]
    _set_par(top, ["positionx"], (x - anchor) * UI_SCALE)
    _set_par(top, ["positiony"], (y - UI_H / 2) * UI_SCALE)
    _set_par(top, ["bgalpha"], 0)
    for index, channel in enumerate("rgb"):
        par = getattr(top.par, "fontcolor" + channel)
        if isinstance(colour, str):
            par.expr = "float(%s)" % (colour % channel)
        else:
            par.val = colour[index]
    return top


#: The meter behind the panel's text. Works in panel units, so the render
#: resolution only changes sharpness.
METER_GLSL = '''// AutoBpm panel graphics. Text is layered over this by Text TOPs.
// Coordinates are panel units (480 x 390, y up); uAux.z is the render scale.

uniform vec4 uState;   // phase 0..1, confidence 0..1, beat in bar 0..3, has a tempo
uniform vec4 uAux;     // autosync threshold, autosync on, render scale, unused
uniform vec4 uRange;   // range min, range max, the detector's latest estimate
uniform vec4 uAccent;  // colour of the current mode, from the palette table

out vec4 fragColor;

const float TAU = 6.28318531;
const vec3 BG = vec3(0.070, 0.072, 0.078);
const vec3 TRACK = vec3(0.165, 0.170, 0.182);
const vec2 STRIP_BPM = vec2(%(strip_lo).1f, %(strip_hi).1f);

float px;  // one output pixel, in panel units

float fill(float d) { return 1.0 - smoothstep(-px, px, d); }

float box(vec2 p, vec2 half_size, float r)
{
    vec2 q = abs(p) - half_size + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

// Red when unsure, amber in between, green once the estimate can be trusted.
vec3 confColour(float c)
{
    vec3 low = vec3(0.95, 0.36, 0.32);
    vec3 mid = vec3(1.00, 0.74, 0.30);
    vec3 high = vec3(0.40, 0.86, 0.56);
    return c < 0.5 ? mix(low, mid, smoothstep(0.15, 0.5, c))
                   : mix(mid, high, smoothstep(0.5, 0.8, c));
}

// Position of a tempo along the range strip, log-scaled.
float stripX(float bpm)
{
    float u = log2(max(bpm, 1.0) / STRIP_BPM.x) / log2(STRIP_BPM.y / STRIP_BPM.x);
    return mix(18.0, 462.0, clamp(u, 0.0, 1.0));
}

// A draggable knob: accent ring around a dark centre.
vec3 knob(vec3 col, vec2 p, vec2 centre, vec3 accent)
{
    float r = length(p - centre);
    col = mix(col, accent, fill(r - 7.0));
    return mix(col, BG, fill(r - 3.0));
}

void main()
{
    px = 1.0 / uAux.z;
    vec2 p = vUV.st * uTDOutputInfo.res.zw * px;
    float phase = uState.x;
    float conf = clamp(uState.y, 0.0, 1.0);
    float beatInBar = uState.z;
    float hasTempo = uState.w;
    vec3 accent = uAccent.rgb;

    vec3 col = BG;

    // Mode pill in the header; its label is a Text TOP.
    col = mix(col, accent, fill(box(p - vec2(418.0, 373.0), vec2(46.0, 10.0), 10.0)));

    // Divider between the readout and the controls.
    col = mix(col, TRACK, fill(box(p - vec2(240.0, 158.0), vec2(228.0, 0.5), 0.0)));

    // Confidence status line. The knob is the Autosync threshold, drag to move it;
    // it dims while Autosync is off.
    vec2 barCentre = vec2(144.0, 202.0);
    vec2 barHalf = vec2(126.0, 3.0);
    float barStart = barCentre.x - barHalf.x;
    float bar = fill(box(p - barCentre, barHalf, 3.0));
    float filled = step(p.x, barStart + 2.0 * barHalf.x * conf);
    col = mix(col, mix(TRACK, confColour(conf), filled), bar);
    vec2 threshold = vec2(barStart + 2.0 * barHalf.x * uAux.x, 202.0);
    col = knob(col, p, threshold, mix(vec3(0.5), vec3(0.95), uAux.y));

    // Beat ring: the arc is the phase, sweeping clockwise from twelve o'clock.
    vec2 q = p - vec2(386.0, 268.0);
    float r = length(q);
    float angle = fract(atan(q.x, q.y) / TAU + 1.0);
    float ring = fill(abs(r - 62.0) - 3.0);
    col = mix(col, TRACK, ring);
    col = mix(col, accent, ring * step(angle, phase) * hasTempo);
    vec2 head = vec2(sin(phase * TAU), cos(phase * TAU)) * 62.0;
    col = mix(col, vec3(1.0), fill(length(q - head) - 6.0) * hasTempo);

    // Centre pulse: brightest on the beat, decaying through the first part of it.
    float pulse = hasTempo * exp(-phase * 6.0);
    float discR = 30.0 + 12.0 * pulse;
    float disc = fill(r - discR);
    float glow = exp(-max(r - discR, 0.0) / 14.0) * (1.0 - disc) * (1.0 - ring);
    col += accent * pulse * 0.45 * glow;
    col = mix(col, mix(TRACK, accent, 0.2 + 0.8 * pulse), disc);

    // Beats in the bar, counted from the last Reset or tap. The first is larger.
    for (int i = 0; i < 4; i++) {
        vec2 centre = vec2(356.0 + 20.0 * float(i), 184.0);
        float on = step(abs(float(i) - beatInBar), 0.5) * hasTempo;
        float d = fill(length(p - centre) - (i == 0 ? 5.0 : 4.0));
        col = mix(col, mix(TRACK, accent, on), d);
    }

    // Detection range strip, with the detector's latest estimate as a white mark -
    // it keeps moving while Hold holds the readout. Drag the handles to set the
    // range. Its scale labels are Text TOPs.
    float xLo = stripX(uRange.x);
    float xHi = stripX(uRange.y);
    float strip = fill(box(p - vec2(240.0, 126.0), vec2(222.0, 3.0), 3.0));
    float inRange = step(xLo, p.x) * step(p.x, xHi);
    col = mix(col, mix(TRACK, accent * 0.8, inRange), strip);
    float mark = fill(box(p - vec2(stripX(uRange.z), 126.0), vec2(1.5, 8.0), 1.0));
    col = mix(col, vec3(1.0), mark * hasTempo);
    col = knob(col, p, vec2(xLo, 126.0), accent);
    col = knob(col, p, vec2(xHi, 126.0), accent);

    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
''' % dict(strip_lo=STRIP_BPM[0], strip_hi=STRIP_BPM[1])


UI_LOGIC = '''# What the AutoBpm panel does when it is clicked, dragged or typed at.
#
# Everything here writes the component's parameters or calls its extension, so the
# panel never holds state of its own - except which range handle is being dragged.

import math

#: The range strip's scale, log-scaled, matching the shader.
STRIP_BPM = (%(strip_lo)d, %(strip_hi)d)

#: The narrowest range the handles can be dragged to, in BPM.
MIN_WIDTH = 4

_grab = None


def comp():
    return parent.AutoBpm


def bpm_at(u):
    """Tempo at a position 0..1 along the range strip."""
    lo, hi = STRIP_BPM
    return lo * (hi / lo) ** min(1.0, max(0.0, u))


def u_of(bpm):
    lo, hi = STRIP_BPM
    return math.log(max(bpm, 1e-6) / lo) / math.log(hi / lo)


_reported = None


def presets():
    """The presets from the presets file, or just Any if it has none."""
    dat = op('presets_file')
    found, problems = comp().op('AutoBpmExt').module.read_presets(dat.text)
    # Report each distinct set of problems once, not on every redraw.
    global _reported
    report = (dat.par.file.eval(), tuple(problems), bool(found))
    if report != _reported:
        _reported = report
        for problem in problems:
            print('[AutoBpm] %%s: %%s' %% (dat.par.file.eval(), problem))
        if not found:
            print('[AutoBpm] no presets in %%r; offering Any only' %% dat.par.file.eval())
    return found or list(comp().op('AutoBpmExt').module.FALLBACK_PRESETS)


def preset_name():
    """The preset the current range matches, or CUSTOM."""
    c = comp()
    current = (c.par.Rangemin.eval(), c.par.Rangemax.eval())
    for name, lo, hi in presets():
        if (lo, hi) == current:
            return name
    return 'CUSTOM'


def _label(name, lo, hi):
    return '%%s   %%g\u2013%%g' %% (name, lo, hi)


def open_preset_menu():
    """Offer every preset in TouchDesigner's pop-up menu, the current one ticked."""
    menu = getattr(op.TDResources, 'PopMenu', None)
    if menu is None:  # no pop-up menu in this build: step through them instead
        next_preset()
        return
    ranges = presets()
    labels = [_label(*preset) for preset in ranges]
    chosen = dict(zip(labels, ranges))
    current = preset_name()
    ticked = [label for label, (name, _, _) in chosen.items() if name == current]

    def picked(info):
        preset = chosen.get(info.get('item'))
        if preset is not None:
            c = comp()
            c.par.Rangemin, c.par.Rangemax = preset[1], preset[2]

    menu.Open(items=labels, callback=picked, checkedItems=ticked,
              title='Detection range')


def next_preset():
    ranges = presets()
    names = [name for name, _, _ in ranges]
    name = preset_name()
    index = (names.index(name) + 1) %% len(ranges) if name in names else 0
    _, lo, hi = ranges[index]
    c = comp()
    c.par.Rangemin, c.par.Rangemax = lo, hi


def set_threshold(u):
    comp().par.Autosyncconfidence = round(min(1.0, max(0.0, u)), 2)


def press_range(u):
    """Grab whichever range handle is nearer the press, and move it there."""
    global _grab
    c = comp()
    to_min = abs(u - u_of(c.par.Rangemin.eval()))
    to_max = abs(u - u_of(c.par.Rangemax.eval()))
    _grab = 'Rangemin' if to_min <= to_max else 'Rangemax'
    drag_range(u)


def drag_range(u):
    c = comp()
    bpm = round(bpm_at(u))
    if _grab == 'Rangemin':
        c.par.Rangemin = min(bpm, c.par.Rangemax.eval() - MIN_WIDTH)
    elif _grab == 'Rangemax':
        c.par.Rangemax = max(bpm, c.par.Rangemin.eval() + MIN_WIDTH)
''' % dict(strip_lo=STRIP_BPM[0], strip_hi=STRIP_BPM[1])


UI_EXEC = '''# Parameter callbacks for the AutoBpm panel's momentary buttons.
#
# Start/Stop and Autosync need none: those buttons are bound to the component's
# Active and Autosync parameters. The rest act on the press and ignore the release.

def onValueChange(par, prev):
    if not par.eval():
        return
    comp = parent.AutoBpm
    name = par.owner.name
    if name == 'btn_tap':
        comp.Tap()
    elif name == 'btn_reset':
        comp.Reset()
    elif name == 'btn_half':
        comp.Half()
    elif name == 'btn_double':
        comp.Double()
    elif name == 'btn_preset':
        op('ui_logic').module.open_preset_menu()
    return
'''


DRAG_EXEC = '''# Panel callbacks for dragging the threshold knob and the range handles.
#
# `u` is the pointer's position across the area, which spans exactly the line it
# sits over. A press grabs; moving with the button held drags.

def _apply(panel, u, pressed):
    logic = op('ui_logic').module
    if panel.name == 'conf_drag':
        logic.set_threshold(u)
    elif panel.name == 'range_drag':
        if pressed:
            logic.press_range(u)
        else:
            logic.drag_range(u)


def onOffToOn(panelValue):
    if panelValue.name == 'lselect':
        panel = panelValue.owner
        _apply(panel, panel.panel.u.val, pressed=True)
    return


def onValueChange(panelValue, prev):
    panel = panelValue.owner
    if panelValue.name == 'u' and panel.panel.lselect.val:
        _apply(panel, panelValue.val, pressed=False)
    return
'''


KEY_CALLBACKS = '''# Keyboard callbacks for the AutoBpm panel: space is TAP.
#
# The Keyboard In DAT only listens while the panel has focus. Repeats from a held
# key are absorbed by Tap() itself, which ignores taps closer than 0.1 s.

def onKey(dat, keyInfo):
    if keyInfo.state and keyInfo.key == 'space':
        parent.AutoBpm.Tap()
    return


def onShortcut(dat, shortcutName, time):
    return
'''


CALLBACKS = '''# Script CHOP callbacks for AutoBpm.

def onCook(scriptOp):
    parent().Cook(scriptOp)
    parent().OnAutosync()
    return
'''


PAR_EXEC = '''# Parameter callbacks for AutoBpm.

LIVE_SETTINGS = %r

def onPulse(par):
    comp = par.owner
    if par.name == 'Reset':
        comp.Reset()
    elif par.name == 'Tap':
        comp.Tap()
    elif par.name == 'Halftempo':
        comp.Half()
    elif par.name == 'Doubletempo':
        comp.Double()
    elif par.name == 'Restartdetector':
        comp.Restart()
    elif par.name == 'Synctempo':
        comp.SyncTempo()
    return


def onValueChange(par, prev):
    # Anything that changes which interpreter or device is used needs a fresh start;
    # the detector's own settings apply live, keeping what it has heard so far.
    if par.name in ('Runtime', 'Envpath', 'Torchdevice', 'Repopath'):
        par.owner.Restart()
    elif par.name in LIVE_SETTINGS:
        par.owner.Configure()
    elif par.name == 'Presetsfile':
        par.owner.PointPresets()
    return
''' % (LIVE_SETTINGS,)


def main():
    print(f"[AutoBpm] repo: {REPO}")
    # Build wherever the caller asked. While iterating it is easier to rebuild
    # straight into the network you are working in than to re-import the .tox:
    #     exec(open(p).read(), {"__file__": p, "PARENT": "/project1"})
    parent_path = globals().get("PARENT") or os.environ.get("TDAUTOBPM_PARENT") or "/"
    comp = build(parent_path, save_to=TOX_PATH)
    print(f"[AutoBpm] built {comp.path}")
    print(f"[AutoBpm] saved {TOX_PATH}")
    print("[AutoBpm] connect an Audio Device In CHOP to this component's input,")
    print("[AutoBpm] then: op('%s').Diagnose()" % comp.path)
    return comp


main()
