"""Build the AutoBpm component and save it as AutoBpm.tox.

Run this *inside TouchDesigner*, from the textport::

    run("/full/path/to/td-autobpmdetector/touchdesigner/build_component.py")

or from any DAT::

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
    op  # noqa: B018  - provided by TouchDesigner
except NameError:  # pragma: no cover - guard for accidental non-TD execution
    raise SystemExit("build_component.py must be run inside TouchDesigner")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

COMP_NAME = "AutoBpm"
TOX_PATH = os.path.join(HERE, "AutoBpm.tox")


def build(parent_path="/"):
    parent_op = op(parent_path)
    if parent_op is None:
        raise RuntimeError(f"no such parent: {parent_path}")

    existing = parent_op.op(COMP_NAME)
    if existing is not None:
        existing.destroy()

    comp = parent_op.create(baseCOMP, COMP_NAME)
    comp.nodeX, comp.nodeY = 0, 0

    # -- extension ---------------------------------------------------------
    ext = comp.create(textDAT, "AutoBpmExt")
    with open(os.path.join(HERE, "AutoBpmExt.py")) as f:
        ext.text = f.read()
    ext.nodeX, ext.nodeY = -400, 200

    comp.par.extension1 = "op('AutoBpmExt').module.AutoBpm(me)"
    comp.par.promoteextension1 = True
    comp.par.reinitextensions.pulse()

    # -- parameters --------------------------------------------------------
    page = comp.appendCustomPage("Auto BPM")

    page.appendToggle("Active", label="Active")[0].default = True

    p = page.appendMenu("Runtime", label="Runtime")[0]
    p.menuNames = ["sidecar", "inprocess"]
    p.menuLabels = ["Sidecar process", "In-process"]
    p.default = "sidecar"

    p = page.appendStr("Envpath", label="Python Env")[0]
    p.default = ""
    # Blank auto-detects: $TDAUTOBPM_PYTHON, $CONDA_PREFIX, $VIRTUAL_ENV, then .venv.

    p = page.appendMenu("Torchdevice", label="Torch Device")[0]
    p.menuNames = ["cpu", "mps", "cuda"]
    p.menuLabels = ["CPU", "MPS (Apple)", "CUDA"]
    p.default = "cpu"

    p = page.appendFloat("Updaterate", label="Update Rate (Hz)")[0]
    p.default, p.normMin, p.normMax = 4.0, 0.5, 20.0

    p = page.appendMenu("Estimate", label="Estimator")[0]
    p.menuNames = ["local_mean", "mode", "mean", "median"]
    p.menuLabels = ["Local mean", "Mode", "Mean (biased)", "Median"]
    p.default = "local_mean"

    p = page.appendFloat("Smoothing", label="Smoothing")[0]
    p.default, p.normMin, p.normMax = 0.90, 0.0, 0.999

    p = page.appendFloat("Resetseconds", label="Reset Every (s)")[0]
    p.default, p.normMin, p.normMax = 5.0, 0.0, 60.0

    p = page.appendFloat("Lockconfidence", label="Lock At Confidence")[0]
    p.default, p.normMin, p.normMax = 0.0, 0.0, 1.0

    page.appendPulse("Reset", label="Reset")
    page.appendPulse("Restartdetector", label="Restart Detector")

    page.appendPulse("Synctempo", label="Sync Tempo")
    page.appendToggle("Autosync", label="Autosync")[0].default = False

    p = page.appendFloat("Autosyncconfidence", label="Autosync Min Confidence")[0]
    p.default, p.normMin, p.normMax = 0.5, 0.0, 1.0

    p = page.appendStr("Status", label="Status")[0]
    p.readOnly = True
    p.default = "idle"

    # -- network -----------------------------------------------------------
    in_chop = comp.create(inCHOP, "audio_in")
    in_chop.nodeX, in_chop.nodeY = -400, 0

    script = comp.create(scriptCHOP, "detect")
    script.nodeX, script.nodeY = -100, 0
    script.inputConnectors[0].connect(in_chop)

    callbacks = comp.create(textDAT, "detect_callbacks")
    callbacks.nodeX, callbacks.nodeY = -100, 200
    callbacks.text = CALLBACKS
    script.par.callbacks = callbacks

    out_chop = comp.create(outCHOP, "bpm_out")
    out_chop.nodeX, out_chop.nodeY = 200, 0
    out_chop.inputConnectors[0].connect(script)

    par_exec = comp.create(parameterexecuteDAT, "par_exec")
    par_exec.nodeX, par_exec.nodeY = 200, 200
    par_exec.par.op = "."
    par_exec.par.pars = "Reset Restartdetector Synctempo Runtime Envpath Torchdevice"
    par_exec.par.valuechange = True
    par_exec.par.pulse = True
    par_exec.text = PAR_EXEC

    comp.par.reinitextensions.pulse()
    return comp


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
    if par.name in ('Runtime', 'Envpath', 'Torchdevice'):
        par.owner.Restart()
    return
'''


def main():
    comp = build()
    comp.save(TOX_PATH)
    print(f"[AutoBpm] built {comp.path} and saved {TOX_PATH}")
    print("[AutoBpm] repo: " + REPO)
    print("[AutoBpm] wire an Audio Device In CHOP into its audio_in, then check Status.")
    return comp


main()
