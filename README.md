# td-autobpmdetector

Real-time BPM detection for TouchDesigner, plus a standalone CLI.

A fork of [shhhum/tdautobpmsync](https://github.com/shhhum/tdautobpmsync), restructured
so that it works with the Python environment you already have — conda, venv, uv, system —
instead of requiring one built on TouchDesigner's own interpreter. The detection model
(a causal TCN, "TempoNet", ~74k parameters) and its trained weights are shhhum's; see
[NOTICE](NOTICE).

Runs at roughly **60× realtime on a CPU core**, so live detection is not demanding.

---

## Quick start

```bash
git clone https://github.com/parsakaz/td-autobpmdetector
cd td-autobpmdetector

uv venv --python 3.11
uv pip install -e '.[live]'

uv run tdautobpm doctor                       # what environments can run this?
uv run tdautobpm analyze test_tracks/yuqt.mp3 # offline tempo of a file
uv run tdautobpm listen                       # live, from a system audio input
```

In TouchDesigner, open the textport and run:

```python
p = "/full/path/to/td-autobpmdetector/touchdesigner/build_component.py"
exec(open(p).read(), {"__file__": p})
```

`run()` will not work here — it takes a string of Python *code*, not a file path. And
passing `__file__` matters: TouchDesigner's textport namespace already defines one,
pointing inside the application bundle, so a bare `exec(open(p).read())` would go looking
for the component files in `TouchDesigner.app`. Setting `TDAUTOBPM_REPO` to the repository
root works too.

That builds `/AutoBpm` and writes `touchdesigner/AutoBpm.tox`. Wire an **Audio Device In**
CHOP into its `audio_in`, and read `bpm`, `confidence`, `beat` and `phase` out of it.
Import that `.tox` into any other project.

---

## Why this fork exists

The upstream project asks you to build a venv on TouchDesigner's bundled interpreter via
`tdPyEnvManager`. On an Apple Silicon Mac that reliably fails partway. Four separate
causes, all verified rather than guessed:

**1. The requirements file fights the installer.** Upstream `requirements.txt` is a 35-line
`pip freeze` that pins `pip==24.0` and `setuptools==65.5.0` — pinning the installer itself,
mid-install. About twenty of those pins are never imported by the model or the inference
code at all: `librosa`, `numba`, `llvmlite`, `scikit-learn`, `scipy`, `pooch`, `audioread`,
`soxr`, `joblib`, `msgpack` and friends. `numba`/`llvmlite` in particular are slow and
fragile to build. The engine actually needs **four** packages: torch, torchaudio, numpy,
soundfile.

**2. A venv on TouchDesigner's interpreter cannot be used outside TouchDesigner.**
`TouchDesigner.app` is codesigned with `com.apple.security.cs.disable-library-validation`,
so third-party `.so` files load fine *inside* the app. The bare `python3.11` binary inside
the app bundle carries no such entitlement, so the identical venv fails from a terminal:

```
dlopen(.../numpy/_core/_multiarray_umath.cpython-311-darwin.so):
mapping process and mapped file (non-platform) have different Team IDs
```

That is why there was no way to test the environment, script it, or run a CLI against it.
`tdautobpm doctor` detects this case by name.

**3. Paths with `~` were not expanded.** Something in the upstream setup flow installed an
entire miniconda into a directory *literally named* `~`. Every path this fork accepts goes
through `expanduser`.

**4. Torch was never the problem.** `torch` installs cleanly against TouchDesigner 2025's
Python 3.11 on Apple Silicon. The install simply died partway through the oversized
requirements list.

---

## Bring your own environment

`Envpath` (in TouchDesigner) or `TDAUTOBPM_PYTHON` (everywhere else) accepts any of:

- a path to a Python interpreter
- a venv directory
- a conda environment directory
- a bare conda environment **name**

When it is blank, resolution order is `TDAUTOBPM_PYTHON` → `$CONDA_PREFIX` →
`$VIRTUAL_ENV` → the project's `.venv`.

`tdautobpm doctor` prints every environment it can find and states, for each, whether it
can run in-process, as a sidecar, or not at all:

```
[conda:codellm-bench]
  /Users/you/miniconda3/envs/codellm-bench/bin/python3
      CPython 3.12.11 arm64  [conda]
      torch, torchaudio, numpy, soundfile
  in-process: no  - host is Python 3.11.15 (cp311), environment is Python 3.12.11 (cp312);
                    compiled extensions are not interchangeable across minor versions
  sidecar:    yes
```

## Two runtimes

| | Sidecar (default) | In-process |
|---|---|---|
| Where torch runs | separate process | inside TouchDesigner |
| Environment must match TD's Python | **no** | yes — same minor version *and* architecture |
| A native crash takes down TD | no | yes |
| Extra latency | one audio block | none |

Sidecar is the default because it removes the constraint that caused all the trouble: any
Python 3.9+ environment works, whatever TouchDesigner happens to ship. Switch with the
`Runtime` parameter.

## CLI

```
tdautobpm analyze FILE...   # offline tempo; --json for machine-readable output
tdautobpm listen            # live from a system input; --device-index N
tdautobpm devices           # list input devices
tdautobpm serve             # run the sidecar by hand
tdautobpm doctor            # diagnose environments
```

## Component outputs

| Channel | Meaning |
|---|---|
| `bpm` | detected tempo; 0 before the first estimate |
| `confidence` | posterior mass near the estimate, 0–1 |
| `beat` | 1 on the cook where the phase wraps, otherwise 0 |
| `phase` | free-running 0–1 beat phase at the detected tempo |

`Sync Tempo` writes `bpm` to the timeline tempo; `Autosync` does it continuously, gated by
`Autosync Min Confidence`.

## What it does and does not do

**Tempo, not beat position.** The model estimates *how fast*, not *where the downbeat is*.
`phase` and `beat` run freely at the detected tempo and are only realigned by `Reset`.
Treat `beat` as a metronome running at the right speed, not as an onset detector. Aligning
to actual beats is beat tracking — a different problem this model does not solve.

**Metrical ambiguity is real.** 140 and 70 BPM describe the same music, and the model
sometimes picks a different level than you would: on the bundled `test_tracks/yuqt.mp3` it
settles around 165 BPM while an autocorrelation baseline says 110 — the 3:2 level of the
same pulse. Neither is wrong. Use the `Estimate`/`Lock At Confidence` parameters, or halve
or double downstream, if you need a particular level.

**Confidence is the signal to trust.** On synthetic click tracks the model is accurate to
under 1 BPM wherever confidence exceeds ~0.5, and it correctly reports *low* confidence on
the cases it gets wrong (bare pulse trains under ~110 BPM). Gate on confidence rather than
assuming every reading is good — that is what `Autosync Min Confidence` is for.

## Changes to the engine

- package-relative imports, so no `sys.path` surgery is needed
- audio is accepted at its native rate (44.1/48 kHz) and resampled by a stateful streaming
  resampler; feeding each block to `torchaudio.functional.resample` independently produces
  a boundary artifact at the block rate, which is exactly the kind of periodicity a tempo
  model will lock onto
- `local_mean` replaces `mean` as the default estimator. The global posterior mean is
  biased: the posterior is broad and multimodal, so the mean lands between the modes and
  drifts toward the middle of the range. It reported 120/119/125 BPM for click tracks at
  90/128/174
- confidence is the probability mass *near* the mode rather than one 1-BPM bin's height,
  which sat near 0.04 even when the estimate was correct
- the 5-second posterior reset is configurable and can latch off once the estimate is
  stable, instead of unconditionally discarding all accumulated evidence
- `torch>=2.2,<3` rather than an exact pin, and pip/setuptools are never pinned

## Development

```bash
uv run pytest          # 71 tests
```

The TouchDesigner component is built by script rather than committed as a hand-made binary,
because `.toe`/`.tox` files are opaque blobs that cannot be reviewed or diffed. See
`touchdesigner/build_component.py`.

`tests/test_td_layer.py` covers the TouchDesigner-facing code that can be checked without
TouchDesigner: neither `AutoBpmExt.py` nor `build_component.py` may trust `__file__` (the
first lives inside the .tox with no file on disk; the second is `exec`'d, inheriting
whatever `__file__` the caller had), and neither may import torch at module scope, since
they run in TouchDesigner's interpreter. Building the component itself still has to be
verified in TouchDesigner.

## Credits

Model architecture, training and the `temponet_ckpt.pt` weights: **shhhum**
([tdautobpmsync](https://github.com/shhhum/tdautobpmsync)). See [NOTICE](NOTICE).
