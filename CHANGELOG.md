# Changelog

## v0.2.1 - 2026-09-12

- Fix a thread conflict when creating an environment: the thread reading the
  installer's output wrote to the log DAT and the textport directly, and
  TouchDesigner objects may only be touched from the main thread. It now collects
  lines for the main thread to report, which also stops the stray thread tracebacks
  that appeared against whichever DAT had last cooked.

## v0.2.0 - 2026-09-12

The .tox now sets itself up, so TouchDesigner users need no terminal, no checkout and
no Python of their own.

- **Install button**: with no environment, the panel offers to build one. It creates a
  Python environment and installs this project's wheel (which carries the model) into
  it, in the background, with progress in the status line and a transcript in the
  `setup_log` DAT.
- It builds on a Python 3.10-3.13 found on the machine, keeping the detector in its
  own process; failing that on TouchDesigner's own Python, switching `Runtime` to
  in-process, which macOS requires for such an environment.
- Environments go in a shared per-user folder, so every project reuses one download.
  New *Setup* page: Environment Folder, Package, Base Python, Create Environment and
  Cancel Setup.
- The component carries the TouchDesigner-side modules and the genre presets, so a
  downloaded .tox works on its own. A checkout, when there is one, still wins.
- Wheels are attached to releases from this version on.

## v0.1.1 - 2026-09-11

Security hardening of the detector process. All of these need someone already able to
run code on the same machine.

- The model checkpoint is loaded as data only, so a checkpoint file cannot execute
  code.
- A connected client can no longer choose which checkpoint the detector loads.
- The detector's Unix socket is owner-only, so other users on the machine cannot
  connect to it; `serve --port` notes that TCP has no such protection.
- Starting the detector replaces a stale socket but never deletes any other file.

## v0.1.0 - 2026-09-11

First release.

**TouchDesigner component** (`touchdesigner/AutoBpm.tox`)

- Tempo, confidence, beat and phase as CHOP channels, from any audio CHOP.
- Control panel: tempo readout coloured by mode, confidence meter with a draggable
  threshold, beat ring with phase arc, beat pulse and beats-in-bar, detection range
  strip with draggable handles.
- Detection range with genre presets read from an editable `presets.csv`. The estimate
  is folded into the range by octaves, so 87 BPM drum & bass reads 174.
- Hold: keeps the last confident tempo while confidence is below the threshold.
- Tap tempo from a button, the space bar or a pulse; a single tap aligns the beat.
- ÷2 / ×2 tempo multiplier.
- Autosync of the timeline tempo, gated on confidence.
- Settings apply to the running detector live, without losing what it has heard.
- Runs the model in a separate process by default, so any Python 3.9+ environment
  works and a crash in torch cannot take TouchDesigner down; the process exits with
  TouchDesigner.
- `Diagnose()` reports the environment, input and detector state.

**Python package and CLI** (`tdautobpm`)

- `analyze`, `listen`, `devices`, `serve` and `doctor` commands, with `--range`.
- Environment resolution across interpreters, venvs and conda environments.
- Streaming resampling from the host's sample rate, a local-mean estimator and a
  confidence measured as probability mass near the estimate.

The TempoNet model and weights are by shhhum, from
[tdautobpmsync](https://github.com/shhhum/tdautobpmsync).
