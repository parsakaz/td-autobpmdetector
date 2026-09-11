# Changelog

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
