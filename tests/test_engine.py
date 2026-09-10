"""Engine and resampler behaviour.

These use synthetic click tracks with a known tempo. Tempo estimation is inherently
ambiguous about the metrical level - 140 BPM and 70 BPM describe the same music - so
accuracy is asserted modulo the usual octave and triplet relationships.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tdautobpm.engine import make_predictor
from tdautobpm.resample import StreamResampler

SR = 44100


def click_track(bpm: float, duration: float = 40.0, sr: int = SR, seed: int = 0):
    """A percussive pulse train at `bpm`, with a light accent every 4th beat."""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    x = np.zeros(n, dtype=np.float32)

    period = 60.0 / bpm
    t = np.arange(int(0.05 * sr)) / sr
    env = np.exp(-t * 40)
    body = np.sin(2 * np.pi * (120 * np.exp(-t * 8)) * t) * env

    for k in range(int(duration / period)):
        i = int(k * period * sr)
        amp = 1.0 if k % 4 == 0 else 0.6
        seg = ((body + 0.3 * rng.standard_normal(len(t)) * env) * amp).astype(np.float32)
        seg = seg[: max(0, n - i)]
        x[i : i + len(seg)] += seg

    return x * 0.5


def metrical_error(estimate: float, truth: float) -> float:
    """Smallest error against the tempo and its common metrical relatives."""
    relatives = [truth, truth / 2, truth * 2, truth * 1.5, truth * 2 / 3,
                 truth / 3, truth * 3, truth * 3 / 4, truth * 4 / 3]
    return min(abs(estimate - r) for r in relatives)


def run(x, sr=SR, block=4096, **kwargs):
    kwargs.setdefault("reset_seconds", None)
    kwargs.setdefault("update_hz", 4.0)
    p = make_predictor(device="cpu", input_sample_rate=sr, **kwargs)
    for i in range(0, len(x), block):
        p.push_audio(x[i : i + block])
    return p


@pytest.mark.parametrize("bpm", [120, 128, 140, 150])
def test_tempo_within_one_bpm_of_a_metrical_level(bpm):
    result = run(click_track(bpm)).predict()
    assert result is not None
    estimate, confidence, _ = result
    assert metrical_error(estimate, bpm) < 1.0, f"got {estimate} for {bpm}"
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize("bpm", [90, 100, 110, 120, 128, 140, 150, 174])
def test_confidence_predicts_accuracy(bpm):
    """The contract users rely on: a confident estimate is a correct one.

    The model is not accurate at every tempo on these synthetic clicks - bare pulse
    trains under about 110 BPM are genuinely ambiguous and it reports, for instance,
    129.6 for a 100 BPM track. What must hold is that it does not report those
    confidently, because Autosync gates on confidence.
    """
    estimate, confidence, _ = run(click_track(bpm)).predict()
    if confidence > 0.5:
        assert metrical_error(estimate, bpm) < 1.0, (
            f"confident ({confidence:.2f}) but wrong: {estimate} for {bpm}"
        )


def test_confidence_is_meaningful_on_a_clean_pulse():
    """A steady click track should not report near-zero confidence.

    Upstream reported the height of a single 1-BPM bin, which sits around 0.04 even
    when the estimate is correct. Confidence is now the mass near the mode.
    """
    _, confidence, _ = run(click_track(128)).predict()
    assert confidence > 0.25


def test_mean_estimator_is_biased_toward_the_range_centre():
    """Documents why `local_mean` replaced `mean` as the default."""
    truth = 174
    biased, _, _ = run(click_track(truth), estimate="mean").predict()
    local, _, _ = run(click_track(truth), estimate="local_mean").predict()
    assert metrical_error(local, truth) < metrical_error(biased, truth)


def test_reset_clears_state():
    p = run(click_track(128))
    assert p.predict() is not None
    p.reset()
    assert p.predict() is None  # buffer empty again


def test_block_size_does_not_change_the_estimate():
    """Streaming must not depend on how the host happens to chunk audio."""
    x = click_track(128)
    a, _, _ = run(x, block=1024).predict()
    b, _, _ = run(x, block=8192).predict()
    assert abs(a - b) < 1.0


def test_accepts_native_sample_rate():
    """48 kHz input is resampled internally rather than misread as 22.05 kHz."""
    estimate, _, _ = run(click_track(128, sr=48000), sr=48000).predict()
    assert metrical_error(estimate, 128) < 1.5


class TestStreamResampler:
    def test_passthrough_when_rates_match(self):
        r = StreamResampler(44100, 44100)
        assert not r.needed
        x = torch.randn(1024)
        assert torch.equal(r.process(x), x)

    def test_output_rate_matches_the_ratio(self):
        r = StreamResampler(48000, 22050)
        total = sum(r.process(torch.randn(1024)).numel() for _ in range(200))
        expected = 200 * 1024 * 22050 / 48000
        assert abs(total - expected) < 0.01 * expected

    def test_no_drift_over_a_long_stream(self):
        """Per-block skip is constant, so error must not accumulate."""
        r = StreamResampler(44100, 22050)
        counts = [r.process(torch.randn(2048)).numel() for _ in range(500)]
        # Ignore the first block, which has no left context yet.
        assert max(counts[1:]) - min(counts[1:]) <= 1

    def test_block_boundaries_do_not_create_artifacts(self):
        """Blockwise output must match resampling the whole signal at once."""
        sr_in, sr_out = 48000, 22050
        t = torch.arange(48000) / sr_in
        x = torch.sin(2 * torch.pi * 440 * t).float()

        import torchaudio

        whole = torchaudio.functional.resample(x, sr_in, sr_out)

        r = StreamResampler(sr_in, sr_out)
        streamed = torch.cat([r.process(x[i : i + 777]) for i in range(0, len(x), 777)])

        n = min(len(whole), len(streamed)) - 64
        # Compare away from the very start, where context is still filling.
        diff = (whole[32:n] - streamed[32:n]).abs().max()
        assert diff < 1e-3, f"max deviation {diff}"

    def test_reset_restores_initial_behaviour(self):
        r = StreamResampler(44100, 22050)
        x = torch.randn(4096)
        first = r.process(x)
        r.reset()
        assert torch.equal(r.process(x), first)
