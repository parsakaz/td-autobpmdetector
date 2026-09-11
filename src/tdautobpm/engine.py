"""Streaming tempo estimation with TempoNet.

Derived from ``temponet/infer.py`` in shhhum's tdautobpmsync
(https://github.com/shhhum/tdautobpmsync), which also provides the TempoNet model and
its trained weights. On top of the original this adds:

* audio at its native rate, through a stateful streaming resampler
  (:mod:`tdautobpm.resample`), instead of requiring 22.05 kHz input;
* a local-mean estimator and a confidence measured as probability mass near the mode;
* an optional, latchable periodic reset of the accumulated posterior;
* a tempo range that the estimate is folded into by octaves (:meth:`set_range`);
* live reconfiguration (:meth:`configure`).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio

from .model import AudioConfig, TempoNet
from .resample import StreamResampler

log = logging.getLogger("tdautobpm.engine")


@dataclass
class MelCfg:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 30.0
    f_max: Optional[float] = None  # None => sample_rate/2
    power: float = 2.0  # power spectrogram
    log_eps: float = 1e-5
    log_base: str = "ln"  # "ln" or "log10"
    center: bool = True  # center=True => pad n_fft//2
    # Normalize each track's logmel
    do_norm: bool = True
    norm_clip: float = 8.0  # clip z-score to [-clip, clip]


def select_logmel_device(
    prefer_device: Optional[Union[str, torch.device]] = None,
) -> torch.device:
    """
    Choose a device for log-mel computation.
    - CUDA preferred when available.
    - MPS is avoided for STFT; falls back to CPU.
    - If prefer_device is set and unavailable, falls back to CPU.
    """
    if prefer_device is not None:
        pref = str(prefer_device).lower()
        if pref.startswith("cuda"):
            return (
                torch.device(pref) if torch.cuda.is_available() else torch.device("cpu")
            )
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "mps":
            # torch.stft has no MPS kernel for this configuration; the model itself
            # still runs on MPS, only the front-end spectrogram falls back.
            log.debug("MPS requested for log-mel; using CPU for STFT (no MPS kernel)")
            return torch.device("cpu")
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_mel_filter(cfg: MelCfg, device: torch.device) -> torch.Tensor:
    f_max = cfg.f_max if cfg.f_max is not None else cfg.sample_rate / 2
    # torchaudio fbanks: [n_freqs, n_mels]
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=cfg.n_fft // 2 + 1,
        f_min=cfg.f_min,
        f_max=f_max,
        n_mels=cfg.n_mels,
        sample_rate=cfg.sample_rate,
        norm="slaney",
        mel_scale="htk",
    ).to(device=device, dtype=torch.float32)
    # We'll multiply: [n_freqs, T] -> [n_mels, T] using fb.T
    return fb.T.contiguous()  # [n_mels, n_freqs]


@torch.no_grad()
def wav_to_logmel(
    wav: torch.Tensor,
    cfg: MelCfg,
    window: torch.Tensor,
    mel_fb_t: torch.Tensor,
    return_cpu: bool = True,
) -> torch.Tensor:
    """
    wav: [T] float32
    returns logmel: [n_mels, T_frames] float32
    """
    if wav.ndim == 2 and wav.shape[0] == 1:
        wav = wav.squeeze(0)
    if wav.ndim != 1:
        raise ValueError(f"Expected 1D wav tensor, got shape {tuple(wav.shape)}")

    # STFT
    spec = torch.stft(
        wav,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop,
        win_length=cfg.win_length,
        window=window,
        center=cfg.center,
        return_complex=True,
    )  # [n_freqs, T]
    mag = spec.abs()
    if cfg.power == 2.0:
        mag = mag * mag
    elif cfg.power != 1.0:
        mag = mag.pow(cfg.power)

    # mel
    mel = mel_fb_t @ mag  # [n_mels, T]
    mel = mel.clamp_min(cfg.log_eps)

    if cfg.log_base == "log10":
        logmel = torch.log10(mel)
    else:
        logmel = torch.log(mel)

    if cfg.do_norm:
        mu = logmel.mean()
        sig = logmel.std().clamp_min(1e-6)
        logmel = (logmel - mu) / sig
        if cfg.norm_clip is not None and cfg.norm_clip > 0:
            logmel = logmel.clamp(-cfg.norm_clip, cfg.norm_clip)

    logmel = logmel.to(dtype=torch.float32)
    if return_cpu:
        return logmel.cpu()
    return logmel


def _mel_cfg_from_audio(cfg: AudioConfig) -> MelCfg:
    f_max = getattr(cfg, "f_max", None)
    if f_max is not None and float(f_max) <= 0:
        f_max = None
    return MelCfg(
        sample_rate=int(cfg.sample_rate),
        n_fft=int(cfg.n_fft),
        hop=int(cfg.hop_length),
        win_length=int(cfg.win_length),
        n_mels=int(cfg.n_mels),
        f_min=float(getattr(cfg, "f_min", 30.0)),
        f_max=(float(f_max) if f_max is not None else None),
        power=2.0,
        log_base="ln",
        do_norm=True,
        norm_clip=8.0,
        center=True,
    )


# -----------------------------
# Streaming predictor (accumulates evidence over time)
# -----------------------------
class TempoStreamPredictorAccum:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        min_window_s: float = 2.6666667,  # 4 beats @ 90 BPM
        max_window_s: float = 5.0,
        ramp_seconds: float = 5.0,
        reset_seconds: Optional[float] = 5.0,
        update_hz: float = 1.0,
        smooth_alpha: float = 0.90,
        estimate: str = "local_mean",
        octave_beta: float = 0.5,
        use_octave_score: bool = True,
        input_sample_rate: Optional[int] = None,
        lock_confidence: float = 0.0,
        range_min: float = 0.0,
        range_max: float = 0.0,
    ):
        """
        Args:
            range_min, range_max: the tempo range the music is known to be in, e.g.
                160-180 for drum and bass. The estimate is folded by octaves into it,
                so a posterior that favours 85 reports 170. ``0`` for either leaves
                the model's full range; see :meth:`set_range`.
            input_sample_rate: rate of the audio handed to :meth:`push_audio`. When it
                differs from the model's rate the stream is resampled internally. Pass
                ``None`` to declare the audio is already at the model rate.
            reset_seconds: how often to flatten the accumulated posterior back to
                uniform. The original fixed this at 5 s, which throws away all evidence
                several times a minute and keeps reported confidence very low. Pass
                ``None`` or ``0`` to accumulate indefinitely.
            lock_confidence: once the posterior mode reaches this confidence, stop
                resetting and keep accumulating. ``0`` disables latching.
        """
        # weights_only: a checkpoint is data, and must not be able to run code the
        # way an arbitrary pickle can (the default before torch 2.6).
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        cfg = AudioConfig(**ckpt["cfg"])
        self.cfg = cfg
        self.device = device

        self.bpm_min = int(ckpt["bpm_min"])
        self.bpm_max = int(ckpt["bpm_max"])
        self.num_bins = self.bpm_max - self.bpm_min + 1

        self.n_mels = cfg.n_mels
        self.hop = cfg.hop_length

        self.model = TempoNet(
            n_mels=cfg.n_mels,
            bpm_min=self.bpm_min,
            bpm_max=self.bpm_max,
        ).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model_device = next(self.model.parameters()).device

        self.min_window_s = float(min_window_s)
        self.max_window_s = float(max_window_s)
        self.ramp_seconds = float(ramp_seconds)
        self.reset_seconds = (
            None if reset_seconds is None or float(reset_seconds) <= 0 else float(reset_seconds)
        )
        self.lock_confidence = float(lock_confidence)
        self._locked = False

        self.max_window_n = int(self.max_window_s * cfg.sample_rate)
        self.min_window_n = int(self.min_window_s * cfg.sample_rate)

        self.update_hz = float(update_hz)
        self.update_n = max(1, int(cfg.sample_rate / self.update_hz))
        self._since_update = 0
        self._since_reset = 0
        self._reset_n = (
            max(1, int(self.reset_seconds * cfg.sample_rate))
            if self.reset_seconds is not None
            else None
        )

        self.smooth_alpha = float(smooth_alpha)

        self.eps = 1e-8
        self.estimate = str(estimate)
        #: half-width, in 1-BPM bins, of the window used by "local_mean"
        self.local_window = 3
        #: half-width, in 1-BPM bins, over which confidence mass is summed
        self.conf_window = 2
        self.octave_beta = float(octave_beta)
        self.use_octave_score = bool(use_octave_score)

        # log-posterior state for stable exponential smoothing
        self.log_posterior = torch.full(
            (self.num_bins,),
            fill_value=-math.log(self.num_bins),
            device=device,
            dtype=torch.float32,
        )

        # Precompute BPM values and octave index maps for optional octave-consistent scoring
        self.bpm_bins = torch.arange(
            self.bpm_min, self.bpm_max + 1, device=device, dtype=torch.float32
        )

        half_bpm = torch.round(self.bpm_bins * 0.5).to(torch.int64)
        dbl_bpm = torch.round(self.bpm_bins * 2.0).to(torch.int64)

        self.idx_half = (half_bpm - self.bpm_min).clamp(0, self.num_bins - 1)
        self.idx_double = (dbl_bpm - self.bpm_min).clamp(0, self.num_bins - 1)

        self.valid_half = (half_bpm >= self.bpm_min) & (half_bpm <= self.bpm_max)
        self.valid_double = (dbl_bpm >= self.bpm_min) & (dbl_bpm <= self.bpm_max)

        self.range = None
        self._fold = None
        self.set_range(range_min, range_max)

        # Cached log-mel components (no checkpoint dependency)
        self.logmel_cfg = _mel_cfg_from_audio(cfg)
        self.logmel_device = select_logmel_device(device)
        logmel_dev = self.logmel_device
        self._stft_window = torch.hann_window(
            int(self.logmel_cfg.win_length),
            periodic=True,
            device=logmel_dev,
            dtype=torch.float32,
        )
        self._mel_fb_t = build_mel_filter(self.logmel_cfg, device=logmel_dev)

        # Accept audio at whatever rate the host runs at.
        self.input_sample_rate = int(input_sample_rate or cfg.sample_rate)
        self._resampler = StreamResampler(self.input_sample_rate, cfg.sample_rate)
        if self._resampler.needed:
            log.debug(
                "resampling input %d Hz -> %d Hz", self.input_sample_rate, cfg.sample_rate
            )

        self._buffer = torch.zeros(self.max_window_n, dtype=torch.float32)
        self._write = 0
        self._filled = 0
        self._start_time = time.time()
        self._last_conf = 0.0

    def set_range(self, range_min: float = 0.0, range_max: float = 0.0) -> None:
        """Restrict the estimate to a tempo range, folding it there by octaves.

        Metrical ambiguity is the usual failure: drum and bass at 170 carries as much
        evidence for 85. Given a range, only bins with an octave image inside it (the
        bin itself, preferably, else half or double it, and so on) can be the mode,
        and the estimate is reported at that image. Evidence is not thrown away:
        the model still sees everything, and the octave-consistent score still pools
        the half and double of each candidate.

        A range with no image of any bin, or a blank one (either end ``<= 0``), leaves
        the full model range.
        """
        lo, hi = float(range_min or 0.0), float(range_max or 0.0)
        self.range = None
        self._fold = None
        if lo <= 0 or hi <= 0 or hi <= lo:
            return

        # Per bin, the power of two that lands it in range, nearest octave first.
        fold = torch.zeros_like(self.bpm_bins)
        for k in (0, 1, -1, 2, -2):
            factor = 2.0 ** k
            image = self.bpm_bins * factor
            fits = (fold == 0) & (image >= lo) & (image <= hi)
            fold = torch.where(fits, torch.full_like(fold, factor), fold)
        if not bool((fold > 0).any()):
            log.warning("tempo range %.0f-%.0f holds no octave of %d-%d BPM; ignoring it",
                        lo, hi, self.bpm_min, self.bpm_max)
            return
        self.range = (lo, hi)
        self._fold = fold

    def configure(self, **cfg) -> None:
        """Change settings on a running predictor, without losing its evidence.

        Covers everything but the checkpoint, device and input rate, which need a new
        predictor. Unknown keys are ignored, so a whole settings dict can be passed.
        """
        if cfg.get("update_hz"):
            self.update_hz = float(cfg["update_hz"])
            self.update_n = max(1, int(self.cfg.sample_rate / self.update_hz))
        if "estimate" in cfg:
            self.estimate = str(cfg["estimate"])
        if "smooth_alpha" in cfg:
            self.smooth_alpha = float(cfg["smooth_alpha"])
        if "lock_confidence" in cfg:
            self.lock_confidence = float(cfg["lock_confidence"])
        if "reset_seconds" in cfg:
            rs = cfg["reset_seconds"]
            self.reset_seconds = None if not rs or float(rs) <= 0 else float(rs)
            self._reset_n = (
                max(1, int(self.reset_seconds * self.cfg.sample_rate))
                if self.reset_seconds is not None
                else None
            )
        if "range_min" in cfg or "range_max" in cfg:
            lo, hi = self.range or (0.0, 0.0)
            self.set_range(cfg.get("range_min", lo), cfg.get("range_max", hi))

    def fold_bpm(self, bpm: float) -> float:
        """Move a tempo into the range by octaves, if it can be; else unchanged."""
        if self.range is None or bpm <= 0:
            return bpm
        lo, hi = self.range
        for k in (0, 1, -1, 2, -2):
            if lo <= bpm * 2.0 ** k <= hi:
                return bpm * 2.0 ** k
        return bpm

    def reset(self) -> None:
        """Flatten the posterior, clear the audio buffer and unlatch."""
        self._reset_posterior()
        self._locked = False
        self._last_conf = 0.0
        self._buffer.zero_()
        self._write = 0
        self._filled = 0
        self._resampler.reset()

    def _reset_posterior(self) -> None:
        self.log_posterior.fill_(-math.log(self.num_bins))
        self._since_update = 0
        self._since_reset = 0
        self._start_time = time.time()

    def push_audio(
        self, chunk: Union[np.ndarray, torch.Tensor]
    ) -> Optional[Tuple[float, float, float]]:
        if isinstance(chunk, np.ndarray):
            chunk_t = torch.from_numpy(chunk.astype(np.float32))
        else:
            chunk_t = chunk.detach().float().cpu()

        if chunk_t.ndim > 1:
            # Mix an interleaved/multi-channel block down to mono.
            chunk_t = chunk_t.mean(dim=-1) if chunk_t.shape[-1] <= 8 else chunk_t.view(-1)

        chunk_t = self._resampler.process(chunk_t)

        n = int(chunk_t.numel())
        if n == 0:
            return None
        i = 0
        while i < n:
            space = self.max_window_n - self._write
            take = min(space, n - i)
            self._buffer[self._write : self._write + take] = chunk_t[i : i + take]
            self._write = (self._write + take) % self.max_window_n
            i += take

        self._filled = min(self.max_window_n, self._filled + n)

        self._since_update += n
        self._since_reset += n
        if self._reset_n is not None and not self._locked and self._since_reset >= self._reset_n:
            self._reset_posterior()
        if self._since_update < self.update_n:
            return None
        self._since_update = 0
        return self.predict()

    def _read_last_n(self, n: int) -> torch.Tensor:
        n = min(n, self._filled)
        if n <= 0:
            return torch.zeros(0, dtype=torch.float32)

        end = self._write
        start = (end - n) % self.max_window_n

        if start < end:
            return self._buffer[start:end].clone()
        else:
            return torch.cat([self._buffer[start:], self._buffer[:end]], dim=0).clone()

    def _current_window_n(self) -> int:
        t = time.time() - self._start_time
        frac = min(1.0, max(0.0, t / self.ramp_seconds))
        n = int(self.min_window_n + frac * (self.max_window_n - self.min_window_n))
        return min(n, self._filled)

    @torch.no_grad()
    def predict(self) -> Optional[Tuple[float, float, float]]:
        n = self._current_window_n()
        if n < self.min_window_n:
            return None

        wav = self._read_last_n(n).to(self.logmel_device)
        if wav.numel() < 1024:
            return None

        # Convert waveform window to log-mel on logmel device
        logmel = wav_to_logmel(
            wav,
            self.logmel_cfg,
            window=self._stft_window,
            mel_fb_t=self._mel_fb_t,
            return_cpu=(self.logmel_device.type == "cpu"),
        )
        logmel = logmel.unsqueeze(0)  # [1, n_mels, T]
        if logmel.device != self.model_device:
            logmel = logmel.to(self.model_device)

        logits = self.model(logmel)
        probs = torch.softmax(logits[0], dim=-1)

        # Exponential smoothing in log-space: log q_t = a log q_{t-1} + (1-a) log p_t
        a = self.smooth_alpha
        self.log_posterior = a * self.log_posterior + (1.0 - a) * torch.log(
            probs.clamp_min(self.eps)
        )

        # Convert to normalized posterior
        posterior = torch.softmax(self.log_posterior, dim=-1)

        # Optional octave-consistent scoring to pick a stable mode
        if self.use_octave_score:
            score = posterior.clone()
            if self.octave_beta > 0:
                score = score + self.octave_beta * torch.where(
                    self.valid_half, posterior[self.idx_half], torch.zeros_like(score)
                )
                score = score + self.octave_beta * torch.where(
                    self.valid_double,
                    posterior[self.idx_double],
                    torch.zeros_like(score),
                )
        else:
            score = posterior
        if self._fold is not None:
            # Only bins with an octave image in the range may be the mode.
            score = torch.where(self._fold > 0, score, torch.full_like(score, -1.0))
        idx_mode = int(torch.argmax(score).item())

        # BPM estimation options.
        #
        # "mean" - the global posterior mean - is the original default and is badly
        # biased: the posterior is broad and usually multimodal (the true tempo plus
        # its half/double octaves), so the mean lands between the modes and drifts
        # toward the middle of the BPM range regardless of the actual tempo. On
        # synthetic click tracks at 90/128/174 BPM it reports 120/119/125. It is kept
        # for compatibility but is not the default.
        #
        # "local_mean" takes the centroid of a narrow window around the mode: it picks
        # the right octave like "mode" does, but keeps sub-BPM resolution.
        if self.estimate == "mean":
            bpm = float((posterior * self.bpm_bins).sum().item())
        elif self.estimate == "median":
            cdf = torch.cumsum(posterior, dim=-1)
            idx_med = int(
                torch.searchsorted(cdf, torch.tensor(0.5, device=cdf.device)).item()
            )
            bpm = float(self.bpm_min + idx_med)
        elif self.estimate == "mode":
            bpm = float(self.bpm_min + idx_mode)
        else:  # "local_mean" (default)
            lo = max(0, idx_mode - self.local_window)
            hi = min(self.num_bins, idx_mode + self.local_window + 1)
            w = posterior[lo:hi]
            mass = w.sum().clamp_min(self.eps)
            bpm = float(((w * self.bpm_bins[lo:hi]).sum() / mass).item())

        # Report the estimate at its image in the range. Mode-based estimates use the
        # mode's own octave, so a local mean of 79.6 around a mode of 80 becomes 159.2
        # rather than being judged on its own and left out of range.
        if self._fold is not None:
            if self.estimate in ("mode", "local_mean"):
                bpm *= float(self._fold[idx_mode].item())
            else:
                bpm = self.fold_bpm(bpm)
            # The local mean can spill a little past an edge; a range is a promise.
            bpm = min(max(bpm, self.range[0]), self.range[1])

        # Confidence as the probability mass near the mode, not the single-bin height.
        # With 141 one-BPM bins a correct-but-slightly-spread estimate scores ~0.04 on
        # the bare peak, which reads as "broken" when it is not.
        lo = max(0, idx_mode - self.conf_window)
        hi = min(self.num_bins, idx_mode + self.conf_window + 1)
        conf = float(posterior[lo:hi].sum().item())
        self._last_conf = conf
        if self.lock_confidence > 0 and conf >= self.lock_confidence:
            self._locked = True

        win_s = float(n / self.cfg.sample_rate)
        return bpm, conf, win_s


# -----------------------------
# Convenience constructors
# -----------------------------
def make_predictor(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    input_sample_rate: Optional[int] = None,
    **kwargs,
) -> "TempoStreamPredictorAccum":
    """Build a predictor, locating the bundled checkpoint when none is given."""
    from .checkpoints import find_checkpoint

    return TempoStreamPredictorAccum(
        find_checkpoint(checkpoint_path),
        device=device,
        input_sample_rate=input_sample_rate,
        **kwargs,
    )


def analyze_file(
    path: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    block: int = 4096,
    accumulate: bool = True,
    update_hz: float = 4.0,
    progress=None,
    range_min: float = 0.0,
    range_max: float = 0.0,
) -> dict:
    """Estimate the tempo of a whole audio file.

    With ``accumulate=True`` the posterior is never reset, so evidence from the entire
    file combines into one estimate - the right behaviour offline, and the reason this
    reports far higher confidence than the live stream does.
    """
    import soundfile as sf

    with sf.SoundFile(path) as f:
        sr = f.samplerate
        pred = make_predictor(
            checkpoint_path,
            device=device,
            input_sample_rate=sr,
            reset_seconds=None if accumulate else 5.0,
            update_hz=update_hz,
            range_min=range_min,
            range_max=range_max,
        )

        history = []
        n_read = 0
        while True:
            data = f.read(block, dtype="float32", always_2d=True)
            if len(data) == 0:
                break
            n_read += len(data)
            out = pred.push_audio(data.mean(axis=1))
            if out is not None:
                history.append(out)
                if progress is not None:
                    progress(n_read / max(1, f.frames), out)

    final = pred.predict() or (history[-1] if history else None)
    if final is None:
        raise ValueError(f"{path}: too short to estimate a tempo")

    bpm, conf, _ = final
    return {
        "path": path,
        "bpm": bpm,
        "confidence": conf,
        "duration_s": n_read / sr if sr else 0.0,
        "sample_rate": sr,
        "updates": len(history),
    }
