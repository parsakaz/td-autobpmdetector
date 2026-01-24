import time
from typing import Optional, Tuple, Union
import math

import numpy as np
import soundfile as sf
import torch
import torchaudio

from model import AudioConfig, TempoNet

from dataclasses import dataclass


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
        device: str = "mps",
        min_window_s: float = 2.6666667,  # 4 beats @ 90 BPM
        max_window_s: float = 5.0,
        ramp_seconds: float = 5.0,
        reset_seconds: float = 5.0,
        update_hz: float = 1.0,
        smooth_alpha: float = 0.90,
        estimate: str = "mean",
        octave_beta: float = 0.5,
        use_octave_score: bool = True,
    ):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
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
        self.reset_seconds = float(reset_seconds)

        self.max_window_n = int(self.max_window_s * cfg.sample_rate)
        self.min_window_n = int(self.min_window_s * cfg.sample_rate)

        self.update_hz = float(update_hz)
        self.update_n = max(1, int(cfg.sample_rate / self.update_hz))
        self._since_update = 0
        self._since_reset = 0
        self._reset_n = max(1, int(self.reset_seconds * cfg.sample_rate))

        self.smooth_alpha = float(smooth_alpha)

        self.eps = 1e-8
        self.estimate = str(estimate)
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

        self._buffer = torch.zeros(self.max_window_n, dtype=torch.float32)
        self._write = 0
        self._filled = 0
        self._start_time = time.time()

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
            chunk_t = chunk_t.view(-1)

        n = int(chunk_t.numel())
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
        if self._since_reset >= self._reset_n:
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
            idx_mode = int(torch.argmax(score).item())
        else:
            idx_mode = int(torch.argmax(posterior).item())

        # BPM estimation options
        if self.estimate == "mean":
            bpm = float((posterior * self.bpm_bins).sum().item())
        elif self.estimate == "median":
            cdf = torch.cumsum(posterior, dim=-1)
            idx_med = int(
                torch.searchsorted(cdf, torch.tensor(0.5, device=cdf.device)).item()
            )
            bpm = float(self.bpm_min + idx_med)
        else:  # "mode"
            bpm = float(self.bpm_min + idx_mode)

        conf = float(posterior[idx_mode].item())
        win_s = float(n / self.cfg.sample_rate)
        return bpm, conf, win_s
