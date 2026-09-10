"""Stateful sample-rate conversion for streaming audio.

TouchDesigner runs at 44.1 or 48 kHz; TempoNet expects 22.05 kHz. Calling
``torchaudio.functional.resample`` on each incoming block independently is wrong:
the polyphase filter has to invent data past both edges of every block, so each
block boundary gets a transient. Feeding that into an onset-sensitive tempo model
produces a periodic artifact at the block rate - exactly the kind of thing a tempo
estimator will happily lock onto.

:class:`StreamResampler` keeps a tail of input samples as left context, resamples
``context + block`` together, and discards the output belonging to the context.
"""

from __future__ import annotations

import math

import torch
import torchaudio


class StreamResampler:
    """Resample a continuous stream block by block, without boundary artifacts.

    The number of samples returned per call varies by a sample or two, but the
    long-run output rate is exact: the context length is constant after the first
    block, so the per-block skip is constant too and no drift accumulates.
    """

    def __init__(
        self,
        orig_freq: int,
        new_freq: int,
        lowpass_filter_width: int = 6,
        dtype: torch.dtype = torch.float32,
    ):
        self.orig_freq = int(orig_freq)
        self.new_freq = int(new_freq)
        self.lowpass_filter_width = int(lowpass_filter_width)
        self.dtype = dtype
        self.ratio = self.new_freq / self.orig_freq

        # Polyphase resampling has an internal phase that cycles with a period of
        # `orig_freq / gcd` input samples. If the context we prepend is not a whole
        # number of those cycles, the output lands on a different grid than it would
        # have in a single-shot resample, and every block is offset. So the context is
        # rounded up to a multiple of the reduced rate - which also makes the number of
        # output samples to discard exact (an integer), leaving no rounding drift.
        g = math.gcd(self.orig_freq, self.new_freq) or 1
        self._orig_r = self.orig_freq // g
        self._new_r = self.new_freq // g

        if self.needed:
            # Support of the sinc kernel, in input samples, on each side.
            half = int(math.ceil(
                self.lowpass_filter_width * self._orig_r / min(self._orig_r, self._new_r)
            ))
            cycles = max(1, int(math.ceil(2 * half / self._orig_r)))
            self.context = cycles * self._orig_r
            self._skip = cycles * self._new_r
        else:
            self.context = 0
            self._skip = 0

        #: left context, always exactly `context` samples
        self._left = torch.zeros(self.context, dtype=dtype)
        #: input received but not yet committed (the right guard)
        self._hold = torch.zeros(0, dtype=dtype)
        # torchaudio.functional.resample rebuilds its sinc kernel on every call; the
        # transform caches it.
        self._transform = None

    @property
    def needed(self) -> bool:
        """False when input and output rates match and we can pass audio through."""
        return self.orig_freq != self.new_freq

    def reset(self) -> None:
        """Forget stream position. The cached kernel is kept."""
        self._left = torch.zeros(self.context, dtype=self.dtype)
        self._hold = torch.zeros(0, dtype=self.dtype)

    def __call__(self, block: torch.Tensor) -> torch.Tensor:
        return self.process(block)

    def process(self, block: torch.Tensor) -> torch.Tensor:
        """Resample one block, returning only samples that are fully determined.

        Output is delayed by ``context`` input samples - 6.7 ms at 48 kHz, 0.5 ms at
        44.1 kHz - because the filter needs samples on *both* sides of each output
        sample. Emitting right up to the end of each block would mean computing those
        samples against the zero padding that follows, which is where per-block
        boundary artifacts come from.
        """
        if block.ndim > 1:
            block = block.reshape(-1)
        block = block.to(dtype=self.dtype)

        if not self.needed:
            return block
        if block.numel() == 0:
            return torch.zeros(0, dtype=self.dtype)

        data = torch.cat([self._hold, block]) if self._hold.numel() else block

        # Withhold `context` samples as right-hand context, and commit a whole number
        # of polyphase cycles so every index below is exact rather than rounded.
        usable = data.numel() - self.context
        n_emit = (usable // self._orig_r) * self._orig_r if usable > 0 else 0
        if n_emit <= 0:
            self._hold = data
            return torch.zeros(0, dtype=self.dtype)

        payload = data[:n_emit]
        self._hold = data[n_emit:]

        buf = torch.cat([self._left, payload, self._hold[: self.context]])
        out = self._resample(buf)

        start = (self.context // self._orig_r) * self._new_r
        count = (n_emit // self._orig_r) * self._new_r
        result = out[start : start + count]

        self._left = torch.cat([self._left, payload])[-self.context :].clone()
        return result

    def _resample(self, buf: torch.Tensor) -> torch.Tensor:
        """Resample, reusing a cached kernel across calls."""
        if self._transform is None:
            self._transform = torchaudio.transforms.Resample(
                self.orig_freq,
                self.new_freq,
                lowpass_filter_width=self.lowpass_filter_width,
                dtype=self.dtype,
            )
        return self._transform(buf)
