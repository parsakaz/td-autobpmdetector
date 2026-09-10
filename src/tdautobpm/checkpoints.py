"""Locate the trained TempoNet checkpoint.

Works both from a source checkout (``models/temponet_ckpt.pt``) and from an installed
wheel, where the file is packaged as ``tdautobpm/_models/temponet_ckpt.pt``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

CHECKPOINT_NAME = "temponet_ckpt.pt"

#: Override the checkpoint location without touching code.
ENV_VAR = "TDAUTOBPM_CHECKPOINT"


def _search_paths() -> list:
    here = Path(__file__).resolve()
    return [
        here.parent / "_models" / CHECKPOINT_NAME,          # installed wheel
        here.parents[2] / "models" / CHECKPOINT_NAME,       # source checkout
        Path.cwd() / "models" / CHECKPOINT_NAME,
    ]


def find_checkpoint(explicit: Optional[str] = None) -> str:
    """Return the checkpoint path, raising FileNotFoundError with the paths tried."""
    if explicit:
        p = Path(os.path.expanduser(str(explicit)))
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"checkpoint not found: {p}")

    if os.environ.get(ENV_VAR):
        p = Path(os.path.expanduser(os.environ[ENV_VAR]))
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"{ENV_VAR} points at a missing file: {p}")

    tried = _search_paths()
    for p in tried:
        if p.is_file():
            return str(p)

    listed = "\n".join(f"  {p}" for p in tried)
    raise FileNotFoundError(
        f"Could not find {CHECKPOINT_NAME}. Looked in:\n{listed}\n"
        f"Set {ENV_VAR} to point at it."
    )
