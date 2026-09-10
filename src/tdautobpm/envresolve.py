"""Locate and validate a Python environment that can run the BPM engine.

This module is imported *inside TouchDesigner*, before any third-party package is
reachable, so it is deliberately stdlib-only. Nothing here may import torch, numpy
or anything else from the target environment - it only ever inspects them from the
outside, by running a probe script in a subprocess.

Background on why this exists at all
------------------------------------
TouchDesigner ships its own CPython. The obvious move - build a venv on that
interpreter - has a trap on macOS. ``TouchDesigner.app`` is codesigned with
``com.apple.security.cs.disable-library-validation``, so third-party ``.so`` files
load fine *inside* the app. The bare ``python3.11`` binary inside the app bundle has
no such entitlement, so the very same venv fails outside TouchDesigner with::

    dlopen(.../numpy/_core/_multiarray_umath.cpython-311-darwin.so):
    mapping process and mapped file (non-platform) have different Team IDs

Which means a TD-interpreter venv can never be tested, scripted or CLI-driven from a
terminal. So we support pointing at *any* environment you already have - conda, venv,
uv, system - and pick a runtime mode that matches what that environment can do.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

#: Modules the engine actually imports at runtime.
REQUIRED_PACKAGES = ("torch", "torchaudio", "numpy", "soundfile")

#: Extra module needed only by ``tdautobpm listen``.
LIVE_PACKAGES = ("sounddevice",)

_PROBE = r"""
import json, platform, sys, sysconfig, importlib.util
mods = {}
for name in %(mods)r:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        spec = None
    mods[name] = spec is not None
paths = sysconfig.get_paths()
print("@@TDAUTOBPM@@" + json.dumps({
    "executable": sys.executable,
    "version": "%%d.%%d.%%d" %% sys.version_info[:3],
    "version_info": list(sys.version_info[:3]),
    "implementation": platform.python_implementation(),
    "machine": platform.machine(),
    "abi_tag": "cp%%d%%d" %% sys.version_info[:2],
    "purelib": paths.get("purelib"),
    "platlib": paths.get("platlib"),
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "modules": mods,
}))
"""

#: Signature macOS emits when library validation rejects a foreign-signed extension.
_TEAM_ID_MARKER = "different Team IDs"


class EnvError(RuntimeError):
    """Raised when an environment cannot be resolved or is unusable."""


@dataclasses.dataclass
class EnvInfo:
    """Everything we know about a candidate interpreter."""

    python: str
    version: str = ""
    version_info: tuple = ()
    implementation: str = ""
    machine: str = ""
    abi_tag: str = ""
    purelib: Optional[str] = None
    platlib: Optional[str] = None
    prefix: str = ""
    base_prefix: str = ""
    modules: dict = dataclasses.field(default_factory=dict)
    kind: str = "unknown"  # venv | conda | system | touchdesigner
    origin: str = ""  # how we found it, for diagnostics
    ok: bool = False
    error: str = ""
    library_validation_blocked: bool = False

    # -- derived ----------------------------------------------------------

    @property
    def missing(self) -> list:
        """Required modules this environment does not have."""
        return [m for m in REQUIRED_PACKAGES if not self.modules.get(m)]

    @property
    def complete(self) -> bool:
        """True when the environment can actually run the engine."""
        return self.ok and not self.missing

    @property
    def is_touchdesigner(self) -> bool:
        return self.kind == "touchdesigner"

    def site_packages(self) -> list:
        """Directories to append to ``sys.path`` for in-process use."""
        out = []
        for p in (self.purelib, self.platlib):
            if p and p not in out and os.path.isdir(p):
                out.append(p)
        return out

    def describe(self) -> str:
        if not self.ok:
            return f"{self.python}\n    unusable: {self.error}"
        mods = ", ".join(
            f"{m}{'' if self.modules.get(m) else ' (MISSING)'}" for m in REQUIRED_PACKAGES
        )
        return (
            f"{self.python}\n"
            f"    {self.implementation} {self.version} {self.machine}  [{self.kind}]\n"
            f"    {mods}"
        )


# ---------------------------------------------------------------------------
# path handling
# ---------------------------------------------------------------------------


def expand(path: str) -> str:
    """Expand ``~`` and ``$VARS`` in a user-supplied path.

    Not optional politeness: a literal, unexpanded ``~`` in the upstream setup flow
    installed an entire miniconda into a directory *named* ``~``. Every path that
    reaches this module from a parameter field or env var goes through here first.
    """
    return os.path.abspath(os.path.expanduser(os.path.expandvars(str(path).strip())))


def _classify(python: str, prefix: str = "", base_prefix: str = "") -> str:
    p = python.lower()
    if ".app/contents" in p and "touchdesigner" in p:
        return "touchdesigner"
    root = Path(prefix or python)
    if (root / "conda-meta").is_dir():
        return "conda"
    if "conda" in p or "miniforge" in p or "mamba" in p:
        return "conda"
    if prefix and base_prefix and prefix != base_prefix:
        return "venv"
    if (root / "pyvenv.cfg").is_file():
        return "venv"
    return "system"


def python_from_spec(spec: str) -> Optional[str]:
    """Turn a user-supplied spec into a python executable path.

    Accepts an interpreter path, a venv directory, a conda env directory, or a bare
    conda environment *name*.
    """
    if not spec:
        return None
    raw = str(spec).strip()
    if not raw:
        return None

    path = Path(expand(raw))

    # A direct interpreter.
    if path.is_file() and os.access(path, os.X_OK):
        return str(path)

    # An environment directory (venv, conda env, or a framework prefix).
    if path.is_dir():
        for rel in ("bin/python3", "bin/python", "Scripts/python.exe", "python.exe"):
            cand = path / rel
            if cand.is_file():
                return str(cand)

    # A bare conda environment name - only if it doesn't look like a path.
    if not any(sep in raw for sep in ("/", "\\")) and not raw.startswith("~"):
        found = conda_env_by_name(raw)
        if found:
            return found

    return None


def conda_env_by_name(name: str) -> Optional[str]:
    """Resolve a conda environment name to its interpreter, or None."""
    for env_dir in _conda_env_dirs():
        cand = Path(env_dir) / name
        for rel in ("bin/python3", "bin/python", "python.exe"):
            exe = cand / rel
            if exe.is_file():
                return str(exe)
    return None


def _conda_env_dirs() -> list:
    """Candidate directories that hold named conda environments."""
    dirs: list = []

    for raw in (os.environ.get("CONDA_ENVS_DIRS") or "").split(os.pathsep):
        if raw.strip():
            dirs.append(expand(raw))

    # Ask conda itself when it is on PATH; it knows about custom envs_dirs.
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if exe:
        try:
            out = subprocess.run(
                [exe, "info", "--json"], capture_output=True, text=True, timeout=20
            )
            if out.returncode == 0:
                info = json.loads(out.stdout)
                for d in info.get("envs_dirs") or []:
                    dirs.append(expand(d))
                # Also accept fully-resolved env paths conda already knows.
                for d in info.get("envs") or []:
                    parent = str(Path(expand(d)).parent)
                    if parent not in dirs:
                        dirs.append(parent)
        except Exception:
            pass

    for root in filter(None, [os.environ.get("CONDA_PREFIX"), "~/miniconda3",
                              "~/lib/miniconda3", "~/anaconda3", "~/miniforge3"]):
        d = Path(expand(root)) / "envs"
        if d.is_dir():
            dirs.append(str(d))

    seen, unique = set(), []
    for d in dirs:
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            unique.append(d)
    return unique


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------


def probe(python: str, origin: str = "", extra_modules: Iterable = ()) -> EnvInfo:
    """Inspect an interpreter by running a probe script inside it."""
    python = expand(python) if os.sep in str(python) else str(python)
    info = EnvInfo(python=python, origin=origin)

    if not os.path.isfile(python):
        info.error = "no such interpreter"
        info.kind = _classify(python)
        return info

    mods = list(REQUIRED_PACKAGES) + [m for m in extra_modules if m not in REQUIRED_PACKAGES]
    script = _PROBE % {"mods": mods}

    try:
        proc = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            # Keep an activated venv/conda from leaking into the probe.
            env={k: v for k, v in os.environ.items()
                 if k not in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")},
        )
    except subprocess.TimeoutExpired:
        info.error = "probe timed out"
        info.kind = _classify(python)
        return info
    except OSError as exc:
        info.error = f"could not execute: {exc}"
        info.kind = _classify(python)
        return info

    stderr = proc.stderr or ""
    marker = "@@TDAUTOBPM@@"
    line = next((l for l in (proc.stdout or "").splitlines() if l.startswith(marker)), None)

    if line is None:
        info.kind = _classify(python)
        if _TEAM_ID_MARKER in stderr:
            info.library_validation_blocked = True
            info.error = (
                "macOS library validation rejected this interpreter's extension modules "
                "(different Team IDs). This interpreter can only load third-party compiled "
                "packages from inside the signed host app."
            )
        else:
            info.error = (stderr.strip().splitlines() or ["probe produced no output"])[-1]
        return info

    data = json.loads(line[len(marker):])
    info.version = data["version"]
    info.version_info = tuple(data["version_info"])
    info.implementation = data["implementation"]
    info.machine = data["machine"]
    info.abi_tag = data["abi_tag"]
    info.purelib = data["purelib"]
    info.platlib = data["platlib"]
    info.prefix = data["prefix"]
    info.base_prefix = data["base_prefix"]
    info.modules = data["modules"]
    info.kind = _classify(python, data["prefix"], data["base_prefix"])
    info.ok = True

    # An interpreter can import-scan fine yet still be unable to *load* the compiled
    # extension. Confirm by actually importing numpy, the smallest real binary dep.
    if info.modules.get("numpy"):
        chk = subprocess.run(
            [python, "-c", "import numpy"], capture_output=True, text=True, timeout=120
        )
        if chk.returncode != 0 and _TEAM_ID_MARKER in (chk.stderr or ""):
            info.library_validation_blocked = True
            info.ok = False
            info.error = (
                "numpy is installed but macOS library validation refuses to load it in this "
                "interpreter (different Team IDs)."
            )

    return info


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def candidates(spec: Optional[str] = None, project_root: Optional[str] = None) -> list:
    """Ordered (origin, spec) pairs to try, best first."""
    out = []
    if spec:
        out.append(("explicit", spec))
    if os.environ.get("TDAUTOBPM_PYTHON"):
        out.append(("TDAUTOBPM_PYTHON", os.environ["TDAUTOBPM_PYTHON"]))
    if os.environ.get("CONDA_PREFIX"):
        out.append(("$CONDA_PREFIX", os.environ["CONDA_PREFIX"]))
    if os.environ.get("VIRTUAL_ENV"):
        out.append(("$VIRTUAL_ENV", os.environ["VIRTUAL_ENV"]))
    root = Path(project_root or default_project_root())
    out.append(("project .venv", str(root / ".venv")))
    return out


def default_project_root() -> str:
    """Repository root, inferred from this file's location."""
    return str(Path(__file__).resolve().parents[2])


def resolve(
    spec: Optional[str] = None,
    project_root: Optional[str] = None,
    require_complete: bool = True,
) -> EnvInfo:
    """Find the best usable environment.

    Raises :class:`EnvError` with an actionable message when nothing works.
    """
    attempts = []
    for origin, raw in candidates(spec, project_root):
        python = python_from_spec(raw)
        if not python:
            attempts.append((origin, raw, "not found"))
            continue
        info = probe(python, origin=origin)
        if info.complete or (info.ok and not require_complete):
            return info
        attempts.append((origin, python, info.error or f"missing: {', '.join(info.missing)}"))

    lines = [f"  {origin}: {what}\n      -> {why}" for origin, what, why in attempts]
    raise EnvError(
        "No usable Python environment found for the BPM engine.\n"
        + "\n".join(lines)
        + "\n\nFix it with either:\n"
        "  uv venv --python 3.11 && uv pip install -e '.[live]'\n"
        "or point at an environment you already have:\n"
        "  export TDAUTOBPM_PYTHON=/path/to/env      # or a conda env name\n"
        "Then run `tdautobpm doctor` to confirm."
    )


# ---------------------------------------------------------------------------
# in-process compatibility
# ---------------------------------------------------------------------------


def host_signature() -> dict:
    """ABI signature of the interpreter this code is running in (i.e. TouchDesigner)."""
    import platform

    return {
        "abi_tag": "cp%d%d" % sys.version_info[:2],
        "machine": platform.machine(),
        "implementation": platform.python_implementation(),
        "version": "%d.%d.%d" % sys.version_info[:3],
    }


def inprocess_incompatibility(info: EnvInfo) -> Optional[str]:
    """Why ``info`` cannot be imported into the current process, or None if it can.

    In-process use means appending the environment's ``site-packages`` to our own
    ``sys.path``. Compiled extensions are built per (CPython minor version, architecture),
    so both must match the host exactly.
    """
    host = host_signature()
    if not info.ok:
        return info.error or "environment is not usable"
    if info.implementation != host["implementation"]:
        return (
            f"host is {host['implementation']}, environment is {info.implementation}"
        )
    if info.abi_tag != host["abi_tag"]:
        return (
            f"host is Python {host['version']} ({host['abi_tag']}), environment is "
            f"Python {info.version} ({info.abi_tag}); compiled extensions are not "
            f"interchangeable across minor versions"
        )
    if info.machine != host["machine"]:
        return f"host is {host['machine']}, environment is {info.machine}"
    if info.missing:
        return "missing packages: " + ", ".join(info.missing)
    return None


def sidecar_incompatibility(info: EnvInfo) -> Optional[str]:
    """Why ``info`` cannot host the sidecar, or None if it can.

    Far more permissive than in-process: a separate process only needs a working
    interpreter with the packages installed. Version and architecture are its own business.
    """
    if not info.ok:
        return info.error or "environment is not usable"
    if info.version_info and info.version_info < (3, 9):
        return f"Python {info.version} is too old; need 3.9+"
    if info.missing:
        return "missing packages: " + ", ".join(info.missing)
    return None
