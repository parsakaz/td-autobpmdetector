"""Creating a Python environment for the detector, for people who have none.

Someone who downloads only the .tox has no checkout, and quite possibly no Python
with torch in it. This module works out where such an environment should go, which
interpreter to build it on, and what to run; the caller runs the commands, because
inside TouchDesigner they have to run without blocking the frame.

Stdlib only: it is imported by TouchDesigner's own interpreter.
"""

from __future__ import annotations

import os
import subprocess
import sys

#: Interpreters to build on, best first. Torch publishes wheels for these, and a
#: version the host has never heard of is a worse bet than one it has.
PREFERRED_VERSIONS = ("3.12", "3.11", "3.13", "3.10")

#: Where to look for them, beyond PATH.
SEARCH_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin")

#: Where released wheels live. Each carries the engine, the sidecar and the model
#: weights, so installing one is all an environment needs.
RELEASES = "https://github.com/parsakaz/td-autobpmdetector/releases/download"


def package_url(version: str) -> str:
    """The wheel for a released version.

    Pinned to an exact version rather than "latest": pip will not take a wheel whose
    filename has no version in it, and a component should install the code it was
    built alongside.
    """
    return "%s/v%s/tdautobpm-%s-py3-none-any.whl" % (RELEASES, version, version)


def app_support_dir() -> str:
    """Per-user application data directory for this project."""
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "tdautobpm")


def default_env_dir() -> str:
    """Where a created environment goes unless the component says otherwise.

    One per user rather than one per project: torch is a large download, and every
    project on the machine can share it.
    """
    return os.path.join(app_support_dir(), "env")


def env_python(env_dir: str) -> str:
    """The interpreter inside an environment directory, whether or not it exists."""
    if os.name == "nt":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")


def _version_of(python: str) -> str:
    """``major.minor`` of an interpreter, or "" if it cannot be run."""
    try:
        out = subprocess.run(
            [python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _has_venv(python: str) -> bool:
    try:
        out = subprocess.run([python, "-c", "import venv, ensurepip"],
                            capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0


def find_base_interpreters(host_python: str = "") -> list:
    """Interpreters that could host a new environment, best first.

    Each is ``(python, version, is_host)``. A separate interpreter comes first: an
    environment built on TouchDesigner's own Python can only be used from inside
    TouchDesigner (macOS refuses to load third-party binaries into the bare
    interpreter, which is what the sidecar would be), so it is the fallback rather
    than the default.
    """
    seen, found = set(), []
    for version in PREFERRED_VERSIONS:
        for directory in SEARCH_DIRS:
            python = os.path.join(directory, "python" + version)
            if not os.path.isfile(python) or not os.access(python, os.X_OK):
                continue
            real = os.path.realpath(python)
            if real in seen:
                continue
            seen.add(real)
            if _version_of(python) == version and _has_venv(python):
                found.append((python, version, False))

    host = host_python or sys.executable
    if host and os.path.realpath(host) not in seen and _has_venv(host):
        found.append((host, _version_of(host), True))
    return found


def create_steps(env_dir: str, base_python: str, package: str) -> list:
    """The commands that build an environment, as ``(label, argv)`` pairs.

    `package` is what to install: a wheel URL, a local wheel, or anything else pip
    accepts, including extra arguments such as ``--no-deps`` or an index URL.

    Two steps rather than one: creating the environment is quick, and installing
    torch is a large download, so a caller showing progress wants them apart.
    """
    import shlex

    python = env_python(env_dir)
    return [
        ("Creating the environment", [base_python, "-m", "venv", env_dir]),
        ("Installing (this downloads torch, a few hundred MB)",
         # No progress bar: its carriage returns are unreadable in a log, and the
         # caller shows elapsed time anyway.
         [python, "-m", "pip", "install", "--upgrade", "--no-input",
          "--progress-bar", "off"]
         + shlex.split(package)),
    ]


def describe_plan(env_dir: str, base_python: str, version: str, is_host: bool) -> str:
    """One line saying what will be built and where, for a confirmation or a log."""
    where = "TouchDesigner's own Python" if is_host else base_python
    return "Python %s environment in %s, built on %s" % (
        version or "?", env_dir, where)
