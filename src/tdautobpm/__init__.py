"""Real-time BPM detection for TouchDesigner.

The heavy pieces (:mod:`tdautobpm.engine`, :mod:`tdautobpm.model`) import torch, so
they are not pulled in here. :mod:`tdautobpm.envresolve` is stdlib-only and safe to
import from inside TouchDesigner before any environment has been set up.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
