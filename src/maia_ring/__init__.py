"""maia-ring: a shared-memory frame ring for passing video frames between processes without copies."""

from .layout import DISCONTINUITY, IFRAME, MOTION, Geometry
from .ring import Frame, FrameMeta, FrameRef, RingPool, RingReader, RingWriter, StaleFrame

__all__ = [
    "DISCONTINUITY",
    "IFRAME",
    "MOTION",
    "Frame",
    "FrameMeta",
    "FrameRef",
    "Geometry",
    "RingPool",
    "RingReader",
    "RingWriter",
    "StaleFrame",
]
__version__ = "0.1.1"
