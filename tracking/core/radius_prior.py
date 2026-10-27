"""Stabilize sphere size without filtering the measured position."""

from collections import deque

import numpy as np


class SphereRadiusPrior:
    def __init__(self):
        self.samples = deque(maxlen=7)
        self.pending_resize = deque(maxlen=3)
        self.radius = None
        self.misses = 0

    def observe(self, radius):
        self.misses = 0
        if radius is None or not np.isfinite(radius) or radius <= 0:
            return
        # A malformed arc or a finger must not resize the tracked sphere.
        if self.radius is not None and abs(radius - self.radius) > 0.2 * self.radius:
            self.pending_resize.append(float(radius))
            # Permit a real change of apparent size after repeated independent
            # arc measurements, rather than locking an obsolete radius forever.
            if len(self.pending_resize) == 3:
                median = float(np.median(self.pending_resize))
                if np.ptp(self.pending_resize) <= 0.05 * median:
                    self.samples.clear()
                    self.samples.extend(self.pending_resize)
                    self.radius = median
                    self.pending_resize.clear()
            return
        self.pending_resize.clear()
        self.samples.append(float(radius))
        if len(self.samples) >= 3:
            median = float(np.median(self.samples))
            self.radius = median if self.radius is None else self.radius + 0.1 * (median - self.radius)

    def missing(self):
        self.misses += 1
        if self.misses >= 5:
            self.samples.clear()
            self.pending_resize.clear()
            self.radius = None
