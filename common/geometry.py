"""Vectorised point-to-segment geometry (all inputs in metres)."""
import numpy as np


def point_seg(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB, and projection fraction t (unclipped).

    Returns (dist, t_unclipped). dist uses the clipped projection.
    All args are broadcastable numpy arrays.
    """
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1), 0.0)
    tc = np.clip(t, 0.0, 1.0)
    qx, qy = ax + tc * dx, ay + tc * dy
    return np.hypot(px - qx, py - qy), t


def bearing(dx, dy):
    """Compass-style angle in degrees [0, 360) of vector (dx, dy) (x=east, y=north)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two angles in degrees -> [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
