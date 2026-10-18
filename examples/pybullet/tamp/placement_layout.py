"""Ordered, separated placement cells for rectangular members (SI units)."""
import math
import numpy as np

PLACEMENT_GAP = 0.12
PLACEMENT_EDGE_MARGIN = 0.15


def original_resting_orientation(dimensions):
    """Keep local +Z vertical and align the longer horizontal axis with world X."""
    dims = np.asarray(dimensions, dtype=float)
    if dims.shape != (3,) or not np.all(np.isfinite(dims)) or np.any(dims <= 0):
        raise ValueError('Member dimensions must be three positive finite lengths')
    rotation = np.eye(3) if dims[0] >= dims[1] else np.array(
        [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    return tuple(abs(rotation) @ dims), rotation


def flat_orientations(dimensions):
    """Longest axis along world X, shortest vertical; allow either broad face down."""
    dims = np.asarray(dimensions, dtype=float)
    if dims.shape != (3,) or not np.all(np.isfinite(dims)) or np.any(dims <= 0):
        raise ValueError('Member dimensions must be three positive finite lengths')
    short, middle, long = np.argsort(dims, kind='stable')
    rotations = []
    for normal in (-1., 1.):
        rotation = np.zeros((3, 3))
        rotation[0, long] = 1.
        rotation[2, short] = normal
        rotation[1, middle] = 1.
        if np.linalg.det(rotation) < 0:
            rotation[1, middle] = -1.
        rotations.append(rotation)
    return tuple(dims[[long, middle, short]]), rotations


def ordered_layout(dimensions, gap=PLACEMENT_GAP, edge=PLACEMENT_EDGE_MARGIN, upright=False):
    """Cells fit ANY group member, so sequence rank never determines member identity.

    Prefer a single column of parallel members. Wrap into columns only for large
    groups, keeping the short direction near 2.5 m when a member fits there.
    No plate-size cap may silently truncate a member or its required slots.
    """
    if gap <= 0 or edge < 0 or not dimensions:
        raise ValueError('A nonempty group, positive gap and nonnegative edge are required')
    footprints = [tuple(d) if upright else flat_orientations(d)[0] for d in dimensions]
    length = max(d[0] for d in footprints)
    width = max(d[1] for d in footprints)
    capacity = max(1, int(math.floor((2.5 - 2*edge + gap)/(width + gap))))
    rows = min(len(footprints), capacity)
    columns = int(math.ceil(len(footprints)/rows))
    required_x = columns*length + (columns-1)*gap + 2*edge
    required_y = rows*width + (rows-1)*gap + 2*edge
    plate_size = (max(2.5, required_x), max(2.5, required_y))
    centers = []
    for rank in range(len(footprints)):
        col, row = divmod(rank, rows)
        centers.append(((col-(columns-1)/2)*(length+gap),
                        (row-(rows-1)/2)*(width+gap)))
    return dict(plate_size=plate_size, required_size=(required_x, required_y),
                cell_size=(length, width), centers=centers,
                gap=float(gap), edge_margin=float(edge))
