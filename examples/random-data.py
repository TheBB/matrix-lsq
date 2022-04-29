from pathlib import Path
from numpy.random import random

from matrix_lsq import DiskStorage, LeastSquares, CompressedSnapshotDiskStorage


def setup():
    """Setup a disk-based storage in the path 'storage',
    with twenty snapshots, each with a 10x10 random lhs
    matrix and a 10-element rhs vector.

    We use four data points for each snapshot, also
    randomly generated.
    """

    storage = DiskStorage(Path('storage'))
    assert len(storage) == 0

    for _ in range(20):
        storage.append(random((4,)), lhs=random((10, 10)), rhs=random((10,)))


def fit():
    """Perform a least-squares fit on the data generated
    by snapsnot(). Return a list of four 10x10 matrices
    for the LHS and four 10-element vectors for the RHS.
    """

    storage = DiskStorage(Path('storage'))
    fitter = LeastSquares(storage)
    return fitter('lhs'), fitter('rhs')


def remove():
    """Remove some snapshots """
    storage = DiskStorage(Path('storage'))
    assert len(storage) != 0

    storage.pop()
    storage.pop(1)

def setup2():
    """Setup a disk-based storage in the path 'storage',
    with twenty Compressedsnapshots, each with a 10x10 random lhs
    matrix and a 10-element rhs vector.

    We use four data points for each snapshot, also
    randomly generated.
    """

    storage = CompressedSnapshotDiskStorage(Path('storage2'))
    assert len(storage) == 0

    for _ in range(20):
        storage.append(random((4,)), lhs=random((10, 10)), rhs=random((10,)))


def fit2():
    """Perform a least-squares fit on the data generated
    by snapsnot(). Return a list of four 10x10 matrices
    for the LHS and four 10-element vectors for the RHS.
    """

    storage = CompressedSnapshotDiskStorage(Path('storage2'))
    fitter = LeastSquares(storage)
    return fitter('lhs'), fitter('rhs')


def remove2():
    """Remove some snapshots """
    storage = CompressedSnapshotDiskStorage(Path('storage2'))
    assert len(storage) != 0

    storage.pop()
    storage.pop(1)


setup()
fit()
remove()

setup2()
fit2()
remove2()
