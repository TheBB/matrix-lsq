from __future__ import annotations

from functools import partial
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Union, Protocol

import numpy as np
import scipy.sparse as sp
import tqdm

__version__ = '0.1.0'

Matrix = Union[np.ndarray, sp.spmatrix]


class Snapshot:
    root: Path

    _data: Optional[np.ndarray]
    _objects: Dict[str, Matrix]

    _storage = [
        (np.ndarray, partial(np.save, allow_pickle=False), partial(np.load, allow_pickle=False), '.npy'),
        (sp.spmatrix, sp.save_npz, sp.load_npz, '.npz'),
    ]

    def __init__(self, root: Path, data: Optional[np.ndarray] = None, **kwargs: Matrix):
        self.root = root
        self._data = data
        self._objects = kwargs
        if data is not None:
            self.save_data()
        for name in kwargs:
            self.save_object(name)

    def objpath_root(self, name: str) -> Path:
        if not self.root:
            raise ValueError("set root first")
        return self.root / f'obj-{name}'

    @property
    def datapath(self) -> Path:
        if not self.root:
            raise ValueError("set root first")
        return self.root / 'data.npy'

    def existing_objpath(self, name: str) -> Optional[Path]:
        rpath = self.objpath_root(name)
        for (_, _, _, suffix) in self._storage:
            if (path := rpath.with_suffix(suffix)).exists():
                return path
        return None

    def save_object(self, name: str):
        if (path := self.existing_objpath(name)) is not None:
            path.unlink()
        obj = self._objects[name]
        for (cls, saver, _, suffix) in self._storage:
            if isinstance(obj, cls):
                saver(self.objpath_root(name).with_suffix(suffix), obj)

    def save_data(self):
        if (path := self.datapath).exists():
            path.unlink()
        if self._data is None:
            return
        np.save(self.datapath, self._data, allow_pickle=False)

    def __getitem__(self, name: str) -> Matrix:
        if name not in self._objects:
            path = self.existing_objpath(name)
            if path is None:
                raise KeyError(name)
            for (_, _, loader, suffix) in self._storage:
                if path.suffix == suffix:
                    self._objects[name] = loader(path)
        return self._objects[name]

    def __setitem__(self, name: str, value: Matrix):
        self._objects[name] = value
        self.save_object(name)

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            path = self.datapath
            if not path.exists():
                raise FileNotFoundError(path)
            self._data = np.load(path, allow_pickle=False)
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value
        self.save_data()


class Storage(Protocol):

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Snapshot:
        ...

    def append(self, data: Optional[np.ndarray] = None, **kwargs: Matrix):
        ...

    def pop(self, index: Optional[int] = None) -> Snapshot:
        ...

    def vipe(self):
        ...


class DiskStorage(Storage):
    root: Path

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def root_of(self, index: int) -> Path:
        return self.root / f'object-{index}'

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob('object-*'))

    def __getitem__(self, index: int) -> Snapshot:
        return Snapshot(self.root_of(index))

    def __iter__(self) -> Iterable[Snapshot]:
        for index in range(len(self)):
            yield self[index]

    def append(self, data: Optional[np.ndarray] = None, **kwargs: Matrix):
        index = len(self)
        root = self.root_of(index)
        root.mkdir(parents=True, exist_ok=True)
        Snapshot(root, data, **kwargs)

    def pop(self, index: Optional[int] = None) -> Snapshot:
        if index is None:
            index = len(self) - 1
        elif index > len(self) - 1:
            raise IndexError("pop index out of range")
        root = self.root_of(index)
        snapshot = Snapshot(root)
        for path in root.glob('*'):
            path.unlink()
        root.rmdir()
        for index_i in range(index + 1, len(self) + 1):
            root_i = self.root_of(index_i)
            root_i_new = self.root_of(index_i - 1)
            root_i.rename(root_i_new)
        return snapshot

    def vipe(self, user_confirm: Optional[bool] = True):
        run_vipe = not user_confirm
        if user_confirm:
            user_input = str(input(f"Do you really want to vipe storage in \'{self.root}\'? This will delete all the "
                                   f"files stored. y: "))
            if user_input.lower() == "y":
                run_vipe = True

        if run_vipe:
            for index in range(len(self)):
                root = self.root_of(index)
                for path in root.glob('*'):
                    path.unlink()
                root.rmdir()
            self.root.rmdir()


class LeastSquares:
    storage: Storage

    def __init__(self, storage: Storage):
        self.storage = storage

    def __call__(self, name: str) -> List[Matrix]:
        rawdata = np.array([snapshot.data for snapshot in self.storage])
        sqrdata = rawdata.T @ rawdata

        if (cond := np.linalg.cond(sqrdata)) > 1e5:
            print(f"warning: data matrix may be ill conditioned: {cond}", file=sys.stderr)

        invdata = np.linalg.inv(sqrdata) @ rawdata.T

        components = [0] * rawdata.shape[0]
        for row, snapshot in tqdm.tqdm(zip(invdata.T, self.storage), desc='Interpolating'):
            obj = snapshot[name]
            for i, coeff in enumerate(row):
                components[i] += coeff * obj

        return components
