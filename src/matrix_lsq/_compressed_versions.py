from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Iterable, Optional, Union, Protocol

import numpy as np
import scipy.sparse as sp

__version__ = '0.1.0'

Matrix = Union[np.ndarray, sp.spmatrix]


class CompressedSnapshot:
    root: Path
    _data: Optional[np.ndarray]
    _numpy_objects: Dict[str, np.ndarray]
    _sparse_objects: Dict[str, sp.spmatrix]

    def __init__(self, root: Path, data: Optional[np.ndarray] = None,
                 support_load_non_compressed_snapshots: bool = True, **kwargs: Matrix):
        self._support_load_non_compressed_snapshots = support_load_non_compressed_snapshots
        self.root = root
        self._data = data
        self._numpy_objects = {}
        self._sparse_objects = {}
        if data is not None:
            self.save_data()
        for name, obj in kwargs.items():
            if isinstance(obj, sp.spmatrix):
                self._sparse_objects[name] = obj
                self.save_sparse_object(name)
            elif isinstance(obj, np.ndarray):
                self._numpy_objects[name] = obj
        if len(self._numpy_objects) != 0:
            self.save_numpy_objects()

    @property
    def datapath(self) -> Path:
        if not self.root:
            raise ValueError("set root first")
        return self.root / 'data.npz'

    def sparse_objpath_root(self, name: str) -> Path:
        if not self.root:
            raise ValueError("set root first")
        return self.root / f'sp-obj-{name}'

    def existing_sparse_objpath(self, name: str) -> Optional[Path]:
        rpath = self.sparse_objpath_root(name)
        if (path := rpath.with_suffix(".npz")).exists():
            return path
        return None

    def save_sparse_object(self, name: str):
        if (path := self.existing_sparse_objpath(name)) is not None:
            path.unlink()
        obj = self._sparse_objects[name]
        sp.save_npz(self.sparse_objpath_root(name).with_suffix(".npz"), obj)

    def numpy_objpath_root(self) -> Path:
        if not self.root:
            raise ValueError("set root first")
        return self.root / f'np-obj'

    def existing_numpy_objpath(self) -> Optional[Path]:
        rpath = self.numpy_objpath_root()
        if (path := rpath.with_suffix(".npz")).exists():
            return path
        return None

    def save_numpy_objects(self):
        if (path := self.existing_numpy_objpath()) is not None:
            path.unlink()
        # save all numpy objects (arrays) to compressed npz file (zip file)
        # Note: we allow save pickle data here, but we will not allow loading it in getitem.
        np.savez_compressed(self.numpy_objpath_root().with_suffix(".npz"), **self._numpy_objects)

    def save_data(self):
        if (path := self.datapath).exists():
            path.unlink()
        if self._data is None:
            return
        # Note: we allow save pickle data here, but we will not allow loading it in getitem.
        np.savez_compressed(self.datapath, data=self._data)

    def _non_compressed_objpath_root(self, name: str) -> Path:
        if not self.root:
            raise ValueError("set root first")
        return self.root / f'obj-{name}'

    def __getitem__(self, name: str) -> Matrix:
        # start checking in sparse objects
        if name not in self._sparse_objects:
            sp_path = self.existing_sparse_objpath(name)
            if sp_path is None:
                if self._support_load_non_compressed_snapshots:
                    old_sp_path = self._non_compressed_objpath_root(name).with_suffix(".npz")
                    if old_sp_path.exists():
                        self._sparse_objects[name] = sp.load_npz(old_sp_path)
            else:
                self._sparse_objects[name] = sp.load_npz(sp_path)
        if name not in self._numpy_objects:
            np_path = self.existing_numpy_objpath()
            if np_path is None:
                if self._support_load_non_compressed_snapshots:
                    old_np_path = self._non_compressed_objpath_root(name).with_suffix(".npy")
                    if old_np_path.exists():
                        self._numpy_objects[name] = np.load(old_np_path, allow_pickle=False)
                    else:
                        raise KeyError(name)
                else:
                    raise KeyError(name)
            else:
                self._numpy_objects[name] = np.load(np_path, allow_pickle=False)[name]
        if name in self._sparse_objects:
            return self._sparse_objects[name]
        else:
            return self._numpy_objects[name]

    def __setitem__(self, name: str, value: Matrix):
        if isinstance(value, sp.spmatrix):
            self._sparse_objects[name] = value
            self.save_sparse_object(name)
        elif isinstance(value, np.ndarray):
            print("warning: Setitem for numpy-ndarrays is inefficient in compressed-format, "
                  "due to needing to copy exiting data.", file=sys.stderr)
            # copy already saved data
            np_path = self.existing_numpy_objpath()
            if np_path is not None:
                numpy_objects = np.load(np_path, allow_pickle=False)
                for file_name in numpy_objects.files:
                    self._numpy_objects[file_name] = numpy_objects[file_name]
            # set value and save
            self._numpy_objects[name] = value
            self.save_numpy_objects()

    @property
    def _non_compressed_datapath(self) -> Path:
        if not self.root:
            raise ValueError("set root first")
        return self.root / 'data.npy'

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            path = self.datapath
            if not path.exists():
                if self._support_load_non_compressed_snapshots:
                    non_compressed_path = self._non_compressed_datapath
                    if not non_compressed_path.exists():
                        raise FileNotFoundError(non_compressed_path)
                    self._data = np.load(non_compressed_path, allow_pickle=False)
                else:
                    raise FileNotFoundError(path)
            else:
                self._data = np.load(path, allow_pickle=False)["data"]
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value
        self.save_data()


class CompressedStorage(Protocol):

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> CompressedSnapshot:
        ...

    def append(self, data: Optional[np.ndarray] = None, **kwargs: Matrix):
        ...

    def pop(self, index: Optional[int] = None) -> CompressedSnapshot:
        ...

    def vipe(self, user_confirm: bool = True):
        ...


class CompressedSnapshotDiskStorage(CompressedStorage):
    root: Path

    def __init__(self, root: Path, support_load_non_compressed_snapshots: bool = True):
        self._support_load_non_compressed_snapshots = support_load_non_compressed_snapshots
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def root_of(self, index: int) -> Path:
        return self.root / f'object-{index}'

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob('object-*'))

    def __getitem__(self, index: int) -> CompressedSnapshot:
        return CompressedSnapshot(self.root_of(index),
                                  support_load_non_compressed_snapshots=self._support_load_non_compressed_snapshots)

    def __iter__(self) -> Iterable[CompressedSnapshot]:
        for index in range(len(self)):
            yield self[index]

    def append(self, data: Optional[np.ndarray] = None, **kwargs: Matrix):
        index = len(self)
        root = self.root_of(index)
        root.mkdir(parents=True, exist_ok=True)
        CompressedSnapshot(root, data, **kwargs)

    def pop(self, index: Optional[int] = None) -> CompressedSnapshot:
        if index is None:
            index = len(self) - 1
        elif index > len(self) - 1:
            raise IndexError("pop index out of range")
        root = self.root_of(index)
        snapshot = CompressedSnapshot(root,
                                      support_load_non_compressed_snapshots=self._support_load_non_compressed_snapshots)
        for path in root.glob('*'):
            path.unlink()
        root.rmdir()
        for index_i in range(index + 1, len(self) + 1):
            root_i = self.root_of(index_i)
            root_i_new = self.root_of(index_i - 1)
            root_i.rename(root_i_new)
        return snapshot

    def vipe(self, user_confirm: bool = True):
        if user_confirm:
            print(f"Do you want to delete ALL files saved in <{self.root}>.")
            choice = "not y or n"
            while choice.lower() not in ("y", "n", "yes", "no"):
                choice = str(input("Delete the storage (y/n): "))
            if choice.lower() in ("n", "no"):
                raise KeyboardInterrupt("User did not want to delete the storage")
        print(f"warning: deleting all files in <{self.root}>.", file=sys.stderr)
        for object_root in self.root.glob("*"):
            for path in object_root.glob("*"):
                path.unlink()
            object_root.rmdir()

