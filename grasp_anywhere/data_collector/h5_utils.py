from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from grasp_anywhere.dataclass.datacollector.config import H5CompressionConfig


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def h5_create_file(path: Path) -> h5py.File:
    ensure_parent_dir(path)
    return h5py.File(str(path), "w")


def h5_write_str_attr(g: h5py.Group, key: str, value: str) -> None:
    g.attrs[key] = value


def h5_write_scalar_attr(g: h5py.Group, key: str, value: int | float) -> None:
    g.attrs[key] = value


def h5_create_extendable_dataset(
    g: h5py.Group,
    name: str,
    *,
    dtype: np.dtype,
    item_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...] | None,
    h5: H5CompressionConfig,
) -> h5py.Dataset:
    """
    Create an extendable dataset with first dimension = time (T).
    """
    shape = (0, *item_shape)
    maxshape = (None, *item_shape)
    chunks = (1, *item_shape) if chunk_shape is None else chunk_shape

    if h5.compression is None:
        return g.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            chunks=chunks,
            dtype=dtype,
            shuffle=bool(h5.shuffle),
        )

    return g.create_dataset(
        name,
        shape=shape,
        maxshape=maxshape,
        chunks=chunks,
        dtype=dtype,
        compression=h5.compression,
        compression_opts=h5.compression_level,
        shuffle=bool(h5.shuffle),
    )


def h5_append_row(ds: h5py.Dataset, row: np.ndarray) -> None:
    n = int(ds.shape[0])
    ds.resize((n + 1, *ds.shape[1:]))
    ds[n] = row


def h5_append_scalar(ds: h5py.Dataset, value: float) -> None:
    n = int(ds.shape[0])
    ds.resize((n + 1,))
    ds[n] = value
