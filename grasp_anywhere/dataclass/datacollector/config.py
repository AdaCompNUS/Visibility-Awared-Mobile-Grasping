from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class H5CompressionConfig:
    """
    HDF5 dataset creation options.

    - `compression=None` means no compression.
    - Common choices: "gzip", "lzf".
    """

    compression: str | None = "gzip"
    compression_level: int | None = 4
    shuffle: bool = True


@dataclass(frozen=True)
class DataCollectionConfig:
    """
    Configuration for writing datasets. All paths/tunables live here.
    """

    run_name: str

    # All collected data is written under this folder.
    # Default: repo-local `data/`
    data_dir: Path = Path("data")

    # Sampling
    record_hz: float = 10.0

    # HDF5 write options
    h5: H5CompressionConfig = H5CompressionConfig()

    # Check if data collection is disabled
    disabled: bool = True

    def run_dir(self) -> Path:
        return self.data_dir / self.run_name
