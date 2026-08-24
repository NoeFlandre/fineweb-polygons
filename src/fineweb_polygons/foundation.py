"""Stable project-storage paths for future processing stages."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/Volumes/Seagate M3/projects/fineweb-polygons")
DATA_ROOT_ENVIRONMENT_VARIABLE = "FINEWEB_POLYGONS_DATA_ROOT"


def _data_root_for(environment: Mapping[str, str]) -> Path:
    configured_root = environment.get(DATA_ROOT_ENVIRONMENT_VARIABLE)
    if not configured_root:
        return DEFAULT_DATA_ROOT
    return Path(configured_root).expanduser()


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Repository and external-volume paths used by future runs."""

    repository_root: Path
    data_root: Path

    @classmethod
    def from_environment(
        cls,
        repository_root: Path,
        environ: Mapping[str, str] | None = None,
    ) -> ProjectPaths:
        """Build paths from the repository root and an optional data-root override."""
        environment = os.environ if environ is None else environ
        return cls(
            repository_root=Path(repository_root).expanduser().resolve(),
            data_root=_data_root_for(environment).resolve(),
        )

    @property
    def raw_dir(self) -> Path:
        """Return the immutable raw-input directory."""
        return self.data_root / "raw"

    @property
    def runs_dir(self) -> Path:
        """Return the directory for run manifests and checkpoints."""
        return self.data_root / "runs"

    @property
    def logs_dir(self) -> Path:
        """Return the directory for run logs."""
        return self.data_root / "logs"

    @property
    def artifacts_dir(self) -> Path:
        """Return the directory for generated artifacts."""
        return self.data_root / "artifacts"

    def ensure_data_layout(self) -> None:
        """Create the external data directories idempotently."""
        for directory in (
            self.raw_dir,
            self.runs_dir,
            self.logs_dir,
            self.artifacts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
