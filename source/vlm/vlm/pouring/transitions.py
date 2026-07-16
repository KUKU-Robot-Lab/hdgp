from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


class PrePourWarmStateBridge:
    """Delegate grasp-to-pour compatibility to the existing pour loader."""

    def __init__(self, hdgp_root: Path, *, bank_class: type[Any] | None = None) -> None:
        self.hdgp_root = hdgp_root.resolve()
        self._bank_class = bank_class

    def load(
        self,
        path: Path | None = None,
        *,
        device: str = "cpu",
        expected_object_spawn_z: float | None = None,
        expected_palm_bounds: tuple[float, float, float, float, float, float] | None = None,
    ) -> Any:
        bank_class = self._bank_class or self._load_existing_bank_class()
        warm_path = path or self.hdgp_root / "data/grasp_warm_tesollo.hdf5"
        return bank_class.from_hdf5_paths(
            (warm_path,),
            device=device,
            expected_object_spawn_z=expected_object_spawn_z,
            expected_palm_bounds=expected_palm_bounds,
        )

    def _load_existing_bank_class(self) -> type[Any]:
        module_path = (
            self.hdgp_root
            / "source/openarm/openarm/tesollo/right/pour_v1/warm_state_bank.py"
        )
        if not module_path.is_file():
            raise FileNotFoundError(f"existing pour warm-state loader not found: {module_path}")
        module_name = "_hdgp_existing_pour_warm_state_bank"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load existing pour warm-state loader: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.PourWarmStateBank
