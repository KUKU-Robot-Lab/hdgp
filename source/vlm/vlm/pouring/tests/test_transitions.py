from __future__ import annotations

from pathlib import Path

from vlm.pouring.transitions import PrePourWarmStateBridge


def test_pre_pour_bridge_loads_existing_grasp_warm_state() -> None:
    root = Path(__file__).resolve().parents[5]
    bridge = PrePourWarmStateBridge(root)
    warm_path = root / "data/grasp_warm_tesollo.hdf5"
    if not warm_path.is_file():
        warm_path = Path("/home/user/rl_ws/hdgp/data/grasp_warm_tesollo.hdf5")

    result = bridge.load(
        warm_path,
        expected_object_spawn_z=0.297,
    )

    assert result.num_states == 2048
    assert result.arm_joint_pos.shape[1] == 7
    assert result.hand_joint_pos.shape[1] == 20


def test_pre_pour_bridge_delegates_to_injected_loader(tmp_path: Path) -> None:
    calls: list[tuple[tuple[Path, ...], str, float | None]] = []

    class FakeBank:
        @classmethod
        def from_hdf5_paths(cls, paths, *, device, expected_object_spawn_z, expected_palm_bounds):
            calls.append((tuple(paths), str(device), expected_object_spawn_z))
            return "bank"

    bridge = PrePourWarmStateBridge(tmp_path, bank_class=FakeBank)

    result = bridge.load(tmp_path / "warm.hdf5", expected_object_spawn_z=0.3)

    assert result == "bank"
    assert calls == [((tmp_path / "warm.hdf5",), "cpu", 0.3)]
