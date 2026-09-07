"""팔 PD 게인 = 벤더값만 (2026-09-06 사용자 확정) — 그 규칙을 코드로 잠근다.

여기서 막는 사고는 하나다: **실기 모터에 들어가지 않는 게인으로 학습하는 것.**
09.03 우팔 d3 가 정확히 그랬고(KUKA kp 300 학습), 배포 불가로 폐기됐다.

세 축으로 검사한다.
    ① 값 자체   — hdgp 사본이 벤더 원본과 같고, MIT 패킷에 실을 수 있는가.
    ② 소비처    — 활성 트랙이 실제로 그 값을 쓰는가(선언이 아니라 **해석 결과**를 본다).
    ③ 드리프트  — 새 파일이 팔 옆에 게인 숫자를 다시 적지 않는가(레거시는 명시 허용).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from openarm.agnostic.modules import vendor_gains as VG

_PKG = Path(VG.__file__).resolve().parents[2]          # openarm/ (패키지 루트)

#: 벤더 파일이 담고 있어야 하는 값. **여기 적힌 숫자가 틀리면 벤더 파일이 바뀐 것**이다
#: (그 자체는 정당한 일이다 — 그때 이 표와 재학습 필요성을 같이 갱신한다).
EXPECTED = {1: (70.0, 2.75), 2: (70.0, 2.50), 3: (70.0, 2.00), 4: (60.0, 2.00),
            5: (10.0, 0.70), 6: (10.0, 0.60), 7: (10.0, 0.50)}

#: ★레거시 — 팔 옆 게인 리터럴이 남아 있는 파일. 전부 **폐지된 트랙**이고 되살릴 때
#:   벤더 게인으로 옮기고 이 목록에서 빼는 것이 조건이다(2026-09-06 기준 동결).
#:   새 파일은 여기 못 들어온다 — 들어오려면 사람이 이 목록을 고쳐야 하고, 그 diff 가
#:   "규칙을 어기고 있다"는 신호가 된다.
LEGACY_ALLOWED = {
    "gripper/left/grasp_sensor_fabrics_ABORTED/grasp_left_env_cfg.py",
    "rh56f1/left/grasp_v2/grasp_left_env_cfg.py",
    "rh56f1/right/grasp_v1/grasp_right_env_cfg.py",
    "rh56f1/right/grasp_v2/grasp_right_env_cfg.py",
    "rh56f1/right/pour_v1/pour_right_env_cfg.py",
    "tesollo/both/pour_sensor/pour_right_env_cfg.py",
    "tesollo/both/pour_v1/pour_right_env_cfg.py",
    "tesollo/left/grasp_v1/grasp_left_env_cfg.py",
    "tesollo/left/grasp_v2/grasp_left_env_cfg.py",
    "tesollo/right/grasp_sensor/grasp_right_env_cfg.py",
    "tesollo/right/grasp_v1/grasp_right_env_cfg.py",
    "tesollo/right/grasp_v2/grasp_right_env_cfg.py",
    "tesollo/right/pour_sensor/pour_right_env_cfg.py",
}

_ARM = re.compile(r"_aj_")
_GAIN_LITERAL = re.compile(r"\b(stiffness|damping)\s*=\s*[0-9]")


# ---------------------------------------------------------------- ① 값
def test_vendored_copy_matches_the_vendor_original():
    """hdgp 사본 ↔ rl_ws 벤더 원본. 원본은 로컬 전용이라 없으면 skip 한다."""
    if not VG.VENDOR_GAINS_SOURCE.is_file():
        pytest.skip(f"벤더 원본 없음({VG.VENDOR_GAINS_SOURCE}) — 학습 서버에는 rl_ws/urdf 가 없다")
    assert VG.load(str(VG.VENDOR_GAINS_SOURCE)) == VG.load(), (
        f"사본이 원본과 갈라졌다: {VG.VENDOR_GAINS_YAML} vs {VG.VENDOR_GAINS_SOURCE}")


def test_gains_are_the_seven_vendor_values():
    assert VG.load() == EXPECTED
    names, kp, kd = VG.as_lists("r")
    assert names == [f"r_aj_{j}" for j in range(1, 8)]
    assert kp == [EXPECTED[j][0] for j in range(1, 8)]
    assert kd == [EXPECTED[j][1] for j in range(1, 8)]
    assert VG.stiffness("l") == {f"l_aj_{j}": EXPECTED[j][0] for j in range(1, 8)}
    assert VG.damping("l") == {f"l_aj_{j}": EXPECTED[j][1] for j in range(1, 8)}


def test_gains_are_realisable_on_the_mit_packet():
    """sim 에서만 도는 값은 벤더값이 아니다 — r2s 적합 kd(7.05)가 이 문에 걸려 폐기됐다."""
    for joint, (kp, kd) in VG.load().items():
        assert kp <= VG.MIT_KP_MAX and kd <= VG.MIT_KD_MAX, f"joint{joint}"


def test_loader_refuses_missing_or_incomplete_files(tmp_path):
    with pytest.raises(VG.VendorGainsError):
        VG.load(str(tmp_path / "nope.yaml"))
    partial = tmp_path / "partial.yaml"
    partial.write_text("joint1:\n  kp: 70.0\n  kd: 2.75\n")
    with pytest.raises(VG.VendorGainsError):
        VG.load(str(partial))
    impossible = tmp_path / "impossible.yaml"
    impossible.write_text("".join(f"joint{j}:\n  kp: 300.0\n  kd: 45.0\n" for j in range(1, 8)))
    with pytest.raises(VG.VendorGainsError, match="MIT"):
        VG.load(str(impossible))


def test_helpers_reject_unknown_side_or_joint():
    with pytest.raises(VG.VendorGainsError):
        VG.stiffness("x")
    with pytest.raises(VG.VendorGainsError):
        VG.subset("r", (0, 9))
    with pytest.raises(VG.VendorGainsError):
        VG.arm_actuators("a", "r", friction={1: 0.1})       # joint2 가 없다


def test_arm_actuators_carry_vendor_gains_per_joint():
    spec = VG.arm_actuators("right_arm", "r", friction=0.0, effort_limit_sim=300.0)
    assert len(spec) == 7
    for j, (kp, kd) in EXPECTED.items():
        entry = spec[f"right_arm_j{j}"]
        assert entry["joint_names_expr"] == [f"r_aj_{j}"]
        assert (entry["stiffness"], entry["damping"]) == (kp, kd)
        assert entry["friction"] == 0.0 and entry["effort_limit_sim"] == 300.0
    assert "friction" not in VG.arm_actuators("a", "l")["a_j1"]


# ---------------------------------------------------------------- ② 소비처
def _arm_entries(specs: dict) -> dict:
    return {k: v for k, v in specs.items() if any("_aj_" in e for e in v["joint_names_expr"])}


def _assert_vendor(specs: dict, where: str) -> None:
    entries = _arm_entries(specs)
    assert entries, f"{where}: 팔 액추에이터가 없다"
    table = {f"{s}_aj_{j}": g for s in ("r", "l") for j, g in VG.load().items()}
    seen = set()
    for name, spec in entries.items():
        for expr in spec["joint_names_expr"]:
            matched = [j for j in table if re.fullmatch(expr, j)]
            assert matched, f"{where}.{name}: '{expr}' 가 어떤 팔 관절도 고르지 않는다"
            for joint in matched:
                kp, kd = table[joint]
                got_kp, got_kd = spec["stiffness"], spec["damping"]
                if isinstance(got_kp, dict):
                    got_kp, got_kd = got_kp[joint], got_kd[joint]
                assert (got_kp, got_kd) == (kp, kd), (
                    f"{where}.{name}[{joint}]: {got_kp}/{got_kd} ≠ 벤더 {kp}/{kd}")
                seen.add(joint)
    assert len(seen) % 7 == 0 and seen, f"{where}: 팔 관절 커버리지가 7의 배수가 아니다 {sorted(seen)}"


def test_robot_registry_arm_actuators_are_vendor():
    from openarm.agnostic.modules import robots

    for side in ("r", "l"):
        _assert_vendor(robots._arm_actuators("active", side), f"robots._arm_actuators({side})")


@pytest.mark.parametrize("track", ["grasp_s2r", "grasp_kp", "grasp_fj"])
def test_active_track_profiles_are_vendor(track):
    module = __import__(f"openarm.agnostic.tasks.{track}.robot_profiles", fromlist=["x"])
    profiles = [v for v in vars(module).values() if hasattr(v, "actuator_specs")]
    assert profiles, f"{track}: 프로필이 없다"
    for profile in profiles:
        _assert_vendor(profile.actuator_specs, f"{track}.{profile.name}")


def test_left_gripper_track_presets_are_vendor():
    from openarm.gripper.left.grasp_sensor import grasp_left_preset as V1
    from openarm.gripper.left.grasp_sensor_v2 import v2_preset as V2

    assert V1.ARM_IK_STIFFNESS == VG.stiffness("l") == V2.LEFT_ARM_VENDOR_STIFFNESS
    assert V1.ARM_IK_DAMPING == VG.damping("l") == V2.LEFT_ARM_VENDOR_DAMPING


# ---------------------------------------------------------------- ①' DG-5F 손
def test_vendored_dg5f_pid_copy_matches_the_vendor_original():
    if not VG.DG5F_PID_SOURCE.is_file():
        pytest.skip(f"DG-5F 벤더 PID 원본 없음({VG.DG5F_PID_SOURCE})")
    assert VG.load_hand(str(VG.DG5F_PID_SOURCE)) == VG.load_hand()


def test_hand_gain_is_the_vendor_driver_pid():
    """2026-09-06 사용자 확정: DG-5F 손도 벤더 기본 p 1.5 / d 0(구 실기 튜닝 4.5 폐기)."""
    assert VG.hand_gains() == (1.5, 0.0)
    hand = VG.load_hand()
    assert len(hand) == 40 and set(hand) >= {"rj_dg_1_1", "lj_dg_5_4"}
    assert len(set(hand.values())) == 1, "관절마다 게인이 다르면 스칼라 한 쌍으로 못 쓴다"


def test_hand_loader_refuses_integral_gain(tmp_path):
    """i 항이 있으면 sim ImplicitActuator 로 옮길 수 없다 — 조용히 버리지 않고 죽는다."""
    bad = tmp_path / "i.yaml"
    bad.write_text("".join(
        f"    {s}j_dg_{f}_{j}: {{ p: 1.5, i: 0.1, d: 0.0 }}\n"
        for s in "rl" for f in range(1, 6) for j in range(1, 5)))
    with pytest.raises(VG.VendorGainsError, match="i="):
        VG.load_hand(str(bad))
    short = tmp_path / "short.yaml"
    short.write_text("    rj_dg_1_1: { p: 1.5, i: 0.0, d: 0.0 }\n")
    with pytest.raises(VG.VendorGainsError, match="40"):
        VG.load_hand(str(short))


#: 손이 **벤더 PD 를 갖지 않는** actuator — `NO_VENDOR_PD` 에 사유가 있다.
# ★09.07 grasp_ua 트랙은 의도적으로 제거됐다(사용자 확정) — 예외 목록도 함께 비운다.
#   목록에 남겨두면 "왜 비어 있나"를 다시 조사하게 된다.
HAND_EXCEPTIONS: set[tuple[str, str]] = set()


@pytest.mark.parametrize("track", ["grasp_s2r", "grasp_kp", "grasp_fj"])
def test_active_track_dg5f_hands_are_vendor(track):
    module = __import__(f"openarm.agnostic.tasks.{track}.robot_profiles", fromlist=["x"])
    checked = 0
    for profile in [v for v in vars(module).values() if hasattr(v, "actuator_specs")]:
        if (track, profile.name) in HAND_EXCEPTIONS:
            continue
        for name, spec in profile.actuator_specs.items():
            if not any("_hj_[a-z]" in e for e in spec["joint_names_expr"]):
                continue
            assert (spec["stiffness"], spec["damping"]) == VG.hand_gains(), (
                f"{track}.{profile.name}.{name}: 손 게인이 벤더값이 아니다")
            checked += 1
    assert checked, f"{track}: 검사한 손 actuator 가 없다"


def test_registry_tesollo_hand_is_vendor_and_rh56f1_is_an_exception():
    from openarm.agnostic.modules import robots

    tesollo = next(iter(robots._tesollo_hand_actuator("a", "r").values()))
    assert (tesollo["stiffness"], tesollo["damping"]) == VG.hand_gains()
    rh56 = next(iter(robots._rh56_hand_actuator("a", "r").values()))
    assert (rh56["stiffness"], rh56["damping"]) != VG.hand_gains()
    assert "rh56f1_hand" in VG.NO_VENDOR_PD and "RS-485" in VG.NO_VENDOR_PD["rh56f1_hand"]


def test_every_no_vendor_pd_exception_states_a_reason():
    """예외는 값이 아니라 **사유**로 관리한다 — 사유 없는 예외는 그냥 드리프트다."""
    assert set(VG.NO_VENDOR_PD) == {"rh56f1_hand", "stock_gripper_jaw", "head_dynamixel"}
    for key, why in VG.NO_VENDOR_PD.items():
        assert len(why) > 60 and "벤더" in why, key


# ---------------------------------------------------------------- ③ 드리프트
def test_no_new_arm_gain_literals_outside_the_legacy_allowlist():
    """팔 관절 표현 옆에 게인 **숫자**를 적은 파일을 찾는다.

    벤더 파일을 고치는 것이 게인을 바꾸는 유일한 방법이다. 태스크 코드가 숫자를
    적기 시작하면 그 순간 "어느 게인으로 학습했는가"가 파일마다 갈린다.
    """
    offenders = {}
    for path in sorted(_PKG.rglob("*.py")):
        if path.name == "vendor_gains.py" or "/tests/" in str(path):
            continue
        lines = path.read_text(errors="ignore").splitlines()
        bad = [i + 1 for i, line in enumerate(lines)
               if not line.lstrip().startswith("#")
               and _GAIN_LITERAL.search(line)
               and _ARM.search(" ".join(lines[max(0, i - 1):i + 2]))]
        if bad:
            offenders[str(path.relative_to(_PKG))] = bad
    new = {f: ln for f, ln in offenders.items() if f not in LEGACY_ALLOWED}
    assert not new, (
        "팔 게인을 코드에 직접 적었다 — `vendor_gains` 를 쓰거나, 되살린 레거시라면 "
        f"LEGACY_ALLOWED 에서 빼고 변환하라: {new}")
    stale = LEGACY_ALLOWED - set(offenders)
    assert not stale, f"이미 정리된 파일이 허용목록에 남아 있다(삭제하라): {sorted(stale)}"
