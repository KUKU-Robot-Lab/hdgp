"""팔 PD 게인의 **유일한 출처** — 벤더 `control_gains.yaml`.

2026-09-06 사용자 확정: **모든 학습·제어에서 벤더 기준 PD 게인만 쓴다.**
팔은 OpenArm `control_gains.yaml`, DG-5F 손은 드라이버 PID(`p=1.5 · d=0`)가 그 기준이다.
따라서 태스크·트랙이 팔 kp/kd 숫자를 직접 적는 일은 없다. 전부 이 모듈을 부른다.
새 값을 넣고 싶으면 벤더 yaml 을 고치는 것이지 태스크 코드를 고치는 게 아니다.
(`tests/test_vendor_gains.py` 가 트랙 소스에 팔 게인 리터럴이 새로 생기면 실패한다.)

왜 벤더만인가
    · 실기는 bringup 이 이 파일의 값을 모터에 1회 주입한다. 다른 값으로 학습한 정책은
      **다른 로봇에서 배운 것**이라 배포할 수 없다(09.03 우팔 d3 = KUKA kp 300 학습 →
      배포 불가 판정, `sim2real/logs/policy/right_d3`).
    · 대안 후보였던 값들은 전부 기각됐다: KUKA 300/45(다른 로봇), sim 기본 400/80
      (실기보다 4~10배 뻣뻣 → 정책 진동이 팔에 그대로 실린다), r2s 적합 kd
      (7.053/4.182/… — MIT 패킷 kd 상한 5.0 을 넘어 실기에 실을 수 없다).

왜 yaml 사본이 hdgp 안에 있나
    원본은 `rl_ws/urdf/vendor/openarm_description/config/arm/v10/control_gains.yaml`
    이지만 **학습 서버에는 rl_ws/urdf 가 없다**(USD 만 배포된다 — 로봇 레지스트리
    계약 테스트도 같은 이유로 URDF 검사를 skip 한다). 그래서 사본을 패키지 안에 둔다.
    원본이 있는 로컬에서는 테스트가 사본↔원본을 대조해 드리프트를 막는다.

단위: kp [N·m/rad], kd [N·m/(rad/s)] — sim ImplicitActuator 와 실기 MIT 패킷이 같은 식
(τ = kp(q*−q) + kd(q̇*−q̇) + τ_ff)을 쓰므로 변환 없이 그대로 들어간다.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

#: 패키지 안 사본(학습 서버 포함 어디서나 존재). 원본은 모듈 docstring 참조.
VENDOR_GAINS_YAML = Path(__file__).resolve().parent / "vendor_arm_control_gains.yaml"
#: 원본(로컬 전용). 드리프트 테스트만 쓴다 — 런타임은 절대 여기를 읽지 않는다.
VENDOR_GAINS_SOURCE = (Path(__file__).resolve().parents[6]
                       / "urdf/vendor/openarm_description/config/arm/v10/control_gains.yaml")
ARM_JOINTS = tuple(range(1, 8))
SIDES = ("r", "l")
#: Damiao MIT 패킷 인코딩 한계(dm_motor_control.cpp) — 밖의 값은 실기에서 실현 불가.
MIT_KP_MAX = 500.0
MIT_KD_MAX = 5.0

# ── DG-5F 손 ────────────────────────────────────────────────────────────────
#: 벤더 드라이버 PID 사본. 원본 `rl_ws/urdf/vendor/delto_m_ros2/dg5f_driver/config/
#: dg5f_both_pid_all_controller.yaml` — 40 관절(좌우 20씩)이 전부 같은 `p 1.5 · i 0 · d 0`.
DG5F_PID_YAML = Path(__file__).resolve().parent / "vendor_dg5f_pid.yaml"
DG5F_PID_SOURCE = (Path(__file__).resolve().parents[6]
                   / "urdf/vendor/delto_m_ros2/dg5f_driver/config/dg5f_both_pid_all_controller.yaml")
_DG5F_GAIN = re.compile(r"^\s+([lr]j_dg_\d_\d):\s*\{\s*p:\s*([0-9.]+),\s*i:\s*([0-9.]+),\s*d:\s*([0-9.]+)")

_ENTRY = re.compile(r"^joint([1-7]):\s*$")
_FIELD = re.compile(r"^\s+(kp|kd):\s*(-?[0-9.]+)\s*$")


class VendorGainsError(RuntimeError):
    """벤더 게인 파일이 없거나 7관절 kp/kd 를 다 담고 있지 않다."""


@lru_cache(maxsize=None)
def load(path: str | None = None) -> dict[int, tuple[float, float]]:
    """`{관절번호 1..7: (kp, kd)}`. 파일이 없거나 불완전하면 **죽는다**(조용한 기본값 금지).

    yaml 라이브러리를 쓰지 않는다 — 이 파일은 `jointN: {kp, kd}` 두 계층뿐이고,
    로봇 레지스트리는 의도적으로 무의존(순수 데이터)이기 때문이다.
    """
    target = Path(path) if path else VENDOR_GAINS_YAML
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise VendorGainsError(f"벤더 게인 파일을 읽을 수 없다: {target} — {exc}") from exc
    out: dict[int, dict[str, float]] = {}
    current: int | None = None
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        entry = _ENTRY.match(line)
        if entry:
            current = int(entry.group(1))
            out.setdefault(current, {})
            continue
        field = _FIELD.match(line)
        if field and current is not None:
            out[current][field.group(1)] = float(field.group(2))
    missing = [j for j in ARM_JOINTS if {"kp", "kd"} - set(out.get(j, {}))]
    if missing:
        raise VendorGainsError(f"{target}: joint{missing} 의 kp/kd 가 없다")
    gains = {j: (out[j]["kp"], out[j]["kd"]) for j in ARM_JOINTS}
    _check_realisable(gains, target)
    return gains


def _check_realisable(gains: dict[int, tuple[float, float]], target: Path) -> None:
    """MIT 패킷에 실을 수 없는 값이면 죽는다 — sim 에서만 도는 게인은 벤더값이 아니다."""
    bad = [f"joint{j}: kp {kp:g} / kd {kd:g}" for j, (kp, kd) in gains.items()
           if kp > MIT_KP_MAX or kd > MIT_KD_MAX]
    if bad:
        raise VendorGainsError(
            f"{target}: MIT 패킷 한계(kp≤{MIT_KP_MAX:g}, kd≤{MIT_KD_MAX:g}) 밖 — {'; '.join(bad)}")


def joint_name(side: str, index: int) -> str:
    if side not in SIDES:
        raise VendorGainsError(f"side 는 {SIDES} 중 하나여야 한다: {side!r}")
    return f"{side}_aj_{index}"


def stiffness(side: str) -> dict[str, float]:
    """`{'r_aj_1': 70.0, …}` — ActuatorCfg.stiffness 에 그대로 넣는다."""
    return {joint_name(side, j): kp for j, (kp, _) in load().items()}


def damping(side: str) -> dict[str, float]:
    return {joint_name(side, j): kd for j, (_, kd) in load().items()}


def gains(side: str) -> dict[str, dict[str, float]]:
    """ActuatorCfg 키워드 두 개를 한 번에: `dict(**vendor_gains.gains('r'))`."""
    return {"stiffness": stiffness(side), "damping": damping(side)}


def subset(side: str, indices) -> dict[str, dict[str, float]]:
    """관절 일부만 담당하는 actuator 그룹용(예: 손목 그룹 = 5,6,7).

    ActuatorCfg 는 `joint_names_expr` 이 고르는 관절만 보므로 dict 에 남는 항목이
    있어도 무해하지만, 그룹별로 잘라 두면 **어느 그룹이 어느 관절을 맡는지**가
    설정 그 자체로 드러난다.
    """
    want = [int(i) for i in indices]
    unknown = [i for i in want if i not in ARM_JOINTS]
    if unknown:
        raise VendorGainsError(f"팔 관절 번호가 아니다: {unknown}")
    table = load()
    return {"stiffness": {joint_name(side, j): table[j][0] for j in want},
            "damping": {joint_name(side, j): table[j][1] for j in want}}


def as_lists(side: str) -> tuple[list[str], list[float], list[float]]:
    """(관절이름, kp, kd) — 배포 계약(`sim2real` deploy_contract.pd.sim_gains) 형식."""
    table = load()
    names = [joint_name(side, j) for j in ARM_JOINTS]
    return names, [table[j][0] for j in ARM_JOINTS], [table[j][1] for j in ARM_JOINTS]


def arm_actuators(prefix: str, side: str, *, friction=None, effort_limit_sim=None) -> dict:
    """한쪽 팔의 ActuatorCfg kwargs 7개(관절당 1개). **게인은 벤더값뿐이다.**

    태스크 설정은 이 함수를 부르고 숫자를 적지 않는다. 관절마다 그룹을 나누는 이유는
    kd 가 관절마다 다르고(2.75…0.5), friction·effort 도 부위마다 다르기 때문이다.

    friction        : float 하나 또는 `{관절번호: 값}`. None 이면 키를 넣지 않는다.
    effort_limit_sim: 모든 관절에 같은 값. None 이면 키를 넣지 않는다(USD maxForce 사용).
    """
    table = load()
    out = {}
    for j in ARM_JOINTS:
        kp, kd = table[j]
        spec = dict(joint_names_expr=[joint_name(side, j)], stiffness=kp, damping=kd)
        if friction is not None:
            value = friction.get(j) if isinstance(friction, dict) else friction
            if value is None:
                raise VendorGainsError(f"friction 에 joint{j} 가 없다: {friction}")
            spec["friction"] = float(value)
        if effort_limit_sim is not None:
            spec["effort_limit_sim"] = float(effort_limit_sim)
        out[f"{prefix}_j{j}"] = spec
    return out


#: real2sim 07.29 우팔 캘리브 — **PD 게인이 아니라 관절 마찰**이라 벤더 규칙 밖이다.
#: 부위별 값이며 태스크가 그대로 넘겨 쓴다.
R2S_FRICTION = {1: 0.213, 2: 0.213, 3: 0.213, 4: 0.493, 5: 0.151, 6: 0.151, 7: 0.151}


# ══════════════════════════════════════════════════════════════════════════════
# DG-5F 손 — 벤더 드라이버 PID
# ══════════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=None)
def load_hand(path: str | None = None) -> dict[str, tuple[float, float]]:
    """`{드라이버 관절명 'rj_dg_1_1': (p, d)}` — 40 관절. 불완전하면 죽는다.

    ★i(적분)는 0 이라 무시한다. sim ImplicitActuator 에는 적분항이 없으므로 0 이 아니면
      옮길 수 없다는 뜻이고, 실제로 벤더는 전 관절 i=0 이다.
    """
    target = Path(path) if path else DG5F_PID_YAML
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise VendorGainsError(f"DG-5F 벤더 PID 파일을 읽을 수 없다: {target} — {exc}") from exc
    out: dict[str, tuple[float, float]] = {}
    for line in text.splitlines():
        m = _DG5F_GAIN.match(line)
        if m:
            name, p_gain, i_gain, d_gain = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
            if i_gain:
                raise VendorGainsError(f"{target}: {name} 의 i={i_gain} — sim 에 옮길 적분항이 없다")
            out[name] = (p_gain, d_gain)
    if len(out) != 40:
        raise VendorGainsError(f"{target}: 손 관절 40개가 아니라 {len(out)}개를 읽었다")
    if len(set(out.values())) != 1:
        raise VendorGainsError(f"{target}: 관절마다 게인이 다르다 — {sorted(set(out.values()))}")
    return out


def hand_gains() -> tuple[float, float]:
    """(p, d) — 모든 DG-5F 관절이 같은 값이라 스칼라 한 쌍이면 충분하다.

    2026-09-06 사용자 확정: **벤더 기본 1.5 로 통일**한다(구 실기 튜닝값 4.5 폐기).
    """
    return next(iter(load_hand().values()))


def hand_actuator(name: str, joint_names_expr, *, effort_limit_sim=None, **extra) -> dict:
    """DG-5F 손 actuator kwargs 하나. 게인은 벤더값, 나머지는 호출자가 준다.

    ⚠벤더 d=0 이다. 실기 손은 기계 마찰이 그 자리를 메우지만 sim 관절에는 마찰이 없다 —
      채터가 보이면 **damping 을 올리지 말고** `friction` 을 넣어라(마찰은 PD 게인이
      아니라서 벤더 규칙 밖이고, damping 을 올리면 그 순간 벤더값이 아니게 된다).
    """
    p_gain, d_gain = hand_gains()
    spec = dict(joint_names_expr=list(joint_names_expr), stiffness=p_gain, damping=d_gain, **extra)
    if effort_limit_sim is not None:
        spec["effort_limit_sim"] = float(effort_limit_sim)
    return {name: spec}


# ══════════════════════════════════════════════════════════════════════════════
# 벤더 PD 가 **없는** 자리 — 규칙의 명시 예외 (2026-09-06 사용자 확정: 현행값 유지)
# ══════════════════════════════════════════════════════════════════════════════
#: 예외 사유. 값을 여기 적지 않는다 — 각 트랙이 쓰던 값을 그대로 두고, 왜 벤더값이
#: 없는지만 한 곳에 기록한다. 새 예외를 추가하려면 사용자 승인이 필요하다.
NO_VENDOR_PD = {
    "rh56f1_hand": (
        "Inspire RH56F1 은 RS-485 위치 서보다. 벤더 스택(vendor/inspire_ws)이 여는 레지스터는 "
        "angleSet/speedSet/forceSet/defaultSpeedSet/defaultForceSet 뿐이고 PD 게인이라는 개념이 "
        "없다. 자산 USD 는 fallback 100.0/1.0, sim 액추에이터는 트랙별 값을 그대로 쓴다."),
    "stock_gripper_jaw": (
        "스톡 2지 그리퍼의 벤더값 GRIPPER_KP 5.0 / GRIPPER_KD 0.1(openarm_real "
        "v10_simple_hardware.hpp)은 **모터축 회전 게인**[N·m/rad]인데 URDF 조는 직동[m]이다. "
        "리드스크류 환산이 없으면 같은 물리량이 아니라 그대로 옮길 수 없다."),
    "head_dynamixel": (
        "머리는 Dynamixel 이고 OpenArm 벤더 파일은 팔 7관절만 담는다. 실기 머리는 위치 모드 + "
        "I게인 400 이며 정책이 명령하지 않는다(상태만 읽는다)."),
}
