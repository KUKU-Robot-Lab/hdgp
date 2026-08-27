"""로봇 프로필 — **이 트랙의 목적이 여기 있다: USD 만 바꿔서 같은 과제를 성공시킨다.**

자매 `grasp_s2r` 의 프로필을 그대로 받고(단일 소스), 자산이 바뀌면서 **실제로 달라지는
필드만** 덮어 새 프로필을 만든다. 자매 파일은 다른 세션 소유라 로봇을 추가할 수 없으므로
레지스트리는 여기서 확장한다.

## 자산 교체가 건드리는 것 (bi_s 사례)

| 필드 | 왜 바뀌나 |
|---|---|
| `usd_relpath` | 자산 그 자체 |
| `fabric_class`/`fabric_robot_dir`/`fabric_params_filename` | Fabrics URDF 가 자산마다 다르다 |
| `init_joint_pos`/`actuator_specs` | **반대편 팔의 구성이 다르다**(sensor=2 지 그리퍼 / bi_s=Tesollo 20 DOF). 없는 관절 이름을 남겨두면 Articulation 조립이 실패한다 |
| `palm_box_min/max`·`object_spawn_center` | palm 오프셋이 54.8mm 다르다 → 도달영역과 컵 배치가 따라 바뀐다 |
| `PALM_NORMAL_COL` | ★★**손바닥 법선축이 자산마다 다르다** — 아래 참조 |

반대로 **바뀌지 않는 것**은 전부 상속한다: 관절/바디 이름 규약, 시너지 자세
(`hand_open_pose`/`hand_grip_pose`), 접촉 그룹, 우팔 액추에이터 게인, 회전 박스.

## ★★손바닥 법선축 (`PALM_NORMAL_COL`)

자매 코드는 `_palm_ee_R()` 의 **열 0 이 손바닥 법선**이라고 가정하고, approach 항이
`palm_normal_dist = |d_local.x|` 로 밀착도를 잰다. 이 가정은 **자산마다 다르다**:

- `openarm_tesollo_sensor_rl` : 법선 = 링크 로컬 **+x** (열 0)
- `openarm_tesollo_bi_s_rl`   : 법선 = 링크 로컬 **+y** (열 1)
  (`modules/robots.py` `palmar_axis_local` 실측 주석: URDF 유도 + probe_palmar_sign
   실측 우팔 +y 합계 +270mm/9-of-10 마디. "자매 sensor 자산은 palmar 가 (1,0,0)이다"
   라고 같은 주석이 명시한다.)

그래서 프로필마다 법선이 몇 번째 열인지 적고, env 가 `_palm_ee_R()` 을 **순환
치환**으로 재정렬해 downstream 전부(approach 거리·케이지 오프셋·obs palm_ax)가 한
번에 맞게 한다. 순환이라 오른손 좌표계(det=+1)가 보존된다.
"""

from __future__ import annotations

import dataclasses

from ...modules import robots as _rb          # 공유 자산/자세 상수 — **읽기만**
from ..grasp_s2r import robot_profiles as _s2r

RobotProfile = _s2r.RobotProfile

_SENSOR_R = _s2r.PROFILES["tesollo_right"]

# ---- bi_s(양팔 Tesollo) 우팔 -------------------------------------------------
# 반대편(좌) 손이 2 지 그리퍼 → Tesollo 20 DOF 로 바뀐다. 없는 관절 이름을 남기면
# Articulation 이 조립에서 죽으므로 **지우고 넣는다**.
_BIS_INIT = {k: v for k, v in _SENSOR_R.init_joint_pos.items()
             if not k.startswith("l_hj_")}
_BIS_INIT.update(_rb._tesollo_hand_rest("l"))

_BIS_ACT = {k: v for k, v in _SENSOR_R.actuator_specs.items() if k != "left_gripper"}
_BIS_ACT.update(_rb._tesollo_hand_actuator("idle", "l"))

BIS_RIGHT = dataclasses.replace(
    _SENSOR_R,
    name="bis_right",
    usd_relpath=_rb.TESOLLO_BI_S.usd_relpath,
    fabric_class=_rb.BIS_RIGHT.fabric_class,
    fabric_robot_dir=_rb.BIS_RIGHT.fabric_robot_dir,
    # bi_s 는 fabric 기본 params 를 쓴다(자매 sensor 자산만 전용 파일이 있다).
    fabric_params_filename=None,
    init_joint_pos=_BIS_INIT,
    actuator_specs=_BIS_ACT,
    # 도달영역·컵 배치는 palm 오프셋(54.8mm 차)을 따라간다 — 공유 프로필 실측값.
    palm_box_min=_rb.BIS_RIGHT.palm_box_min,
    palm_box_max=_rb.BIS_RIGHT.palm_box_max,
    palm_box_verified=_rb.BIS_RIGHT.palm_box_verified,
    object_spawn_center=_rb.BIS_RIGHT.object_spawn_center,
)

# 손바닥 법선이 `_palm_ee_R()` 의 몇 번째 열인가 — 파일 상단 설명 참조.
PALM_NORMAL_COL: dict[str, int] = {
    "tesollo_right": 0,
    "gripper_left": 0,
    "bis_right": 1,
}

_OURS: dict[str, RobotProfile] = {"bis_right": BIS_RIGHT}

# 자매 것 + 우리 것. 자매 프로필을 **덮어쓰지 않는다**(같은 이름이면 fail-loud).
_dupe = set(_OURS) & set(_s2r.PROFILES)
if _dupe:
    raise RuntimeError(
        f"자매 프로필과 이름이 겹친다: {sorted(_dupe)} — 자매 정의를 가리게 된다")
PROFILES: dict[str, RobotProfile] = {**_s2r.PROFILES, **_OURS}

_missing = [n for n in PROFILES if n not in PALM_NORMAL_COL]
if _missing:
    raise RuntimeError(
        f"PALM_NORMAL_COL 미선언 프로필 {_missing} — 손바닥 법선축은 자산마다 다르므로 "
        "추정하면 안 된다(probe_palmar_sign 으로 실측할 것)")

# ★자매 `GraspS2REnv.__init__` 이 자기 모듈의 `PROFILES[cfg.profile_name]` 을 직접
#   읽는다. 우리 프로필을 그 레지스트리에 **등록**해야 상속이 성립한다(자매 *파일* 은
#   건드리지 않는다 — 런타임 딕셔너리 등록이다). 기존 키는 절대 덮지 않는다.
for _n, _p in _OURS.items():
    if _n in _s2r.PROFILES and _s2r.PROFILES[_n] is not _p:
        raise RuntimeError(f"자매 레지스트리의 '{_n}' 을 덮으려 한다 — 금지")
    _s2r.PROFILES[_n] = _p
