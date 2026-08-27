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

반대로 **바뀌지 않는 것**은 전부 상속한다: 관절/바디 이름 규약, 시너지 자세
(`hand_open_pose`/`hand_grip_pose`), 접촉 그룹, 우팔 액추에이터 게인, 회전 박스,
그리고 **손바닥 법선축**.

## 손바닥 법선은 `palm_ee` **+x** — 자산 무관 (사용자 확정 08.27)

자매 approach 가 `palm_normal_dist = |d_local.x|` 로 밀착도를 재는 근거이고, 이
규약은 자산이 바뀌어도 같다(palm 계열 body 는 회전이 동일하고 위치만 다르다).

★혼동 주의: `modules/robots.py` 의 `palmar_axis_local` 은 **손가락 마디 링크**의
  손바닥면 방향(마디별 dict)이지 palm body 의 법선이 아니다. 그 값이 자산마다
  (0,1,0)/(1,0,0) 로 갈리는 것은 **손가락 링크 프레임** 이야기다 — palm 법선축을
  자산별로 바꾸면 approach 가 엉뚱한 축을 재게 된다.
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

_OURS: dict[str, RobotProfile] = {"bis_right": BIS_RIGHT}

# 자매 것 + 우리 것. 자매 프로필을 **덮어쓰지 않는다**(같은 이름이면 fail-loud).
_dupe = set(_OURS) & set(_s2r.PROFILES)
if _dupe:
    raise RuntimeError(
        f"자매 프로필과 이름이 겹친다: {sorted(_dupe)} — 자매 정의를 가리게 된다")
PROFILES: dict[str, RobotProfile] = {**_s2r.PROFILES, **_OURS}

# ★자매 `GraspS2REnv.__init__` 이 자기 모듈의 `PROFILES[cfg.profile_name]` 을 직접
#   읽는다. 우리 프로필을 그 레지스트리에 **등록**해야 상속이 성립한다(자매 *파일* 은
#   건드리지 않는다 — 런타임 딕셔너리 등록이다). 기존 키는 절대 덮지 않는다.
for _n, _p in _OURS.items():
    if _n in _s2r.PROFILES and _s2r.PROFILES[_n] is not _p:
        raise RuntimeError(f"자매 레지스트리의 '{_n}' 을 덮으려 한다 — 금지")
    _s2r.PROFILES[_n] = _p
