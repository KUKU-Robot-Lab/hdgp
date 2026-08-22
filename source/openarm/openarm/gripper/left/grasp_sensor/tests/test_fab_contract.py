# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fabrics 변형(`open-grip_l_grasp_sensor_fab`) 정적 계약 — Isaac 불필요.

각 계약은 실측으로 태운 함정 하나씩에 대응한다. 통과가 목적이 아니라 재발 방지가 목적.
"""

import math
import re
from pathlib import Path

import pytest

from openarm.gripper.left.grasp_sensor import grasp_left_preset as P

_PKG = Path(__file__).resolve().parents[1]
_HDGP = Path(__file__).resolve().parents[7]
_FABRICS = _HDGP / "source/FABRICS/src/fabrics_sim"


def _src(name: str) -> str:
    return (_PKG / name).read_text(encoding="utf-8")


def test_fabric_rotation_goes_through_quaternion_not_euler():
    """★기준 palm 자세 (0, π/2, 0) 은 euler_zyx 짐벌 특이점 정확히 위다.

    euler 로 지령하면 표현이 퇴화한다(08.21 회전 계단 오버슈트 19~32% 실측이 그 정체).
    set_features 에는 quaternion (B,7) 경로로만 넘긴다.
    """
    src = _src("grasp_left_fabric_action.py")
    assert '"quaternion"' in src
    assert '"euler_zyx"' not in src, "euler 경로가 되살아났다"
    # set_features 의 quaternion 규약은 xyzw ([6,3,4,5] 재배열) — wxyz 를 그대로 넘기면
    # 조용히 다른 회전이 된다.
    assert "q_target[:, 1:4]" in src and "q_target[:, 0]" in src, "wxyz→xyzw 변환이 없다"


def test_fabric_rest_pose_is_this_tasks_home_not_aborted_home():
    """★fabric cspace rest 는 **이 태스크의 홈**이어야 한다.

    내장 기본값은 ABORTED 트랙 홈(j7=+1.356)이고, j7>0.7 은 l_al_5↔l_al_7 자기충돌
    여유가 9 mm 아래로 떨어지는 구간이다(08.21 감사 스윕). 홈이 다르면 자기충돌 결론이
    이식되지 않는다 — fabric 이 팔을 옛 홈으로 끌면 안 된다.
    """
    src = _src("grasp_left_fabric_action.py")
    assert "default_config_override=home" in src
    assert 'P.LEFT_ARM_HOME_JOINT_POS[f"l_aj_{i}"]' in src
    # FABRICS 쪽 파라미터가 실제로 존재하는지 (없으면 TypeError 로 런타임에 죽는다)
    fab_src = (_FABRICS / "fabrics/openarm_tesollo_pose_fabric.py").read_text()
    m = re.search(
        r"class OpenArmGripperLeftPoseFabric.*?def __init__\((.*?)\):", fab_src, re.S
    )
    assert m and "default_config_override" in m.group(1), (
        "OpenArmGripperLeftPoseFabric 이 default_config_override 를 받지 않는다"
    )


def test_fabric_world_is_the_left_arm_one():
    """★우팔용 world 를 쓰면 좌팔이 자기 대역물과 잡을 컵에서 밀려난다(08.21 실측)."""
    assert P.FABRIC_WORLD_FILENAME == "open_gripper_left_boxes_no_table"
    import yaml as _yaml
    world = _yaml.safe_load(
        (_FABRICS / "worlds" / f"{P.FABRIC_WORLD_FILENAME}.yaml").read_text()
    )
    # ⚠ 문자열 검색이 아니라 **키**를 본다 — 파일 머리말 주석이 두 이름을 설명하고 있다.
    assert "left_target_cup" not in world, "잡을 컵이 장애물로 등록돼 있다"
    assert "left_arm_body" not in world, "좌팔 자기 대역물이 남아 있다"
    assert "right_arm_body" in world, "유휴 우팔 대역물이 빠졌다"


def test_fabric_integrates_in_process_actions_once_per_env_step():
    """★apply_actions 는 physics decimation 횟수만큼 불린다 — 거기서 적분하면
    fabric 시간이 2배로 흐른다(agnostic 트랙에서 실증된 함정)."""
    src = _src("grasp_left_fabric_action.py")
    proc = src[src.index("def process_actions"):src.index("def apply_actions")]
    appl = src[src.index("def apply_actions"):src.index("def reset")]
    assert "_integrator.step" in proc
    assert "_integrator.step" not in appl
    assert "set_joint_velocity_target" in appl, "속도목표 0 배선이 빠졌다(agnostic 규약)"


def test_palm_box_covers_spawn_and_goal_regions():
    """액션 박스가 컵 스폰 접근 영역과 확장된 목표 영역을 못 덮으면 정책이 도달해야 할
    곳을 지령할 수 없다 — 보상은 있는데 액션이 못 가는 조용한 불능."""
    spawn_x = (P.CUP_SPAWN_X_CENTER - P.CUP_SPAWN_X_RANGE,
               P.CUP_SPAWN_X_CENTER + P.CUP_SPAWN_X_RANGE)
    spawn_y = (P.CUP_SPAWN_Y_CENTER - P.CUP_SPAWN_Y_RANGE,
               P.CUP_SPAWN_Y_CENTER + P.CUP_SPAWN_Y_RANGE)
    for lo, hi, box in [
        (spawn_x[0], spawn_x[1], P.PALM_BOX_X),
        (spawn_y[0], spawn_y[1], P.PALM_BOX_Y),
        (P.CUP_SPAWN_Z, P.CUP_SPAWN_Z, P.PALM_BOX_Z),
        (P.GOAL_POS_X[0], P.GOAL_POS_X[1], P.PALM_BOX_X),
        (P.GOAL_POS_Y[0], P.GOAL_POS_Y[1], P.PALM_BOX_Y),
        (P.GOAL_POS_Z[0], P.GOAL_POS_Z[1], P.PALM_BOX_Z),
    ]:
        assert box[0] <= lo and hi <= box[1], (
            f"PALM_BOX {box} 가 요구 구간 [{lo}, {hi}] 를 못 덮는다"
        )


def test_goal_region_is_the_user_specified_box():
    """08.22 사용자 지정: 목표 영역 x±5 y±7 z±5 cm. 하한은 리프트 임계 위여야
    '먼저 들어라 → 옮겨라' 순서가 유지된다(기존 계약과 동일 논리)."""
    assert P.GOAL_JITTER == (0.05, 0.07, 0.05)
    assert P.GOAL_POS_Z[0] > P.MINIMAL_LIFT_HEIGHT


def test_fab_action_dim_is_seven():
    """팔 6D(위치 3 + 축각 3) + 이진 그리퍼 1 = 7. 관절공간판(8)과 다르다 —
    체크포인트를 트랙 간에 섞어 쓰면 안 된다."""
    src = _src("grasp_left_fabric_action.py")
    assert "return 6" in src
    # 그리퍼는 부모 것을 그대로 쓴다 — fab cfg 가 arm_action 만 바꾸는지
    cfg = _src("grasp_left_fab_env_cfg.py")
    assert "arm_action" in cfg
    assert "gripper_action" not in cfg, "그리퍼 액션까지 덮어썼다 — 부모 검증이 무효가 된다"


def test_ref_quat_is_the_measured_home_palm_orientation():
    """★기준 파지 자세 = **홈의 실측 palm 자세**여야 한다 — 임의 자세가 아니다.

    Ry(90°) 를 썼다가 G2 게이트가 죽었다(홈이 접근축 기준 180° 뒤집힌 자세라 위치 유지
    지령만 줘도 fabric 이 손목을 뒤집으려 들었다, hold 44 mm 실측). 홈을 바꾸면 이 값도
    재실측해야 한다 — 여기서는 정규화와 "홈 접근축이 +X 를 향한다"는 기하 성질만 고정.
    """
    import torch
    w, x, y, z = P.PALM_REF_QUAT_WXYZ
    assert math.isclose(w * w + x * x + y * y + z * z, 1.0, abs_tol=1e-6)
    # R[:,2] = 접근축. 실측 (+0.94, +0.26, −0.24) — x 성분이 지배해야 한다.
    q = torch.tensor([w, x, y, z])
    ww, xx, yy, zz = q
    approach = torch.tensor([
        2 * (xx * zz + ww * yy),
        2 * (yy * zz - ww * xx),
        1 - 2 * (xx * xx + yy * yy),
    ])
    assert approach[0] > 0.8, f"접근축이 +X 를 향하지 않는다: {approach.tolist()}"


def test_fab_gym_registration_points_to_real_symbols():
    """등록 실패는 glob 임포트가 조용히 삼킨다 — entry point 문자열을 정적으로 검증."""
    cfg_init = (_PKG / "config" / "__init__.py").read_text()
    assert "open-grip_l_grasp_sensor_fab" in cfg_init
    assert "grasp_left_fab_env_cfg:GraspLeftGripperFabEnvCfg" in cfg_init
    assert (_PKG / "config" / "agents" / "rl_games_ppo_fab_cfg.yaml").is_file()
    yaml_txt = (_PKG / "config" / "agents" / "rl_games_ppo_fab_cfg.yaml").read_text()
    assert "name: open-grip_l_grasp_sensor_fab" in yaml_txt
    env_cfg = _src("grasp_left_fab_env_cfg.py")
    assert "class GraspLeftGripperFabEnvCfg(" in env_cfg
    assert "class GraspLeftGripperFabEnvCfg_PLAY(" in env_cfg
