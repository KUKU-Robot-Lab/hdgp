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


def test_fabric_rotation_uses_reference_euler_zyx_convention():
    """★★fab_test21: 회전은 원본 kuka 와 같은 **euler_zyx 절대** 규약이다.

    ⚠ 이 계약은 08.21 의 반대 계약을 **의도적으로 뒤집은 것**이다. 당시 계약은
      "quaternion 경로만" 을 요구했고 근거는 기준 palm 자세가 (0, π/2, 0) 으로 euler_zyx
      짐벌 특이점 **정확히 위**라는 것, 그리고 회전 계단 오버슈트 19~32% 실측이었다.
      뒤집는 근거 둘:
        ① 기준 자세가 바뀌어 ey 중심이 −76.09° 다(특이점에서 14°). 정확히 위가 아니다.
        ② 그 19~32% 는 **결함 있는 플랜트**(vel_ff 0 · fabric 60% 속도 · damping 하드끝)
           위에서 잰 값이다. 셋을 원본으로 되돌렸으므로 재측정 대상이다.
      ★재측정으로 오버슈트가 재현되면 이 계약을 되돌리고 사유를 여기 적을 것.

    ⚠ 짐벌 특이점은 불연속이 아니다 — 전방 사상은 연속이고 비용은 조건수 저하다.
    """
    src = _src("grasp_left_fabric_action.py")
    assert '"euler_zyx"' in src, "set_features 에 euler_zyx 규약을 넘기지 않는다"
    assert '"quaternion"' not in src
    # (B, 6) = [xyz, ez, ey, ex]
    assert "torch.zeros(num_envs, 6, device=device)" in src, "palm pose 버퍼가 6D 가 아니다"

    # 중심은 로봇별(우리 기준 파지 자세), 반폭은 원본 실사용값 45°
    import math as _m
    assert len(P.PALM_EULER_ZYX_CENTER) == 3
    assert abs(P.PALM_MAX_POSE_ANGLE - _m.radians(45.0)) < 1e-9, (
        "원본 kuka 실사용 max_pose_angle 은 45° 다(env.max_pose_angle=45.0)"
    )
    # 중심이 기준 quat 과 같은 자세를 가리키는지 — 규약이 어긋나면 홈 자세가 통째로 틀어진다
    ez, ey, ex = P.PALM_EULER_ZYX_CENTER
    cz, sz = _m.cos(ez), _m.sin(ez)
    cy, sy = _m.cos(ey), _m.sin(ey)
    cx, sx = _m.cos(ex), _m.sin(ex)
    R_e = [[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
           [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
           [-sy, cy * sx, cy * cx]]
    w, x, y, z = P.PALM_REF_QUAT_WXYZ
    R_q = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
           [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
           [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    err = max(abs(R_e[i][j] - R_q[i][j]) for i in range(3) for j in range(3))
    assert err < 1e-6, f"euler 중심이 기준 quat 자세와 다르다 (max |ΔR| = {err:.2e})"


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
    assert "set_joint_velocity_target" in appl, "속도목표 배선이 빠졌다"


def test_velocity_feedforward_is_wired_not_zeroed():
    """★★fab_test21: fabric 속도를 PD 속도목표로 **넘겨야** 한다. 0 을 넣으면 안 된다.

    옛 계약은 정확히 반대였다 — "속도목표 0 배선이 빠졌다(agnostic 규약)". 그 규약이
    어디서 왔는지 근거가 없었고, DEXTRAH 원본은 `vel_scale * fabric_qd` 를 넣는다.
    0 을 넣으면 PD 감쇠항이 움직임을 반대로 민다:
        vel_ff 없음: kp·err = kd·v + τ_마찰 → err ≈ (kd/kp)·v = 0.2·v [rad]
        vel_ff 있음: kp·err = τ_마찰만      → err ≈ 0.0012 rad = 0.07°
    실측 정합: 관절속도 0.855 rad/s 일 때 예측 171 mrad vs 실측 |fabric_q−q| 140 mrad.

    ⚠ 이 테스트가 존재하는 이유는 **계약이 결함을 화석으로 굳혔기 때문**이다. 옛 계약은
      근거 없는 전제("agnostic 규약")를 못박아 두 트랙에 걸쳐 결함을 보호했다.
      계약은 실측이나 원본 소스에 묶어야 한다 — 관행에 묶으면 안 된다.
    """
    src = _src("grasp_left_fabric_action.py")
    appl = src[src.index("def apply_actions"):src.index("def reset")]
    assert "torch.zeros_like" not in appl, (
        "속도목표에 0 을 넣고 있다 — 속도에 비례해 뒤처지도록 배선한 것이다"
    )
    assert "_fabric_qd" in appl, "적분된 fabric 속도를 쓰지 않는다"
    assert "_vel_ff_scale" in appl, "ADR 이 조절할 배율이 곱해지지 않았다"

    # ADR 방향: 레벨 0 = 완전 피드포워드(쉬움) → 최고 레벨 = 0(어려움). DEXTRAH (1.0, 0.0).
    assert P.ADR_VEL_FF_SCALE[0] > P.ADR_VEL_FF_SCALE[1], (
        f"ADR 방향이 뒤집혔다 {P.ADR_VEL_FF_SCALE} — 처음부터 드래그가 걸려 결함을 재현한다"
    )
    assert P.ADR_VEL_FF_SCALE[0] == P.FABRIC_VEL_FF_SCALE, (
        "ADR 레벨 0 값과 기본값이 다르면 ADR 적용 전후로 제어가 바뀐다"
    )
    assert 0.0 < P.FABRIC_VEL_FF_SCALE <= 1.0

    cur = _src("grasp_left_curriculums.py")
    assert "vel_ff_scale" in cur, "ADR 이 vel_ff 를 조절하지 않는다"


def test_palm_box_covers_spawn_and_goal_regions():
    """액션 박스가 컵 스폰 접근 영역과 확장된 목표 영역을 못 덮으면 정책이 도달해야 할
    곳을 지령할 수 없다 — 보상은 있는데 액션이 못 가는 조용한 불능."""
    spawn_x = (P.CUP_SPAWN_X_CENTER - P.CUP_SPAWN_X_RANGE,
               P.CUP_SPAWN_X_CENTER + P.CUP_SPAWN_X_RANGE)
    spawn_y = (P.CUP_SPAWN_Y_CENTER - P.CUP_SPAWN_Y_RANGE,
               P.CUP_SPAWN_Y_CENTER + P.CUP_SPAWN_Y_RANGE)
    # ★★fab_test35 정정. 이 계약은 **팜 지령 z ≈ 물체 z** 를 전제하고 있었는데
    #   `probe_palm_z_transfer.py` 실측이 그 전제를 깼다 — 팔이 내려갈수록 압축된다:
    #       지령 z 0.445 → 턱 z 0.465 (목표 상단)   지령 0.315 → 턱 0.325 (목표 하단)
    #       지령 0.202  → 턱 0.2488 (파지)          지령 ≤0.13 → 턱 0.2300 에서 포화
    #   그래서 z 만 **지령 축으로 환산한 값**으로 검사한다. x·y 는 1:1 에 가까워 그대로 둔다.
    #   ⚠ 이 상수들은 전달표에서 읽은 실측이다. 기하로 추정하면 또 틀린다.
    GOAL_CMD_Z = (0.315, 0.445)
    GRASP_CMD_Z = 0.202
    for lo, hi, box in [
        (spawn_x[0], spawn_x[1], P.PALM_BOX_X),
        (spawn_y[0], spawn_y[1], P.PALM_BOX_Y),
        (GRASP_CMD_Z, GRASP_CMD_Z, P.PALM_BOX_Z),
        (P.GOAL_POS_X[0], P.GOAL_POS_X[1], P.PALM_BOX_X),
        (P.GOAL_POS_Y[0], P.GOAL_POS_Y[1], P.PALM_BOX_Y),
        (GOAL_CMD_Z[0], GOAL_CMD_Z[1], P.PALM_BOX_Z),
    ]:
        assert box[0] <= lo and hi <= box[1], (
            f"PALM_BOX {box} 가 요구 구간 [{lo}, {hi}] 를 못 덮는다"
        )


def test_goal_region_is_the_user_specified_box():
    """★fab_test17 ADR(사용자 지시 "모드을"=보수적): 목표 영역 x±8 **y±9** z±7 cm (워크스페이스 스캔 실측 전역).
    y 만 크게 넓힌 이유 — 작업면 Y 는 90cm 로 X(40cm)의 2.25배이고, x 는 테이블 앞모서리까지
    10mm 여유뿐이라 못 넓힌다. z 는 파지·리프트 기하에 종속이라 소폭만.
    하한은 리프트 임계 위여야 '먼저 들어라 → 옮겨라' 순서가 유지된다(기존 계약과 동일 논리)."""
    assert P.GOAL_JITTER == (0.08, 0.09, 0.07)
    assert P.GOAL_POS_Z[0] > P.MINIMAL_LIFT_HEIGHT


def test_adr_ranges_match_the_measured_workspace():
    """★★fab_test18 ADR — 스폰·목표 범위는 **전부 실측이 정한 값**이어야 한다.

    세 실측이 각 경계를 잡았다:
      · 관통 스윕(smoke 1e): 스폰 x [0.36,0.42] 25/25 조용 · 스폰 y 를 올리면 관통
        (y 0.26 에서 4/5, y 0.29 에서 5/5) → 스폰 y 상한 0.21
      · 도달성(probe_adr_reach): 스폰·목표 y 0.09·0.14 는 미달(최대 198 mm)
      · 워크스페이스 스캔 140점: y 0.23~0.41 이 스위트스팟(중앙값 1.0~4.6 mm),
        y ≤0.17 급락(56.9~128.6), x 0.30~0.46 유사, z 0.33~0.51 유사
    """
    # 스폰 x — 관통 경계(아래)와 테이블 앞모서리−컵반경(위) 사이
    assert P.CUP_SPAWN_X_CENTER - P.CUP_SPAWN_X_RANGE >= P.SPAWN_X_SAFE_MIN - 1e-9
    assert P.CUP_SPAWN_X_CENTER + P.CUP_SPAWN_X_RANGE <= P.WORK_SURFACE_X[1] - 0.044 + 1e-9
    # 스폰 y — 양쪽이 막혀 있다(아래 도달성, 위 관통)
    assert P.CUP_SPAWN_Y_CENTER - P.CUP_SPAWN_Y_RANGE >= 0.17 - 1e-9
    assert P.CUP_SPAWN_Y_CENTER + P.CUP_SPAWN_Y_RANGE <= 0.21 + 1e-9
    # 목표 — 스캔이 확인한 도달 영역 안
    assert P.GOAL_POS_X[0] >= 0.30 - 1e-9 and P.GOAL_POS_X[1] <= 0.46 + 1e-9
    assert P.GOAL_POS_Y[0] >= 0.23 - 1e-9 and P.GOAL_POS_Y[1] <= 0.41 + 1e-9
    assert P.GOAL_POS_Z[1] <= 0.51 + 1e-9
    # ADR 만렙이 위 실측 범위와 일치해야 한다 — 아니면 ADR 이 검증 안 된 곳으로 넓힌다
    assert P.ADR_SPAWN_X_RANGE[1] == P.CUP_SPAWN_X_RANGE
    assert P.ADR_SPAWN_Y_RANGE[1] == P.CUP_SPAWN_Y_RANGE
    assert P.ADR_GOAL_JITTER_SCALE[1] == 1.0


def test_adr_starts_neutral_and_ramps_on_dwell():
    """★★DR 초기값은 **중립**이어야 한다 — 처음부터 미끄럽거나 무거우면 파지를 못 배운다.
    fab_test14 가 jerk 로 정확히 같은 실수를 했다(초기 탐색 41% 사망)."""
    assert P.ADR_CUP_STATIC_FRICTION[0] == (1.0, 1.0)
    assert P.ADR_CUP_DYNAMIC_FRICTION[0] == (1.0, 1.0)
    assert P.ADR_CUP_MASS_SCALE[0] == (1.0, 1.0)
    assert P.ADR_CUP_MAX_LINEAR_ACCEL[0] == 0.0
    # 만렙은 실제로 어려워져야 한다
    assert P.ADR_CUP_DYNAMIC_FRICTION[1][0] < 1.0, "만렙에도 미끄럽지 않으면 DR 이 아니다"
    assert P.ADR_CUP_MASS_SCALE[1][1] > 1.0
    assert P.ADR_CUP_MAX_LINEAR_ACCEL[1] > 0.0

    # ⚠ 확장 간격은 **env 스텝** 단위여야 한다(common_step_counter). epoch 을 그대로 쓰면
    #   1 epoch = horizon_length 스텝이라 확장이 그만큼 빨라진다.
    # ★★fab_test22: 이제 env 스텝으로 **직접** 정의한다(원본과 같은 기준). epoch×horizon
    #   환산은 나머지 때문에 원본값 3000 에 못 맞았다(187×16 = 2992).
    #   대략 몇 epoch 인지는 아래로 검산한다 — 두 자리 수 이상 어긋나면 뭔가 잘못된 것이다.
    assert abs(P.ADR_MIN_STEPS_BETWEEN / P.ADR_HORIZON_STEPS - P.ADR_MIN_EPOCHS_BETWEEN) < 1.0
    yaml_text = (_PKG / "config/agents/rl_games_ppo_fab_cfg.yaml").read_text(encoding="utf-8")
    m = re.search(r"horizon_length:\s*(\d+)", yaml_text)
    assert m and int(m.group(1)) == P.ADR_HORIZON_STEPS, "horizon 환산이 yaml 과 어긋났다"

    cfg = _src("grasp_left_fab_env_cfg.py")
    for term in ("cup_physics_material", "cup_mass", "cup_disturbance"):
        assert f"self.events.{term} = EventTermCfg(" in cfg, f"{term} 이 배선되지 않았다"
    assert "curriculums.adr_expand_on_dwell" in cfg
    cur = _src("grasp_left_curriculums.py")
    assert "self._level += 1" in cur and "self._level < levels" in cur, "레벨 상한이 없다"


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


def test_between_jaws_reward_is_wired_and_uses_enclose_not_closure():
    """★★08.22 fab_test1 실패의 직접 처방 — 이 항이 빠지면 같은 실패가 재발한다.

    실측: fab 정책은 개도 3.1 mm 로 **주먹을 쥔 채** 컵 옆구리를 눌렀다(수직 최선 36.5 mm
    ≈ 컵 반경). 성공한 test17 은 0.4 mm / 26.5 mm.
    ⚠ 옛 `closure = 1 − drive/open` 을 곱하면 **닫을수록 커져** 그 실패 행동을 보상한다.
    """
    cfg = _src("grasp_left_env_cfg.py")
    # ★★fab_test32: 접근 보상을 agnostic 트랙 `approach_reward` 로 이식했다.
    #   TCP 수평 배수(`reach_with_tcp_level`)는 폐기 — 접근 신호를 1/5 로 떨어뜨렸다.
    #   대역 겨냥은 이제 `grasp_offset` 이 담당한다(파지중심 = 컵 원점 + 오프셋).
    assert "rewards.approach_opposed" in cfg, "접근 보상이 이식본이 아니다"
    assert "def approach_opposed" in _src("grasp_left_rewards.py")
    assert '"grasp_offset": P.CUP_ORIGIN_TO_GRASP_Z' in cfg, (
        "파지중심이 컵 원점 그대로다 — 대역이 아니라 원점을 겨냥하게 된다"
    )
    assert "func=rewards.reach_with_tcp_level" not in cfg, "폐기한 배수판이 아직 배선돼 있다"
    assert "self.rewards.cup_between_jaws" in cfg, "턱 사이 진입 보상이 배선되지 않았다"
    rew = _src("grasp_left_rewards.py")
    assert "def cup_between_jaws(" in rew
    assert "def gripper_closure_on_cup(" not in rew, (
        "닫을수록 커지는 옛 식이 되살아났다 — 주먹으로 누르기가 감싸기보다 높아진다"
    )
    # enclose = 두 손가락이 컵 축 양쪽에 있는가
    assert "s_l" in rew and "s_r" in rew and "torch.minimum" in rew


def test_between_jaws_averages_along_and_lateral_never_multiplies():
    """★CLAUDE.md 설계 규칙: 곱할 항은 게이트 성격만, 연속 품질은 평균한다.

    곱하면 fab 현재 상태 품질이 0.0159 로 떨어져 다른 항에 묻힌다(test10/test11 을
    죽인 구조). 평균이면 0.170 — 10 배 차이.
    """
    rew = _src("grasp_left_rewards.py")
    # 08.23 기하가 공용 헬퍼로 옮겨졌다 — 두 항(정렬/폐쇄)이 같은 자를 쓰게 하기 위해서다.
    body = rew[rew.index("def _jaw_geometry("):]
    assert "align = 0.5 * (1.0 - torch.tanh(along / along_std)) + 0.5 * (" in body


def test_between_jaws_std_matches_measured_scale():
    """★임계는 실측 규모에서 정한다 — 옛 20/20 mm 는 fab 현재 상태 품질이 0.00000 이라
    배선해도 죽은 항이 된다(test10/test11 과 같은 함정)."""
    import math as _m
    # fab_test1 실측 평균: along 27.1 mm · lateral 108.9 mm
    q = 0.5 * (1 - _m.tanh(0.0271 / P.JAW_ALONG_STD)) + 0.5 * (
        1 - _m.tanh(0.1089 / P.JAW_LATERAL_STD)
    )
    assert q > 0.10, f"현재 상태에서 정렬 품질이 {q:.4f} — gradient 가 죽는다"
    # 목표 상태(test17 최선: along 0 · lateral 0.4 mm)에서는 만점 근처여야 개선 폭이 생긴다
    q_goal = 0.5 * 1.0 + 0.5 * (1 - _m.tanh(0.0004 / P.JAW_LATERAL_STD))
    assert q_goal > 0.95 and q_goal / q > 4.0, "현재↔목표 개선 폭이 부족하다"


def test_between_jaws_weight_stays_a_stepping_stone():
    """★상한이 lifting+goal+settle 대비 작아야 "감싸고 가만히 있기" 국소최적이 안 생긴다."""
    assert P.BETWEEN_JAWS_REWARD_WEIGHT <= 0.15 * (
        15.0 + 16.0 + P.SETTLE_REWARD_WEIGHT
    )


def test_reach_reward_targets_the_graspable_band_not_the_cup_origin():
    """★★컵 원점은 상면 +92 mm 인데 그리퍼 통과 대역은 +10~85 mm 다.

    원점 높이의 컵 지름(88 mm)이 개구(84.5 mm)보다 넓어 **턱이 못 들어간다.**
    레퍼런스 `object_ee_distance` 를 그대로 쓰면 도달 보상이 들어갈 수 없는 높이를
    가리킨다(G3 실측: 진입 TCP 오차 100.2 mm → 대역 겨냥 시 70.7 mm).
    """
    lo, hi = P.GRASP_HEIGHT_BAND
    origin_h = P.CUP_SPAWN_Z - P.TABLE_SURFACE_Z
    assert not (lo <= origin_h <= hi), "전제가 바뀌었다 — 컵 원점이 파지 대역 안이면 이 교정은 불필요"
    grasp_h = P.GRASP_TARGET_Z - P.TABLE_SURFACE_Z
    assert lo < grasp_h < hi, f"파지 목표 높이 {grasp_h * 1e3:.0f} mm 가 대역 밖이다"
    cfg = _src("grasp_left_env_cfg.py")
    assert "self.rewards.reaching_object" in cfg
    rew = _src("grasp_left_rewards.py")
    # ★오프셋은 컵 로컬 축을 따라야 한다 — world z 면 컵이 기울 때 파지점이 컵 밖으로 나간다
    body = rew[rew.index("def ee_grasp_point_distance("):]
    assert "matrix_from_quat" in body and "cup_z * grasp_offset" in body


def test_physx_aggregate_pair_buffers_cover_measured_demand():
    """★★부족해도 **죽지 않고 접촉만 조용히 놓친다** — 가장 위험한 종류다.

    vision-3090 2048 env 실측: PhysX 가 foundLostAggregatePairsCapacity **4,562,626** 을
    요구했다. fab_test1 은 2 * 1024 * 1024 로 돌아 내내 상호작용을 놓치고 있었고,
    모니터링 grep 이 "Patch buffer|buffer overflow" 만 봐서 드러나지 않았다.
    """
    src = _src("grasp_left_env_cfg.py")
    ns: dict = {}
    for line in src.splitlines():
        t = line.strip()
        if t.startswith("self.sim.physx.gpu_") and "=" in t:
            key, val = t.split("=", 1)
            try:
                ns[key.strip().rsplit(".", 1)[-1]] = eval(val.strip())  # noqa: S307
            except Exception:  # noqa: BLE001
                pass
    assert ns["gpu_found_lost_aggregate_pairs_capacity"] >= 4_562_626, (
        "실측 요구치 미만 — PhysX 가 접촉을 조용히 놓친다"
    )
    assert (
        ns["gpu_total_aggregate_pairs_capacity"]
        >= ns["gpu_found_lost_aggregate_pairs_capacity"] // 2
    ), "total 이 found_lost 대비 지나치게 작다"
    # 같은 런에서 함께 드러난 두 버퍼 (실측 요구치)
    assert ns["gpu_max_rigid_contact_count"] >= 3_191_536, "contact 버퍼가 실측 요구치 미만"
    assert ns["gpu_collision_stack_size"] >= 68_960_016, (
        "collisionStackSize 가 실측 요구치 미만 — 'Contacts have been dropped' 가 난다"
    )



def test_jaw_reference_line_sits_at_the_measured_grasp_depth():
    """★★파지 기준점은 TCP 도 턱 중점도 아니다 — **손가락 패드 중앙**이다.

    test17(성공) 정책의 **실제로 들고 있는 13,058 샘플** 실측: 컵 축의 base z 중앙값
    **+46.9 mm**. 손가락 강체 원점은 +15.0 mm, TCP 는 +80.0 mm 다.
    기준선을 안 옮기면 보상이 "컵을 손바닥까지 32 mm 더 밀어넣어라"를 가리킨다.
    ⚠ 여기서 두 번 틀렸다: TCP 기준(진입오차 70.7 mm) → 턱 중점 기준(141.9 mm, 악화).
      lateral 의 min(최선 1 샘플)을 깊이로 읽은 게 원인 — 분포를 봐야 한다.
    """
    assert P.JAW_FINGER_BODY_Z < P.GRASP_DEPTH_IN_BASE_Z < P.TCP_OFFSET_IN_BASE_Z, (
        "파지 깊이가 손가락 원점과 TCP 사이에 있어야 한다"
    )
    assert abs(P.JAW_PAD_OFFSET - 0.0319) < 1e-6
    assert abs(P.TCP_TO_GRASP_DEPTH - 0.0331) < 1e-6
    rew = _src("grasp_left_rewards.py")
    # 08.23 프레임 계산이 _jaw_frame 으로 한 번 더 분리됐다 — lift 게이트도 같은 자를 쓴다.
    body = rew[rew.index("def _jaw_frame("):]
    assert "approach * pad_offset" in body, "기준선을 패드 중앙으로 옮기지 않았다"
    cfg = _src("grasp_left_env_cfg.py")
    assert '"pad_offset": P.JAW_PAD_OFFSET' in cfg


def test_palm_box_has_room_for_gravity_droop_lead():
    """★★fabric 은 물리적 중력 처짐을 모른다 — 지령을 앞당겨야 자세가 나온다.

    G4 폐루프 실측: 파지 자세를 실제로 내려면 접근축으로 **47~117 mm 선행**이 필요하고,
    그렇게 하면 TCP 오차 0.2 mm 로 수렴한다. 박스가 그 선행을 못 담으면 클램프되어
    정책이 필요한 지령을 **표현할 수 없다**(0.50 일 때 8.7 mm 에서 정체).
    """
    deepest_x = P.CUP_SPAWN_X_CENTER + P.CUP_SPAWN_X_RANGE + P.TCP_TO_GRASP_DEPTH
    assert P.PALM_BOX_X[1] >= deepest_x + 0.13, (
        f"x 상한 {P.PALM_BOX_X[1]} 이 파지점 {deepest_x:.3f} + 처짐 선행 여유를 못 담는다"
    )


def test_closure_reward_is_gated_by_enclosure_with_no_floor():
    """★★닫기 보상은 **감싼 상태에서만** 나와야 한다.

    fab_test4 실측: enclose 0.845 로 턱은 잘 감쌌는데 '열기' 지령 78.0% · 거의 닫힘 0.0%
    — 닫는 보상이 없어 **한 번도 닫지 않았다**(닫으면 컵이 밀려 정렬 보상만 잃는다).
    ⚠ 옛 `gripper_closure_on_cup` 처럼 closure 를 약한 straddle 에만 곱하면 허공·주먹
      폐쇄가 만점이 된다(fab_test1 이 정확히 그 행동). enclose 를 곱하고 **바닥값은 없다.**
    """
    cfg = _src("grasp_left_env_cfg.py")
    assert "self.rewards.grip_closure_when_enclosed" in cfg
    rew = _src("grasp_left_rewards.py")
    body = rew[rew.index("def grip_closure_when_enclosed("):]
    body = body[: body.index("\ndef ")] if "\ndef " in body else body
    assert "align * enclose * closure" in body, "enclose 게이트가 없다"
    assert "enclose_floor" not in body, "폐쇄 보상에 바닥값을 주면 허공 폐쇄가 보상된다"


def test_both_jaw_rewards_share_one_geometry_helper():
    """두 항이 다른 기하를 쓰면 한쪽만 고쳤을 때 조용히 어긋난다."""
    rew = _src("grasp_left_rewards.py")
    assert "def _jaw_geometry(" in rew
    for fn in ("cup_between_jaws", "grip_closure_when_enclosed"):
        body = rew[rew.index(f"def {fn}("):]
        body = body[: body.index("\ndef ", 10)] if "\ndef " in body[10:] else body
        assert "_jaw_geometry(" in body, f"{fn} 이 공용 기하를 안 쓴다"


def test_closure_weight_stays_a_stepping_stone():
    """★closure 는 컵 지름에서 포화한다(58mm 단면 → 약 0.32) → 실현 상한 약 1.4.
    lifting 15 + goal 16 + settle 15 대비 충분히 작아야 파지·이송이 우선이다."""
    realistic_max = 0.32 * P.CLOSURE_WHEN_ENCLOSED_WEIGHT
    assert realistic_max <= 0.10 * (15.0 + 16.0 + P.SETTLE_REWARD_WEIGHT)


def test_lift_ramp_is_gated_by_grasp_ok_not_enclose():
    """★★08.24 이 계약을 **뒤집었다.** 전에는 램프에 `enclose` 를 곱하는 것을 고정했는데,
    `enclose` 는 판별력이 없다는 것이 fab_test11 로 드러났다:

        정책               along    lateral   enclose
        test17(파지 성공)   13.5 mm   21.7 mm   0.804
        fab_test11(옆 대기) 12.0 mm  **85.5 mm** **0.824**   ← 성공과 구분 불가

    턱이 벌어져 있으면 컵 축에서 8.5 cm 떨어져도 "축이 턱 사이를 지난다"가 성립한다.
    fab_test11 은 그 상태로 4000 epoch 을 돌며 컵을 0.2 mm 도 못 들었다.
    → 게이트를 `grasp_ok`(lateral 을 직접 보는 하드 술어)로 바꾼다.
    """
    rew = _src("grasp_left_rewards.py")
    body = rew[rew.index("def _held("):rew.index("def held_with_good_pose(")]
    # ★fab_test33: 이진 `grasp_ok` → 연속 `grasp_quality`(같은 lateral/along 측정).
    #   판별력 요구는 그대로다 — 위 실측 대비는 `test_grasp_gate_is_continuous_...` 가 검산한다.
    assert "grasp_quality(env" in body, "리프트 게이트가 lateral 기반 술어를 쓰지 않는다"
    assert "_enclose(env" not in body, "판별력 없는 enclose 게이트가 되살아났다"
    assert "lifted * held * (near & upright).float()" in body


def test_grasp_ok_separates_measured_success_from_failure():
    """★게이트 문턱은 **실측 분포를 가르는 값**이어야 한다.

    성공 lateral 20.2~21.7 mm · 실패 78.6~85.5 mm — 4배 격차, 겹침 없음.
    along 은 성공 12.2~13.5 · 실패 12.0~27.8 로 **단독 판별력이 없다**(겹침).
    """
    assert P.GRASP_GATE_LATERAL_OK > 0.022, "성공 실측(21.7mm)을 못 통과시킨다"
    assert P.GRASP_GATE_LATERAL_OK < 0.078, "실패 실측(78.6mm)을 통과시킨다"
    assert P.GRASP_GATE_RELEASE_LAT > P.GRASP_GATE_LATERAL_OK, (
        "해제 문턱이 진입 문턱보다 작으면 히스테리시스가 아니라 채터링이 된다"
    )
    rew = _src("grasp_left_rewards.py")
    body = rew[rew.index("def grasp_ok("):]
    body = body[: body.index("\ndef ")]
    assert "lateral < lat_ok" in body, "lateral 조건이 없다 — enclose 만으로는 못 가른다"
    # ★clamp 전 축 좌표를 써야 한다. clamp 된 값은 대역 밖이어도 경계로 접혀 항상 참이 된다.
    assert "axis_t > P.CUP_GRASP_BAND_AXIS[0]" in body


def test_gripper_action_is_hard_gated_open_before_approach():
    """★★접근 성공 전에는 그리퍼를 **강제로 연다**.

    근거: Fabrics 가 우연한 리프트를 없앴다(적용 관절 목표 변화 test17 2.79 rad/s vs
    fab_test5 0.38 rad/s → 컵 상승 +138 vs +17 mm). 정책이 "열기·위치·닫기·들기" 연접을
    우연히 맞춰야 하는 문제를, 앞 두 칸을 코드가 강제해 없앤다.
    실패 이력: fab_test1 주먹(개도 3.1mm) · fab_test11 옆에서 좁게 닫음(16.3mm) —
    강제 개방은 둘 다 **구조적으로 불가능**하게 만든다.
    """
    act = _src("grasp_left_actions.py")
    assert "class GatedBinaryJointPositionAction(BinaryJointPositionAction)" in act
    body = act[act.index("def process_actions", act.index("class GatedBinary")):]
    body = body[: body.index("def reset")]
    assert "self._phase | ok" in body, "래치가 없다 — 닫는 순간 문턱을 넘나들면 컵을 놓는다"
    assert "lateral < self.cfg.release_lateral" in body, "히스테리시스 해제가 없다"
    assert "self._open_command" in body, "phase 0 에서 강제 개방하지 않는다"
    # ★판정은 process_actions 에서만. apply_actions 는 decimation 만큼 불린다.
    assert "grasp_ok" not in act[act.index("def reset", act.index("class GatedBinary")):]
    rst = act[act.index("def reset", act.index("class GatedBinary")):]
    assert "_phase" in rst, "reset 에서 래치를 지우지 않는다 — 리셋 오염 네 번째"
    cfg = _src("grasp_left_env_cfg.py")
    assert "GatedBinaryJointPositionActionCfg" in cfg


def test_all_gated_terms_share_one_jaw_cfg_instance():
    """★SceneEntityCfg 는 매니저가 제자리 변경하는 가변 객체다 — term 마다 새 인스턴스."""
    cfg = _src("grasp_left_env_cfg.py")
    assert cfg.count("jaw_cfg") >= 3, "jaw_cfg 배선이 빠진 term 이 있다"
    assert cfg.count('SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES))') >= 4, (
        "SceneEntityCfg 를 term 간 공유하면 매니저가 제자리 변경해 두 번째 term 이 죽는다"
    )


def test_left_gripper_colliders_stay_decomposition():
    """★★파지 도구인 좌 그리퍼 3개는 **convexDecomposition** 이어야 한다.

    agnostic 트랙의 `_armhull` 은 좌 그리퍼까지 hull 로 바꾼다 — 그쪽은 우손으로 잡는
    태스크라 무해하지만 우리에겐 파지 부위다. hull 이 되면 개구(84.5 mm 실측)와
    파지 대역(10~85 mm)이 무효가 된다.

    physics 레이어가 바이너리(usdc)라 직접 못 읽는다 → 자산 옆 매니페스트를 본다.
    갱신: ./isaaclab.sh -p scripts/assets_tools/write_collider_manifest.py <asset>
    """
    import json
    assert P.ROBOT_ASSET_DIR.endswith("_lgrip"), (
        f"자산이 {P.ROBOT_ASSET_DIR} 다 — _armhull 은 좌 그리퍼를 hull 로 바꾼다"
    )
    root = _HDGP / "assets" / "robot" / P.ROBOT_ASSET_DIR
    assert (root / "openarm_tesollo_sensor_rl.usd").exists(), f"자산이 없다: {root}"
    man = root / "collider_manifest.json"
    assert man.is_file(), f"매니페스트가 없다 — write_collider_manifest.py 를 돌릴 것: {man}"
    links = json.loads(man.read_text())["links"]
    for link in P.GRIPPER_COLLIDER_LINKS:
        assert links.get(link) == "convexDecomposition", (
            f"{link} 가 {links.get(link)} 다 — 파지 기하가 바뀐다"
        )
    # 나머지는 hull 이어야 속도 이득이 난다(우손 27개가 가장 큰 몫).
    hull = sum(1 for v in links.values() if v == "convexHull")
    assert hull >= 40, f"hull 이 {hull}개뿐 — 속도 이득이 안 난다"


def test_settle_velocity_thresholds_match_the_measured_regime():
    """★임계는 **현재 실측 규모**에서 정한다 — test10/test11 을 죽인 규칙.

    fab_test7 결정론 실측 0.193 m/s · 1.473 rad/s. 옛 임계(0.40/3.00)는 test8 시절
    0.444 m/s 기준이라 이미 품질 0.55 로 포화 — "더 멈춰라"는 압력이 없다.
    ⚠ 세 항(near·lin·ang)을 동시에 조이면 곱이 죽는다. 위치 std 는 그대로 둔다.
    """
    import math
    def q(v, std):
        return 1.0 - math.tanh(v / std)
    now = 0.5 * q(0.193, P.SETTLE_LIN_VEL_STD) + 0.5 * q(1.473, P.SETTLE_ANG_VEL_STD)
    goal = 0.5 * q(0.05, P.SETTLE_LIN_VEL_STD) + 0.5 * q(0.30, P.SETTLE_ANG_VEL_STD)
    assert now > 0.05, f"현재 상태 품질 {now:.3f} — 너무 낮으면 항이 죽는다"
    assert goal / now >= 3.0, f"현재→목표 개선 폭 {goal / now:.1f}배 — 압력이 부족하다"
    assert math.isclose(P.SETTLE_POS_STD, 0.15), "위치 std 를 같이 조이면 곱이 죽는다"


def test_gravity_droop_compensation_is_wired_and_bounded():
    """★★Fabrics 는 순수 기구학이라 중력을 모른다 — PD 가 중력 부하만큼 뒤처진다.

    실측: 좌팔 PD 추종오차 32.9 mrad → 파지 자세 TCP 40~48 mm 처짐.
    G4 폐루프로 47~117 mm 앞당기면 TCP 오차 0.2 mm 로 수렴한다.
    지금까지는 **정책이 그 선행량을 스스로 학습**했고 그만큼이 목표 정확도에서 빠졌다.

    관절공간 보상이라 프레임 정합 문제가 없다. 세 가지가 반드시 함께 있어야 한다:
      · 상한 = effort/강성 (그 이상은 토크 포화 → windup)
      · 저역통과 (순간 오차를 쓰면 가속 구간의 속도 지연까지 보상해 팔이 과격해진다)
      · reset 시 초기화 (직전 에피소드의 처짐이 남으면 첫 스텝이 튄다)
    """
    src = _src("grasp_left_fabric_action.py")
    assert "self._droop" in src, "처짐 보상이 배선되지 않았다"
    assert "P.GRAVITY_COMP_ENABLED" in src
    # 상한 = ARM_IK_MAX_TRACKING_ERROR (effort/강성)
    assert "_droop_limit" in src and "ARM_IK_MAX_TRACKING_ERROR" in src, "상한이 없다"
    assert "clamp(-self._droop_limit, self._droop_limit)" in src, "clamp 가 빠졌다"
    # 저역통과 + env step 당 1회 갱신 (apply_actions 는 decimation 번 불린다)
    proc = src[src.index("def process_actions"):src.index("def apply_actions")]
    appl = src[src.index("def apply_actions"):src.index("def reset")]
    assert "GRAVITY_COMP_GAIN" in proc, "적분 갱신이 process_actions 에 없다"
    assert "GRAVITY_COMP_GAIN" not in appl, (
        "apply_actions 에서 갱신하면 decimation 배로 빨리 수렴한다"
    )
    rst = src[src.index("def reset"):]
    assert "self._droop[env_ids] = 0.0" in rst, "reset 초기화가 빠졌다"
    # 상한값 자체 — effort/강성 관계가 유지되는가
    assert P.ARM_IK_MAX_TRACKING_ERROR["l_aj_[5-7]"] == 7.0 / P.ARM_IK_STIFFNESS
    assert 0.0 < P.GRAVITY_COMP_GAIN <= 0.2, "적분 이득이 너무 크면 과도 구간에서 진동한다"
    # ★저역통과가 아니라 적분이어야 한다 — 저역통과는 정확히 절반만 상쇄한다(실측)
    assert "self._droop + P.GRAVITY_COMP_GAIN * err" in src, "적분 형태가 아니다"


def test_cup_axis_point_is_clamped_to_the_graspable_band():
    """★★보상 구멍 — 컵 축은 **무한 직선**이라 컵 위 허공에서 감싸도 만점이 나온다.

    fab_test10 실측(epoch 500): 턱 중점 축방향 높이 **+157.6 mm**(컵 원점 기준,
    컵 상단은 +83 mm) = 75 mm 허공. 그 상태로 cup_between_jaws **2.15/3.0** 과 closure 를
    받으면서 **컵은 0.1 mm 도 움직이지 않았다**(상승 +0.9 mm).
    학습 지표로도 between 2.1 인데 reach 0.064(TCP 가 파지점에서 171 mm)로 모순이었다.
    ⚠ 이 구멍이 fab_test9·10 의 "lift 가 0" 을 설명한다 — 리미터 탓이라던 판정이 틀렸다.

    → 최근접점을 잡을 수 있는 높이 대역으로 clamp. 세 항(between·closure·lift 게이트)이
      모두 `_jaw_frame` 을 쓰므로 한 곳만 고치면 전부 닫힌다.
    """
    rew = _src("grasp_left_rewards.py")
    body = rew[rew.index("def _jaw_frame("):rew.index("def _enclose(")]
    assert "clamp(" in body and "CUP_GRASP_BAND_AXIS" in body, (
        "컵 축 최근접점이 clamp 되지 않았다 — 허공에서 감싸도 만점이 된다"
    )
    lo, hi = P.CUP_GRASP_BAND_AXIS
    # 대역은 컵 몸통 안이어야 한다(원점 기준 바닥 −92.09 mm ~ 상단 +82.91 mm)
    assert -P.CUP_BOTTOM_TO_ORIGIN <= lo < hi <= 0.0, (
        f"대역 ({lo:.5f}, {hi:.5f}) 이 컵 몸통 밖이거나 뒤집혔다"
    )
    # 실측 실패 지점(+157.6 mm)이 대역 밖으로 확실히 걸러져야 한다
    assert hi < 0.1576, "실측된 허공 straddle 높이가 대역 안이다 — 구멍이 그대로다"
    # GRASP_HEIGHT_BAND 에서 파생돼야 한다(리터럴 금지)
    assert abs(lo - (P.GRASP_HEIGHT_BAND[0] - P.CUP_BOTTOM_TO_ORIGIN)) < 1e-9
    assert abs(hi - (P.GRASP_HEIGHT_BAND[1] - P.CUP_BOTTOM_TO_ORIGIN)) < 1e-9


# ── fab_test13: 리미터 ON + dwell 보너스 + σ 재팽창 억제 ─────────────────────


def test_no_rate_limiter_reference_alignment():
    """★★fab_test21: rate limiter 는 **꺼져 있어야** 한다 — DEXTRAH 원본에 없다.

    원본 `compute_actions` 는 절대 palm pose 를 박스로 스케일·clamp 할 뿐이고 변화율
    상한은 fabric 이 정한다. 우리가 fab_test13 에 리미터를 붙인 이유("지령이 팔보다
    빠르다")의 굼뜸은 원본 대비 자초한 것이었다(08.25 대조):
        · fabric 시간이 원본의 60%   · fabric damping 이 원본 ADR 하드끝(20 vs 10)
        · 속도 피드포워드 0(원본 1.0)
    ⚠ 순서 계약이다 — 셋을 원본으로 되돌린 **뒤에만** 리미터를 뗄 수 있다. 원인을 안
      고친 채 증상 억제기만 떼면 자매 트랙 leash 사고가 반복된다(palm_err 51~65 → 90mm).
    """
    # ★★fab_test25 뒤집음. 원본에 리미터가 없다는 사실은 그대로지만, t24 실측이
    #   리미터의 **두 번째 기능**(접촉 속도 제한)을 드러냈다 — 없이 돌리니 초기 낙하가
    #   역대 최고(0.914)로 뛰고 정책이 컵 회피를 학습했다(reach 0.033 = t16 의 1/17).
    #   보상 구조는 t16 과 동일했으므로 원인은 제어다. 이 계약은 "리미터가 없어야 한다"가
    #   아니라 "**떼려면 셋을 먼저 원본으로 되돌려야 한다**"는 순서 계약으로 남긴다.
    assert P.PALM_CMD_RATE_LIMIT_ENABLED is True, (
        "리미터가 꺼졌다 — t24 에서 컵 회피 국소최적을 낳았다"
    )
    assert P.PALM_CMD_RATE_LIMIT == 0.10, "지령 변화율 상한이 지정값(100 mm/step)이 아니다"

    # 셋이 전부 원본값이어야 리미터 제거가 성립한다
    assert P.FABRIC_VEL_FF_SCALE == 1.0, "속도 피드포워드가 원본값(1.0)이 아니다"
    assert P.FABRIC_DAMPING_GAIN == P.ADR_FABRIC_DAMPING_GAIN[0] == 10.0, (
        f"fabric damping {P.FABRIC_DAMPING_GAIN} — 원본 ADR 레벨 0 은 10 이다"
    )
    src = _src("grasp_left_fabric_action.py")
    assert "self._fabric_dt = float(env.step_dt)" in src, (
        "fabric 시간이 원본 비율이 아니다 — 원본은 fabrics_dt = 정책 스텝, "
        "decimation 회 적분 → 벽시계의 2배속"
    )
    proc = src[src.index("def process_actions"):src.index("def apply_actions")]
    assert "PALM_CMD_RATE_LIMIT" in proc, "리미터가 배선되지 않았다(상수만 있고 코드가 없다)"
    # ⚠ 리셋 직후에는 걸면 안 된다 — 홈에서 첫 지령까지는 "변화"가 아니라 초기화다.
    assert "_cmd_primed" in proc, "리셋 직후 예외가 없다 — 리셋마다 팔이 끌려간다"


def test_arm_pd_gains_match_reference_taper():
    """★★fab_test21: 팔 PD 는 DEXTRAH kuka 원본의 **테이퍼**여야 한다.

    원본 iiwa7: j1-4 300/45 · j5 100/20 · j6 50/15 · j7 25/15.
    구값 균일 400/80 은 DEXTRAH open_tesollo 판본에서 **학습에 쓰지 않는 반대쪽 팔을
    잠그는** 게인과 같은 값이다 — 구동축 게인이 아니었다.
    """
    assert P.ARM_FABRIC_STIFFNESS == {
        "l_aj_[1-4]": 300.0, "l_aj_5": 100.0, "l_aj_6": 50.0, "l_aj_7": 25.0}
    assert P.ARM_FABRIC_DAMPING == {
        "l_aj_[1-4]": 45.0, "l_aj_5": 20.0, "l_aj_6": 15.0, "l_aj_7": 15.0}
    # kd/kp 비가 근위→원위로 커져야 한다(원본 0.15 → 0.60)
    ratios = [P.ARM_FABRIC_DAMPING[k] / P.ARM_FABRIC_STIFFNESS[k]
              for k in ("l_aj_[1-4]", "l_aj_5", "l_aj_6", "l_aj_7")]
    assert ratios == sorted(ratios), f"kd/kp 가 단조 증가하지 않는다 {ratios}"

    fab_src = _src("grasp_left_fab_env_cfg.py")
    assert "P.ARM_FABRIC_STIFFNESS" in fab_src and "P.ARM_IK_STIFFNESS" not in fab_src, (
        "fab 변형이 아직 IK 용 균일 게인을 쓴다"
    )


def test_reference_adr_items_are_present_and_start_easy():
    """★★원본 ADR 항목을 **빠뜨리거나 하드 끝값으로 고정**하지 않았는가.

    이 트랙은 같은 실수를 세 번 했다:
        속도 피드포워드 (1.0 → 0.0) 에서 **0.0** 을 고정값으로
        fabric damping  (10 → 20)   에서 **20**  을 고정값으로
        PD 게인 DR ×(0.5, 2.0)      을 **아예 누락**
    전부 "난이도를 올리는 요소를 과제 성립 전에 최대로" 라는 같은 유형이다.
    """
    for name, pair in (("vel_ff", P.ADR_VEL_FF_SCALE),
                       ("fabric_damping", P.ADR_FABRIC_DAMPING_GAIN)):
        assert pair[0] != pair[1], f"{name} ADR 범위가 퇴화했다 {pair}"
    # 게인/마찰 DR 은 레벨 0 이 중립이어야 한다
    assert P.ADR_ARM_GAIN_SCALE[0] == (1.0, 1.0), "PD 게인 DR 레벨 0 이 중립이 아니다"
    assert P.ADR_ARM_FRICTION[0] == (0.0, 0.0), "관절 마찰 DR 레벨 0 이 중립이 아니다"
    # 최고 레벨은 원본 범위와 같아야 한다
    assert P.ADR_ARM_GAIN_SCALE[1] == (0.5, 2.0)
    assert P.ADR_ARM_FRICTION[1] == (0.0, 5.0)

    fab_src = _src("grasp_left_fab_env_cfg.py")
    assert "self.events.arm_gains" in fab_src and "self.events.arm_friction" in fab_src
    cur = _src("grasp_left_curriculums.py")
    for k in ("ADR_FABRIC_DAMPING_GAIN", "ADR_ARM_GAIN_SCALE", "ADR_ARM_FRICTION"):
        assert k in cur, f"ADR 이 {k} 를 적용하지 않는다"


def test_dwell_bonus_thresholds_sit_between_measured_states():
    """★dwell 임계는 실측 규모에서(test10/test11 사고 재발 방지):
        fab_test12 final 실측 q ≈ 0.03 → 절대 못 받아야 하고
        중간 달성 상태(60mm·0.10m/s·0.8rad/s) q ≈ 0.29 → 받을 수 있어야 한다.
    임계가 이 사이를 벗어나면 죽은 항(너무 높음) 또는 공짜 항(너무 낮음)이 된다.
    """
    # 실측 상태를 q 로 환산 (object_settled_at_goal 과 같은 식, gate=1 가정)
    def q(dist, lin, ang):
        near = 1.0 - math.tanh(dist / P.SETTLE_POS_STD)
        still = 0.5 * (1.0 - math.tanh(lin / P.SETTLE_LIN_VEL_STD)) + 0.5 * (
            1.0 - math.tanh(ang / P.SETTLE_ANG_VEL_STD))
        return near * still

    q_current = q(0.162, 0.225, 1.864)   # fab_test12 final 결정론 실측
    q_mid = q(0.060, 0.100, 0.800)       # 리미터 켠 제어로 도달 가능한 중간 상태
    assert q_current < P.DWELL_Q_THRESH < q_mid, (
        f"임계 {P.DWELL_Q_THRESH} 가 실측 구간 ({q_current:.3f}, {q_mid:.3f}) 밖이다"
    )
    # 순회 통과(~5스텝)와 머무름을 가르려면 hold 는 충분히 길어야 한다
    assert P.DWELL_HOLD_STEPS >= 10
    # 보너스이지 주항이 아니다 — settle/lifting 을 넘으면 우선순위가 뒤집힌다
    assert P.DWELL_REWARD_WEIGHT <= P.SETTLE_REWARD_WEIGHT


def test_dwell_term_is_wired_with_counter_reset_and_fresh_jaw_cfg():
    """카운터는 스텝 간 상태 — reset 누락(리셋 오염, 이 트랙에서 4회)과
    SceneEntityCfg 공유(가변 객체)를 소스에서 잡는다."""
    rew = _src("grasp_left_rewards.py")
    body = rew[rew.index("class DwellSettledAtGoal"):rew.index("def _jaw_frame(")]
    assert "def reset(" in body and "_count[env_ids] = 0" in body, "카운터 reset 이 없다"
    assert "object_settled_at_goal(" in body, "순간 품질은 settle 항과 같은 식이어야 한다"
    assert "clamp(" in body, "램프 없이 이진이면 절벽이 된다"

    cfg = _src("grasp_left_env_cfg.py")
    block = cfg[cfg.index("dwell_at_goal"):]
    block = block[:block.index("grasp_pose")]
    assert "DwellSettledAtGoal" in block and "DWELL_REWARD_WEIGHT" in block
    assert 'SceneEntityCfg("robot"' in block, "jaw_cfg 를 다른 term 과 공유하면 안 된다"


def test_gate_rate_diagnostic_actually_logs():
    """⚠ RewardManager 는 w·term·dt 를 로깅한다 — weight 0 이면 항상 0.0000 이 찍혀
    fab_test12 내내 1차 관전 지표(게이트 개방률)를 잃었다. 0 이 아니고 무해해야 한다."""
    cfg = _src("grasp_left_env_cfg.py")
    m = re.search(r"gate_rate = RewTerm\(func=rewards\.gripper_gate_rate, weight=([0-9.e-]+)", cfg)
    assert m, "gate_rate 진단 항이 없다"
    w = float(m.group(1))
    assert 0.0 < w <= 0.01, f"gate_rate weight {w} — 0 은 로깅 불가, 크면 학습 오염"


def test_agent_yaml_matches_kuka_reference():
    """★★fab_test22: 에이전트 설정은 **kuka 원본 그대로**여야 한다(사용자 지시).

    ⚠ 이 계약은 fab_test19 의 반대 계약을 뒤집은 것이다. 당시 계약은
      `bounds_loss_coef == 0.001` 을 요구했고 근거는 실측이었다 — 0.005 로 올렸더니
      t18 의 σ 가 ep700대 0.312 로 성공한 t16(0.457)보다 32% 낮았고 dwell 이 0 이 됐다.
      뒤집는 근거: **그 측정은 다른 학습기 위에서 났다.** 당시는 MLP [256,128,64] ·
      대칭 critic · gamma 0.99 · lr 1e-4 adaptive 였고, 지금은 LSTM 1024 + 비대칭
      critic + gamma 0.998 + lr 3e-4 linear 다. 탐색 동역학이 통째로 다르다.
      ★재학습에서 σ 가 다시 조기 고갈되면 이 항목만 0.001 로 되돌리고 사유를 여기 적을 것.
    """
    yaml_text = (_PKG / "config/agents/rl_games_ppo_fab_cfg.yaml").read_text(encoding="utf-8")
    # ★★fab_test31: **`gamma` 를 의도적으로 kuka 와 다르게 둔다(0.998 → 0.99).**
    #   근거는 실측이다. bisect 로 t16 env 위에서 최적화 하이퍼파라미터만 kuka 로 바꾸니
    #   `lifting` 이 4.03 → 0.000 이 됐고(B 그룹), 그 안에서 다시 B2(γ·horizon·batch)가
    #   범인으로 갈렸다. 그리고 γ 는 절단 보정 결함과 곱해진다 —
    #   γ^(에피소드 길이): γ0.99×250 = 0.08 vs γ0.998×600 = 0.30 (**3.7 배**).
    #   ⚠ 나머지 B2 항목(horizon 16 · mini_epochs 4 · minibatch 16384)은 **아직 kuka 값**이다.
    #     γ 만으로 회복되는지 먼저 보고, 안 되면 그때 하나씩 되돌린다.
    ref = {
        "tau": "0.95", "learning_rate": "3e-4",
        "kl_threshold": "0.013", "horizon_length": "16", "minibatch_size": "16384",
        "mini_epochs": "4", "critic_coef": "4", "e_clip": "0.2",
        "entropy_coef": "0.002", "bounds_loss_coef": "0.005", "grad_norm": "1.0",
        "clip_observations": "5.0", "clip_actions": "1.0", "seq_length": "16",
    }
    for k, v in ref.items():
        assert re.search(rf"{k}:\s*{re.escape(v)}\b", yaml_text), (
            f"{k} 가 kuka 원본값 {v} 이 아니다"
        )
    assert re.search(r"gamma:\s*0\.99\b", yaml_text), "gamma 는 0.99 여야 한다(위 주석)"
    assert re.search(r"value_bootstrap:\s*True", yaml_text), "절단 보정은 켜 둔다(kuka 는 False)"
    # ★★fab_test36: `lr_schedule` 과 `bound_loss_type` 은 **의도적으로 kuka 와 다르다.**
    #   근거는 `test_learner_is_actually_on` 의 실측(a_loss 42 배 · lr 11.6 배 · bounds 0).
    #   kuka 원본은 linear + `regularization`(오타) 인데, 후자는 rl_games 에서 아무 분기와도
    #   안 맞아 bounds loss 가 통째로 꺼진다 — 원본을 그대로 옮기면 그 결함까지 옮긴다.
    assert re.search(r"lr_schedule:\s*adaptive", yaml_text)
    assert re.search(r"schedule_type:\s*standard", yaml_text)
    assert re.search(r"bound_loss_type:\s*bound\b", yaml_text)
    assert re.search(r"zero_rnn_on_done:\s*True", yaml_text)

    # actor LSTM 1024 + critic LSTM 2048 (비대칭)
    assert re.search(r"rnn:\s*\n\s*name:\s*lstm\s*\n\s*units:\s*1024", yaml_text), \
        "actor LSTM 1024 가 없다"
    assert "central_value_config:" in yaml_text, "비대칭 critic(central value)이 없다"
    assert re.search(r"units:\s*2048", yaml_text), "critic LSTM 2048 이 없다"
    assert re.search(r"units:\s*\[512, 512\]", yaml_text), "actor MLP 가 [512,512] 가 아니다"
    assert re.search(r"units:\s*\[1024, 512\]", yaml_text), "critic MLP 가 [1024,512] 가 아니다"

    # ADR 의 epoch→env step 환산은 이 horizon 과 같아야 한다
    m = re.search(r"horizon_length:\s*(\d+)", yaml_text)
    assert int(m.group(1)) == P.ADR_HORIZON_STEPS, (
        f"ADR_HORIZON_STEPS({P.ADR_HORIZON_STEPS}) != horizon_length({m.group(1)}) — "
        "ADR 확장 주기가 의도한 epoch 수와 어긋난다"
    )


def test_reference_obs_includes_fabric_state_and_asymmetric_critic():
    """★★fab_test22: 정책이 **fabric 내부 상태**를 봐야 한다 + critic 은 비대칭이어야 한다.

    원본 kuka 는 `fabric_q · fabric_qd · fabric_qdd` 를 actor·critic 양쪽에 넣는다.
    우리는 셋 다 없었고, 그래서 정책은 자기가 낸 지령이 실현됐는지를 관측할 수단이
    아예 없었다(08.25 실측: 이송 중 |cmd−TCP| 90 mm).
    """
    fab_src = _src("grasp_left_fab_env_cfg.py")
    for t in ("fabric_q", "fabric_qd"):
        assert f"self.observations.policy.{t} = ObsTerm" in fab_src, f"policy obs 에 {t} 누락"
    # ★★fab_test31 제거: `fabric_qdd`(가속도) · `hand_vel` · critic `arm_torque`.
    #   `clip_observations 5.0` 아래에서 실측 포화율이 각각 39.2% · — · 41.1% 였다.
    #   단위가 rad/s² · N·m 라 애초에 clip 5 짜리 obs 에 들어갈 수 없는 값이었고,
    #   잘린 차원은 정보가 아니라 상수에 가까운 가짜 신호다.
    assert "policy.fabric_qdd = ObsTerm" not in fab_src, "가속도가 obs 에 남아 있다 — 39% 포화"
    assert "policy.hand_vel = ObsTerm" not in fab_src, "hand_vel 이 남아 있다 — 중복"
    assert "func=obs_mdp.arm_applied_torque" not in fab_src, "arm_torque 가 남아 있다 — 41% 포화"
    assert "self.observations.critic = _CriticCfg()" in fab_src, "비대칭 critic 그룹이 없다"
    # critic 은 특권 정보를 봐야 한다(실측 토크·접촉력·물체 속도)
    for t in ("finger_contact_forces", "object_lin_ang_vel"):
        assert t in fab_src, f"critic 특권 관측 {t} 누락"
    # critic 은 노이즈를 받지 않는다
    assert "self.enable_corruption = False   # critic" in fab_src
    # policy 는 노이즈를 받는다 — 단 `ObsTerm.noise` 가 아니라 전용 모듈이 건다.
    # ★원본은 폭을 env 마다 다시 뽑고 에피소드 고정 bias 를 얹는데 Unoise 는 둘 다
    #   표현 못 한다. 경로가 둘이면 "노이즈를 껐는데 왜 흔들리지"를 겪는다 — 하나로 둔다.
    assert "self.observations.policy.enable_corruption = False" in fab_src
    for t in ("joint_pos_noisy", "joint_vel_noisy", "object_position_noisy"):
        assert t in fab_src, f"policy 노이즈 관측 {t} 누락"
    # ★원본 policy obs 에 있던 손 직교 상태·물체 자세
    # ★fab_test31: `hand_body_vel` 은 뺐다(joint_vel·fabric_qd 중복, 12 차원).
    for t in ("hand_body_pos", "object_rotation"):
        assert t in fab_src, f"원본 policy obs 항목 {t} 누락"


def test_adr_schedule_matches_kuka_reference():
    """★★fab_test22: ADR 규칙을 kuka 원본과 맞춘다.

    원본(`dextrah_kuka_allegro_env.py:1119`):
        step_since_last_dr_change >= min_steps_for_dr_change   (= 5 × 에피소드 스텝 = 3000)
        AND in_success_region.float().mean() > success_for_adr (= 0.4)
        num_adr_increments = 50 · starting_adr_increments = 0
    우리는 5 레벨 · 2400 스텝 · **dwell 보상 크기 EMA ≥ 1.0** 이었다. 셋 다 달랐고,
    특히 판정 지표가 '크기' 라 보상 weight 를 손질할 때마다 임계 의미가 흔들렸다.
    """
    assert P.ADR_LEVELS == 50, f"레벨 수 {P.ADR_LEVELS} — 원본 num_adr_increments 는 50"
    assert abs(P.ADR_TRIGGER - 0.4) < 1e-9, "원본 success_for_adr 는 0.4(비율)"
    # 에피소드 스텝 = episode_length_s / (decimation × sim_dt). 60 Hz 이므로 5.0 s → 300.
    # ★fab_test37: 값을 박지 않는다 — 에피소드 길이는 로그 근거로 5.0 s 이고
    #   (`test_episode_is_short_enough_that_engaging_beats_idling`), 여기서 검사할 것은
    #   "확장 간격이 **5 × 에피소드**인가"라는 kuka 규약뿐이다.
    episode_steps = int(round(P.EPISODE_LENGTH_S / (2 * (1.0 / 120.0))))
    assert episode_steps == 300
    assert P.ADR_MIN_STEPS_BETWEEN == 5 * episode_steps, (
        f"확장 간격 {P.ADR_MIN_STEPS_BETWEEN} — 원본은 5 × 에피소드 스텝 = {5 * episode_steps}"
    )
    # 판정이 '비율' 이어야 한다 — 크기 EMA 면 weight 에 따라 임계가 흔들린다
    cur = _src("grasp_left_curriculums.py")
    assert "> 0.0).float().mean()" in cur, "ADR 판정이 성공 env 비율이 아니다"


def test_reference_physics_and_solver_settings():
    """★★fab_test22: 씬 마찰·솔버·강체 속성이 kuka 원본값이어야 한다."""
    assert P.SCENE_STATIC_FRICTION == 1.0 and P.SCENE_DYNAMIC_FRICTION == 1.0
    assert P.ARTICULATION_SOLVER_POSITION_ITER == 8
    assert P.ARTICULATION_SOLVER_VELOCITY_ITER == 0
    assert P.RIGID_MAX_DEPENETRATION_VELOCITY == 1000.0
    assert P.PHYSX_BOUNCE_THRESHOLD_VELOCITY == 0.2
    assert P.PHYSX_GPU_MAX_RIGID_PATCH_COUNT == 4 * 5 * 2 ** 15
    # ★★fab_test37: 에피소드 길이는 **의도적으로 kuka(10.0)와 다르다.**
    #   근거는 `test_episode_is_short_enough_that_engaging_beats_idling` 의 로그 전수.
    assert P.EPISODE_LENGTH_S == 5.0
    # 외란 주기 — 원본 wrench_trigger_every = 1 초
    assert P.ADR_DISTURB_INTERVAL_S == (1.0, 1.0)
    fab_src = _src("grasp_left_fab_env_cfg.py")
    for k in ("SCENE_STATIC_FRICTION", "ARTICULATION_SOLVER_POSITION_ITER",
              "RIGID_MAX_DEPENETRATION_VELOCITY", "PHYSX_BOUNCE_THRESHOLD_VELOCITY"):
        assert k in fab_src, f"{k} 가 배선되지 않았다"


def test_workspace_exit_termination_and_reset_noise():
    """★★fab_test22: 원본 `_get_dones` 의 작업공간 이탈 종료 + `robot_spawn` 리셋 노이즈."""
    fab_src = _src("grasp_left_fab_env_cfg.py")
    assert "self.terminations.object_out_of_workspace = DoneTerm(" in fab_src
    assert "self.events.arm_spawn_noise = EventTermCfg(" in fab_src
    # 작업공간 박스는 스폰과 목표를 모두 덮어야 한다 — 아니면 정상 동작이 종료된다
    for lo, hi, c, r, name in (
        (*P.OBJECT_WORKSPACE_X, P.CUP_SPAWN_X_CENTER, P.CUP_SPAWN_X_RANGE, "spawn x"),
        (*P.OBJECT_WORKSPACE_Y, P.CUP_SPAWN_Y_CENTER, P.CUP_SPAWN_Y_RANGE, "spawn y"),
    ):
        assert lo < c - r and c + r < hi, f"{name} 이 작업공간 박스를 벗어난다"
    gx, gy = P.GOAL_POINT[0], P.GOAL_POINT[1]
    assert P.OBJECT_WORKSPACE_X[0] < gx - P.GOAL_JITTER[0] and gx + P.GOAL_JITTER[0] < P.OBJECT_WORKSPACE_X[1]
    assert P.OBJECT_WORKSPACE_Y[0] < gy - P.GOAL_JITTER[1] and gy + P.GOAL_JITTER[1] < P.OBJECT_WORKSPACE_Y[1]


def test_action_jerk_is_not_wired_anywhere():
    """★★jerk(2차 차분) 항은 **기각**됐다 — 어느 변형에도 배선하면 안 된다.

    fab_test19 층 분해 실측이 기각 근거다(결정론, 32 env, 목표 10 cm 이내 분리 계측):
        ① 정책 raw 액션    |Δ|0.217 |Δ²|0.205  방향반전 **18.8%**
        ② 리미터 통과 지령 |Δ|11.5mm            방향반전 8.4%
        ③ fabric 관절 목표 |Δ|15.5mrad          방향반전 **0.0%**
        ④ 실제 팔 관절     |Δ|17.1mrad          방향반전 **0.0%**
    2차 성분은 ③ 에서 완전히 사라진다 — 팔은 떨지 않으므로 벌할 것이 없다.
    실제로 물린 벌금은 |Δ²a| 가 큰 접근·이송 구간이었고, fab_test19 는 래치(ep325) 직후
    −2.17(그 시점 정밀신호 합의 88%)을 물고 dwell 1.02 → 0.005 로 무너졌다.

    ⚠ 이 계약은 "게이트를 달았으니 괜찮다"는 재시도를 막는다. fab_test16 은 게이트가
      있었는데도 살아남은 게 아니라, 래치 시점(ep988)에 raw jerk 가 이미 작아
      페널티가 −0.33 에 그쳤을 뿐이다. 게이트는 **시점**만 정하고 **크기**는 못 정한다.
    """
    for fname in ("grasp_left_fab_env_cfg.py", "grasp_left_env_cfg.py"):
        src = _src(fname)
        assert "self.rewards.action_jerk = RewTerm(" not in src, (
            f"{fname} 에 jerk 항이 배선됐다 — 기각된 경로다(fab_test19 실측)"
        )
        assert "func=rewards.ActionJerkL2" not in src, f"{fname} 에 ActionJerkL2 배선"
        assert "self.curriculum.jerk_after_dwell" not in src


def test_palm_command_rate_penalty_is_first_order_and_self_gated():
    """★★jerk 를 대체하는 항의 계약 — **1차**이고 **게이트가 항 안에** 있어야 한다.

    fab_test14/19 실패의 공통 구조는 "억제 항이 과제 성립 전 구간에 걸린 것"이었다.
    커리큘럼 래치는 그 시점을 늦출 뿐 크기를 정하지 못했다(위 테스트 참조). 이 항은
    게이트(들고 + 목표 근접)를 항 자체에 곱해, 접근·이송 구간에서 **구조적으로 0** 이다.

    보이는 진동이 1차라는 근거: dwell 구간 지령 배회 1.16~5.2 mm/step(초당 60~260 mm).
    """
    fab_src = _src("grasp_left_fab_env_cfg.py")
    assert "self.rewards.palm_cmd_rate = RewTerm(" in fab_src, "대체 항이 배선되지 않았다"
    assert "func=rewards.palm_command_rate_at_goal" in fab_src
    block = fab_src[fab_src.index("self.rewards.palm_cmd_rate = RewTerm("):]
    block = block[:block.index("\n        )")]
    # 1) 게이트가 항 안에 있어야 한다 — 목표·파지 판정 인자가 전달돼야 성립한다
    for key in ('"command_name"', '"jaw_cfg"', '"std"', '"minimal_height"'):
        assert key in block, f"게이트 인자 {key} 누락 — 게이트 없는 억제 항은 t14/t19 재발"
    # 2) 리미터 상한으로 정규화해야 weight 가 상금 대비로 검산 가능하다
    assert '"rate_limit": P.PALM_CMD_RATE_REF' in block

    # 3) 억제가 상금을 넘으면 안 된다 — t14/t19 는 정확히 이걸 넘겨서 죽었다.
    #    항이 0~1 로 정규화돼 있으므로 최악의 경우 벌금 = |weight| 이고,
    #    그 순간 함께 들어오는 상금은 settle(15) + dwell(10) 이다.
    w = abs(P.PALM_CMD_RATE_PENALTY_WEIGHT)
    assert w > 0.0, "weight 가 0 이면 항이 없는 것과 같다"
    assert w <= 0.5 * (P.SETTLE_REWARD_WEIGHT + P.DWELL_REWARD_WEIGHT), (
        f"|weight| {w} 가 상금(settle {P.SETTLE_REWARD_WEIGHT} + dwell "
        f"{P.DWELL_REWARD_WEIGHT})의 절반을 넘는다 — fab_test14/19 재발 조건"
    )
    assert P.PALM_CMD_RATE_PENALTY_WEIGHT < 0.0, "억제 항인데 weight 가 음수가 아니다"

    # 4) 관절공간 변형에는 palm 지령 자체가 없다 — 새어 들어가면 안 된다
    assert "self.rewards.palm_cmd_rate" not in _src("grasp_left_env_cfg.py")

    # 5) 액션항이 "적용된" 지령 이동량을 노출해야 한다(raw 액션이 아니라 리미터 통과 후)
    act_src = _src("grasp_left_fabric_action.py")
    assert "def cmd_step_norm" in act_src, "지령 이동량 노출 프로퍼티가 없다"
    assert "self._cmd_step_norm[env_ids] = 0.0" in act_src, (
        "리셋에서 초기화하지 않으면 에피소드 첫 스텝이 텔레포트 차분으로 벌을 받는다"
    )


def test_settle_and_gate_share_one_ruler():
    """★같은 판정을 두 함수가 각자 다시 짜면 조용히 어긋난다 — 이 트랙에서 네 번 당했다.

    `palm_command_rate_at_goal` 의 게이트는 `object_settled_at_goal` 과 **같은**
    `held_and_near_goal` 을 써야 한다(정지 정도만 뺀 값).
    """
    r = _src("grasp_left_rewards.py")
    assert "def held_and_near_goal(" in r
    gate_block = r[r.index("def palm_command_rate_at_goal("):]
    assert "held_and_near_goal(" in gate_block, "게이트가 공용 함수를 쓰지 않는다"


def test_grasp_pose_pressure_raised_via_curve_not_weight():
    """★★fab_test14: 접근축 pitch↔컵기울기 거의 1:1 실측(19.9→20.3·9.8→13.0·20~21→21.2~21.3)
    근거로 압력을 올리려 했으나, weight 5.0→8.0 은 계약 테스트(test_lift_contract.py 의 1/3
    상한)가 막았다 — `grasp_pose`는 `settled_at_goal`(w15, 동일 게이트)과 달리 목표 근접을
    요구하지 않아, weight 를 올리면 "아무 데서나 똑바로 들고 서 있기"가 매력적인 국소최적이
    된다. **weight 는 그대로 두고 zero_at 곡선만 조인다** — 국소최적 위험은 weight(만점 상한)
    가 결정하므로 곡선 재척도는 그 위험을 키우지 않는다."""
    assert P.GRASP_POSE_REWARD_WEIGHT == 5.0, "weight 는 1/3 상한 그대로 — 곡선만 조인다"
    assert P.CUP_UPRIGHT_ZERO_AT_DEG == 27.0
    q_at_measured = (math.cos(math.radians(21.2)) - P.CUP_UPRIGHT_ZERO_AT_COS) / (
        1.0 - P.CUP_UPRIGHT_ZERO_AT_COS
    )
    assert 0.15 < q_at_measured < 0.45, (
        "실측 상태(21.2°)에서 죽지도(0.15 미만) 압력이 안 오르지도(0.45 초과) 않아야 한다"
    )


# ═══════════════════════════════════════════════════════════════════════════
# fab_test23 — kuka 3차 대조에서 남아 있던 항목
# ═══════════════════════════════════════════════════════════════════════════


def test_arm_gravity_is_disabled_like_reference():
    """★★원본 kuka 는 팔 중력을 끄고 중력 보상 항이 **없다**.

    우리는 중력을 켠 채 `_droop` 적분항으로 처짐을 흡수했는데, 그 적분항이 08.25 에
    속도 피드포워드 결손을 가려서 진단을 늦췄다(관절 오차를 자기가 먹어치웠다).
    `disable_gravity` 와 `GRAVITY_COMP_ENABLED` 는 **짝으로** 움직여야 한다 —
    한쪽만 되돌리면 팔이 무중력인데 보정까지 들어가 목표를 넘어 밀린다.
    """
    assert P.ROBOT_DISABLE_GRAVITY is True
    assert P.GRAVITY_COMP_ENABLED is False, "중력을 껐으면 보상 적분항도 꺼야 한다"
    fab_src = _src("grasp_left_fab_env_cfg.py")
    assert "self.scene.robot.spawn.rigid_props.disable_gravity = P.ROBOT_DISABLE_GRAVITY" in fab_src
    for attr in ("retain_accelerations", "sleep_threshold", "stabilization_threshold"):
        assert attr in fab_src, f"kuka articulation 속성 {attr} 미이식"
    assert "JointDrivePropertiesCfg" in fab_src and P.ROBOT_DRIVE_TYPE == "force"


def test_disturbance_wrench_matches_reference_distribution():
    """★★외란은 등방·질량비례·토크 포함이어야 하고, 손이 가까울 때만 걸려야 한다.

    IsaacLab 기본 `apply_external_force_torque` 는 `force_range=(0, F)` 를 성분마다
    균등 추출해 힘이 **항상 +x+y+z 한 팔분면**으로만 간다 — 정책이 외란 방향을 외운다.
    원본은 정규분포 방향을 정규화하고, 크기를 `mass × U(0, a_max)` 로 준다(질량 DR 과
    외란 DR 이 서로를 상쇄하지 않게).
    """
    src = _src("grasp_left_events.py")
    assert "def apply_object_wrench" in src
    assert "normalize" in src, "등방 방향이 아니다"
    assert "get_masses" in src, "외란 크기가 질량에 비례하지 않는다"
    assert "torsional_radius" in src, "토크 외란이 없다"
    assert "hand_dist_threshold" in src, "손–물체 거리 게이트가 없다"
    assert P.DISTURB_TORSIONAL_RADIUS == 0.01
    assert P.DISTURB_HAND_DIST_THRESHOLD == 0.3
    assert P.ADR_CUP_MAX_LINEAR_ACCEL == (0.0, 10.0)
    fab_src = _src("grasp_left_fab_env_cfg.py")
    assert "func=events.apply_object_wrench" in fab_src
    assert "mdp.apply_external_force_torque" not in fab_src, "기본 항이 남아 있다"


def test_observation_noise_has_per_env_width_and_episode_bias():
    """★★원본 관측 노이즈는 **두 층**이다 — per-step 노이즈 + per-episode bias.

    per-step 노이즈는 정책이 시간축으로 평균 내 지울 수 있지만 bias 는 못 지운다.
    실기에서 실제로 문제가 되는 쪽(엔코더 오프셋·extrinsics)은 bias 다. 우리에겐
    이 층이 통째로 없었다. 폭이 env 마다 다르다는 점도 원본의 핵심이다.
    """
    src = _src("grasp_left_obs_noise.py")
    assert "def resample" in src and "def corrupt" in src and "def set_level_value" in src
    assert "torch.rand(n, device=env.device)" in src, "폭이 env 별로 뽑히지 않는다"
    for pair in (P.ADR_OBS_OBJ_POS_BIAS, P.ADR_OBS_OBJ_ROT_BIAS,
                 P.ADR_OBS_JOINT_POS_BIAS, P.ADR_OBS_JOINT_VEL_BIAS):
        assert pair[0] == 0.0 and pair[1] > 0.0, "bias 는 0 에서 시작해 ADR 이 키운다"
    assert P.ADR_OBS_OBJ_POS_BIAS == (0.0, 0.02)
    assert P.ADR_OBS_OBJ_ROT_BIAS == (0.0, 0.08)
    assert P.ADR_OBS_JOINT_POS_BIAS == (0.0, 0.08)
    assert P.ADR_OBS_JOINT_VEL_BIAS == (0.0, 0.08)
    fab_src = _src("grasp_left_fab_env_cfg.py")
    assert "func=obs_noise.resample" in fab_src, "리셋마다 재추첨하는 이벤트가 없다"
    cur_src = _src("grasp_left_curriculums.py")
    assert "obs_noise.set_level_value" in cur_src, "ADR 이 노이즈 폭을 못 올린다"


def test_restitution_starts_at_reference_value():
    """★kuka EventCfg 는 반발계수를 **1.0** 에서 시작한다(우리는 0.0 이었다)."""
    assert P.ADR_ROBOT_RESTITUTION[0] == (1.0, 1.0)
    assert P.ADR_CUP_RESTITUTION[0] == (1.0, 1.0)
    assert P.ADR_ROBOT_RESTITUTION[1] == (0.8, 1.0)
    assert P.ADR_CUP_RESTITUTION[1] == (0.8, 1.0)


def test_adr_interpolation_uses_reference_denominator():
    """★kuka 보간 분모는 `num_increments`(=50)이고 카운터는 0..50 (51 단계)다.

    우리는 분모를 `levels-1`(=49), 최고 레벨을 49 로 썼다. 끝값은 같지만 단계 폭이
    2% 어긋난다 — 그리고 최고 난이도에 **도달하지 못한다**(frac 이 1.0 이 안 된다).
    """
    src = _src("grasp_left_curriculums.py")
    assert "self._level / n" in src, "분모가 num_increments 가 아니다"
    assert "self._level < levels" in src and "self._level < levels - 1" not in src
    assert P.ADR_LEVELS == 50


def test_command_and_actual_positions_are_logged():
    """★★fab_test26: 지령 위치와 실제 위치가 **둘 다** TB 에 찍혀야 한다.

    이 트랙은 둘을 나란히 본 적이 없어서 진단이 전부 사후 프로브였다 —
    추종오차 90 mm 도, t24 의 컵 회피 국소최적(reach 0.033 = t16 의 1/17)도
    런이 끝난 뒤 아카이브를 파서야 알았다.

    weight 0 은 IsaacLab 에서 log-only 이므로 총보상을 오염시키지 않는다.
    """
    src = _src("grasp_left_fab_env_cfg.py")
    for name in ("diag_cmd_", "diag_jaw_", "diag_cmd_jaw_gap",
                 "diag_cmd_step", "diag_jaw_cup_dist"):
        assert name in src, f"진단 항 {name} 이 배선되지 않았다"
    # 진단 항은 반드시 weight 0 이어야 한다 — 아니면 보상 지형을 바꾼다
    seg = src[src.index("진단 항 (weight 0"):src.index("도메인 랜덤화")]
    assert seg.count("weight=0.0") >= 5, "진단 항 중 weight 0 이 아닌 것이 있다"
    r = _src("grasp_left_rewards.py")
    for fn in ("def diag_palm_cmd", "def diag_jaw_pos", "def diag_cmd_jaw_gap",
               "def diag_cmd_step", "def diag_jaw_cup_dist"):
        assert fn in r, f"{fn} 이 없다"
    # ★두 값이 같은 프레임이어야 비교가 성립한다(지령은 base 기준, body 는 world 기준)
    assert "env.scene.env_origins" in r, "env 로컬 변환이 없다 — 두 값의 프레임이 다르다"


def test_log_only_reward_support_is_guarded():
    """★★fab_test27: weight 0 진단 항이 **조용히 0 이 되는 것**을 가드로 막는다.

    학습 호스트의 IsaacLab 이 upstream 이면 weight==0 항의 func 을 호출조차 하지 않아
    `diag_*` 가 전부 정확히 0.0000 으로 찍힌다. 에러가 없어서 몇 시간을 태운 뒤에야
    안다 — 이 저장소가 죽은 접촉센서로 이미 당한 서명이다.
    """
    src = _src("grasp_left_fab_env_cfg.py")
    assert "_require_log_only_reward_terms" in src, "log-only 가드가 없다"
    assert "_require_log_only_reward_terms()" in src, "가드가 임포트 시점에 실행되지 않는다"
    assert "raise RuntimeError" in src, "가드가 조용히 통과한다 — fail-loud 여야 한다"


def test_cmd_primed_is_actually_set():
    """★★fab_test29: `_cmd_primed` 가 **실제로 True 가 되는지**.

    이 플래그가 False 로 고정돼 있으면 두 가지가 조용히 죽는다:
      ① rate limiter 가 한 번도 안 걸린다(게이트가 이 플래그다)
      ② `cmd_step_norm` 이 항상 0 → `palm_cmd_rate` 보상이 상시 0
    둘 다 에러 없이 "0"으로만 나타나서, t27 에서 "리미터를 넣었는데 안 낫는다"는
    가짜 판정을 할 뻔했다 — 넣은 적이 없었다.
    """
    src = _src("grasp_left_fabric_action.py")
    proc = src[src.index("def process_actions"):src.index("def apply_actions")]
    assert "self._cmd_primed[:] = True" in proc, (
        "_cmd_primed 를 True 로 만드는 곳이 process_actions 에 없다 — "
        "리미터와 cmd_step_norm 이 조용히 죽는다"
    )
    # 리셋에서는 반드시 다시 False 여야 한다(홈→첫 지령은 '변화'가 아니다)
    assert "self._cmd_primed[env_ids] = False" in src, "리셋에서 초기화하지 않는다"


def test_truncation_does_not_contaminate_the_gradient():
    """★★fab_test31: 만기(절단)를 **진짜 종료로 학습하면 안 된다.**

    `value_bootstrap: False` 면 rl_games 의 절단 보정이 실행되지 않아
    (`a2c_common.py`: `if self.value_bootstrap and 'time_outs' in infos`),
    `time_out` 과 `object_dropping` 이 구분되지 않고 만기 시각의 가치가 0 이 된다.
    "에피소드가 끝났다"가 아니라 "세상이 끝났다"를 학습하는 것이다.

    ⚠ 왜곡 크기는 **γ^(에피소드 길이)** 다 — γ0.99×250 은 0.08 이라 거의 무해했지만
      γ0.998×600 은 0.30 으로 3.7 배다. 그래서 감마와 짝으로 고정한다.
    ⚠ 배선 전제: `is_finite_horizon=False` 여야 wrapper 가 `time_outs` 를 넣는다.
    """
    import yaml
    cfg = yaml.safe_load(
        (_PKG / "config/agents/rl_games_ppo_fab_cfg.yaml").read_text())["params"]["config"]
    assert cfg["value_bootstrap"] is True, "절단 보정이 꺼져 있다 — 만기가 죽음으로 학습된다"
    assert cfg["gamma"] == 0.99, (
        f"gamma {cfg['gamma']} — 0.998 은 600 스텝 에피소드에서 만기 왜곡을 3.7 배로 키운다"
    )
    # 종료 항 분류: 만기만 절단이어야 한다
    src = _src("grasp_left_fab_env_cfg.py")
    assert "time_out=True" not in src, (
        "이 파일에서 종료 항에 time_out=True 를 붙이면 안 된다 — "
        "물체 낙하·작업공간 이탈은 **진짜 종료**다"
    )


def test_contact_filter_points_at_the_rigid_body():
    """★★fab_test33: 접촉 필터는 **RigidBodyAPI 가 붙은 프림**을 가리켜야 한다.

    `/Object`(프림 루트)를 가리키면 PhysX 가 GPU 접촉 필터를 지원하지 못하고
    `force_matrix_w` 가 **정확히 0** 이 된다. 조용히 죽는 게 아니라 시뮬레이터가
    env 마다 경고를 찍는데(1024 env × 2 센서 = 2,048 줄) 로그가 길어 묻힌다.

        [omni.physx.tensors.plugin] GPU contact filter for collider
        '/World/envs/env_N/Object' is not supported

    컵 자산의 강체는 `/object_shaker_body/baseLink` 다.
    """
    src = _src("grasp_left_fab_env_cfg.py")
    assert "Object/{P.CUP_BODY_NAME}" in src or "P.CUP_BODY_NAME}\"]" in src, (
        "접촉 필터가 강체 프림(baseLink)을 가리키지 않는다 — force_matrix_w 가 0 이 된다"
    )
    assert 'filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]' not in src, (
        "필터가 프림 루트를 가리킨다 — GPU 접촉 필터 미지원"
    )


def test_grasp_gate_is_continuous_and_still_discriminates():
    """★★fab_test33: 파지 게이트가 **연속**이면서 판별력을 유지해야 한다.

    `_held()` 안의 이진 `grasp_ok` 가 다섯 항(`lifting` · `goal_tracking(+fine)` ·
    `settled` · `dwell` · `grasp_pose`)의 **공통 목**이라, 파지 전엔 전부 정확히 0 이었다.
    연속화하되 두 가지를 잃으면 안 된다:
      ① 판별력 — 성공 기하와 실패 기하가 충분히 벌어져야 한다
      ② 던지기 차단 — `near`·`upright` 게이트는 **그대로 남아야** 한다(test3 사고)
    """
    import math
    src = _src("grasp_left_rewards.py")
    assert "def grasp_quality" in src
    held = src[src.index("def _held("):src.index("def held_with_good_pose")]
    assert "grasp_quality(env" in held, "_held 가 아직 이진 게이트를 쓴다"
    assert "grasp_ok(env" not in held, "_held 에 이진 게이트가 남아 있다"
    assert "(near & upright)" in held, "던지기 차단 게이트가 사라졌다 — test3 사고 재발"

    # 실측 기하로 판별력 검산 (성공 vs 두 실패 모드)
    def q(lat_mm, alo_mm):
        v = (math.exp(-lat_mm / 1e3 / P.GRASP_GATE_LATERAL_OK)
             * math.exp(-alo_mm / 1e3 / P.GRASP_GATE_ALONG_OK))
        return min(v / P.GRASP_QUALITY_REF, 1.0)
    ok = q(20.0, 13.0)          # test17 · fab_test8 성공
    fist = q(78.6, 27.8)        # fab_test1 주먹
    beside = q(85.5, 12.0)      # fab_test11 컵 옆 대기
    assert ok > 0.95, f"성공 기하에서 {ok:.3f} — 1.0 이어야 가중 균형이 유지된다"
    assert ok / max(fist, beside) > 8.0, (
        f"판별력 부족: 성공 {ok:.3f} vs 실패 {max(fist, beside):.3f}"
    )


def test_approach_sharpness_matches_our_working_distance():
    """★fab_test34: 접근 커널 sharpness 는 **우리 작업 거리**에 맞춰야 한다.

    이식 원본(agnostic)은 8.0 인데 그 트랙은 홈이 물체에 훨씬 가깝다. 우리 작업 구간은
    `d_palm + d_side ≈ 0.15~0.40 m` 이고, 8.0 이면 d=0.40 에서 값 0.041 · gradient 0.33/m
    로 신호가 바닥이다 — t33 이 936 epoch 동안 턱-컵 0.18~0.21 m 에서 정체했다.

    `s·exp(−s·d)` 의 gradient 최대점은 `s = 1/d` 다. 우리 대역이면 s ∈ [2.5, 6.7].
    """
    assert 2.5 <= P.APPROACH_SHARPNESS <= 6.7, (
        f"sharpness {P.APPROACH_SHARPNESS} 가 작업 거리(0.15~0.40 m)의 최적 대역 밖이다"
    )
    # 성공 지점 대비가 살아 있어야 한다 — 너무 낮추면 평지가 된다
    import math
    near, far = 0.08, 0.40
    ratio = math.exp(-P.APPROACH_SHARPNESS * near) / math.exp(-P.APPROACH_SHARPNESS * far)
    assert ratio >= 3.0, f"성공/현재 대비 {ratio:.1f}배 — 3배 미만이면 평지다"


def test_grasp_pose_is_inside_the_action_box():
    """★★★fab_test35: **파지 자세가 액션 박스 안에 있어야 한다.**

    당연해 보이지만 t24~t34 열 판이 이걸 어긴 채 돌았다. `probe_palm_z_transfer.py` 실측:
        지령 z 0.220(구 박스 바닥) → 턱 z 0.2553  (파지 목표 0.2475 보다 +7.8 mm)
        지령 z 0.202               → 턱 z 0.2488  ← 이게 필요한데 박스 밖이었다
    파지 가능 턱 높이 [0.230, 0.285] 을 내는 지령은 [0.13, 0.267]. 구 박스에서는
    액션 z ∈ [−1.00, −0.72] 로 **범위의 14% 이고 경계에 붙어** 있었다.

    ⚠ 팜 지령 z 와 턱 z 는 1:1 이 아니다(팔이 내려갈수록 압축되고 0.2300 에서 포화).
      아래 상수는 그 전달표에서 읽은 값이다 — 기하로 추정하지 말 것.
    """
    CMD_FOR_BAND = (0.13, 0.267)     # 턱 z [0.230, 0.285] 을 내는 팜 지령 (실측)
    lo, hi = P.PALM_BOX_Z
    center, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    a_lo = max((CMD_FOR_BAND[0] - center) / half, -1.0)
    a_hi = (CMD_FOR_BAND[1] - center) / half
    assert a_hi <= 1.0, "파지 대역 상단이 박스 위로 나갔다"
    width = a_hi - a_lo
    assert width >= 0.5, (
        f"파지 가능 액션 z 창이 {width:.2f} 뿐이다 — 경계에 붙어 탐색이 못 찾는다"
    )
    assert a_hi <= -0.2, "파지 창이 액션 0 근처면 목표 유지와 겹쳐 구분이 안 된다"
    # 목표 유지도 여전히 도달 가능해야 한다 (턱 z 0.465 ← 지령 0.445)
    assert hi >= 0.46, f"박스 상한 {hi} — 목표 유지 지령(0.445)에 여유가 없다"


def test_learner_is_actually_on():
    """★★★fab_test36: 학습기가 **실제로 작동하는 설정**인지 고정한다.

    t35 는 학습기가 원래 세기의 1/40 로 켜져 있었다(ep100-200 실측, t16ctl 대비):
        losses/a_loss 0.00718 → 0.00017 (42 배) · info/last_lr 0.00334 → 0.000289 (11.6 배)
        losses/bounds_loss 6.18 → 0.000 (죽음)

    ① `lr_schedule` 은 **adaptive** — linear 고정 감쇠는 KL 이 임계를 넘어도 lr 을 못 올려
       국소최적에서 못 나온다.
    ② `bound_loss_type` 은 rl_games 가 **`'regularisation'`(영국식 s)** 과 `'bound'` 만
       받는다. 미국식 `regularization` 은 어느 분기에도 안 맞아 **조용히 0** 이 된다.
       ⚠ `'regularisation'` 을 쓰면 안 된다 — `reg_loss = mu²` 가 mu 를 0(=박스 중심,
         우리 경우 목표 근처)으로 당겨 "목표로 가는 기동"을 보상 밖에서 만든다.
    """
    import yaml
    cfg = yaml.safe_load(
        (_PKG / "config/agents/rl_games_ppo_fab_cfg.yaml").read_text())["params"]["config"]
    assert cfg["lr_schedule"] == "adaptive", (
        "lr_schedule 이 adaptive 가 아니다 — lr 이 묶여 a_loss 가 42 배 작아진다"
    )
    assert cfg["bound_loss_type"] == "bound", (
        f"bound_loss_type={cfg['bound_loss_type']!r} — rl_games 는 'bound' 또는 "
        "'regularisation'(영국식) 만 받는다. 그 외는 조용히 0 이 된다. "
        "그리고 'regularisation'(mu²)은 mu 를 박스 중심=목표 쪽으로 당겨 해롭다."
    )


def test_episode_is_short_enough_that_engaging_beats_idling():
    """★★★fab_test37: 에피소드가 길면 **컵을 안 건드리는 것이 최적**이 된다.

    `object_dropping` 은 페널티가 아니라 **종료**다. 그래서 수익이 다르게 스케일한다 —
    가만히 있으면 `r_idle × T`(길이에 비례), 접근하다 낙하하면 `r_near × k`(낙하 시각,
    T 와 무관). 길수록 회피가 유리해진다.

    ⚠ 이 계약은 **모델이 아니라 로그**로 고정한다. 아카이브 23 런 + 오늘 10 런에서
      리프트한 10 개는 **전부 5.0 s** 이고, 10.0 s 인 열두 판(t22~t36)은 예외 없이
      `lifting_object` 정확히 0 이었다. 그리고 그 열두 판은 전부 `drop`(ep50-200)이
      0.002~0.034 로 죽었다 — 컵을 만지지 않으니 파지를 찾을 표본이 없다.
    ⚠ 교락 주의: 10 s 런은 전부 kuka env 이고 5 s 런은 전부 t16 env 라 지금 데이터로는
      길이만 따로 분리되지 않는다. 이 상수를 5.0 으로 두는 것이 그 교락을 푸는
      단일 변수 실험이다. 5 s 인데도 drop 이 죽은 런이 있으므로(t1·t9·t10·t11)
      길이만으로 전부 설명되지는 않는다.
    """
    assert P.EPISODE_LENGTH_S <= 5.0, (
        f"에피소드 {P.EPISODE_LENGTH_S} s — 리프트한 10 런은 전부 5.0 s 였고 "
        "10.0 s 인 열두 판은 전부 lifting 0 이었다"
    )


def test_contact_reward_pays_for_touching_the_cup():
    """★★★fab_test38: 컵을 **건드리는 것 자체**에 값이 있어야 한다.

    t22~t37 열세 판의 공통 서명이 `drop`(ep50-200) 0.000 이었다. 로그 전수에서
    `drop` 은 리프트의 **필요조건**이다 — ≥0.02 인 10 런은 전부 리프트했고, <0.02 인
    9 런은 전부 lifting 0 이다. 컵을 안 만지면 파지를 찾을 표본이 없다.

    만지지 않는 이유는 **만져서 얻는 게 없기 때문**이다: 낙하는 페널티가 아니라 종료라
    위험만 있고, 파지 계열은 `grasp_quality` 를 지나야 하는데 거기 닿으려면 이미 잘
    잡고 있어야 한다. DexPour ablation 의 Config.2 가 같은 실패를 기록한다.

    ⚠ 이 항은 **접촉 센서가 살아 있을 때만** 의미가 있다. 08.26 까지 필터가 프림 루트를
      가리켜 `force_matrix_w` 가 최대까지 정확히 0 이었다 — 그때 넣었으면 죽은 항이다.
      `test_contact_filter_points_at_the_rigid_body` 와 짝으로 봐야 한다.
    """
    cfg = _src("grasp_left_fab_env_cfg.py")
    assert "self.rewards.contact_engage = RewTerm" in cfg, "접촉 보상이 배선되지 않았다"
    assert "func=rewards.contact_engage" in cfg
    src = _src("grasp_left_rewards.py")
    assert "def contact_engage" in src
    assert "finger_contact_forces" in src, "접촉 보상이 실제 센서를 읽지 않는다"

    # 가중은 기존 항을 지배하면 안 된다 — t37 실측 순간율 합 0.486
    LIVE_RATE = 0.486
    contrib = P.CONTACT_ENGAGE_WEIGHT * 0.5 * (1.0 / 3.0)   # 한 턱 · 에피소드 1/3
    assert 0.25 * LIVE_RATE < contrib < 1.0 * LIVE_RATE, (
        f"기여 {contrib:.3f} vs 기존 {LIVE_RATE:.3f} — 너무 작으면 무시되고 "
        "너무 크면 접근 보상을 밀어낸다"
    )
    assert P.CONTACT_FORCE_THRESHOLD > 0.0, "문턱 0 이면 수치 잡음도 접촉으로 센다"
