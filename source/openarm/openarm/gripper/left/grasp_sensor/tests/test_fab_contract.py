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
    """★★fab_test61: 회전은 원본 kuka 와 같은 **euler_zyx 절대** 규약이다.

    ⚠ 이 계약은 t60(성공기 복원)의 반대 계약을 **의도적으로 뒤집은 것**이다.
      t60 계약은 "quaternion 경로만" 이었고 근거는 08.21 실측 — 당시 기준 palm 자세가
      (0, π/2, 0) 으로 euler_zyx 짐벌 특이점 **정확히 위**였고 회전 계단 오버슈트가
      19~32% 였다.

    뒤집는 이유는 이번 판의 **질문 자체**다(사용자 지시): "t60 의 나머지 세팅을 그대로
    두고 fabric 배선만 t59 로 했을 때도 리프트가 되는가." 회전 규약은 그 배선 8종
    중 하나이므로 t59 값이어야 한다. 부수 근거 둘도 유효하다:
      ① 현 기준 자세의 ey 중심은 −76.09° 로 특이점에서 14° 떨어져 있다.
      ② 그 19~32% 는 결함 있는 플랜트(vel_ff 0 · fabric 60% 속도 · damping 하드끝)
         위에서 잰 값인데, 이 판은 셋을 전부 t59(=원본) 배선으로 되돌린다.

    ★fab_test61 이 리프트에 실패하고 회전이 원인으로 지목되면 이 계약을 t60 판으로
      되돌리고 사유를 여기 적을 것. 되돌릴 원문은 git 8b62997 에 있다.

    ⚠ 짐벌 특이점은 불연속이 아니다 — 전방 사상(euler→R)은 ey=±90° 에서도 연속이고
      비용은 조건수 저하다(ez 와 ex 가 같은 회전을 만들어 액션 1D 가 국소 중복).
    """
    src = _src("grasp_left_fabric_action.py")
    assert '"euler_zyx"' in src, "set_features 에 euler_zyx 규약을 넘기지 않는다"
    assert '"quaternion"' not in src, "구 quaternion 경로가 남아 있다"
    assert "PALM_EULER_ZYX_CENTER" in src, "euler 중심 상수를 안 쓴다"

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
    # ★★fab_test77: E2(t75/76) 확대를 **되돌린다**. 산포를 키우면 조건부 추종 압력이
    #   커진다는 가설이 실측으로 기각됐다 — 목표→지령 기울기가
    #     t73(옛 상자) x 0.109 · y 0.297 · z 0.053
    #     t75(넓힌 상자) x 0.099 · y 0.016 · z 0.006   ← **오히려 나빠졌다**
    #   병목은 목표 분포가 아니라 **액션 축 포화**였다(t75 best 프로브: y mu 1.504 포화
    #   99.1% · z mu 1.319 포화 86.4% · 덜 포화된 x 만 기울기가 산다). clamp 미분이 0 이라
    #   포화된 축은 목표를 따라갈 수 없다. ⇒ 상자는 t73 기준선으로 되돌리고 처방은
    #   `bounds_loss_coef` 로 건다(한 판에 한 변수).
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


def test_between_jaws_reward_is_wired_and_uses_enclose_not_closure():
    """★★08.22 fab_test1 실패의 직접 처방 — 이 항이 빠지면 같은 실패가 재발한다.

    실측: fab 정책은 개도 3.1 mm 로 **주먹을 쥔 채** 컵 옆구리를 눌렀다(수직 최선 36.5 mm
    ≈ 컵 반경). 성공한 test17 은 0.4 mm / 26.5 mm.
    ⚠ 옛 `closure = 1 − drive/open` 을 곱하면 **닫을수록 커져** 그 실패 행동을 보상한다.
    """
    cfg = _src("grasp_left_env_cfg.py")
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
    assert "self.rewards.reaching_object" in cfg and "ee_grasp_point_distance" in cfg
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
    assert "grasp_ok(" in body, "리프트 게이트가 grasp_ok 를 쓰지 않는다"
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
    assert "in_band = (axis_t > _band[0]) & (axis_t < _band[1])" in body
    # ★대역은 호출자가 줄 수 있다(v2 는 판 위 80~150 mm). 안 주면 **v1 값**이어야
    #   한다 — 09.03 이전엔 환경변수가 이 모듈 상수를 통째로 바꿔 v1 까지 오염시켰다.
    assert "band if band is not None else P.CUP_GRASP_BAND_AXIS" in body


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
    # ★fab_test66: effort/강성 파생 관계는 끊겼다(위 주석 참조). 값 자체를 고정한다.
    assert P.ARM_IK_MAX_TRACKING_ERROR["l_aj_[5-7]"] == 0.0175
    assert isinstance(P.ARM_IK_STIFFNESS, dict), "관절별 kp 테이퍼가 풀렸다"
    assert P.GRAVITY_COMP_ENABLED, "포화가 풀린 뒤의 중력 보상이 이 판의 전제다"
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


def test_mu_drift_is_held_by_hinge_not_by_squashing():
    """★★t67(표류)·t68(동결) 두 실패의 처방 — 둘 중 하나로 풀면 다른 하나가 재발한다.

    t67: 선형 mu + `bounds_loss_coef` 1e-4 → mu 가 y=3.11 · z=2.04 로 표류.
         샘플이 전부 clamp 에 뭉개져(포화 y 99.7%) 그 축의 학습 신호가 사라졌다.
    t68: mu 를 tanh 로 가뒀더니 zmu 가 -0.97 에 붙어 **동결**됐다.
         tanh'(0.97)=0.059 로 gradient 가 17 배 감쇠하는데, 하필 이 태스크의
         접근 자세가 거기다(t67 접근 지령 z=0.180 = a -0.909). 전 구간 lift 0.00.
    ⇒ mu 는 자유롭게(None) 두고, 표류만 hinge 벌점으로 막는다. `bound_loss_type`
      기본값 'bound' 는 |mu| ≤ 1.1 에서 정확히 0 이라 박스 안 행동을 안 건드린다.
    """
    yaml_txt = (_PKG / "config" / "agents" / "rl_games_ppo_fab_cfg.yaml").read_text()
    assert "mu_activation: None" in yaml_txt, "tanh 로 가두면 t68 의 동결이 재발한다"
    assert "mu_activation: tanh" not in yaml_txt
    coef = [ln for ln in yaml_txt.splitlines()
            if ln.strip().startswith("bounds_loss_coef:")]
    assert len(coef) == 1
    assert float(coef[0].split(":")[1]) >= 5e-3, (
        "복원력이 없으면 표류가 재발한다 — t71 의 ymu 는 -4.04 까지 갔다"
    )


def test_obs_set_default_is_the_winning_pose_variant():
    """★★t70 vs t71 귀속 결과. 지령(`palm_cmd`)과 실측(`tcp_pos`)은 공선이라 둘 다 주면
    안 되고(t68·t69 가 lift 0.00 으로 죽었다), 둘 중 **실측**이 이겼다(fine 0.57 vs 0.28).
    """
    src = _src("grasp_left_fab_env_cfg.py")
    assert 'os.environ.get("HDGP_OBS_SET", "pose")' in src


def test_all_three_action_axes_are_logged():
    """★t67 의 진짜 병목은 y 였는데 z 만 찍고 있어 판이 끝난 뒤에야 알았다."""
    src = _src("grasp_left_env_cfg.py")
    assert 'for _ax, _i in (("x", 0), ("y", 1)):' in src
    assert "diag_act_z_mu" in src


def test_absolute_action_has_pose_feedback_in_obs():
    """★★절대 태스크공간 액션은 **피드백 없이는 닫힌 루프가 아니다**.

    t67 의 obs 에는 raw `last_action` 뿐이라 정책이 (a) 자기 palm 위치와
    (b) 리미터가 옮겨 놓은 실제 지령을 못 봤다. (b) 는 적분기 상태라
    메모리 없는 MLP 에는 원리적으로 관측 불가능했다.
    """
    src = _src("grasp_left_fab_env_cfg.py")
    for term in ("palm_cmd", "tcp_pos", "palm_rot"):
        assert f"self.observations.policy.{term} = ObsTerm(" in src, f"{term} 관측이 빠졌다"
    obs_src = _src("grasp_left_observations.py")
    # ★지령은 raw 가 아니라 리미터를 통과한 값이어야 한다 — 그게 이 항의 존재 이유다.
    assert "term.processed_actions" in obs_src
    assert "raw_actions" not in obs_src, "raw 액션은 last_action 과 중복이라 의미가 없다"
    # ★euler 로 자세를 내면 roll 중심 3.095 rad 가 ±π 경계에서 6.28 을 널뛴다.
    assert "matrix_from_quat" in obs_src


def test_command_state_is_reset_with_the_episode():
    """★지령 상태가 obs 에 들어간 순간 그것도 리셋 오염원이 된다(이 트랙 5번째)."""
    src = _src("grasp_left_fabric_action.py")
    reset = src[src.index("def reset("):src.index("def _debug_vis_callback")
                if "def _debug_vis_callback" in src else len(src)]
    assert "_palm_pose_target[env_ids" in reset, "리셋 직후 첫 관측이 직전 에피소드 지령이다"


def test_body_scoped_obs_cfg_is_passed_through_params():
    """★IsaacLab 함정: `SceneEntityCfg` 는 **params 에 있는 것만** resolve 된다.

    기본 인자로 두면 `body_ids` 가 slice(None) 인 채 들어와 인덱싱에서 죽는다
    (fab_test68 첫 기동이 여기서 죽었다).
    """
    src = _src("grasp_left_fab_env_cfg.py")
    blk = src[src.index("palm_rot = ObsTerm("):]
    assert 'params={"robot_cfg": SceneEntityCfg(' in blk[:400]
    obs_src = _src("grasp_left_observations.py")
    assert "isinstance(robot_cfg.body_ids, slice)" in obs_src, "resolve 누락이 조용히 지나간다"


def test_goal_is_scored_on_tcp_but_judged_on_cup():
    """★★fab_test73(사용자 지시): 목표 상자는 **TCP 제약 IK** 로 도달 가능한 곳만 골라
    만든 것이라, 채점도 TCP 로 해야 검증한 바디와 채점하는 바디가 같아진다.

    ⚠ 대신 컵이 게이트 `near`(80 mm) 만큼 벌어질 수 있으므로 **합격 판정은 컵**이다.
      두 진단이 함께 있어야 벌어지는 순간을 본다.
    """
    src = _src("grasp_left_rewards.py")
    # ⚠ 슬라이스 끝은 **다음 함수**다 — fab_test74 가 그 사이에 height 판을 끼워 넣었고,
    #   그건 의도적으로 컵 원점을 쓴다(모드마다 기준이 다르다).
    blk = src[src.index("def object_goal_distance_when_held"):
              src.index("def object_goal_distance_height_gated")]
    assert "ee_frame.data.target_pos_w" in blk, "held 모드의 목표 채점이 TCP 가 아니다"
    assert "obj.data.root_pos_w" not in blk.split('"""')[-1], "컵 원점으로 되돌아갔다"
    cfg = _src("grasp_left_env_cfg.py")
    for d in ("diag_cup_goal_dist", "diag_tcp_goal_dist"):
        assert f"self.rewards.{d} = RewTerm(" in cfg, f"{d} 진단이 빠졌다"


def test_goal_gate_ab_switch_and_cup_basis():
    """★★fab_test74(E1): goal 보상의 **신호 시점**을 가르는 A/B.

    IsaacLab 레퍼런스는 게이트가 `cube.z > 0.04` 하나뿐인데 스폰이 0.055라
    **step 0 부터 참**이다 — 정책이 조건부 목표 추종을 맨 처음부터 배운다.
    우리는 `_held`(파지·리프트 완성) 뒤에야 신호가 돌아, t73 실측 조건부 추종
    기울기가 x 0.109 · y 0.297 · z 0.053 이었다(1.0 이어야 한다).

    ⚠⚠ `height` 모드의 거리는 **반드시 컵 원점**이다. TCP 로 재면 빈 그리퍼를
      목표에 놓기만 해도 만점이라 컵을 아예 안 든다. TCP 채점은 `_held` 가 파지를
      요구하는 `held` 모드에서만 안전하다 — 게이트와 거리 기준은 한 쌍이다.
    """
    cfg = _src("grasp_left_env_cfg.py")
    assert '_os.environ.get("HDGP_GOAL_GATE", "held")' in cfg, "기본값이 held 가 아니다"
    assert "rewards.object_goal_distance_height_gated" in cfg
    assert "rewards.object_goal_distance_when_held" in cfg, "기존 모드가 사라졌다"
    assert '"gate_height": P.OBJECT_DROP_HEIGHT' in cfg

    src = _src("grasp_left_rewards.py")
    blk = src[src.index("def object_goal_distance_height_gated"):
              src.index("def object_settled_at_goal")]
    body = blk.split('"""')[-1]
    assert "obj_pos_w = obj.data.root_pos_w" in body and "des_pos_w - obj_pos_w" in body, (
        "거리 기준이 컵 원점이 아니다 — 빈 그리퍼 해킹이 열린다"
    )
    assert "target_pos_w" not in body, "height 모드에서 TCP 를 쓰면 안 된다"
    assert "_held(" not in body, "레퍼런스형이면 파지 게이트가 없어야 한다"

def test_action_rate_curriculum_is_disabled_but_joint_vel_is_not():
    """★`action_rate` 커리큘럼(−1e-4 → −1e-1, 1000 배)만 꺼져 있어야 한다.

    이 항은 목적을 달성하지 못한다 — 저장소 실측: "action_rate_l2 는 액션공간 통계라
    탐색 노이즈(σ)에 오염돼, 옵티마이저가 σ 만 줄이고 정책 평균의 평활도는 1000 epoch
    동안 평탄했다". 그리고 t73/t75 가 정확히 발동 시점(36000 step ÷ horizon 24 = ep1500)에
    꺾였다(t75 fine 0.320→0.156 · t73 rew 124→92, cupd 131→180 mm).

    기전: 표류한 축(mu 1.5)에서 goal 은 clamp 미분 0 이라 gradient 가 없는데
    `action_rate_l2` 는 clamp **이전** raw 액션을 재므로 살아 있다. 발동 후 그 축에 남는
    유일한 힘이 "흔들지 마라"가 되고, σ 가 줄면 포화가 굳는다.

    ⚠ `joint_vel` 은 유지한다 — 관절속도는 물리량이라 σ 오염이 훨씬 덜하고, 실측 크기가
      `action_rate` 의 1/14.6(−0.047 vs −0.68)이다. 둘을 같이 끄면 무엇이 들었는지 못 가린다.
    ⚠ 항 자체와 base weight −1e-4 는 남는다 — TFEvents 로 채터를 계속 관측해야 한다.
    """
    src = _src("grasp_left_env_cfg.py")
    assert "self.curriculum.action_rate = None" in src, (
        "action_rate 커리큘럼이 다시 켜졌다 — ep1500 에 1000 배 승격이 돌아온다"
    )
    assert 'self.curriculum.joint_vel.params["num_steps"]' in src, (
        "joint_vel 커리큘럼까지 끄면 단일 변수가 깨진다"
    )
    # 항 자체는 살아 있어야 로깅된다 (레퍼런스 정의를 지운 것이 아니다)
    ref = _src("_vendored_lift_openarm_env_cfg.py")
    assert "action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)" in ref
