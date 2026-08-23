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

"""학습된 정책이 컵을 **무엇으로** 들고 있는지 측정한다.

왜 필요한가: test3(1500 epoch)에서 `lifting_object` 가 에피소드의 91% 를 차지하는데
`reaching_object` 는 평탄했다(TCP–컵 약 19 cm). 즉 리프트 판정은 계속 참인데 **그리퍼는
컵 근처에 없다**. `mdp.object_is_lifted` 는 파지를 요구하지 않고 z 만 보므로, 팔뚝·손등처럼
그리퍼가 아닌 부위로 떠받쳐도 만점이 나온다.

여기서는 컵에 가장 가까운 **링크가 무엇인지**를 시계열로 세어 그 가설을 확정하거나 기각한다.
"그리퍼로 잡았다"면 최근접 링크가 손가락이어야 하고, "얹었다"면 팔뚝/손등이 나온다.

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_lift_left_policy_contact.py \
        --checkpoint <path.pth>
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=250)
# ★태스크를 고정하면 안 된다. 관절공간판(obs 36·action 8)과 태스크공간 IK 판(35·7)은
#   체크포인트 모양이 달라, 하드코딩하면 IK 체크포인트가 size mismatch 로 죽는다.
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import openarm.tasks  # noqa: F401
from isaaclab.utils.math import matrix_from_quat
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

TASK = args.task


def _quat_tilt_deg(quat: torch.Tensor) -> torch.Tensor:
    """물체 로컬 +z 와 월드 +z 사이 각도."""
    w, x, y, z = quat.unbind(-1)
    axis_z_w = 1 - 2 * (x * x + y * y)
    return torch.rad2deg(torch.acos(axis_z_w.clamp(-1.0, 1.0)))


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")

    env = gym.make(TASK, cfg=env_cfg)
    raw = env.unwrapped
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math_inf := float("inf"))
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math_inf)
    wrapped = RlGamesVecEnvWrapper(env, args.device, clip_obs, clip_act)

    vecenv.register("IsaacRlgWrapper", lambda cfg_name, n, **kw: RlGamesGpuEnv(cfg_name, n, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})

    agent_cfg["params"]["config"]["env_info"] = wrapped.get_number_of_agents and {
        "observation_space": wrapped.observation_space,
        "action_space": wrapped.action_space,
        "agents": 1,
    }
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args.checkpoint)
    agent.reset()

    robot = raw.scene["robot"]
    obj = raw.scene["object"]
    ee = raw.scene["ee_frame"]
    origins = raw.scene.env_origins
    left = [(i, n) for i, n in enumerate(robot.body_names) if n.startswith(("l_hl_", "l_al_"))]
    idx = [i for i, _ in left]
    names = [n for _, n in left]
    grip_ids, _ = robot.find_joints(P.GRIPPER_JOINT_NAMES, preserve_order=True)
    base_i = robot.body_names.index(P.GRIPPER_BASE_BODY)
    finger_ids = [robot.body_names.index(n) for n in P.GRIPPER_FINGER_BODIES]

    def _tensor(o):
        # RlGamesVecEnvWrapper 는 {'obs': tensor} 를 준다. player 는 텐서를 기대한다.
        return o["obs"] if isinstance(o, dict) else o

    arm_ids, _ = robot.find_joints([f"l_aj_{i}" for i in range(1, 8)], preserve_order=True)
    prev_act = None
    prev_delta = None
    obs = _tensor(wrapped.reset())
    # ★play.py 와 같은 준비 절차. 이게 없으면 player 가 배치를 1개로 보고
    #   (1, num_envs*obs_dim) 로 flatten 해 행렬곱이 깨진다.
    agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()
    nearest_count: dict[str, int] = {n: 0 for n in names}
    lifted_steps = 0
    held_steps = 0
    total = 0
    grip_open_when_lifted = []
    tcp_when_lifted = []
    jaw_tilt = []          # jaw 축이 수평면에서 벗어난 각(도). 0 = 완전 수평
    approach_pitch = []    # 접근축(base +z)이 수평면에서 벗어난 각(도)
    cup_tilt_held = []     # 쥐고 있을 때 컵이 세워져 있는가
    axis_angle = []        # ★TCP z축 ↔ 컵 z축 사이 각(도). **90° 가 올바른 파지**
    lin_speed = []         # 쥐고 있을 때 컵 선속도 (m/s)
    ang_speed = []         # 쥐고 있을 때 컵 각속도 (rad/s)
    goal_dist = []         # 컵 ↔ 목표 거리 (m)
    # ★진동 진단: 제어는 IK 가 아니라 **관절 위치 델타**(JointPositionAction)다.
    #   정책이 매 스텝 관절 목표를 내므로, 그 목표가 스텝마다 흔들리면 팔이 멈추지 않는다.
    act_delta = []         # |a_t − a_{t−1}| (1차 차분 — action_rate 가 벌하는 양)
    act_jerk = []          # |Δa_t − Δa_{t−1}| (2차 차분 — 레퍼런스에 **없는** 항)
    act_flips = []         # 액션 성분별 부호 반전 비율 (진동이면 높다)
    arm_speed = []         # 팔 관절 속도 크기 (rad/s)
    tgt_delta = []         # 적용된 관절 목표의 스텝간 변화 (rad) — 제한기 이후
    prev_tgt = [None]
    late_lin = []          # 에피소드 후반(목표 근처)에서의 컵 선속도
    # ★리프트 게이트가 **놓인 상태에서 열려 있으므로**(레퍼런스 정합) 'lifted 비율' 만으로는
    #   들었는지 알 수 없다 — 가만히 있어도 100% 다. 컵 원점 z 를 직접 잰다.
    cup_z = []             # 컵 원점 z (env-local)
    cup_dxy = []           # 스폰 위치에서의 수평 이동
    spawn_xy = [None]
    # ★그리퍼 지령의 **시계열**이 필요하다. 평균만 보면 '컵에 막혀 멈춘 것'과 '아예
    #   안 닫는 것'을 구분할 수 없다 — 둘 다 중간값으로 나온다.
    # ★★straddle 실측 — "컵이 턱 사이에 들어왔는가" 의 물리 규모.
    #   `rewards.cup_between_jaws` 와 **같은 식**으로 잰다. 그래야 여기서 나온 수치를
    #   그대로 std 산정에 쓸 수 있다(CLAUDE.md: 새 항의 임계는 실측 규모를 재고 정한다).
    sd_along = []          # 턱 축 방향 어긋남 (m) — 전 env 평균
    sd_lateral = []        # 턱 축 선까지의 수직거리 (m) — 전 env 평균
    sd_along_best = []     # 그 스텝에서 가장 잘 맞춘 env (최소값)
    sd_lateral_best = []
    # ★★"목표에 와서도 못 멈춘다"를 층으로 가르는 계측. G3(고정 지령)가 0.054 m/s 를
    #   냈으므로 제어기는 멈출 수 있다 — 정책이 지령을 계속 흔드는지, 아니면 지령은
    #   멈췄는데 아래층이 못 따라오는지를 구분해야 처방이 갈린다.
    #   사슬: 정책 지령(palm 목표) → fabric_q → 실제 관절 → 컵
    gz_cmd = []            # 목표 근처에서 palm **위치 지령**의 스텝간 변화 (mm/step)
    gz_fab = []            # 같은 구간 fabric_q 변화 (mrad/step)
    gz_qd = []             # 같은 구간 실제 관절 속도 (rad/s)
    gz_cup = []            # 같은 구간 컵 선속도 (m/s)
    gz_prev_cmd = [None]
    gz_prev_fab = [None]
    sd_enclose = []        # 두 손가락이 컵 축 양쪽에 있는 정도 (0~1)
    sd_axis_h = []         # 턱 중점의 **컵 축 방향 높이** (컵 원점 기준, m)
    sd_term = []           # `cup_between_jaws` 항의 실제 값 (weight 곱하기 전)
    grip_series = []       # 구동 관절 위치 (m)
    grip_cmd = []          # 이진 그리퍼 액션의 부호 (>0 = 열기 지령)

    for _ in range(args.steps):
        with torch.inference_mode():
            act = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
        if prev_act is not None:
            d = act - prev_act
            act_delta.append(float(d.norm(dim=-1).mean()))
            if prev_delta is not None:
                act_jerk.append(float((d - prev_delta).norm(dim=-1).mean()))
                # 부호 반전 = 방향을 되돌린 성분의 비율. 일정 방향 이동이면 0 에 가깝다.
                act_flips.append(float(((d * prev_delta) < 0).float().mean()))
            prev_delta = d.clone()
        prev_act = act.clone()
        obs, _, _, _ = wrapped.step(act)
        obs = _tensor(obs)
        arm_speed.append(float(robot.data.joint_vel[:, arm_ids].norm(dim=-1).mean()))
        grip_series.append(float(robot.data.joint_pos[:, grip_ids[0]].mean()))
        grip_cmd.append(float((act[:, -1] > 0).float().mean()))
        # ★변화율 제한기가 있으면 raw 액션의 차분은 무의미하다(clamp 되어 버려진다).
        #   실제로 로봇에 간 **관절 목표**의 스텝간 변화를 따로 잰다 — 이게 평활도다.
        tgt = robot.data.joint_pos_target[:, arm_ids]
        if prev_tgt[0] is not None:
            tgt_delta.append(float((tgt - prev_tgt[0]).abs().amax(dim=-1).mean()))
        prev_tgt[0] = tgt.clone()

        cup = obj.data.root_pos_w - origins
        if spawn_xy[0] is None:
            spawn_xy[0] = cup[:, :2].clone()
        cup_z.append(float(cup[:, 2].mean()))
        cup_dxy.append(float((cup[:, :2] - spawn_xy[0]).norm(dim=-1).mean()))
        # straddle: 턱 중점 ↔ 컵 축. 보상 함수와 동일한 기하.
        _f = robot.data.body_pos_w[:, finger_ids, :]
        # ★기준선을 손가락 패드 중앙으로 (보상 함수와 동일). 손가락 원점 그대로면
        #   "컵을 손바닥까지 밀어넣어라"를 재는 자가 된다 — 실측 파지 깊이는 base z=+46.9mm.
        _ap = matrix_from_quat(robot.data.body_quat_w[:, finger_ids[0], :])[:, :, 2]
        _f = _f + (_ap * P.JAW_PAD_OFFSET).unsqueeze(1)
        _mid = 0.5 * (_f[:, 0, :] + _f[:, 1, :])
        _jaw = _f[:, 1, :] - _f[:, 0, :]
        _u = _jaw / _jaw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        _cz = matrix_from_quat(obj.data.root_quat_w)[:, :, 2]
        _tm = _mid - obj.data.root_pos_w
        _cpt = obj.data.root_pos_w + _cz * (_tm * _cz).sum(-1, keepdim=True)
        _d = _cpt - _mid
        _al = (_d * _u).sum(-1).abs()
        _lat = (_d - _u * (_d * _u).sum(-1, keepdim=True)).norm(dim=-1)
        sd_along.append(float(_al.mean())); sd_along_best.append(float(_al.min()))
        sd_lateral.append(float(_lat.mean())); sd_lateral_best.append(float(_lat.min()))
        # enclose: 두 손가락이 컵 축 **양쪽**에 있는가 (보상 함수와 동일)
        _sl = ((_cpt - _f[:, 0, :]) * _u).sum(-1)
        _sr = ((_f[:, 1, :] - _cpt) * _u).sum(-1)
        _enc = (torch.minimum(_sl, _sr) / P.JAW_ENCLOSE_HALF_WIDTH).clamp(0.0, 1.0)
        _align = 0.5 * (1 - torch.tanh(_al / P.JAW_ALONG_STD)) + 0.5 * (
            1 - torch.tanh(_lat / P.JAW_LATERAL_STD))
        sd_enclose.append(float(_enc.mean()))
        # ★★턱이 컵 **몸통**에 있는가, 아니면 축 연장선(허공)에 있는가.
        #   `cup_between_jaws` 의 cup_pt 는 컵 축의 **무한 직선** 위 최근접점이라,
        #   컵 위 허공에서 축을 감싸도 만점이 나온다. 그 구멍을 직접 잰다.
        _mid_h = (_mid - obj.data.root_pos_w)  # 컵 원점 기준
        _cz2 = matrix_from_quat(obj.data.root_quat_w)[:, :, 2]
        sd_axis_h.append(float((_mid_h * _cz2).sum(-1).mean()))
        sd_term.append(float((_align * (P.JAW_ENCLOSE_FLOOR
                                        + (1 - P.JAW_ENCLOSE_FLOOR) * _enc)).mean()))
        # 램프가 0 을 벗어나는 지점 = 실제로 뜨기 시작한 높이
        lifted = cup[:, 2] > P.LIFT_RAMP_ZERO_Z
        tcp_w = ee.data.target_pos_w[:, 0, :] - origins
        held = lifted & ((tcp_w - cup).norm(dim=-1) < P.GRASP_MAX_EE_DISTANCE)
        held_steps += int(held.sum())
        total += int(lifted.numel())
        lifted_steps += int(lifted.sum())
        if not bool(lifted.any()):
            continue
        pos = robot.data.body_pos_w[:, idx, :] - origins.unsqueeze(1)
        d = (pos - cup.unsqueeze(1)).norm(dim=-1)          # (E, L)
        near = d.argmin(dim=-1)
        for e in torch.nonzero(lifted).flatten().tolist():
            nearest_count[names[int(near[e])]] += 1
        tcp = ee.data.target_pos_w[:, 0, :] - origins
        tcp_when_lifted.append(float((tcp - cup).norm(dim=-1)[lifted].mean()))
        grip_open_when_lifted.append(float(robot.data.joint_pos[:, grip_ids[0]][lifted].mean()))

        # ★파지 **자세**. 2 지 그리퍼가 원통을 제대로 잡으려면 jaw 축(두 손가락을 잇는
        #   방향 = gripper_base 의 y 축)이 **수평**이어야 두 접촉점이 컵 지름 양끝에 놓인다.
        if bool(held.any()):
            q = robot.data.body_quat_w[:, base_i, :]
            w, x, y, z = q.unbind(-1)
            jaw_z = 2 * (y * z + w * x)            # R[2,1] : base y 축의 world z 성분
            appr_z = 1 - 2 * (x * x + y * y)       # R[2,2] : base z 축의 world z 성분
            jaw_tilt.append(float(torch.rad2deg(torch.asin(jaw_z.abs().clamp(max=1.0)))[held].mean()))
            approach_pitch.append(float(torch.rad2deg(torch.asin(appr_z.abs().clamp(max=1.0)))[held].mean()))
            cup_tilt_held.append(float(_quat_tilt_deg(obj.data.root_quat_w)[held].mean()))

            # ★★TCP z축(그리퍼 접근축) 과 컵 z축(원통 축) 사이 각.
            #   원통을 **옆에서** 물어야 제대로 된 파지이므로 90° 여야 한다.
            #   0° 면 컵 축 방향으로 내려꽂은 것이라 두 손가락이 지름을 잡지 못한다.
            tcp_axis = torch.stack(
                [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1
            )
            cw, cx, cy, cz = obj.data.root_quat_w.unbind(-1)
            cup_axis = torch.stack(
                [2 * (cx * cz + cw * cy), 2 * (cy * cz - cw * cx), 1 - 2 * (cx * cx + cy * cy)],
                dim=-1,
            )
            dot = (tcp_axis * cup_axis).sum(dim=-1).abs().clamp(max=1.0)
            axis_angle.append(float(torch.rad2deg(torch.acos(dot))[held].mean()))

            # ★"목표로 옮겨 정지"를 보상하려면 실제 속도 규모를 알아야 임계를 정할 수 있다.
            lin_speed.append(float(obj.data.root_lin_vel_w.norm(dim=-1)[held].mean()))
            ang_speed.append(float(obj.data.root_ang_vel_w.norm(dim=-1)[held].mean()))
            cmd = raw.command_manager.get_command("object_pose")
            from isaaclab.utils.math import combine_frame_transforms  # noqa: PLC0415
            des_w, _ = combine_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, cmd[:, :3]
            )
            gd = (des_w - obj.data.root_pos_w).norm(dim=-1)
            goal_dist.append(float(gd[held].mean()))
            # 목표에 이미 가까운(≤10 cm) env 만 골라 "도달 후에도 움직이는가"를 본다
            close = held & (gd < 0.10)
            if bool(close.any()):
                late_lin.append(float(obj.data.root_lin_vel_w.norm(dim=-1)[close].mean()))

            # ★층 분해 — 목표 10 cm 이내 & 쥐고 있는 env 만
            if bool(close.any()):
                _at = raw.action_manager.get_term("arm_action")
                _cmd = _at.processed_actions[:, :3]
                _fab = _at._fabric_q
                if gz_prev_cmd[0] is not None:
                    gz_cmd.append(float((_cmd - gz_prev_cmd[0]).norm(dim=-1)[close].mean()) * 1e3)
                    gz_fab.append(float((_fab - gz_prev_fab[0]).abs().amax(dim=-1)[close].mean()) * 1e3)
                gz_prev_cmd[0] = _cmd.clone(); gz_prev_fab[0] = _fab.clone()
                gz_qd.append(float(robot.data.joint_vel[:, arm_ids].norm(dim=-1)[close].mean()))
                gz_cup.append(float(obj.data.root_lin_vel_w.norm(dim=-1)[close].mean()))

    print("\n=== 리프트 판정 중 컵에 가장 가까운 링크 ===")
    print(f"  z 만 보는 판정(레퍼런스): {lifted_steps / max(total, 1):.1%}")
    print(f"  쥐고 있음까지 요구(신규):   {held_steps / max(total, 1):.1%}"
          f"   ← 이 정책의 처내기가 새 게이트로 얼마나 무효화되는가")
    ranked = sorted(nearest_count.items(), key=lambda kv: -kv[1])
    shown = sum(v for _, v in ranked) or 1
    for n, c in ranked[:8]:
        if c == 0:
            continue
        kind = "그리퍼" if "gripper" in n else "팔"
        print(f"  {n:<28} {c / shown:6.1%}  ({kind})")
    if tcp_when_lifted:
        print(f"\n  리프트 중 TCP–컵 거리 평균 {sum(tcp_when_lifted) / len(tcp_when_lifted) * 1e3:.1f} mm")
        print(f"  리프트 중 그리퍼 개도 평균 {sum(grip_open_when_lifted) / len(grip_open_when_lifted) * 1e3:.1f} mm "
              f"(닫힘 0 ~ 열림 {P.GRIPPER_OPEN_POS * 1e3:.0f})")
        print("  → 최근접이 손가락이 아니고 TCP 가 멀면 **그리퍼가 아닌 부위로 떠받친 것**이다.")
    if jaw_tilt:
        n = len(jaw_tilt)
        print("\n=== 쥐고 있을 때의 파지 자세 ===")
        print(f"  jaw 수평 이탈    {sum(jaw_tilt) / n:6.1f}°   (0 = 완전 수평. 두 접촉점이 컵 지름 양끝)")
        print(f"  접근축 pitch     {sum(approach_pitch) / n:6.1f}°   (0 = 수평 접근, 90 = 위에서 내려잡기)")
        print(f"  컵 기울기        {sum(cup_tilt_held) / n:6.1f}°   (0 = 세워진 채로 들림)")
        print(f"  ★TCP z ↔ 컵 z   {sum(axis_angle) / n:6.1f}°   "
              f"(**90° = 원통을 옆에서 문 올바른 파지**, 0° = 축 방향으로 내려꽂음)")
        print("  → jaw 수평 이탈이 크면 컵을 비스듬히 물어 접촉이 한쪽으로 몰린다.")
    if lin_speed:
        n = len(lin_speed)
        v = sum(lin_speed) / n
        w_ = sum(ang_speed) / n
        g = sum(goal_dist) / n
        print("\n=== 정지 보상 임계 산정용 실측 ===")
        print(f"  컵 선속도 {v:.3f} m/s   각속도 {w_:.3f} rad/s   목표까지 {g * 1e3:.0f} mm")
        # ★프리셋에서 읽는다. 여기에 리터럴을 박아두면 프리셋이 바뀐 뒤에도
        #   낡은 임계로 "품질 0" 을 찍어 실제보다 나쁘게 오보한다(실제로 그랬다).
        for name, val, std in (
            ("선속도", v, P.SETTLE_LIN_VEL_STD),
            ("각속도", w_, P.SETTLE_ANG_VEL_STD),
            ("목표거리", g, P.SETTLE_POS_STD),
        ):
            import math as _m
            q = 1.0 - _m.tanh(val / std)
            print(f"    {name}: 현재 std={std} → 품질 {q:.4f}")
        print("  → 품질이 0 에 가까우면 보상 신호가 없어 gradient 가 생기지 않는다.")
    if cup_z:
        import statistics as _st
        nz = len(cup_z)
        print("\n=== 컵이 실제로 들렸는가 (게이트가 놓인 상태에서 열려 있으므로 필수) ===")
        print(f"  스폰 원점 z {P.CUP_SPAWN_Z:.5f} · 램프 0→1 구간 +{P.LIFT_RAMP_SPAN * 1e3:.0f} mm")
        print(f"  {'구간':<10}{'컵 z':>10}{'스폰대비':>12}{'xy이동':>10}")
        for a, b, lab in [(0, nz // 5, "0~20%"), (nz // 5, 2 * nz // 5, "20~40%"),
                          (2 * nz // 5, 3 * nz // 5, "40~60%"), (3 * nz // 5, 4 * nz // 5, "60~80%"),
                          (4 * nz // 5, nz, "80~100%")]:
            z = _st.mean(cup_z[a:b]); dd = _st.mean(cup_dxy[a:b])
            print(f"  {lab:<10}{z:10.5f}{(z - P.CUP_SPAWN_Z) * 1e3:+9.1f} mm{dd * 1e3:8.1f} mm")
        up = sum(1 for z in cup_z if z > P.CUP_SPAWN_Z + 0.01) / nz
        print(f"  최대 컵 z {max(cup_z):.5f} (스폰 대비 {(max(cup_z) - P.CUP_SPAWN_Z) * 1e3:+.1f} mm)"
              f" · 스폰보다 1 cm 이상 올라간 스텝 {up:.1%}")
        print("  → 이 값이 0 에 가까우면 **컵을 안 들고 곁에 서 있는 것**이다.")

    if gz_cmd:
        import statistics as _st
        print("\n=== ★목표 근처(10 cm)에서 무엇이 움직이는가 — 층 분해 ===")
        print(f"  샘플 {len(gz_cmd)} 스텝 · 한 스텝 = {1000 * 0.02:.0f} ms")
        print(f"  {'① palm 위치 지령 변화':<26}{_st.mean(gz_cmd):8.2f} mm/step"
              f"  = {_st.mean(gz_cmd) / 0.02 / 1000:.3f} m/s")
        print(f"  {'② fabric_q 변화(최대관절)':<26}{_st.mean(gz_fab):8.2f} mrad/step"
              f"  = {_st.mean(gz_fab) / 0.02 / 1000:.3f} rad/s")
        print(f"  {'③ 실제 관절 속도':<26}{_st.mean(gz_qd):8.3f} rad/s")
        print(f"  {'④ 컵 선속도':<26}{_st.mean(gz_cup):8.3f} m/s")
        print("  → ①이 크면 **정책이 지령을 흔드는 것**(처방: 액션 쪽).")
        print("     ①이 작은데 ③④가 크면 **제어기가 못 멈추는 것**(처방: fabric/PD 쪽).")
        print(f"     참고: G3 고정 지령 실측 잔류 0.054 m/s = 제어기 하한")

    if sd_along:
        import math as _m
        import statistics as _st
        na = len(sd_along)
        am, lm = _st.mean(sd_along), _st.mean(sd_lateral)
        ab, lb = min(sd_along_best), min(sd_lateral_best)
        print("\n=== ★컵이 턱 사이에 들어왔는가 (straddle 실측) ===")
        print(f"  {'':<14}{'평균':>10}{'최선(min)':>12}")
        print(f"  {'턱축 어긋남':<14}{am * 1e3:9.1f} mm{ab * 1e3:11.1f} mm")
        print(f"  {'턱축까지 수직':<14}{lm * 1e3:9.1f} mm{lb * 1e3:11.1f} mm")
        print(f"  현재 프리셋 std: along {P.JAW_ALONG_STD * 1e3:.0f} mm / "
              f"lateral {P.JAW_LATERAL_STD * 1e3:.0f} mm")
        for lab, v, std in (("평균 상태", (am, lm), None), ("최선 상태", (ab, lb), None)):
            q = (1 - _m.tanh(v[0] / P.JAW_ALONG_STD)) * (1 - _m.tanh(v[1] / P.JAW_LATERAL_STD))
            print(f"    {lab} straddle 품질 = {q:.5f}")
        _ah = _st.mean(sd_axis_h)
        print(f"  {'턱 중점 축방향 높이':<14}{_ah * 1e3:9.1f} mm  (컵 원점 기준. 파지점은 "
              f"{-44.6:.1f} mm, 컵 상단은 +{(0.175 - 0.09209) * 1e3:.0f} mm)")
        if _ah > (0.175 - 0.09209):
            print("     ★★턱이 **컵 상단보다 위** = 축 연장선(허공)을 감싸고 있다 — 보상 구멍!")
        print(f"  {'enclose (턱 양쪽)':<14}{_st.mean(sd_enclose):9.3f}    "
              f"(1 = 두 손가락이 컵 축을 사이에 둠, 0 = 주먹)")
        print(f"  ★cup_between_jaws 항 = {_st.mean(sd_term):.4f} "
              f"→ 보상 기여 {_st.mean(sd_term) * P.BETWEEN_JAWS_REWARD_WEIGHT:.3f}"
              f" (상한 {P.BETWEEN_JAWS_REWARD_WEIGHT:.1f})")
        print("  → 항 값이 0 에 가까우면 gradient 가 없다(test10/test11 과 같은 함정).")

    if grip_series:
        import statistics as _st
        ng = len(grip_series)
        closed = sum(1 for g in grip_series if g < 0.005) / ng
        opened = sum(1 for g in grip_series if g > 0.040) / ng
        print("\n=== 그리퍼가 실제로 무엇을 하는가 ===")
        print(f"  개도  최소 {min(grip_series) * 1e3:5.1f} · 평균 {_st.mean(grip_series) * 1e3:5.1f}"
              f" · 최대 {max(grip_series) * 1e3:5.1f} mm  (완전닫힘 0 ~ 완전열림 44)")
        print(f"  거의 닫힘(<5 mm) 스텝 {closed:.1%} · 거의 열림(>40 mm) 스텝 {opened:.1%}")
        print(f"  '열기' 지령을 낸 스텝 {_st.mean(grip_cmd):.1%}")
        print("  → 지령이 계속 '열기'면 **닫을 생각이 없는 것**이고, 개도가 컵 지름 근처에서")
        print("     멈춰 있으면 **닫다가 컵에 막힌 것**(= 실제로 물고 있다)이다.")

    if act_delta:
        mode = "태스크공간 diff-IK" if "_ik" in TASK else "관절 위치 델타"
        print(f"\n=== 진동 진단 (제어 = {mode}) ===")
        print(f"  1차 차분 |Δa|          {sum(act_delta) / len(act_delta):.4f}"
              f"   ← action_rate 가 벌하는 양 (범위 ±1)")
        if act_jerk:
            print(f"  2차 차분 |Δ²a| (jerk)  {sum(act_jerk) / len(act_jerk):.4f}"
                  f"   ← 레퍼런스에 **없는** 항")
            print(f"  방향 반전 비율          {sum(act_flips) / len(act_flips):.1%}"
                  f"   (일정 방향 이동이면 0%, 진동이면 50% 근처)")
        if tgt_delta:
            d = sum(tgt_delta) / len(tgt_delta)
            print(f"  ★적용된 관절 목표 변화  {d:.5f} rad/스텝 = {d / 0.02:.2f} rad/s"
                  f"   (관절 속도 한계 2.175~2.61)")
        print(f"  팔 관절 속도            {sum(arm_speed) / len(arm_speed):.3f} rad/s")
        if late_lin:
            print(f"  목표 10 cm 이내에서 컵 선속도 {sum(late_lin) / len(late_lin):.3f} m/s"
                  f"   (도달 후에도 이 값이 크면 **멈추지 못하는 것**)")
        print("  → 액션 변화가 크면 정책이 목표를 계속 바꾸는 것이고, 팔 속도만 크면"
              " 목표는 안정한데 추종이 흔들리는 것이다.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
