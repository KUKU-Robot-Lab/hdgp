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
        lifted = cup[:, 2] > P.MINIMAL_LIFT_HEIGHT
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
        print(f"  스폰 원점 z {P.CUP_SPAWN_Z:.5f} · 리프트 게이트 {P.MINIMAL_LIFT_HEIGHT:.5f}")
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
