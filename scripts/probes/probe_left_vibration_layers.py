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

"""컵의 진동이 **어느 층에서 생기는가**를 분해 측정한다.

사용자 질문(그대로): "action 이 진동해서 fabric 이 진동하고 제어가 진동하는 건지?
아니면 action 은 고정되고 있는데 fabric 과 제어가 그걸 못 따라가는 건지?"
답에 따라 jerk 항의 존폐가 갈린다 — 전자면 정책을 벌하는 게 맞고, 후자면 이미
매끄러운 신호를 벌하는 헛일이라 제거하고 게인·리미터를 손봐야 한다.

측정하는 다섯 층 (각각 1차차분 RMS · 2차차분 RMS · 방향반전율):
  ① a_raw      정책 raw 액션 6D          — 정책이 내는 지령 자체
  ② cmd_pos    리미터 통과 후 palm 위치   — 실제로 fabric 에 들어간 목표
  ③ fab_q      fabric 관절 목표 7D        — fabric 이 만든 궤적
  ④ q          실제 팔 관절 7D            — PD 제어가 따라간 결과
  ⑤ cup        컵 world 위치 3D           — 최종 결과

추적 오차도 함께 본다:
  |cmd_pos − TCP|   지령 → 실제 도달 (fabric+제어 전체의 추종 오차)
  |fab_q − q|       fabric 목표 → 실제 관절 (PD 만의 추종 오차)

★결정적 실험 — 액션 동결(freeze):
  `--freeze_at N` 스텝 이후 정책 출력을 **그 시점 값으로 고정**한다. 리미터 때문에
  지령은 곧 정지하므로, 그 뒤에도 컵이 계속 흔들리면 원인은 제어·fabric 층이고
  잠잠해지면 원인은 정책 층이다. 이게 사용자 질문에 대한 직접 답이다.

실행:
    PYTHONUNBUFFERED=1 TERM=xterm ./isaaclab.sh -p \
        scripts/probes/probe_left_vibration_layers.py \
        --task open-grip_l_grasp_sensor_fab --checkpoint <path.pth> \
        --num_envs 32 --steps 250 --freeze_at 150
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--freeze_at", type=int, default=150,
                    help="이 스텝부터 정책 출력을 동결한다. 0 이면 동결하지 않는다.")
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor_fab")
# ★바닥값의 원인을 프로브에서 직접 끈다(학습 코드는 건드리지 않는다).
parser.add_argument("--no_gravity_comp", action="store_true",
                    help="중력보상 적분항을 끈다 — 이산 적분기 한계주기 가설 검증")
parser.add_argument("--arm_damping", type=float, default=None, help="팔 PD damping 덮어쓰기")
parser.add_argument("--arm_stiffness", type=float, default=None, help="팔 PD stiffness 덮어쓰기")
parser.add_argument("--rate_limit", type=float, default=None,
                    help="PALM_CMD_RATE_LIMIT 덮어쓰기 — 천장 측정 시 리미터를 풀어야 한다")
parser.add_argument("--vel_ff", type=float, default=None,
                    help="fabric 속도 피드포워드 배율 덮어쓰기 (0=옛 배선, 1=DEXTRAH 원본)")
parser.add_argument("--fabric_damping", type=float, default=None,
                    help="fabric 적분기 damping 덮어쓰기 — 잔류속도 바닥값의 유력 후보")
# ★교란 제거: 학습값(fd=20)으로 동일한 파지 상태까지 굴린 **뒤에** damping 을 바꾼다.
#   시작부터 바꾸면 정책이 다른 곳에 도달해 fd 값끼리 비교가 성립하지 않는다(실측 확인).
parser.add_argument("--freeze_damping", type=float, default=None,
                    help="동결 시점에 fabric damping 을 이 값으로 교체")
parser.add_argument("--ramp_at", type=int, default=0,
                    help="이 스텝부터 먼 목표를 지령해 리미터를 포화시킨다 — 지속 지령 상한 측정")
parser.add_argument("--step_at", type=int, default=0,
                    help="이 스텝에 지령을 계단 입력한다 — 응답성 비용 측정")
parser.add_argument("--step_delta", type=float, default=0.15,
                    help="계단 입력 크기(정규화 액션 단위)")
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


class Chan:
    """한 층의 시계열을 받아 1차/2차 차분과 방향반전율을 누적한다.

    ⚠ 리셋 직후 스텝은 버린다 — 텔레포트가 만든 거대한 차분이 평균을 삼킨다.
      (이 트랙에서 리셋 오염에 네 번 당했다.)
    """

    def __init__(self, name: str, unit: str):
        self.name, self.unit = name, unit
        self.prev = None
        self.prev_d = None
        self.d1 = []
        self.d2 = []
        self.rev = []

    def push(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        x = x.detach()
        if self.prev is not None:
            d = x - self.prev
            sel = d if mask is None else d[mask]
            if sel.numel():
                self.d1.append(float(sel.norm(dim=-1).mean()))
            if self.prev_d is not None:
                dd = d - self.prev_d
                s2 = dd if mask is None else dd[mask]
                if s2.numel():
                    self.d2.append(float(s2.norm(dim=-1).mean()))
                dot = (d * self.prev_d).sum(-1)
                sr = dot if mask is None else dot[mask]
                if sr.numel():
                    self.rev.append(float((sr < 0).float().mean()))
            self.prev_d = d
        self.prev = x.clone()

    def drop_last(self, n: int) -> None:
        """리셋 경계에서 오염된 표본을 뒤에서부터 지운다."""
        for lst in (self.d1, self.d2, self.rev):
            del lst[-n:]

    def report(self, window: slice = slice(None)) -> str:
        def m(lst):
            w = lst[window]
            return sum(w) / len(w) if w else float("nan")
        return (f"  {self.name:<10} |Δ|={m(self.d1):9.5f} {self.unit}"
                f"   |Δ²|={m(self.d2):9.5f}   방향반전={m(self.rev) * 100:5.1f}%")


def main() -> None:
    if args.no_gravity_comp:
        P.GRAVITY_COMP_ENABLED = False
    if args.fabric_damping is not None:
        P.FABRIC_DAMPING_GAIN = args.fabric_damping
    if args.vel_ff is not None:
        P.FABRIC_VEL_FF_SCALE = args.vel_ff
    if args.rate_limit is not None:
        P.PALM_CMD_RATE_LIMIT = args.rate_limit
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    if args.arm_damping is not None:
        env_cfg.scene.robot.actuators["left_arm"].damping = args.arm_damping
    if args.arm_stiffness is not None:
        env_cfg.scene.robot.actuators["left_arm"].stiffness = args.arm_stiffness
    # ★프로브가 리셋 오염으로 죽지 않도록 에피소드를 길게 잡는다(이 트랙의 상습 함정).
    env_cfg.episode_length_s = 1.0e6
    agent_cfg = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")

    env = gym.make(TASK, cfg=env_cfg)
    raw = env.unwrapped
    inf = float("inf")
    wrapped = RlGamesVecEnvWrapper(
        env, args.device,
        agent_cfg["params"]["env"].get("clip_observations", inf),
        agent_cfg["params"]["env"].get("clip_actions", inf),
    )
    vecenv.register("IsaacRlgWrapper", lambda cfg_name, n, **kw: RlGamesGpuEnv(cfg_name, n, **kw))
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped}
    )
    agent_cfg["params"]["config"]["env_info"] = {
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
    arm_ids, _ = robot.find_joints([f"l_aj_{i}" for i in range(1, 8)], preserve_order=True)
    base_i = robot.body_names.index(P.GRIPPER_BASE_BODY)
    act_term = raw.action_manager.get_term("arm_action")

    def tensor(o):
        return o["obs"] if isinstance(o, dict) else o

    obs = tensor(wrapped.reset())
    agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    ch = {
        "a_raw":   Chan("①a_raw", "  "),
        "cmd_pos": Chan("②cmd_pos", "m "),
        "fab_q":   Chan("③fab_q", "rad"),
        "q":       Chan("④q", "rad"),
        "cup":     Chan("⑤cup", "m "),
    }
    track_cmd_tcp = []      # |cmd_pos − TCP|  (m)
    track_fab_q = []        # |fab_q − q|      (rad)
    cup_speed = []          # 컵 선속도 (m/s)
    droop_n = []            # 중력보상 적분항 노름 (rad) — 포화하면 상수로 눕는다
    qd_n = []               # 팔 관절 속도 노름 (rad/s)
    ramp_tcp, ramp_cmd = [], []   # 램프 구간 TCP/지령 스텝당 변위 (m)
    prev_tcp = None
    anchor: dict = {}
    tcp_speed = []          # TCP 선속도 (m/s) — 팔 자체가 떠는가
    lifted_frac = []
    near_frac, near_cmd_tcp, near_cup_v, near_tcp_v, near_cmd_step = [], [], [], [], []
    prev_cmd = None
    frozen_action = None

    with torch.inference_mode():
        for t in range(args.steps):
            # ★ADR 커리큘럼이 리셋 때마다 vel_ff_scale 을 레벨 0 값으로 덮어쓴다.
            #   override 를 매 스텝 다시 강제하지 않으면 A/B 가 같은 값이 된다(실제로 당했다).
            if args.vel_ff is not None:
                act_term.vel_ff_scale = args.vel_ff
            act = agent.get_action(obs, is_deterministic=True)
            if args.freeze_at and t >= args.freeze_at:
                if frozen_action is None:
                    frozen_action = act.clone()
                    if args.freeze_damping is not None:
                        act_term._damping[:] = args.freeze_damping
                if args.step_at and t == args.step_at:
                    frozen_action[:, 0] += args.step_delta
                if args.ramp_at and t == args.ramp_at:
                    # ★자매 트랙 probe_armscale 과 같은 방법 — 지령이 리미터 상한으로
                    #   **계속** 움직이게 두고 실제 TCP 변위율을 본다. 계단 첨두(전이)가
                    #   아니라 **지속 가능한** 속도라, 리미터를 정할 때 필요한 값은 이쪽이다.
                    frozen_action[:, 0] = -1.0   # 박스 한쪽 끝 = 항상 멀리 있는 목표
                act = frozen_action
            obs, _, _, _ = wrapped.step(act)
            obs = tensor(obs)

            q = robot.data.joint_pos[:, arm_ids]
            base_p = robot.data.body_pos_w[:, base_i] - raw.scene.env_origins
            base_R = matrix_from_quat(robot.data.body_quat_w[:, base_i])
            tcp = base_p + base_R[:, :, 2] * P.TCP_OFFSET_IN_BASE_Z
            cup = obj.data.root_pos_w - raw.scene.env_origins
            lifted = cup[:, 2] > (P.CUP_SPAWN_Z + 0.03)
            lifted_frac.append(float(lifted.float().mean()))
            m = lifted if bool(lifted.any()) else None
            # ★목표 근처만 따로 본다 — 이송 중에 지령이 앞서 나가는 것은 정상이고,
            #   문제가 되는 것은 "멈춰야 할 때 못 멈추는" dwell 구간뿐이다.
            goal_w = raw.command_manager.get_command("object_pose")[:, :3]
            near = lifted & ((cup - goal_w).norm(dim=-1) < 0.10)
            near_frac.append(float(near.float().mean()))
            if bool(near.any()):
                near_cmd_tcp.append(float((act_term._prev_cmd_pos - tcp).norm(dim=-1)[near].mean()))
                near_cup_v.append(float(obj.data.root_lin_vel_w.norm(dim=-1)[near].mean()))
                near_tcp_v.append(float(robot.data.body_lin_vel_w[:, base_i].norm(dim=-1)[near].mean()))
                near_cmd_step.append(float((act_term._prev_cmd_pos - prev_cmd).norm(dim=-1)[near].mean())
                                     if prev_cmd is not None else float("nan"))
            prev_cmd = act_term._prev_cmd_pos.clone()

            ch["a_raw"].push(act_term.raw_actions, m)
            ch["cmd_pos"].push(act_term._prev_cmd_pos, m)
            ch["fab_q"].push(act_term._fabric_q, m)
            ch["q"].push(q, m)
            ch["cup"].push(cup, m)

            sel = lifted if bool(lifted.any()) else torch.ones_like(lifted)
            track_cmd_tcp.append(float((act_term._prev_cmd_pos - tcp).norm(dim=-1)[sel].mean()))
            track_fab_q.append(float((act_term._fabric_q - q).norm(dim=-1)[sel].mean()))
            cup_speed.append(float(obj.data.root_lin_vel_w.norm(dim=-1)[sel].mean()))
            droop_n.append(float(act_term._droop.norm(dim=-1)[sel].mean())
                           if hasattr(act_term, "_droop") else 0.0)
            qd_n.append(float(robot.data.joint_vel[:, arm_ids].norm(dim=-1)[sel].mean()))
            if prev_tcp is not None:
                ramp_tcp.append(float((tcp - prev_tcp).norm(dim=-1)[sel].mean()))
                ramp_cmd.append(float(act_term.cmd_step_norm[sel].mean()))
            prev_tcp = tcp.clone()
            # ★순변위 대조 — 순간속도가 크고 순변위가 작으면 "이동"이 아니라 스텝 이하
            #   주파수의 진동이고, 그때는 잔류속도 기준 자체가 그 진동을 재는 셈이 된다.
            if t == args.freeze_at + 50:
                anchor["tcp"], anchor["cup"] = tcp.clone(), cup.clone()
            if t == args.steps - 1 and "tcp" in anchor:
                anchor["net_tcp"] = float((tcp - anchor["tcp"]).norm(dim=-1)[sel].mean())
                anchor["net_cup"] = float((cup - anchor["cup"]).norm(dim=-1)[sel].mean())
                anchor["n"] = args.steps - 1 - (args.freeze_at + 50)
            tcp_speed.append(
                float(robot.data.body_lin_vel_w[:, base_i].norm(dim=-1)[sel].mean())
            )

    F = args.freeze_at
    # Chan 은 첫 스텝에 표본을 못 만들므로 인덱스가 스텝보다 2 작다. 여유 있게 자른다.
    pre = slice(20, max(21, F - 5))
    post = slice(F + 5, None) if F else slice(0, 0)

    def avg(lst, s):
        w = lst[s]
        return sum(w) / len(w) if w else float("nan")

    print("\n" + "=" * 78)
    print(f"진동 층 분해 — {args.checkpoint.split('/')[-1]}")
    print(f"gravity_comp={'OFF' if args.no_gravity_comp else 'ON'} "
          f"kp={args.arm_stiffness or P.ARM_IK_STIFFNESS} kd={args.arm_damping or P.ARM_IK_DAMPING} "
          f"fabric_damping={P.FABRIC_DAMPING_GAIN} "
          f"vel_ff(실효)={act_term.vel_ff_scale} rate_limit={P.PALM_CMD_RATE_LIMIT}")
    print(f"envs={args.num_envs} steps={args.steps} freeze_at={F}  "
          f"리프트 비율 {avg(lifted_frac, slice(20, None)) * 100:.1f}%")
    print("=" * 78)
    print("\n[A] 정책 구동 구간 (스텝 20~%d) — 리프트된 env 만" % max(21, F - 5))
    for c in ch.values():
        print(c.report(pre))
    print(f"\n  추종오차 |cmd−TCP| = {avg(track_cmd_tcp, pre) * 1000:7.2f} mm")
    print(f"  추종오차 |fab_q−q| = {avg(track_fab_q, pre) * 1000:7.2f} mrad")
    print(f"  컵 속도            = {avg(cup_speed, pre):7.4f} m/s")
    print(f"  TCP 속도           = {avg(tcp_speed, pre):7.4f} m/s")

    def navg(lst, lo, hi):
        w = [v for v in lst[lo:hi] if v == v]
        return sum(w) / len(w) if w else float("nan")

    print(f"\n[A2] ★목표 10cm 이내(dwell 구간)만 — 표본 비율 "
          f"{avg(near_frac, slice(20, None)) * 100:.1f}%")
    hi = max(21, F - 5) if F else None
    print(f"  추종오차 |cmd−TCP| = {navg(near_cmd_tcp, 20, hi) * 1000:7.2f} mm")
    print(f"  지령 이동 |Δcmd|   = {navg(near_cmd_step, 20, hi) * 1000:7.2f} mm/step "
          f"(리미터 상한 {P.PALM_CMD_RATE_LIMIT * 1000:.0f})")
    print(f"  컵 속도            = {navg(near_cup_v, 20, hi):7.4f} m/s")
    print(f"  TCP 속도           = {navg(near_tcp_v, 20, hi):7.4f} m/s")

    if args.ramp_at:
        R = args.ramp_at
        lo, hi = R + 15, R + 65      # 전이 15스텝 버리고 50스텝(=1.0s) 창
        if len(track_cmd_tcp) > hi:
            cmd_rate = navg(ramp_cmd, lo, hi) * 1000.0
            tcp_rate = navg(ramp_tcp, lo, hi) * 1000.0
            print(f"\n[F] 지속 램프 (스텝 {lo}~{hi}, 리미터 {P.PALM_CMD_RATE_LIMIT * 1000:.1f} mm/step)")
            print(f"      지령 이동율  {cmd_rate:6.3f} mm/step")
            print(f"      실제 TCP 변위율 {tcp_rate:6.3f} mm/step")
            print(f"      ★달성률      {tcp_rate / cmd_rate * 100 if cmd_rate else float('nan'):5.1f} %")

    if "net_tcp" in anchor:
        n = anchor["n"]
        print(f"\n[E] 순변위 대조 (동결+50 ~ 끝, {n} 스텝 = {n * 0.02:.1f} s)")
        print(f"      TCP 순변위 {anchor['net_tcp'] * 1000:7.2f} mm  "
              f"→ 평균 {anchor['net_tcp'] / (n * 0.02) * 1000:6.2f} mm/s   "
              f"(순간속도 {avg(tcp_speed, slice(-50, None)) * 1000:.1f} mm/s)")
        print(f"      컵  순변위 {anchor['net_cup'] * 1000:7.2f} mm  "
              f"→ 평균 {anchor['net_cup'] / (n * 0.02) * 1000:6.2f} mm/s   "
              f"(순간속도 {avg(cup_speed, slice(-50, None)) * 1000:.1f} mm/s)")
        print(f"      팔 관절속도 |qd| = {avg(qd_n, slice(-50, None)):.4f} rad/s")

    if args.step_at:
        S = args.step_at
        base = avg(track_cmd_tcp, slice(S - 20, S))
        print(f"\n[D] 계단응답 (스텝 {S}, Δa={args.step_delta}, "
              f"fabric_damping={args.freeze_damping or P.FABRIC_DAMPING_GAIN})")
        print(f"      계단 직전 |cmd−TCP| = {base * 1000:6.2f} mm")
        peak = max(track_cmd_tcp[S:S + 10]) if len(track_cmd_tcp) > S + 10 else float("nan")
        print(f"      계단 직후 최대 오차 = {peak * 1000:6.2f} mm")
        settled = None
        for k in range(S, len(track_cmd_tcp)):
            if track_cmd_tcp[k] <= base + 0.005:
                settled = k - S
                break
        print(f"      base+5mm 복귀까지  = "
              f"{settled if settled is not None else '미복귀'} 스텝")
        print(f"      최종 |cmd−TCP|     = {avg(track_cmd_tcp, slice(-20, None)) * 1000:6.2f} mm")
        print(f"      최종 컵 속도       = {avg(cup_speed, slice(-20, None)):7.4f} m/s")
        print(f"      최종 TCP 속도      = {avg(tcp_speed, slice(-20, None)):7.4f} m/s")
        # ★리미터 값을 정하려면 "팔이 실제로 낼 수 있는 속도"가 필요하다.
        #   큰 계단을 주고 그 직후 최대 TCP 속도를 본다 = 이 제어기의 실현 가능 상한.
        win = tcp_speed[S:S + 40]
        cwin = cup_speed[S:S + 40]
        print(f"      계단 직후 최대 TCP 속도 = {max(win) if win else float('nan'):7.4f} m/s"
              f"  → 리미터 환산 {max(win) * 0.02 * 1000 if win else 0:6.2f} mm/step")
        print(f"      계단 직후 최대 컵 속도  = {max(cwin) if cwin else float('nan'):7.4f} m/s")

    if F:
        print(f"\n[B] 액션 동결 구간 (스텝 {F + 5}~{args.steps}) — 정책 지령 고정")
        for c in ch.values():
            print(c.report(post))
        print(f"\n  추종오차 |cmd−TCP| = {avg(track_cmd_tcp, post) * 1000:7.2f} mm")
        print(f"  추종오차 |fab_q−q| = {avg(track_fab_q, post) * 1000:7.2f} mrad")
        print(f"  컵 속도            = {avg(cup_speed, post):7.4f} m/s")
        print(f"  TCP 속도           = {avg(tcp_speed, post):7.4f} m/s")
        print("\n[C] 동결 후 감쇠 곡선 (10스텝 평균) — 꼬리인가 바닥인가")
        print("      스텝   컵속도    TCP속도   |cmd−TCP|  |droop|")
        for lo in range(F, args.steps, 10):
            hi = min(lo + 10, args.steps)
            print(f"    {lo:4d}~{hi - 1:<4d} {avg(cup_speed, slice(lo, hi)):7.4f}  "
                  f"{avg(tcp_speed, slice(lo, hi)):7.4f}   "
                  f"{avg(track_cmd_tcp, slice(lo, hi)) * 1000:7.2f}mm  "
                  f"{avg(droop_n, slice(lo, hi)) * 1000:7.2f}mrad")
        print("\n[판정] 동결 구간에서 ⑤cup 의 |Δ²| 와 컵 속도가")
        print("       · 크게 줄면  → 진동의 원인은 **정책**(jerk 항 유효)")
        print("       · 그대로면  → 원인은 **fabric/제어**(jerk 항 무효, 게인·리미터 문제)")
    print("=" * 78 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
