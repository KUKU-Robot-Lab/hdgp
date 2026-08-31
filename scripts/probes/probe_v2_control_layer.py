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

"""제어 층 진단 — 정책은 **고정**하고 fabric·지령 배선만 잰다.

★★08.29 사용자 가설 두 개를 한 롤아웃에서 가른다.

  **H1 — cspace 앵커의 위치가 틀렸다.**
    `default_config` 는 모든 env 공용 홈 하나이고, 그 FK palm 은
        (0.171, −0.101, 0.185)
    로 **액션 박스 y[0.10,0.43] 바깥**(−201 mm)이다. 컵 스폰에서 374 mm,
    목표에서 486 mm. 즉 널스페이스 복원력이 **정책이 지령할 수조차 없는 방향**으로
    상시 걸리고, 그 세기는 자매 트랙의 3 배다(`conical_gain` 3.0 vs 1.0).
    ⇒ 리프트 후가 앵커에서 가장 먼 국면이라 이 토크가 최대가 된다.

  **H2 — palm 지령 *회전* 이 안 따라가서 TCP 가 목표를 못 본다.**
    `PALM_MAX_POSE_ANGLE = ±20°` 는 축별 **하드 클램프**다. 목표를 보려면 그보다
    더 돌려야 하는데 못 돌리면 정확히 이 증상이 된다.
    ⚠ **회전 축의 포화는 한 번도 로깅된 적이 없다** — `diag_act_*` 는 위치 축
      0·1·2 만 등록돼 있고 액션은 6 차원(위치 3 + 회전 3)이다. v1 에서 y 축 포화
      99.1% 로 죽은 이력이 있는데 회전에는 같은 계측이 없다.

무엇을 재는가 (전부 **리프트 전/후 국면 분리**):
  ① 위치 지령 → TCP 전달률 (전체 · 축별, 특히 z)
  ② 회전 지령 → 실제 palm 회전 오차, 회전 액션 축별 포화, 회전 리미터 포화
  ③ 앵커: 현재 palm ↔ 앵커 palm 거리 · 관절공간 거리
  ④ TCP 가 목표를 향하는가 (손바닥 법선과 목표 방향의 각도)

`HDGP_FABRIC_PARAMS` 로 조건을 바꿔 같은 정책에 대해 A/B 한다:
    openarm_gripper_left_pose_params.yaml     C0 현행 (conical_gain 3.0)
    openarm_gripper_left_pose_params_f1.yaml  C1 conical_gain 1.0 (자매 정합)
    openarm_gripper_left_pose_params_f2.yaml  C3 C1 + palm damping 50

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_v2_control_layer.py \
        --checkpoint <path.pth> --num_envs 1024 --steps 250
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor_v2")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from openarm.gripper.left.grasp_sensor_v2 import v2_preset as P  # noqa: E402
from openarm.gripper.left.grasp_sensor_v2 import v2_stages as S  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

TASK = args.task
DEG = 180.0 / math.pi


def _q(t: torch.Tensor, p: float) -> float:
    return torch.quantile(t.float(), p).item() if t.numel() else float("nan")


def _row(lab: str, pre: torch.Tensor, post: torch.Tensor, sc: float = 1.0,
         fmt: str = "8.3f") -> None:
    def s(t):
        return (f"{t.mean().item()*sc:{fmt}} (p50 {_q(t,0.5)*sc:{fmt}})"
                if t.numel() else "        —")
    print(f"  {lab:<32} 리프트前 {s(pre)}   리프트後 {s(post)}")


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")

    env = gym.make(TASK, cfg=env_cfg)
    raw = env.unwrapped
    inf = float("inf")
    wrapped = RlGamesVecEnvWrapper(
        env, args.device,
        agent_cfg["params"]["env"].get("clip_observations", inf),
        agent_cfg["params"]["env"].get("clip_actions", inf))
    vecenv.register("IsaacRlgWrapper", lambda cfg_name, n, **kw: RlGamesGpuEnv(cfg_name, n, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["config"]["env_info"] = {
        "observation_space": wrapped.observation_space,
        "action_space": wrapped.action_space, "agents": 1}
    hz = int(agent_cfg["params"]["config"].get("horizon_length", 24))
    agent_cfg["params"]["config"]["minibatch_size"] = args.num_envs * hz

    runner = Runner(); runner.load(agent_cfg)
    agent = runner.create_player(); agent.restore(args.checkpoint); agent.reset()

    dev = args.device
    obj = raw.scene["object"]
    ee = raw.scene["ee_frame"]
    robot_cfg = SceneEntityCfg("robot"); robot_cfg.resolve(raw.scene)
    obj_cfg = SceneEntityCfg("object"); obj_cfg.resolve(raw.scene)

    # ★액션 term 핸들 — 지령(processed_actions)·원액션·fabric 을 직접 읽는다.
    aterm = None
    for _n, _tm in raw.action_manager._terms.items():
        if hasattr(_tm, "processed_actions") and hasattr(_tm, "cmd_step_norm"):
            aterm = _tm
            print(f"[probe] action term = {_n} ({type(_tm).__name__})")
            break
    if aterm is None:
        raise RuntimeError("palm fabric 액션 term 을 못 찾았다")
    fab = aterm._fabric
    anchor_q = fab.default_config[:, :7].clone()
    print(f"[probe] fabric params = {P.FABRIC_PARAMS_FILENAME}")
    print(f"[probe] 앵커 관절값(env0) = {[round(v,4) for v in anchor_q[0].tolist()]}")
    print(f"[probe] 앵커가 env 마다 다른가: "
          f"{bool((anchor_q - anchor_q[0:1]).abs().max() > 1e-9)}")

    rec = {k: [] for k in ("mask_post", "cmd_d", "tcp_d", "cmd_dz", "tcp_dz",
                           "rot_err", "a_rot_sat", "rot_rate_sat", "rot_box_sat",
                           "anchor_qd", "face_ang", "lim_sat",
                           # ★축별 **부호 있는** 회전 액션 — 한쪽 쏠림(중심 오류) vs
                           #   양쪽 포화(범위 부족)를 가른다. 처방이 완전히 갈린다.
                           "a3", "a4", "a5", "e3", "e4", "e5")}

    def _t(o):
        return o["obs"] if isinstance(o, dict) else o

    obs = _t(wrapped.reset())
    agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    prev_cmd = None
    prev_tcp = None
    with torch.inference_mode():
        for _ in range(args.steps):
            act = agent.get_action(obs, is_deterministic=True)
            obs, _, _, _ = wrapped.step(act)
            obs = _t(obs)

            cmd = aterm.processed_actions            # (N,6) [xyz, ez,ey,ex]
            a = aterm.raw_actions                    # (N,6) 원액션
            tcp = ee.data.target_pos_w[..., 0, :] - raw.scene.env_origins
            cup_z = obj.data.root_pos_w[:, 2]
            post = cup_z > P.MINIMAL_LIFT_HEIGHT     # ★리프트 후 국면

            if prev_cmd is not None:
                dcmd = cmd[:, :3] - prev_cmd[:, :3]
                dtcp = tcp - prev_tcp
                rec["cmd_d"].append(dcmd.norm(dim=1))
                rec["tcp_d"].append(dtcp.norm(dim=1))
                rec["cmd_dz"].append(dcmd[:, 2])
                rec["tcp_dz"].append(dtcp[:, 2])
                # 회전 리미터 포화 — |Δeuler| 가 상한에 붙는 비율
                de = (cmd[:, 3:6] - prev_cmd[:, 3:6]).abs()
                rec["rot_rate_sat"].append(
                    (de >= P.PALM_ROT_RATE_LIMIT * 0.99).float().max(dim=1).values)
                rec["lim_sat"].append(
                    (aterm.cmd_step_norm >= P.PALM_CMD_RATE_LIMIT * 0.99).float())
                rec["mask_post"].append(post.clone())
            prev_cmd, prev_tcp = cmd.clone(), tcp.clone()

            if len(rec["mask_post"]) == 0:
                continue
            # 회전 액션 포화(축 3·4·5) — **한 번도 로깅된 적 없는 값**
            rec["a_rot_sat"].append((a[:, 3:6].abs() > 0.99).float().mean(dim=1))
            for i, k in enumerate(("a3", "a4", "a5")):
                rec[k].append(a[:, 3 + i].clamp(-1, 1).clone())
            # 실제 palm euler_zyx — 정책이 **실현한** 자세 범위(박스와 대조)
            e_act = _R_to_euler_zyx(matrix_from_quat(ee.data.target_quat_w[..., 0, :]))
            for i, k in enumerate(("e3", "e4", "e5")):
                rec[k].append((e_act[:, i] - aterm._euler_center[:, i]).clone())
            # euler 지령이 박스(±MAX_POSE_ANGLE) 경계에 붙은 비율
            off = (cmd[:, 3:6] - aterm._euler_center).abs()
            rec["rot_box_sat"].append(
                (off >= P.PALM_MAX_POSE_ANGLE * 0.99).float().max(dim=1).values)
            # 지령 회전 vs 실제 palm 회전 — 각도 오차
            R_cmd = _euler_zyx_to_R(cmd[:, 3:6])
            R_act = matrix_from_quat(ee.data.target_quat_w[..., 0, :])
            rec["rot_err"].append(_R_angle(R_cmd, R_act))
            # 앵커까지 관절공간 거리
            rec["anchor_qd"].append(
                (raw.scene["robot"].data.joint_pos[:, :7] - anchor_q).norm(dim=1))
            # 손바닥 법선이 목표를 향하는 각도
            goal = S.goal_pos_w(raw, "object_pose", robot_cfg) - raw.scene.env_origins
            to_goal = goal - tcp
            to_goal = to_goal / to_goal.norm(dim=1, keepdim=True).clamp(min=1e-6)
            rec["face_ang"].append(
                torch.acos((R_act[:, :, 0] * to_goal).sum(dim=1).clamp(-1, 1)))

    n = min(len(v) for v in rec.values() if v)
    M = torch.stack(rec["mask_post"][:n])
    def split(key):
        X = torch.stack(rec[key][:n])
        return X[~M], X[M]

    print("\n" + "=" * 96)
    print(f"제어 층 진단 — {args.num_envs} env · {args.steps} step · 결정론 · "
          f"fabric={P.FABRIC_PARAMS_FILENAME}")
    print("=" * 96)

    print("\n① 위치 지령 → TCP 전달")
    cd_pre, cd_post = split("cmd_d"); td_pre, td_post = split("tcp_d")
    _row("지령 이동량 (mm/step)", cd_pre, cd_post, 1000.0)
    _row("TCP 이동량 (mm/step)", td_pre, td_post, 1000.0)
    print(f"  {'★전달률 (TCP/지령)':<32} 리프트前 {(td_pre.mean()/cd_pre.mean()).item():8.3f}"
          f"              리프트後 {(td_post.mean()/cd_post.mean()).item():8.3f}")
    cz_pre, cz_post = split("cmd_dz"); tz_pre, tz_post = split("tcp_dz")
    _row("지령 z 증분 (mm/step)", cz_pre, cz_post, 1000.0)
    _row("TCP  z 증분 (mm/step)", tz_pre, tz_post, 1000.0)
    ls_pre, ls_post = split("lim_sat")
    _row("위치 리미터 포화율", ls_pre, ls_post)

    print("\n② 회전 — ★회전 축은 지금까지 계측이 없었다")
    ar_pre, ar_post = split("a_rot_sat")
    _row("회전 액션 |a|>0.99 비율", ar_pre, ar_post)
    rb_pre, rb_post = split("rot_box_sat")
    _row(f"euler 박스(±{P.PALM_MAX_POSE_ANGLE*DEG:.0f}°) 포화율", rb_pre, rb_post)
    rr_pre, rr_post = split("rot_rate_sat")
    _row("회전 리미터 포화율", rr_pre, rr_post)
    re_pre, re_post = split("rot_err")
    _row("지령↔실제 palm 회전 오차 (°)", re_pre, re_post, DEG)

    print("\n②b ★축별 부호 — 한쪽 쏠림(중심 오류)인가 양쪽 포화(범위 부족)인가")
    print(f"  {'축':>4} {'국면':>8} {'평균 a':>8} {'a=+1 비율':>10} {'a=−1 비율':>10}"
          f" {'실제 palm 편차(°)':>18}")
    for i, k in enumerate(("a3", "a4", "a5")):
        ap, aq = split(k); ep_, eq = split(("e3", "e4", "e5")[i])
        for lab, A, E in (("리프트前", ap, ep_), ("리프트後", aq, eq)):
            pos_f = (A > 0.99).float().mean().item()
            neg_f = (A < -0.99).float().mean().item()
            print(f"  {['ez','ey','ex'][i]:>4} {lab:>8} {A.mean().item():>8.3f}"
                  f" {pos_f:>10.3f} {neg_f:>10.3f}"
                  f" {E.mean().item()*DEG:>10.1f} (p50 {_q(E,0.5)*DEG:>5.1f})")
    print(f"  ⇒ 한쪽만 1.0 에 가까우면 **중심이 틀린 것**, 양쪽이 고르면 **범위가 좁은 것**.")
    print(f"     박스는 중심 ±{P.PALM_MAX_POSE_ANGLE*DEG:.0f}° 다.")

    print("\n③ cspace 앵커 (H1)")
    aq_pre, aq_post = split("anchor_qd")
    _row("앵커까지 관절거리 ‖q−q_a‖ (rad)", aq_pre, aq_post)

    print("\n④ TCP 가 목표를 향하는가 (H2)")
    fa_pre, fa_post = split("face_ang")
    _row("손바닥 법선↔목표 방향 각 (°)", fa_pre, fa_post, DEG)

    print("\n" + "=" * 96)
    print("PROBE_CONTROL_DONE")
    env.close()


def _R_to_euler_zyx(R: torch.Tensor) -> torch.Tensor:
    """R = Rz·Ry·Rx 의 역변환 → (N,3) [ez, ey, ex]. ey=±90° 부근은 퇴화하므로
    ★중심 자세(ey ≈ −85°)가 그 근처다 — 편차 해석 시 주의."""
    sy = (-R[:, 2, 0]).clamp(-1.0, 1.0)
    ey = torch.asin(sy)
    ez = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    ex = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    return torch.stack((ez, ey, ex), dim=1)


def _euler_zyx_to_R(e: torch.Tensor) -> torch.Tensor:
    """(N,3) [ez,ey,ex] → R = Rz·Ry·Rx. fabric 의 `euler_zyx` 규약과 같다."""
    cz, sz = torch.cos(e[:, 0]), torch.sin(e[:, 0])
    cy, sy = torch.cos(e[:, 1]), torch.sin(e[:, 1])
    cx, sx = torch.cos(e[:, 2]), torch.sin(e[:, 2])
    R = torch.zeros(e.shape[0], 3, 3, device=e.device, dtype=e.dtype)
    R[:, 0, 0] = cz * cy
    R[:, 0, 1] = cz * sy * sx - sz * cx
    R[:, 0, 2] = cz * sy * cx + sz * sx
    R[:, 1, 0] = sz * cy
    R[:, 1, 1] = sz * sy * sx + cz * cx
    R[:, 1, 2] = sz * sy * cx - cz * sx
    R[:, 2, 0] = -sy
    R[:, 2, 1] = cy * sx
    R[:, 2, 2] = cy * cx
    return R


def _R_angle(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """두 회전행렬 사이의 각 (rad)."""
    tr = torch.einsum("nij,nij->n", A, B)          # trace(AᵀB)
    return torch.acos(((tr - 1.0) * 0.5).clamp(-1.0, 1.0))


if __name__ == "__main__":
    main()
    simulation_app.close()
