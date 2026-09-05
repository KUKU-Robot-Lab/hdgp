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

"""우팔 s2r env 의 actor obs(155D)와 **배포 코어**가 만든 obs 를 항목별로 대조한다.

★★왜 이렇게 나눠 재는가 (09.03). 좌팔에서 배운 것: 155 를 통째로 비교하면 "다르다"
  까지밖에 못 간다. 그래서 **항 경계로 잘라** 어디가 어긋나는지까지 짚는다.

  그리고 **FK 를 두 단계로 분리**한다:
    ①  `--fk sim`   — sim 의 body pose(palm·손끝)를 코어에 그대로 주입한다.
                      여기서 어긋나면 원인은 **관측 조립**이다.
    ②  `--fk fabric`— 배포가 실제로 쓸 Fabrics FK 를 주입한다.
                      ①이 맞고 ②만 어긋나면 원인은 **FK 프레임 차이**다.
  한 번에 재면 두 원인이 섞여 무엇도 확정할 수 없다.

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_s2r_obs_vs_deploy.py \\
        --checkpoint <g1_ep17000.pth> --run <sim2real/logs/policy/right_g1> --steps 3
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="open-sens_r_grasp_s2r-lstm")
parser.add_argument("--run", type=str,
                    default="/home/user/rl_ws/sim2real/logs/policy/right_g1")
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--object", default=None,
                    help="물체를 이 위치(x,y,z env-local)로 강제한다 — 실기 컵 좌표를 "
                         "넣어 '그 자리에서 정책이 쥐는가'를 sim 에서 판별한다")
parser.add_argument("--trace", action="store_true",
                    help="obs 대조 대신 게이트·폐쇄도·손액션을 스텝마다 찍는다")
parser.add_argument("--height_only", action="store_true",
                    help="sim 의 TCP·손끝 높이만 찍는다")
parser.add_argument("--fk", choices=("sim", "fabric"), default="sim",
                    help="코어에 주입할 FK — 'sim' 은 관측 조립만, 'fabric' 은 배포 경로")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.num_envs = 1

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

SIM2REAL = Path("/home/user/rl_ws/sim2real")
sys.path.insert(0, str(SIM2REAL / "scripts"))

#: 항 경계 — `grasp_s2r_obs_builder.SEGMENTS` 와 같아야 한다.
SLOTS = [
    ("arm_q", 0, 7), ("arm_qd", 7, 14), ("hand_q", 14, 34), ("hand_qd", 34, 54),
    ("palm_pos", 54, 57), ("palm_ax", 57, 63), ("tips_rel_palm", 63, 78),
    ("palm_to_obj", 78, 81), ("obj_to_tips", 81, 96), ("tip_force", 96, 111),
    ("joint_err", 111, 131), ("actions", 131, 152), ("goal_rel", 152, 155),
]


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg)
    raw = env.unwrapped

    inf = float("inf")
    wrapped = RlGamesVecEnvWrapper(
        env, args.device,
        agent_cfg["params"]["env"].get("clip_observations", inf),
        agent_cfg["params"]["env"].get("clip_actions", inf))
    vecenv.register("IsaacRlgWrapper",
                    lambda cfg_name, n, **kw: RlGamesGpuEnv(cfg_name, n, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["config"]["env_info"] = {
        "observation_space": wrapped.observation_space,
        "action_space": wrapped.action_space, "agents": 1}
    hz = int(agent_cfg["params"]["config"].get("horizon_length", 24))
    agent_cfg["params"]["config"]["minibatch_size"] = 1 * hz

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args.checkpoint)
    agent.reset()

    from grasp_s2r_core import GraspS2RCore, S2RSensors  # noqa: E402

    robot = raw.robot
    obj = raw.object
    origin = raw.scene.env_origins[0]

    def env_local(p):
        return (p - origin).detach().cpu().numpy().astype(float)

    # ── FK 주입 ─────────────────────────────────────────────────────────
    def fk_sim_palm(_q27):
        pos = env_local(robot.data.body_pos_w[0, raw.palm_idx])
        from isaaclab.utils.math import euler_xyz_from_quat
        r, p, y = euler_xyz_from_quat(robot.data.body_quat_w[:, raw.palm_idx])
        return np.array([*pos, float(y[0]), float(p[0]), float(r[0])])

    def fk_sim_tips(_q27):
        return env_local(robot.data.body_pos_w[0, raw._tip_ids_t])

    # ★★Fabrics 는 **자기 관절 순서**를 쓴다(`_fab_t`). DOF 순을 그대로 넘기면 손끝이
    #   최대 148 mm 어긋난다(09.03 실측). env 가 `_syn_to_fab_idx` 로 변환하는 이유다.
    def _q_fab():
        return robot.data.joint_pos[:, raw._fab_t].contiguous()

    def fk_fab_palm(_q27):
        with torch.inference_mode():
            return raw.fabric.get_palm_pose(
                _q_fab(), "euler_zyx")[0].cpu().numpy().astype(float)

    def fk_fab_tips(_q27):
        with torch.inference_mode():
            return raw.fabric.get_fingertip_positions(
                _q_fab())[0].cpu().numpy().astype(float)

    palm_fn, tips_fn = ((fk_sim_palm, fk_sim_tips) if args.fk == "sim"
                        else (fk_fab_palm, fk_fab_tips))

    # ★★관측 노이즈를 0 으로 — 안 끄면 잡음이 전 항에 섞여 **조립 오차와 구별되지
    #   않는다**(첫 측정에서 arm_q 0.019·qd 0.118 이 전부 잡음이었다).
    for _a in ("_adr_obs_noise_qpos", "_adr_obs_noise_qvel", "_adr_obs_noise_object"):
        if hasattr(raw, _a):
            setattr(raw, _a, 0.0)
    for _c in ("obs_noise_body", "obs_noise_qpos", "obs_noise_qvel", "obs_noise_object"):
        if hasattr(raw.cfg, _c):
            setattr(raw.cfg, _c, 0.0)
    lim = robot.data.soft_joint_pos_limits[0].detach().cpu().numpy()

    from grasp_s2r_synergy import HAND_JOINT_NAMES  # noqa: E402
    jn = list(robot.joint_names)
    prof_ids = [jn.index(nm) for nm in HAND_JOINT_NAMES]
    arm_ids = [jn.index(f"r_aj_{k}") for k in range(1, 8)]
    dof_ids = list(raw._hand_ids_t.detach().cpu().numpy())

    core = GraspS2RCore(
        policy=lambda o: np.zeros(21),
        fabric_palm_pose=palm_fn, fabric_tips=tips_fn,
        fabric_step=lambda p6, n=0: np.zeros(7),
        run_dir=args.run, goal3=np.zeros(3), soft_limits=lim[prof_ids],
    )

    def _t(o):
        return o["obs"] if isinstance(o, dict) else o

    obs = _t(wrapped.reset())
    agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    # ★goal 은 **리셋 뒤**에 읽어야 한다 — 그 전에는 0 이라 `goal_rel` 이 통째로 틀린다.
    goal3 = raw.goal_pos[0].detach().cpu().numpy().astype(float)
    core.goal3 = goal3
    q = robot.data.joint_pos[0].detach().cpu().numpy().astype(float)
    core.reset(arm_q=q[arm_ids], hand_q=q[dof_ids],
               object_pos=env_local(obj.data.root_pos_w[0]))
    print(f"[probe] FK={args.fk} · goal {np.round(goal3, 4).tolist()} · "
          f"r_cage {core._r_cage:.4f}", flush=True)

    # ── 손끝 FK 격차 진단 — sim body(센서팁) vs Fabrics taskmap ──────────────
    _q27 = np.concatenate([q[arm_ids], q[dof_ids]])
    _sim_t = fk_sim_tips(_q27)
    _fab_t = fk_fab_tips(_q27)
    _names = [robot.body_names[i] for i in raw.tip_ids]
    print("[probe] 손끝 비교 (sim body ↔ fabric taskmap)", flush=True)
    for _k in range(5):
        _d = _fab_t[_k] - _sim_t[_k]
        print(f"   {_names[_k]:>22} sim {np.round(_sim_t[_k], 4).tolist()} "
              f"fab {np.round(_fab_t[_k], 4).tolist()} "
              f"Δ {np.linalg.norm(_d) * 1000:6.1f} mm", flush=True)
    # 순서 뒤바뀜 가능성 — 교차 거리행렬의 최근접 대응을 본다.
    _dm = np.linalg.norm(_fab_t[:, None, :] - _sim_t[None, :, :], axis=-1)
    print(f"   최근접 대응(fab→sim): {_dm.argmin(axis=1).tolist()} "
          f"· 대각 {np.round(np.diag(_dm) * 1000, 1).tolist()} mm", flush=True)

    with torch.inference_mode():
        for step in range(args.steps):
            q = robot.data.joint_pos[0].detach().cpu().numpy().astype(float)
            qd = robot.data.joint_vel[0].detach().cpu().numpy().astype(float)
            # ★손끝 힘은 sim 이 world 로 갖고 있고 팁 자세로 로컬 변환한다 —
            #   배포 빌더와 같은 입력을 주려면 world 힘 + 팁 quat 을 그대로 넘긴다.
            f_w = []
            for k, finger in enumerate(raw._finger_names):
                s = raw._finger_sensors[finger][-1]
                f_w.append(s.data.force_matrix_w.view(1, -1, 3).sum(dim=1)[0])
            f_w = torch.stack(f_w).detach().cpu().numpy().astype(float)
            tq = robot.data.body_quat_w[0, raw.tip_ids].detach().cpu().numpy().astype(float)

            # ★sim 이 **이 obs 로** 낼 액션을 먼저 뽑아 코어에도 그대로 먹인다.
            #   안 그러면 코어의 시너지 목표가 갈라져 `joint_err` 이 인공적으로 어긋난다.
            act = agent.get_action(obs, is_deterministic=True)
            act_np = act[0].detach().cpu().numpy().astype(float)
            core.policy = lambda _o, _a=act_np: _a

            mine = core.step(S2RSensors(
                arm_q=q[arm_ids], arm_qd=qd[arm_ids],
                hand_q=q[dof_ids], hand_qd=qd[dof_ids],
                object_pos=env_local(obj.data.root_pos_w[0]),
                tip_force_world=f_w, tip_quat=tq,
            )).obs
            sim = obs[0].detach().cpu().numpy().astype(float)

            if args.trace:
                _sc = raw._syn_close[0].detach().cpu().numpy()
                _mv = raw._syn_movable.detach().cpu().numpy()
                _obj = env_local(obj.data.root_pos_w[0])
                _palm = env_local(robot.data.body_pos_w[0, raw.palm_idx])
                _pt = raw.palm_targets[0].detach().cpu().numpy()
                print(f" [{step:4d}] 팔액션 " +
                      " ".join(f"{v:+5.2f}" for v in act_np[:6]) +
                      f" · palm목표 {np.round(_pt[:3], 3).tolist()}"
                      f" · gate {float(raw._close_gate[0]):.2f}"
                      f" · 케이지거리 {float(raw._cage_ctr_dist[0])*1000:5.0f} mm"
                      f" · 폐쇄 {_sc[_mv].mean():.3f}"
                      f" · 손액션평균 {act_np[6:].mean():+.3f}"
                      f" · palm→컵 {np.linalg.norm(_obj-_palm)*1000:5.0f} mm"
                      f" · |joint_err| 평균 {np.abs(sim[111:131]).mean():.3f}"
                      f" 최대 {np.abs(sim[111:131]).max():.3f}", flush=True)
                obs, _, _, _ = wrapped.step(act)
                obs = _t(obs)
                continue

            print(f"\n=== step {step} ===", flush=True)
            print(f"{'항':>16} {'슬롯':>10}  {'최대차':>10}   sim / 배포")
            for nm, a, b in SLOTS:
                d = np.abs(sim[a:b] - mine[a:b])
                k = int(d.argmax())
                flag = "  ★" if d.max() > 1e-3 else ""
                print(f"{nm:>16} [{a:3d}:{b:3d}] {d.max():10.6f}   "
                      f"{sim[a+k]:+.5f} / {mine[a+k]:+.5f}{flag}")

            obs, _, _, _ = wrapped.step(act)
            obs = _t(obs)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
