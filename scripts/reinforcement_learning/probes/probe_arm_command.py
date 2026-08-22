"""팔 지령 분석 — 정책이 palm 목표를 **얼마나 빨리** 움직이고, 팔이 그중 얼마를 따라가나.

fab_test9 실측: zero-action 추종은 8mm 인데 학습 중 palm 오차가 평균 200mm · p95 290mm.
제어기는 정상이므로 원인은 지령이다. 다만 `action/arm_step_delta`(0.6)는 위치 3축과
자세 3축을 **섞은 평균**이라 무엇이 문제인지 못 가른다.

또 매핑이 비대칭이다 — `home_palm` 이 박스 중심이 아니라서 액션 단위당 이동량이
방향마다 최대 7.5배 다르다(y: a=+1 이 0.300m, a=-1 이 0.040m).

Part A: 학습된 정책 롤아웃 — 축별 |Δa|, 지령 palm 이동[mm/step, deg/step], 실제 추종률
Part B: 추종 대역 — 목표를 여러 속도로 계단 이동시켜 팔이 따라오는 한계를 잰다

    isaaclab.sh -p .../probe_arm_command.py --checkpoint <path> [--self_collisions --gravity]
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--checkpoint", default=None, help="없으면 Part B 만")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--self_collisions", action="store_true")
parser.add_argument("--gravity", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry   # noqa: E402
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper                    # noqa: E402

import openarm.tasks         # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
from openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg import resolve_cfg  # noqa: E402
env_cfg.enable_self_collisions = bool(args.self_collisions)
env_cfg.enable_gravity = bool(args.gravity)
resolve_cfg(env_cfg)
print(f"[probe] self_collisions={args.self_collisions} · gravity={args.gravity}")

agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
_env = gym.make(args.task, cfg=env_cfg)
raw = _env.unwrapped
AX = ["x", "y", "z", "ez", "ey", "ex"]

# ---- 매핑 비대칭 (액션 단위당 물리 이동량) ---------------------------------------
lo, hi, hm = raw.palm_lo[0], raw.palm_hi[0], raw.home_palm[0]
print("\n" + "=" * 78)
print("액션 단위당 이동량 (home 이 박스 중심이 아니라 방향마다 다르다)")
print("=" * 78)
print(f"{'축':>4s}{'a=-1 쪽':>12s}{'a=+1 쪽':>12s}{'비대칭':>9s}")
for k, n in enumerate(AX):
    dn, up = float(hm[k] - lo[k]), float(hi[k] - hm[k])
    u = "m" if k < 3 else "rad"
    print(f"{n:>4s}{dn:10.3f}{u:>2s}{up:10.3f}{u:>2s}{max(up, dn)/max(min(up, dn), 1e-9):8.1f}x")

# ================= Part A : 정책 롤아웃 =========================================
if args.checkpoint:
    from rl_games.torch_runner import Runner                       # noqa: E402
    from rl_games.common import env_configurations, vecenv         # noqa: E402
    env = RlGamesVecEnvWrapper(_env, args.device,
                               agent_cfg["params"]["env"].get("clip_observations", 5.0),
                               agent_cfg["params"]["env"].get("clip_actions", 1.0))
    vecenv.register("IsaacRlgWrapper", lambda cn, ne, **kw: env)
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: env})
    runner = Runner()
    agent_cfg["params"]["config"]["num_actors"] = args.num_envs
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args.checkpoint)
    agent.has_batch_dimension = True
    agent.batch_size = args.num_envs
    agent.reset()

    def _t(o):
        if isinstance(o, tuple):
            o = o[0]
        return o["policy"] if isinstance(o, dict) and "policy" in o else (
            o["obs"] if isinstance(o, dict) and "obs" in o else o)

    obs = _t(env.reset())
    prev_a = None
    prev_tgt = None
    da = torch.zeros(6, device=raw.device)
    dtgt = torch.zeros(6, device=raw.device)
    dpalm = torch.zeros(3, device=raw.device)
    err = []
    n = 0
    SKIP = args.steps // 4
    for i in range(args.steps):
        with torch.no_grad():
            act = agent.get_action(obs, is_deterministic=True)
        obs = _t(env.step(act)[0])
        tgt = raw.palm_targets.clone()
        palm = raw.robot.data.body_pos_w[:, raw.palm_idx] - raw.scene.env_origins
        if prev_a is not None and i >= SKIP:
            da += (raw.actions[:, :6] - prev_a).abs().mean(0)
            dtgt += (tgt - prev_tgt).abs().mean(0)
            dpalm += (palm - prev_palm).abs().mean(0)
            err.append((tgt[:, :3] - palm).norm(dim=-1).mean().item())
            n += 1
        prev_a = raw.actions[:, :6].clone()
        prev_tgt, prev_palm = tgt, palm.clone()

    print("\n" + "=" * 78)
    print(f"Part A · 정책 지령 (결정론적, 마지막 {n} 스텝, 1스텝 = 16.7ms)")
    print("=" * 78)
    print(f"{'축':>4s}{'|Δa|':>8s}{'지령 이동/스텝':>16s}{'실제 palm 이동':>16s}{'추종률':>8s}")
    for k, nm in enumerate(AX):
        u, sc = ("mm", 1000.0) if k < 3 else ("deg", 57.2958)
        cmd = float(dtgt[k] / n) * sc
        act_m = float(dpalm[k] / n) * 1000.0 if k < 3 else float("nan")
        r = (act_m / cmd) if k < 3 and cmd > 1e-6 else float("nan")
        print(f"{nm:>4s}{float(da[k]/n):8.3f}{cmd:12.1f}{u:>4s}"
              + (f"{act_m:12.1f}mm{r:8.2f}" if k < 3 else " " * 24))
    print(f"\n  위치 지령 속도 합 {sum(float(dtgt[k]/n) for k in range(3))*1000*60:8.0f} mm/s"
          f"   (zero-action 추종 8mm 기준 제어기는 정상)")
    print(f"  palm 추종 오차 평균 {sum(err)/len(err)*1000:.1f} mm")

# ================= Part B : 추종 대역 ===========================================
print("\n" + "=" * 78)
print("Part B · 추종 대역 — 목표를 일정 속도로 밀 때 팔이 따라오는가")
print("=" * 78)
raw.reset()
zero = torch.zeros(args.num_envs, raw.cfg.action_space, device=raw.device)
for _ in range(40):
    raw.step(zero)
base = raw.home_palm.clone()
print(f"{'속도 mm/s':>10s}{'정상상태 오차 mm':>18s}{'지연 mm':>10s}")
for speed in (20.0, 50.0, 100.0, 200.0, 500.0, 1000.0):
    raw.reset()
    for _ in range(30):
        raw.step(zero)
    errs = []
    for i in range(90):                      # 1.5s
        t = (i + 1) / 60.0
        d = min(speed * t / 1000.0, 0.12)    # +x 로 최대 12cm
        tg = base.clone(); tg[:, 0] += d
        raw.palm_targets = tg
        raw.fabric.set_features(raw._fabric_hand_cmd, raw.palm_targets, "euler_zyx",
                                raw.fabric_q.detach(), raw.fabric_qd.detach(),
                                raw._world_ids, raw._world_indicator, raw._fabric_damping)
        raw._step_fabric()
        for _ in range(raw.cfg.decimation):
            raw._apply_action()
            raw.scene.write_data_to_sim()
            raw.sim.step(render=False)
            raw.scene.update(dt=raw.physics_dt)
        p = raw.robot.data.body_pos_w[:, raw.palm_idx] - raw.scene.env_origins
        errs.append((tg[:, :3] - p).norm(dim=-1).mean().item())
    print(f"{speed:10.0f}{errs[-1]*1000:16.1f}{max(errs)*1000:12.1f}")
print("\n  ★정책 지령 속도가 이 표의 어디에 있는지가 판정이다.")
_env.close()
app.close()
