"""grasp_lift_fabric 스모크 probe — 학습 전에 물리로 확인해야 하는 것들.

플랜 §5 probe 1/2/3/6 을 한 번에 잰다:
  1. 자산 계약  — 관절/바디 해석 결과 vs 프로필 선언, Fabrics num_joints
  2. 홈 자세    — 리셋 직후 palm pose, 물체가 밀리는지 (★1스텝 후 캡처: 버퍼 stale)
  3. Fabrics 추종 — 고정 목표에 대한 정상상태 오차(78mm 이력 재현 여부)
  6. zero-action + fps

사용:
    isaaclab.sh -p scripts/reinforcement_learning/probes/probe_grasp_lift_fabric_smoke.py \
        --task open-bis_r_grasp_lift_fab --num_envs 16 --steps 150
"""
from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app = AppLauncher(args).app

import gymnasium as gym          # noqa: E402
import torch                     # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks             # noqa: E402,F401  (자동 등록)


def banner(t):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}", flush=True)


# train.py 와 같은 경로로 cfg 를 만들어 넘긴다(gym.make 만으로는 cfg 가 안 채워진다).
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
p = env.profile

banner("1. 자산 계약")
print(f"  프로필      : {p.name}  (자산 {p.asset.name}, 태그 {p.asset.tag})")
print(f"  팔 관절     : 해석 {len(env.arm_ids):2d} / 선언 {p.num_arm_joints:2d}")
print(f"  손 관절     : 해석 {len(env.hand_ids):2d} / 선언 {p.num_hand_joints:2d}")
print(f"  손가락      : {len(env._fingers)}  {env._fingers}")
print(f"  Fabrics     : {p.fabric_class}({p.fabric_robot_dir}) num_joints={env.fabric.num_joints}")
print(f"  물체 뱅크   : {env.bank.name} ({len(env.bank)}종)")
print(f"  액션/관측   : {env.cfg.action_space} / {env.cfg.observation_space} "
      f"(critic {env.cfg.state_space})")

# 로봇 전 DOF 가 actuator 로 덮였는지 (커버리지 누락 = 조용한 free-spin)
n_dof = env.robot.num_joints
covered = sum(len(a.joint_indices) for a in env.robot.actuators.values())
print(f"  actuator 커버: {covered}/{n_dof} " + ("OK" if covered == n_dof else "★누락!"))

obs, _ = env.reset()
env.step(torch.zeros(args.num_envs, env.cfg.action_space, device=env.device))  # 버퍼 갱신

# ★접촉 센서가 살아 있는지 — 죽은 센서로 학습하면 보상 7항 중 6항이 상시 0 이다.
banner("1b. 접촉 센서 생존 (손만 완전 폐합)")
_p = env.robot.data.body_pos_w[:, env.palm_idx]
_t = env.robot.data.body_pos_w[:, env._tip_t].mean(dim=1)
_r = torch.zeros(args.num_envs, 13, device=env.device)
_r[:, :3] = 0.5 * (_p + _t); _r[:, 3] = 1.0
env.object.write_root_state_to_sim(_r)
_a = torch.zeros(args.num_envs, env.cfg.action_space, device=env.device)
for _i in range(60):
    _a[:, 6:] = min(1.0, _i / 24)
    env.step(_a)
_f, _ = env._contact()
print(f"  손 완전 폐합 시 접촉력 최대 {_f.max():.3f} N · >1N 손가락 "
      f"{( _f > 1.0).float().sum(dim=1).mean():.2f}")
print(f"  → {'PASS' if _f.max() > 0.1 else '★FAIL — 접촉 필터 prim 을 확인할 것'}")
env.reset(); env.step(torch.zeros_like(_a))

banner("2. 홈 자세 (★리셋 1스텝 후 — 위치 버퍼가 stale 하다)")
palm_w = env.robot.data.body_pos_w[:, env.palm_idx] - env.scene.env_origins
obj_w = env.object.data.root_pos_w - env.scene.env_origins
tips = env.robot.data.body_pos_w[:, env._tip_t] - env.scene.env_origins[:, None, :]
print(f"  palm (env-local) : {palm_w[0].tolist()}")
print(f"  손끝 z 범위      : {tips[:, :, 2].min():.4f} ~ {tips[:, :, 2].max():.4f}")
print(f"  물체 위치        : {obj_w[0].tolist()}")
print(f"  스폰 대비 이동   : {(obj_w[:, :2] - env.object_spawn_pos[:, :2]).norm(dim=-1).max():.4f} m "
      "(★0 에 가까워야 한다 — 홈 팔이 스폰 박스를 점유하면 컵이 밀린다)")
print(f"  palm 박스 lo     : {env.palm_lo[0].tolist()}")
print(f"  palm 박스 hi     : {env.palm_hi[0].tolist()}")
print(f"  액션 기준점(a=0 → 홈) : {[f'{v:.3f}' for v in env.home_palm[0, :3].tolist()]}")
print(f"  홈 실측과의 차이       : {(palm_w[0] - env.home_palm[0, :3]).norm():.4f} m "
      "(0 이어야 정상 — a=0 이 곧 '홈 유지')")

banner("3~6. zero-action 롤아웃")
zero = torch.zeros(args.num_envs, env.cfg.action_space, device=env.device)
t0 = time.time()
qd_max, perr_hist, obj_move = 0.0, [], []
for i in range(args.steps):
    obs, rew, term, trunc, info = env.step(zero)
    qd = env.robot.data.joint_vel[:, env._arm_t].abs().max().item()
    qd_max = max(qd_max, qd)
    pw = env.robot.data.body_pos_w[:, env.palm_idx] - env.scene.env_origins
    perr_hist.append((env.palm_targets[:, :3] - pw).norm(dim=-1).mean().item())
    ow = env.object.data.root_pos_w - env.scene.env_origins
    obj_move.append((ow[:, :2] - env.object_spawn_pos[:, :2]).norm(dim=-1).mean().item())
dt = time.time() - t0
fps = args.steps * args.num_envs / dt

print(f"  스텝 {args.steps} × env {args.num_envs} = {dt:.1f}s → {fps:,.0f} fps "
      f"(2048 env 환산 {fps / args.num_envs * 2048:,.0f})")
print(f"  팔 관절속도 최대  : {qd_max:.2f} rad/s  (종료 임계 20)")
print(f"  Fabrics palm 오차 : 초기 {perr_hist[0]:.4f} → 최종 {perr_hist[-1]:.4f} m")
print("    ↑ ★정상상태 오차. pour_v1 에서 78mm 이력이 있다. 크면 워크스페이스 박스를 넓힌다.")
print(f"  물체 xy 이동      : 최종 {obj_move[-1]:.4f} m (zero-action 이므로 0 이어야 정상)")
# ★작업면 정합: 컵이 어디에 안착하는지 **재서** 확인한다(자산 상면 계산과 대조).
_rest = (env.object.data.root_pos_w - env.scene.env_origins)[:, 2]
print(f"  컵 안착 z (실측)  : 평균 {_rest.mean():.4f}  범위 {_rest.min():.4f}~{_rest.max():.4f}")
_fall = (env.object_spawn_pos[:, 2] - _rest).mean().item()
print(f"    스폰 z 평균 {env.object_spawn_pos[:, 2].mean():.4f} · 계산 안착 z "
      f"{env._object_rest_z.mean():.4f} · 실제 낙하량 {_fall:+.4f} m")
print("    ↑ ★낙하량이 크면 lift 보상 기준선이 그만큼 어긋난다(기준선 = 스폰 z).")
print(f"  보상 평균         : {rew.mean().item():+.4f}")
for k in ("task/contact_gate", "task/envelope_frac", "obj/xy_displacement",
          "obj/tilt_deg", "fabric/palm_err_mean", "task/respawn_rate"):
    v = info.get("log", info).get(k, env.extras.get(k))
    if v is not None:
        print(f"    {k:24s} = {float(v):+.4f}")

env.close()
app.close()
