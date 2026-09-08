"""thumb_1 이 관절 한계를 넘는가 — 접촉 없는 자유공간에서 먼저 잰다.

왜 이 프로브가 있나. fj_c1/c2/c3 재생에서 `r_hj_thumb_1` 지령은 하한 −0.384 에 붙어
멈춰 있는데 실측이 −4.07 rad 까지 갔다(한계 밖 93%). URDF·USD 한계는 둘 다 정상이라
"한계가 없다"와 "접촉이 한계를 뚫는다"가 남는다. 컵이 손 안에 있으면 둘이 섞이므로,
손끝이 컵에서 98mm 떨어진 리셋 자세에서 **엄지만** 하한으로 밀어 자유공간 추종을 본다.

    isaaclab.sh -p .../probe_fj_thumb1_limit.py --steps 200
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-sens_r_grasp_fj-play-lstm-sapg")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--joint", default="r_hj_thumb_1")
parser.add_argument("--mode", default="alone", choices=["alone", "close", "beyond"],
                    help="alone=이 관절만 하한 · close=손 전체 닫기 · beyond=액션한계를 넓혀 하드한계 **밖** 목표를 준다")
parser.add_argument("--self_collision", action="store_true",
                    help="enable_self_collisions=True 로 다시 켜고 잰다(기본 False)")
parser.add_argument("--armature", type=float, default=None,
                    help="손 액추에이터 armature(회전자 관성). SimToolReal 은 0.00012~0.0032, 우리는 없음")
parser.add_argument("--joint_friction", type=float, default=None,
                    help="손 관절 마찰. SimToolReal 은 0.0038~0.132, 우리는 없음")
parser.add_argument("--vel_iters", type=int, default=None,
                    help="solver_velocity_iteration_count 덮어쓰기(기본 0)")
parser.add_argument("--max_depen", type=float, default=None,
                    help="max_depenetration_velocity 덮어쓰기(기본 1000)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym   # noqa: E402
import torch              # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks      # noqa: E402,F401

cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
cfg.seed = args.seed
if args.self_collision:
    # robot_cfg 는 파생 구조라 값만 바꾸면 no-op — 재조립 훅을 반드시 다시 부른다.
    cfg.enable_self_collisions = True
    if hasattr(cfg, "finalize_after_overrides"):
        cfg.finalize_after_overrides()
_ha = cfg.robot_cfg.actuators.get("hand")
if args.armature is not None and _ha is not None:
    _ha.armature = args.armature
if args.joint_friction is not None and _ha is not None:
    _ha.friction = args.joint_friction
_sp = cfg.robot_cfg.spawn
if args.vel_iters is not None:
    _sp.articulation_props.solver_velocity_iteration_count = args.vel_iters
if args.max_depen is not None:
    _sp.rigid_props.max_depenetration_velocity = args.max_depen
print(f"[PROBE] solver pos/vel iters = "
      f"{_sp.articulation_props.solver_position_iteration_count}/"
      f"{_sp.articulation_props.solver_velocity_iteration_count} · "
      f"max_depen = {_sp.rigid_props.max_depenetration_velocity} · "
      f"self_coll = {_sp.articulation_props.enabled_self_collisions} · "
      f"hand armature = {getattr(_ha, 'armature', None)} · "
      f"friction = {getattr(_ha, 'friction', None)}", flush=True)
env = gym.make(args.task, cfg=cfg)
raw = env.unwrapped
env.reset()

lo, hi = raw._act_lo, raw._act_hi
_names0 = [raw.robot.data.joint_names[i] for i in raw._syn_ids]
if args.mode == "beyond":
    # ★한계가 실제로 걸려 있는지 가르는 유일한 방법. 지금까지의 시험은 목표가 늘
    #   한계 안(clamp)이라 "한계가 막은 것"과 "목표가 거기까지였던 것"을 못 가른다.
    _jb = _names0.index(args.joint)
    raw._act_lo = raw._act_lo.clone(); raw._act_lo[_jb] = -2.5
    raw._act_span = raw._act_hi - raw._act_lo
    lo, hi = raw._act_lo, raw._act_hi
q0 = raw._hand_reset_q                       # 리셋 시드(액션한계로 clamp 된 값)
a_hold = 2.0 * (q0 - lo) / (hi - lo) - 1.0   # 그 자세를 유지하는 액션
names = [raw.robot.data.joint_names[i] for i in raw._syn_ids]
j = names.index(args.joint)

act = torch.zeros((args.num_envs, raw.cfg.action_space), device=raw.device)
if args.mode == "close":
    # 굴곡 관절(_2/_3/_4)은 상한이 굽힘 방향 — 전부 +1 로 닫는다.
    act[:, 7:] = 1.0
else:
    act[:, 7:] = a_hold.unsqueeze(0)
act[:, 7 + j] = -1.0                         # 대상 관절은 하한으로

hard = raw.robot.data.joint_pos_limits[0, raw._syn_ids, :]
print(f"[PROBE] {args.joint}: 액션한계 [{lo[j]:+.3f},{hi[j]:+.3f}] · "
      f"하드한계 [{hard[j,0]:+.3f},{hard[j,1]:+.3f}] · 리셋 {q0[j]:+.3f}", flush=True)

_viol_max = 0.0
_viol_tail = []
for t in range(args.steps):
    env.step(act)
    _q = raw.robot.data.joint_pos[:, raw._syn_ids][:, j]
    _v = (hard[j, 0] - _q).clamp(min=0.0)          # 하한을 넘어간 양(rad)
    _viol_max = max(_viol_max, float(_v.max()))
    if t >= args.steps - 100:
        _viol_tail.append(float(_v.mean()))
    if t % 20 == 0 or t == args.steps - 1:
        q = raw.robot.data.joint_pos[:, raw._syn_ids][:, j]
        tg = raw._syn_target[:, j]
        ft = getattr(raw, "_start_ft_last", None)
        print(f"  t={t:4d}  target={tg.mean():+.4f}  q_mean={q.mean():+.4f}  "
              f"q_min={q.min():+.4f}  q_max={q.max():+.4f}  "
              f"위반={(q < hard[j,0] - 1e-3).float().mean():.2f}", flush=True)

import statistics as _st
print(f"[RESULT] seed={args.seed} n={args.num_envs} | 최대이탈={_viol_max:.4f} rad · "
      f"말미100 평균이탈={_st.mean(_viol_tail):.4f} rad", flush=True)
env.close()
app.close()
