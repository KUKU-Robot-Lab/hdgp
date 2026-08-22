"""Fabrics palm 추종을 **2층으로 분해**해 측정한다.

지금까지 잰 "추종 오차"는 전부 합성값이었다. 층이 다르면 고치는 노브가 다르다:

  L1 (fabric 내부) = FK(fabric_q) vs 지령 palm pose
      → attractor 수렴 오차. `openarm_gripper_left_pose_params.yaml` 의 palm_attractor
        게인과 cfg.fabrics_damping_gain 이 지배한다.
  L2 (물리 추종)   = 실제 TCP vs FK(fabric_q)
      → 관절 PD 추종 오차. 이 환경은 로봇 중력이 꺼져 있고(400/80) 거의 0 이어야 한다.
        여기서 크게 나오면 fabric 게인이 아니라 액추에이터·마찰 문제다.

★ 직전 scratchpad/fabric_hold.py 는 pregrasp 버퍼의 **위치 3성분만** 현재 TCP 로 덮어쓰고
  자세 3성분을 파지자세(-155, 75, 180)° 로 남겨두었다. palm_position_only=False 라 자세도
  attractor 에 들어가므로 fabric 은 "손목을 돌려라"를 정상 수행했고, 그때 따라온 TCP 병진을
  처짐(16.6 mm)으로 읽었다. 그 수치는 무효다. 여기서는 **6D 전체**를 지령한다.

사용:
  python scripts/probes/probe_fabric_tracking.py --mode hold
  python scripts/probes/probe_fabric_tracking.py --mode step --axis z   --delta 0.05
  python scripts/probes/probe_fabric_tracking.py --mode step --axis rot --delta 20
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=["hold", "step"], default="hold")
parser.add_argument("--axis", choices=["x", "y", "z", "rot"], default="z",
                    help="step 모드에서 계단을 줄 축. rot 은 euler_zyx 의 ez.")
parser.add_argument("--delta", type=float, default=0.05,
                    help="계단 크기. 위치축은 m, rot 은 deg.")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--no_cup", action="store_true",
                    help="컵을 멀리 치운다 — 로봇이 컵에 닿아 생기는 정적 접촉력 격리용.")
parser.add_argument("--no_world_mesh", action="store_true",
                    help="fabric world model 의 물체(컵) 반발을 끈다 — 편차원인 격리용.")
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from openarm.gripper.left.grasp_sensor_fabrics_ABORTED.grasp_left_env import (  # noqa: E402
    GraspLeftGripperEnv,
)
from openarm.gripper.left.grasp_sensor_fabrics_ABORTED.grasp_left_env_cfg import (  # noqa: E402
    GraspLeftGripperEnvCfg,
)
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402

cfg = GraspLeftGripperEnvCfg()
cfg.scene.num_envs = args.num_envs
env = GraspLeftGripperEnv(cfg)
env.reset()

_pidx = env.robot.body_names.index(P.GRIPPER_BASE_BODY)
_off = torch.tensor([0.0, 0.0, P.TCP_OFFSET_IN_BASE_Z], device=env.device).repeat(env.num_envs, 1)


def tcp_measured() -> torch.Tensor:
    """물리 시뮬레이션의 실제 TCP 위치 (로봇 base 프레임)."""
    return (
        env.robot.data.body_pos_w[:, _pidx, :]
        + quat_apply(env.robot.data.body_quat_w[:, _pidx, :], _off)
    ) - env.scene.env_origins


def palm_from_fabric() -> torch.Tensor:
    """fabric 자신의 FK 로 본 palm 6D pose (x,y,z, ez,ey,ex)."""
    return env.fabric.get_palm_pose(env.fabric_q.detach(), "euler_zyx")


n_act = env.cfg.action_space
zero = torch.zeros(env.num_envs, n_act, device=env.device)

# 버퍼가 채워지도록 한 스텝 굴린 뒤 기준 자세를 잡는다.
env.step(zero)
base_pose = palm_from_fabric().clone()          # 홈에서의 palm 6D
# ★자산 재빌드 때 홈 TCP 가 움직였는지 확인용. preset LEFT_HOME_TCP_POS 와 대조한다.
# 구 자산(08.21 재빌드 전)에서 실측한 홈 TCP. 자산 갱신 후 이동량을 보는 기준선.
_ref = torch.tensor((0.2391, 0.2443, 0.2947), device=env.device)
print(f"  홈 palm(fabric FK) {[round(v, 4) for v in base_pose[0, :3].tolist()]} · "
      f"preset 대비 {float((base_pose[0, :3] - _ref).norm())*1e3:.2f} mm · "
      f"물리 TCP 대비 {float((tcp_measured()[0] - base_pose[0, :3]).norm())*1e3:.2f} mm")

target = base_pose.clone()
if args.mode == "step":
    if args.axis == "rot":
        target[:, 3] += math.radians(args.delta)
    else:
        target[:, {"x": 0, "y": 1, "z": 2}[args.axis]] += args.delta

# 액션 0 이 곧 이 목표가 되도록 pregrasp 기준점을 **6D 전부** 덮어쓴다.
# (delta 범위가 대칭이라 scale(0, -d, +d) = 0 이므로 palm_pose == pregrasp 가 된다.)
env.pregrasp_palm_pose_buf[:] = target

accel_limit = torch.tensor(
    [7.5, 7.5, 10.0, 10.0, 10.0, 20.0, 20.0], device=env.device
)  # yaml joint_limits.acceleration

l1_hist, l2_hist, pos_hist, qdd_frac, qerr_hist = [], [], [], [], []
# ★에피소드 중간에 리셋된 env 는 fabric_q 가 홈으로 튀어 L1 을 오염시킨다(같은 설정
#   두 실행이 2.2 vs 13.3 으로 갈린 원인). episode_length_buf 가 되감기면 제외한다.
alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
_prev_len = env.episode_length_buf.clone()
n_reset = 0
for _ in range(args.steps):
    # 접촉 래치가 목표를 리프트 램프로 가로채지 못하게 막는다(순수 추종만 본다).
    env.lift_latched_buf[:] = False
    if args.no_cup:
        _cs = env.cup.data.root_state_w.clone()
        _cs[:, 0] = 5.0; _cs[:, 1] = 5.0; _cs[:, 7:] = 0.0
        env.cup.write_root_state_to_sim(_cs)
    if args.no_world_mesh:
        env.object_indicator.zero_()
    env.step(zero)

    fab = palm_from_fabric()
    cmd = env.palm_pose_targets
    meas = tcp_measured()

    alive &= env.episode_length_buf >= _prev_len
    _prev_len = env.episode_length_buf.clone()
    n_reset = int((~alive).sum())
    _a = alive.float()
    _n = _a.sum().clamp(min=1.0)
    # ★L2 를 프레임 무관하게 교차검증: 관절공간 오차. 여기가 0 인데 L2 가 크면
    #   추종 실패가 아니라 **TCP 프레임 정의 불일치**다.
    _qerr = (env.robot.data.joint_pos[:, env.arm_dof_indices] - env.fabric_q).abs().max(dim=1).values
    qerr_hist.append(float((_qerr * _a).sum() / _n))
    l1_hist.append(float(((fab[:, :3] - cmd[:, :3]).norm(dim=-1) * _a).sum() / _n))
    l2_hist.append(float(((meas - fab[:, :3]).norm(dim=-1) * _a).sum() / _n))
    # ★pos_hist·qdd 도 alive 마스크를 씌운다. 안 씌우면 홈으로 튄 리셋 env 가 평균을
    #   끌어내려 오버슈트 131%·정상상태오차 31.8 mm 같은 허수가 나온다(실제로 나왔다).
    # ★rot 은 euler ez = index 3 이다. .get(axis, 2) 로 두면 회전 계단인데 z **위치**를
    #   재게 되어 "회전 0, 오버슈트 216%" 같은 허수가 나온다(실제로 나왔다).
    _ax = {"x": 0, "y": 1, "z": 2, "rot": 3}[args.axis]
    pos_hist.append(float((fab[:, _ax] * _a).sum() / _n))
    qdd_frac.append(float(((env.fabric_qdd.abs() / accel_limit).amax(dim=1) * _a).sum() / _n))

tail = max(1, args.steps // 10)


def _mm(v):
    return v * 1e3


print(f"\n  리셋으로 제외된 env {n_reset} / {env.num_envs}")
print(f"\n=== Fabric 추종 2층 분해 · mode={args.mode}"
      + (f" axis={args.axis} delta={args.delta}" if args.mode == "step" else "")
      + f" · {args.steps} 스텝 ({args.steps/60:.1f} s) ===")
print(f"  L1  fabric 내부 (FK(fabric_q) vs 지령)   "
      f"평균 {_mm(sum(l1_hist)/len(l1_hist)):7.2f} mm · "
      f"정상상태 {_mm(sum(l1_hist[-tail:])/tail):7.2f} mm")
print(f"  L2  물리 추종  (실제 TCP vs FK(fabric_q)) "
      f"평균 {_mm(sum(l2_hist)/len(l2_hist)):7.2f} mm · "
      f"정상상태 {_mm(sum(l2_hist[-tail:])/tail):7.2f} mm")
_qdd_tail = sum(qdd_frac[-tail:]) / tail
print(f"  관절공간 오차 (physics q vs fabric_q, 최대관절)  "
      f"평균 {sum(qerr_hist)/len(qerr_hist)*1e3:7.2f} mrad · "
      f"정상상태 {sum(qerr_hist[-tail:])/tail*1e3:7.2f} mrad")
print(f"  fabric_qdd / accel 상한  최대 {max(qdd_frac)*100:5.1f} % · "
      f"정상상태 {_qdd_tail*100:5.1f} %"
      + ("   ← 정상상태에 상한 포화 = 한계주기(채터)" if _qdd_tail > 0.5 else ""))

# 시계열: 수렴/발산/진동 구분. 합계 하나로는 못 가른다.
_marks = [0, 9, 29, 59, 119, 199, args.steps - 1]
print("  L1 시계열 (mm)  " + "  ".join(
    f"{m+1}스텝={_mm(l1_hist[m]):.1f}" for m in _marks if m < len(l1_hist)))
print("  qdd 시계열 (%)  " + "  ".join(
    f"{m+1}스텝={qdd_frac[m]*100:.0f}" for m in _marks if m < len(qdd_frac)))

if args.mode == "step":
    start = pos_hist[0]
    final = sum(pos_hist[-tail:]) / tail
    span = final - start
    if abs(span) > 1e-9:
        # 90% 상승시간
        t90 = next((i for i, v in enumerate(pos_hist)
                    if (v - start) / span >= 0.9), None)
        peak = max(pos_hist, key=lambda v: (v - start) / span)
        overshoot = ((peak - start) / span - 1.0) * 100.0
        cmd_span = args.delta if args.axis != "rot" else math.radians(args.delta)
        sse = cmd_span - span
        print(f"  계단응답  90% 상승 "
              f"{('%d 스텝 (%.2f s)' % (t90, t90/60)) if t90 is not None else '미도달'}"
              f" · 오버슈트 {overshoot:5.1f} % · 정상상태오차 {_mm(sse):7.2f} "
              + ("mm" if args.axis != "rot" else "mrad"))

# ── 어느 항이 미는가 ───────────────────────────────────────────────
# hold 인데 L1 이 크면 fabric 이 자기 홈에서 평형이 아니라는 뜻이다. 후보는
# joint_limit_repulsion(한계 근처) 과 body_repulsion(engage_depth 0.30 m) 이다.
# 관절별 이동과 한계 여유를 같이 찍어 구분한다.
_q = env.fabric_q.detach()
_home = env.q_home_arm.unsqueeze(0)
_lo = env.robot.data.soft_joint_pos_limits[:, env.arm_dof_indices, 0]
_hi = env.robot.data.soft_joint_pos_limits[:, env.arm_dof_indices, 1]
_tq = env.robot.data.applied_torque[:, env.arm_dof_indices]
_lim = env.robot.data.joint_effort_limits[:, env.arm_dof_indices]
_qe = (env.robot.data.joint_pos[:, env.arm_dof_indices] - env.fabric_q).abs()
print("\n  관절별   추종오차(mrad)  토크/한계(%)   effort limit")
for _j in range(_qe.shape[1]):
    print(f"    l_aj_{_j+1}   {float(_qe[:, _j].mean())*1e3:10.1f}   "
          f"{float((_tq[:, _j].abs()/_lim[:, _j].clamp(min=1e-6)).mean())*100:8.1f}   "
          f"{float(_lim[:, _j].mean()):8.2f}")
print("\n  관절별 (rad)   홈에서 이동   하한여유   상한여유")
for _j in range(_q.shape[1]):
    print(f"    l_aj_{_j+1}      {float((_q[:, _j]-_home[:, _j]).mean()):+8.4f}   "
          f"{float((_q[:, _j]-_lo[:, _j]).mean()):8.4f}   "
          f"{float((_hi[:, _j]-_q[:, _j]).mean()):8.4f}")
try:
    _col = env.fabric.collision_status()
    print(f"  body_repulsion 충돌상태 최대침투 {float(_col.max()):.4f}")
except Exception as _e:
    print(f"  collision_status 사용불가: {_e}")

# ★손가락 접촉력 — 중력 OFF 인데 손목 토크가 정적으로 큰 원인이 접촉인지 확인한다.
try:
    _f = torch.stack([sn.data.net_forces_w[:, 0, :].norm(dim=-1) for sn in env._finger_sensors], dim=1)
    print(f"  손가락 접촉력  평균 {float(_f.mean()):.3f} N · 최대 {float(_f.max()):.3f} N · "
          f"접촉 env {int((_f.max(dim=1).values > 0.1).sum())}/{env.num_envs}")
except Exception as _e:
    print(f"  접촉센서 읽기 실패: {_e}")

print("\n  판정 기준: hold 는 L1·L2 < 2 mm. 계단은 오버슈트 < 5%, 정상상태오차 < 2 mm.")

env.close()
app.close()
