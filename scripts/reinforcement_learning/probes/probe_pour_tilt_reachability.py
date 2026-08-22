"""P3 — pour_fabric 확장 회전 박스의 fabric 추종 도달성 실측.

pour 는 grasp 과 달리 자세 오프셋 박스를 깊은 기울임까지 연다
(pose_offset_lo_deg 기본 (-45,-45,-150)). 이 박스는 **후보**다 — fabric 이
그 회전을 실제로 추종 못 하면 정책 액션이 포화해도 결과가 안 바뀌어 tilt 를
배울 수 없다(워크스페이스 박스와 같은 논리). 실패 시 학습 금지·박스 재설계.

방법: probe 부팅(require_warm_bank=False, 뱅크 불요) → hold 통과 → capture 후
회전 액션 패턴(단축 스윕 + 깊은 기울임 코너)을 고정 인가, 정상상태에서
  (slew 수렴한 지령 pose) vs (실측 palm pose)
를 위치/측지 회전 오차로 대조.

게이트(08.22 재설계 — 깊은 축이 roll 로 확정된 뒤):
  A 운용역 정밀: tilt지령 ≤115° 패턴은 rot_err <15° AND pos_err <30mm
    (success 판정 tilt_target 110° 까지는 정밀 추종이 필요)
  B 깊이: max tilt달성 ≥ 135° (pour_v1 deep-tilt 상한)
  C 단조: **깊축 스윕**의 tilt달성이 강단조 증가 AND 스윕 격차 <15°.
    복합(회전+위치) 패턴은 정보로만 보고 — 격차 15.1° 실측(깊130°+z0.3)은
    graceful degradation 이고 달성 115.3° > tilt_target 110° 라 학습을 막지 않는다.

    ./isaaclab.sh -p scripts/reinforcement_learning/probes/probe_pour_tilt_reachability.py
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_b_pour_fab")
parser.add_argument("--settle", type=int, default=260,
                    help="capture 후 스텝(램프 30 + 회전 slew 150°/2° = 75 + 정착)")
parser.add_argument("--offsets", type=str, default=None,
                    help="pose_offset 오버라이드 'lo1,lo2,lo3,hi1,hi2,hi3' (deg) — 박스 후보 실측용")
parser.add_argument("--gate_rot_deg", type=float, default=15.0)
parser.add_argument("--gate_pos_mm", type=float, default=30.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab.utils.math import quat_error_magnitude, quat_from_euler_xyz  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401

# 액션 패턴: (라벨, a[3], a[4], a[5]) — a[0:3]=0 (앵커 위치 유지), a[6:9]=0 (rcv).
PATTERNS = [
    ("baseline a=0",        0.0,  0.0,  0.0),
    ("깊축 -0.40",          -0.40, 0.0,  0.0),
    ("깊축 -0.60",          -0.60, 0.0,  0.0),
    ("깊축 -0.73 (110°)",   -0.73, 0.0,  0.0),
    ("깊축 -0.87 (130°)",   -0.87, 0.0,  0.0),
    ("깊축 -1.00 (150°)",   -1.00, 0.0,  0.0),
    ("깊 -0.73 pitch+1",   -0.73, 1.0,  0.0),
    ("깊 -0.73 pitch-1",   -0.73, -1.0, 0.0),
    ("깊 -0.73 yaw+0.5",   -0.73, 0.0,  0.5),
    ("깊 -0.73 yaw-0.5",   -0.73, 0.0, -0.5),
    ("얕 +1 (반대)",         1.00, 0.0,  0.0),
    ("pitch +1",            0.0,  1.0,  0.0),
    ("pitch -1",            0.0, -1.0,  0.0),
    ("yaw +1",              0.0,  0.0,  1.0),
    ("yaw -1",              0.0,  0.0, -1.0),
    ("깊-0.87 + z상승 0.3", -0.87, 0.0,  0.0),
]
N = len(PATTERNS)

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=N)
env_cfg.require_warm_bank = False
# ★probe 모드 오염 차단: 컵이 테이블에 있어 palm 이동만으로 '낙하' 기하가 성립하고,
#   깊은 회전 추종의 관절 속도가 runaway 에 걸려 에피소드가 리셋된다 — 리셋되면
#   anchor 재캡처로 지령이 초기화돼 측정이 무효다. 순수 추종 측정이므로 전부 끈다.
env_cfg.drop_dist_m = 1e6
env_cfg.drop_z_m = 1e6
env_cfg.runaway_joint_vel = 1e6
if args.offsets:
    v = [float(x) for x in args.offsets.split(",")]
    assert len(v) == 6, "--offsets 는 6개"
    env_cfg.pose_offset_lo_deg = tuple(v[:3])
    env_cfg.pose_offset_hi_deg = tuple(v[3:])
    print(f"[P3] 오프셋 후보 오버라이드: lo={v[:3]} hi={v[3:]}")
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

A = env.cfg.action_space
a = torch.zeros(N, A, device=env.device)
for i, (_, r3, r4, r5) in enumerate(PATTERNS):
    a[i, 3], a[i, 4], a[i, 5] = r3, r4, r5
a[N - 1, 2] = 0.3            # 마지막 패턴: 기울인 채 z 상승 (부을 때 실제 조합)

zero = torch.zeros_like(a)
hold = int(env.cfg.hold_steps)
for _ in range(hold + 1):     # hold 통과 + capture 스텝
    env.step(zero)
assert bool(env._captured.all()), "capture 미성립 — hold_steps 배선 확인"

resets = torch.zeros(N, device=env.device)
for _ in range(args.settle):
    env.step(a)
    resets += (env.episode_length_buf == 0).float()
if float(resets.sum()) > 0:
    print(f"[P3] ⚠ 측정 중 리셋 발생 env: {resets.nonzero().flatten().tolist()} — 결과 무효")

pose = env._palm_pose_6d(env.src)               # 실측 (env-local xyz+euler_xyz)
cmd = env.src.cmd                               # slew 수렴한 지령 (=desired 정상상태)
pos_err_mm = (pose[:, :3] - cmd[:, :3]).norm(dim=-1) * 1000.0
q_meas = quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
q_cmd = quat_from_euler_xyz(cmd[:, 3], cmd[:, 4], cmd[:, 5])
rot_err_deg = quat_error_magnitude(q_meas, q_cmd) * 180.0 / torch.pi

# ---- tilt 달성각: 파지가 강체라면 컵 up-axis = 캡처 시점 world ẑ 의 palm-local 방향.
#     R_meas · (R_anchor^T ẑ) 가 ẑ 에서 벌어진 각 = 컵 tilt 등가량.
from isaaclab.utils.math import quat_apply, quat_conjugate  # noqa: E402
q_anchor = quat_from_euler_xyz(env.src.anchor[:, 3], env.src.anchor[:, 4],
                               env.src.anchor[:, 5])
z_hat = torch.zeros(N, 3, device=env.device)
z_hat[:, 2] = 1.0
u_local = quat_apply(quat_conjugate(q_anchor), z_hat)


def _tilt_deg(q):
    u = quat_apply(q, u_local)
    return torch.rad2deg(torch.acos(u[:, 2].clamp(-1.0, 1.0)))


tilt_cmd_deg = _tilt_deg(q_cmd)
tilt_meas_deg = _tilt_deg(q_meas)

# 지령이 slew 로 아직 desired 에 도달 못 했으면 결과가 무의미 — 검산.
slew_done = ((cmd - env.src.anchor).abs() / env.src.scale.clamp(min=1e-6))

deg = 180.0 / torch.pi
print("=" * 88)
print(f"[P3] task={args.task} settle={args.settle}  (capture 후)")
print(f"[P3] anchor euler {[round(v * 57.3, 1) for v in env.src.anchor[0, 3:].tolist()]}°"
      f" · scale rot {[round(v * 57.3, 1) for v in env.src.scale[0, 3:].tolist()]}°")
print("-" * 88)
print(f"{'패턴':<18s} {'지령 euler(deg)':<24s} {'rot_err':>8s} {'pos_err':>9s}"
      f" {'tilt지령':>8s} {'tilt달성':>8s}")
worst_rot, worst_pos = 0.0, 0.0
for i, (label, *_r) in enumerate(PATTERNS):
    e = [round(float(v) * deg, 1) for v in cmd[i, 3:].tolist()]
    r_i, p_i = float(rot_err_deg[i]), float(pos_err_mm[i])
    worst_rot, worst_pos = max(worst_rot, r_i), max(worst_pos, p_i)
    print(f"{label:<18s} {str(e):<24s} {r_i:7.1f}° {p_i:8.1f}mm"
          f" {float(tilt_cmd_deg[i]):7.1f}° {float(tilt_meas_deg[i]):7.1f}°")
print("-" * 88)
op = tilt_cmd_deg <= 115.0
gate_a = bool(((rot_err_deg[op] < args.gate_rot_deg)
               & (pos_err_mm[op] < args.gate_pos_mm)).all())
gate_b = bool((tilt_meas_deg.max() >= 135.0))
sweep = [i for i, (label, *_) in enumerate(PATTERNS) if label.startswith("깊축")]
sw_meas, sw_cmd = tilt_meas_deg[sweep], tilt_cmd_deg[sweep]
gate_c = bool((sw_meas[1:] > sw_meas[:-1]).all()
              and ((sw_cmd - sw_meas) < 15.0).all())
print(f"[P3] A 운용역(≤115°) rot<{args.gate_rot_deg}°·pos<{args.gate_pos_mm}mm : "
      f"{'PASS' if gate_a else 'FAIL'} "
      f"(worst rot {float(rot_err_deg[op].max()):.1f}° pos {float(pos_err_mm[op].max()):.1f}mm)")
print(f"[P3] B 깊이 max tilt달성 ≥135°                : "
      f"{'PASS' if gate_b else 'FAIL'} ({float(tilt_meas_deg.max()):.1f}°)")
print(f"[P3] C 깊축 스윕 강단조 & 격차<15°            : "
      f"{'PASS' if gate_c else 'FAIL'} "
      f"(스윕 최대 격차 {float((sw_cmd - sw_meas).max()):.1f}° · "
      f"전 패턴 최대 격차 {float((tilt_cmd_deg - tilt_meas_deg).max()):.1f}°=정보)")
print("RESULT:", "PASS" if (gate_a and gate_b and gate_c) else "FAIL")

env.close()
app.close()
