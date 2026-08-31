"""side-to-side 파지 자세의 도달 지도 — **자세 검증 포함** 범용 프로브.

무엇을 재는가
    (x, y, z) × roll 그리드의 각 지점에 side-to-side 자세(홈 회전 ± roll)를
    지령하고 정착시킨 뒤, 두 겹의 성공 판정을 나란히 낸다:
      ① 위치만   — palm 3D 추종오차 < pos_tol
      ② 자세까지 — ① AND 법선(palm_ee +x)이 world_z 와 수직(기울기 < normal_tol)
                    AND 롤 축(palm_ee +y)이 연직(ZX 기울기 < roll_tol)
    ②가 진짜 기준이다. fabric 은 도달이 빠듯하면 **회전을 포기하고 위치만**
    맞추는 해로 도망간다 — 08.26 실측에서 위치 오차 0 인데 palm_ee 가 위로
    들린 셀이 나왔고, 자세 미검증 지도는 그런 셀을 성공으로 오분류했다.

왜 palm 기준인가(손끝이 아니라)
    손끝은 손 자세의 함수라 지도가 손 액션에 종속된다. 파지 기하는
    "palm 이 컵 옆·같은 높이에 side-to-side 로 선다"로 정의되고(자매 실측
    cup_palm 수직 −7mm), 손끝 도달은 그 위에서 손끝 워크스페이스(실측 박스)가
    보장한다. 컵 배치 결정에 필요한 것은 palm 지도다.

다른 환경에서 쓰는 법
    --task 만 바꾸면 된다. 전제: env 에 palm_lo/hi(지령 박스)·home_palm·
    _tcp_idx(palm_ee)·palm_idx 가 있고 절대 박스 매핑을 쓴다. 그리드·임계는
    인자로 조정. 박스·slew 는 프로브가 스스로 풀었다가 지도만 재고 끝낸다.

실행(서버):
    CUDA_VISIBLE_DEVICES=1 python scripts/probes/probe_sidegrasp_reach_map.py \
        --task open-bis_l_grasp_lift_fab --num_envs 64
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_l_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--settle", type=int, default=240,
                    help="★80 은 수렴 미달이었다(90→80 차이로 같은 셀이 12→46mm). "
                         "중간(2/3 지점)과 끝의 오차 차로 수렴을 검증한다")
parser.add_argument("--z_cmd_offsets", type=str, default="0,0.03,0.06",
                    help="지령 z 보정 후보[m] — fabric 은 중력 처짐을 모르므로 "
                         "지령을 위로 띄워야 실 palm 이 목표 z 에 앉는다"
                         "(액션박스=목표+선행량 규약). 판정은 항상 **목표** 기준")
parser.add_argument("--pos_tol_mm", type=float, default=30.0)
parser.add_argument("--normal_tol_deg", type=float, default=15.0,
                    help="법선이 수평에서 벗어나도 되는 각")
parser.add_argument("--roll_tol_deg", type=float, default=20.0,
                    help="롤 축(palm_y)이 연직에서 벗어나도 되는 각(ZX 기울기)")
parser.add_argument("--xs", type=str, default="0.20,0.26,0.32,0.38")
parser.add_argument("--ys", type=str, default="0.10,0.16,0.22,0.28,0.34",
                    help="팔쪽 |y|. **음수 = 정중선 교차**(반대편)로 해석된다")
parser.add_argument("--zs", type=str, default="0.24,0.28,0.32,0.36,0.40")
parser.add_argument("--rolls_deg", type=str, default="0,15,-15")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import math                                    # noqa: E402
import gymnasium as gym                        # noqa: E402
import torch                                   # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import openarm.tasks                           # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.episode_length_s = 1.0e9
env = gym.make(args.task, cfg=env_cfg).unwrapped
N, A = args.num_envs, env.cfg.action_space
dev = env.device
env.reset()
env.step(torch.zeros(N, A, device=dev))

is_left = "_l_" in args.task
env.palm_lo[0, :3] = torch.tensor(
    [0.05, (0.02 if is_left else -0.55), 0.16], device=dev)
env.palm_hi[0, :3] = torch.tensor(
    [0.55, (0.55 if is_left else -0.02), 0.70], device=dev)
env._slew_on = False
home6 = env.home_palm[0].clone()
up = torch.tensor([0.0, 0.0, 1.0], device=dev)

xs = [float(v) for v in args.xs.split(",")]
ys = [float(v) for v in args.ys.split(",")]
zs = [float(v) for v in args.zs.split(",")]
rolls = [math.radians(float(v)) for v in args.rolls_deg.split(",")]
z_offs = [float(v) for v in args.z_cmd_offsets.split(",")]
pts = [(x, y, z) for z in zs for y in ys for x in xs]
print(f"\n지도 그리드 {len(pts)}점 × roll {len(rolls)} × z오프셋 {len(z_offs)} — env {N} 병렬",
      flush=True)

# key -> best-by-②: (pos_err_mm, normal_tilt_deg, zx_deg, roll_deg, ok_pos, ok_full)
results: dict = {}
_conv_gap_max = 0.0
for r in rolls:
  for zo in z_offs:
    for i0 in range(0, len(pts), N):
        chunk = pts[i0:i0 + N]
        env.reset()
        env.step(torch.zeros(N, A, device=dev))
        want = home6.unsqueeze(0).repeat(N, 1)      # ★판정 기준 = 목표(오프셋 없음)
        for k, (x, y, z) in enumerate(chunk):
            want[k, 0], want[k, 1], want[k, 2] = x, (y if is_left else -y), z
            want[k, 3] = home6[3] + (r if is_left else -r)
        tgt = want.clone()
        tgt[:, 2] += zo                              # 지령만 위로(중력 처짐 보상)
        a_arm = ((2.0 * tgt - (env.palm_hi + env.palm_lo))
                 / (env.palm_hi - env.palm_lo)).clamp(-1.0, 1.0)
        _mid = max(1, (2 * args.settle) // 3)
        perr_mid = None
        for _t in range(args.settle):
            a = torch.zeros(N, A, device=dev)
            a[:, :6] = a_arm
            env.step(a)
            if _t == _mid - 1:
                _ppm = env.robot.data.body_pos_w[:, env.palm_idx] - env.scene.env_origins
                perr_mid = (want[:, :3] - _ppm).norm(dim=-1) * 1000.0
        pp = env.robot.data.body_pos_w[:, env.palm_idx] - env.scene.env_origins
        perr = (want[:, :3] - pp).norm(dim=-1) * 1000.0
        # 수렴 검증 — 2/3 지점과 끝의 차가 크면 settle 부족(지도가 수렴 한계로 오염).
        _conv_gap_max = max(_conv_gap_max, float((perr_mid - perr).abs().max()))
        R = matrix_from_quat(env.robot.data.body_quat_w[:, env._tcp_idx])
        # 법선 수평성: palm_x·up 은 0 이어야 한다 — asin(|·|) = 수평에서 벗어난 각
        n_tilt = torch.rad2deg(torch.asin(
            (R[:, :, 0] * up).sum(-1).abs().clamp(max=1.0)))
        # 롤 축 연직성(ZX 기울기): acos(|palm_y·up|)
        zx = torch.rad2deg(torch.acos(
            (R[:, :, 1] * up).sum(-1).abs().clamp(max=1.0)))
        for k, key in enumerate(chunk):
            pe, nt, zz = float(perr[k]), float(n_tilt[k]), float(zx[k])
            ok_pos = pe < args.pos_tol_mm
            ok_full = ok_pos and nt < args.normal_tol_deg and zz < args.roll_tol_deg
            cand = (pe, nt, zz, math.degrees(r), ok_pos, ok_full)
            cur = results.get(key)
            # ② 성공 우선, 다음 ① 성공, 다음 pos_err 최소
            def rank(t):
                return (not t[5], not t[4], t[0])
            if cur is None or rank(cand) < rank(cur):
                results[key] = cand
  print(f"  roll {math.degrees(r):+.0f}° (z오프셋 전부) 완료", flush=True)

print("\n" + "=" * 84)
print(f"수렴 검증: |err(2/3지점) − err(끝)| 최대 {_conv_gap_max:.1f}mm "
      f"{'✓수렴' if _conv_gap_max < 10.0 else '★settle 부족 — 지도 무효'}")
print(f"side-to-side 도달 지도 — {'좌' if is_left else '우'}팔")
print(f"  ✓=자세까지(위치<{args.pos_tol_mm:.0f}mm ∧ 법선수평<{args.normal_tol_deg:.0f}° "
      f"∧ ZX<{args.roll_tol_deg:.0f}°) · p=위치만 · 숫자=위치오차mm")
print("=" * 84)
for z in zs:
    print(f"\n  z={z:.2f}   x→  " + "   ".join(f"{x:.2f}" for x in xs))
    for y in ys:
        row = []
        for x in xs:
            pe, nt, zz, rr, okp, okf = results[(x, y, z)]
            row.append(" ✓  " if okf else (" p  " if okp else f"{min(pe,999):3.0f} "))
        print(f"    |y|={y:.2f}  " + "   ".join(row))
full = [(k, v) for k, v in results.items() if v[5]]
posonly = [(k, v) for k, v in results.items() if v[4] and not v[5]]
print(f"\n②자세까지 성공 {len(full)}/{len(results)} · ①위치만(자세 포기) {len(posonly)}")
if posonly:
    print("★위치만 성공(자세 미검증 지도의 오분류 후보):")
    for k, v in sorted(posonly, key=lambda t: t[0][2])[:6]:
        print(f"  ({k[0]:.2f},|y|{k[1]:.2f},z{k[2]:.2f}) 법선기움 {v[1]:.0f}° ZX {v[2]:.0f}°")
if full:
    print(f"최저 도달 z(자세까지) = {min(k[2] for k, _ in full):.2f}")
    for k, v in sorted(full, key=lambda t: (t[0][2], t[1][0]))[:8]:
        print(f"  ({k[0]:.2f}, |y|{k[1]:.2f}, z{k[2]:.2f})  위치 {v[0]:.0f}mm "
              f"법선기움 {v[1]:.0f}° ZX {v[2]:.0f}° roll {v[3]:+.0f}°")
print("=" * 84 + "\n", flush=True)
env.close(); app.close()
