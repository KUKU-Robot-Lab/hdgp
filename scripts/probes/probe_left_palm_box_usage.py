#!/usr/bin/env python3
"""좌팔 fab — **palm 액션 박스의 실사용 범위**를 실측한다.

무엇을 결정하나
---------------
`PALM_BOX_{X,Y,Z}` 를 얼마나 좁힐 수 있는가. 이게 왜 중요한지는 한 줄로 정해진다:

    액션은 절대 규약이라 정책 지터 = σ × box_half.
    파지 성패는 개구 방향 여유 **±13.25 mm** (개구 84.5 − 파지대역 단면 58, /2)에서 갈린다.
    t16 best 의 σ 0.348 × half 0.19 = **66 mm** — 결정 경계보다 5 배 거칠다.

박스를 좁히면 같은 σ 로도 해상도가 그만큼 올라간다. 그래서 "실제로 필요한 최소 박스"가
얼마인지를 **추정하지 않고 잰다**. 이 트랙은 도달범위를 눈대중했다가 두 번 틀렸다.

왜 정책 롤아웃이 아닌가
-----------------------
t16 이하 체크포인트는 obs 차원이 다르다(35 → 85). 현재 코드로 못 돌린다. 그리고 정책이
쓴 범위는 "필요한 범위"가 아니라 "그 정책이 쓸 줄 알았던 범위"다 — 박스를 정하는 근거로는
**과제가 요구하는 범위**가 옳다. 여기서 재는 것은 후자다:

    파지 시점  jaw 중점이 컵 스폰 박스 전 구간의 파지점에 놓이는 palm 지령
    이송 시점  jaw 중점이 목표 박스 전 구간의 파지점에 놓이는 palm 지령
    홈         리셋 직후 palm 이 실제로 있는 위치 (박스가 이걸 품어야 첫 지령이 텔레포트가 아니다)

방법 — **순방향 스윕** (역해가 아니라 정방향 사상을 잰다)
--------------------------------------------------------
★★역방향(목표 jaw 중점 → 필요 지령)을 폐루프로 풀려던 두 판이 다 실패했다:
   1차 매 스텝 보정 → **발산**(|err| 103→147 mm, 2/256). fabric 추종 지연을 위치 오차로
     읽고 또 밀어서 생긴 와인드업이다.
   2차 정착 후 보정 → 이송 목표는 115/128 수렴했는데 **파지 높이 목표는 9/128** 뿐.
     컵을 치워도 그대로였으니 물리 충돌이 아니다. 원인은 프로브 자신에 있었다 —
     palm **자세를 공칭값에 고정**해 놓고 위치만 풀었다. 자세를 고정하면 도달집합이
     2-D 패치가 아니라 곡선으로 쪼그라든다(probe_left_gripper_reach 가 같은 함정을
     기록해 뒀는데 그대로 다시 밟았다). 정책은 ±45° 를 쓸 수 있다.

그래서 방향을 뒤집는다. **지령을 무작위로 뿌리고(자세 포함) 턱이 어디 가는지 본다.**
역해도, 수렴도, 자세 가정도 필요 없다. 그다음 "과제가 요구하는 턱 위치"에 실제로 도달한
표본만 남기면, 그 지령들의 포락이 곧 **쓸모 있는 박스**다. 남지 않은 구역은 아무 목표도
못 내는 곳이므로 박스에서 빼도 잃는 게 없다.
사용:
  TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH=<hdgp>/source/openarm \\
    ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_left_palm_box_usage.py --num_envs 256
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rounds", type=int, default=40, help="스윕 라운드 수")
parser.add_argument("--settle", type=int, default=30, help="라운드당 정착 스텝 (0.5 s)")
parser.add_argument("--pad", type=float, default=0.010,
                    help="요구 영역 허용 여유 (m)")
parser.add_argument("--quantile", type=float, default=0.01,
                    help="포락 분위 — 극단 표본 하나가 박스를 넓히지 않게")
parser.add_argument("--margin", type=float, default=0.020,
                    help="제안 박스에 더할 여유 (m)")
parser.add_argument("--rot_scale", type=float, default=1.0,
                    help="회전 액션 범위 배율. 1.0 = PALM_MAX_POSE_ANGLE(±45°, kuka 값).\n                         팜 프레임에서 턱까지 약 140 mm 이므로 회전 ±45° 는 그것만으로\n                         턱을 ±107 mm 움직인다 — 위치 박스와 같은 크기의 지렛대다.")
parser.add_argument("--keep_object", action="store_true",
                    help="컵을 남긴다. 기본은 치운다 — 팔·fabric 의 도달 능력을 재는 것이지\n                         컵을 피하는 능력을 재는 게 아니다(1차 실행에서 파지 높이 목표가\n                         128 개 중 3 개만 수렴했고, 원인 후보가 컵과의 물리 충돌이었다).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.gripper.left.grasp_sensor  # noqa: F401,E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402

TASK = "open-grip_l_grasp_sensor_fab"



def jaw_mid(env) -> torch.Tensor:
    """패드 중앙 보정을 넣은 턱 중점 (env 로컬). 보상의 `_jaw_frame` 과 같은 자다."""
    robot = env.scene["robot"]
    ids = [robot.body_names.index(b) for b in P.GRIPPER_FINGER_BODIES]
    pos = robot.data.body_pos_w[:, ids, :]
    approach = matrix_from_quat(robot.data.body_quat_w[:, ids[0], :])[:, :, 2]
    pos = pos + (approach * P.JAW_PAD_OFFSET).unsqueeze(1)
    return pos.mean(dim=1) - env.scene.env_origins


def required_regions():
    """과제가 요구하는 **턱 중점** 영역 — (lo, hi) 두 박스."""
    gx, gy = P.CUP_SPAWN_X_CENTER, P.CUP_SPAWN_Y_CENTER
    rx, ry = P.CUP_SPAWN_X_RANGE, P.CUP_SPAWN_Y_RANGE
    # 파지: 컵 스폰 박스 위, 파지 대역 높이. z 는 대역 폭만큼 허용한다.
    grasp = ((gx - rx, gy - ry, P.TABLE_SURFACE_Z + P.GRASP_HEIGHT_BAND[0]),
             (gx + rx, gy + ry, P.TABLE_SURFACE_Z + P.GRASP_HEIGHT_BAND[1]))
    # 이송: 목표 박스. 컵 원점이 목표에 있을 때 턱은 CUP_ORIGIN_TO_GRASP_Z 만큼 아래를 문다.
    goal = tuple(
        tuple(P.GOAL_POINT[i] + s * P.GOAL_JITTER[i] + (P.CUP_ORIGIN_TO_GRASP_Z if i == 2 else 0.0)
              for i in range(3))
        for s in (-1.0, 1.0))
    return grasp, goal


def inside(pts: torch.Tensor, box, pad: float) -> torch.Tensor:
    lo = torch.tensor(box[0], device=pts.device) - pad
    hi = torch.tensor(box[1], device=pts.device) + pad
    return ((pts >= lo) & (pts <= hi)).all(dim=-1)


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    # ★리셋 오염 차단 — 이 트랙에서 네 번 당했다.
    cfg.episode_length_s = 1.0e9
    cfg.terminations.time_out = None
    cfg.terminations.object_dropping = None
    cfg.terminations.object_out_of_workspace = None
    cfg.curriculum.adr = None
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    dev = env.device
    act = env.action_manager.get_term("arm_action")
    n = env.num_envs

    if not args.keep_object:
        obj = env.scene["object"]
        st = obj.data.default_root_state.clone()
        st[:, :3] = env.scene.env_origins + torch.tensor([0.0, 0.0, -5.0], device=dev)
        obj.write_root_pose_to_sim(st[:, :7])
        obj.write_root_velocity_to_sim(torch.zeros_like(st[:, 7:]))

    print(f"[홈] 턱 중점 = {[round(v, 4) for v in jaw_mid(env).mean(0).tolist()]}")
    grasp_box, goal_box = required_regions()
    print(f"[요구] 파지 턱 영역 {tuple(round(v,3) for v in grasp_box[0])} ~ "
          f"{tuple(round(v,3) for v in grasp_box[1])}")
    print(f"[요구] 이송 턱 영역 {tuple(round(v,3) for v in goal_box[0])} ~ "
          f"{tuple(round(v,3) for v in goal_box[1])}")

    cmds, jaws = [], []
    a = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
    a[:, 6:] = -1.0
    for r in range(args.rounds):
        a[:, :6] = torch.rand(n, 6, device=dev) * 2.0 - 1.0
        a[:, 3:6] *= args.rot_scale
        for _ in range(args.settle):
            env.step(a)
        cmds.append((act._box_center + a[:, :3] * act._box_half).clone())
        jaws.append(jaw_mid(env).clone())
        if (r + 1) % 5 == 0:
            print(f"  라운드 {r+1:3d}/{args.rounds}  표본 {(r+1)*n}")

    cmd = torch.cat(cmds)
    jaw = torch.cat(jaws)
    print(f"\n표본 {cmd.shape[0]}개  (회전 범위 ×{args.rot_scale} = "
          f"±{P.PALM_MAX_POSE_ANGLE*args.rot_scale*57.2958:.1f}°)")

    hit_g = inside(jaw, grasp_box, args.pad)
    hit_t = inside(jaw, goal_box, args.pad)
    print(f"파지 영역 적중 {int(hit_g.sum())}  ·  이송 영역 적중 {int(hit_t.sum())}  "
          f"(허용 ±{args.pad*1000:.0f} mm)")

    def envelope(mask, name):
        if int(mask.sum()) < 3:
            print(f"[{name}] 표본 {int(mask.sum())}개 — 포락을 낼 수 없다")
            return None
        c = cmd[mask]
        q = torch.tensor([args.quantile, 1.0 - args.quantile], device=dev)
        mn = torch.quantile(c, q[0], dim=0)
        mx = torch.quantile(c, q[1], dim=0)
        print(f"[{name}] n={int(mask.sum())}  (분위 {args.quantile:.3f}~{1-args.quantile:.3f})")
        for j, ax in enumerate("xyz"):
            print(f"   {ax}  [{mn[j]:.4f}, {mx[j]:.4f}]  폭 {(mx[j]-mn[j])*1000:6.1f} mm")
        return mn, mx

    envelope(hit_g, "파지를 낼 수 있는 palm 지령")
    envelope(hit_t, "이송을 낼 수 있는 palm 지령")
    both = envelope(hit_g | hit_t, "합집합 = 쓸모 있는 박스")

    if both is None:
        env.close(); simulation_app.close(); return
    mn, mx = both
    home = jaw_mid(env)   # 홈도 품어야 첫 지령이 텔레포트가 아니다
    mn = torch.minimum(mn, home.min(0).values)
    mx = torch.maximum(mx, home.max(0).values)
    cur = (P.PALM_BOX_X, P.PALM_BOX_Y, P.PALM_BOX_Z)
    print("\n=== 권고 ===")
    print(f"{'축':<3}{'현재':>20}{'half':>9}{'제안(여유 ±20mm)':>26}{'half':>9}"
          f"{'σ0.35 지터':>12}{'개선':>7}")
    prop = []
    for j, ax in enumerate("xyz"):
        c0, c1 = cur[j]
        p0, p1 = float(mn[j]) - args.margin, float(mx[j]) + args.margin
        ch, ph = (c1 - c0) / 2, (p1 - p0) / 2
        prop.append((p0, p1))
        print(f"{ax:<3}({c0:.3f}, {c1:.3f}){'':>2}{ch*1000:7.0f}mm"
              f"      ({p0:.3f}, {p1:.3f}){'':>2}{ph*1000:7.0f}mm"
              f"{ph*0.348*1000:10.0f}mm{(1-ph/ch)*100:6.0f}%")
    print("\n★비교: 개구 방향 물리 여유 ±13.25 mm (개구 84.5 − 파지대역 단면 58, /2)")
    print("제안 박스:")
    for j, ax in enumerate("XYZ"):
        print(f"  PALM_BOX_{ax} = ({prop[j][0]:.3f}, {prop[j][1]:.3f})")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
