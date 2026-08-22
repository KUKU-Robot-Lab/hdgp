"""`open-grip_l_grasp_sensor_fab` 학습 전 게이트 G1~G3 — Isaac 실측.

G1 도달성: PALM_BOX 안의 격자 목표(스폰 접근 + 목표 박스 8 꼭짓점)로 fabric 을 수렴시켜
   TCP 오차와 관절 한계 여유를 잰다. 오차가 크거나 한계에 붙으면 박스를 줄여야 한다.
G2 hold/계단: 08.21 튜닝은 ABORTED 홈·dt 1/60 에서 했다 — **이 태스크 홈·env dt** 에서
   hold 오차와 z 계단 응답을 재확인한다.
G3 스크립트 파지-리프트-이송: RL 없이 fabric 지령 시퀀스(컵 옆→폐쇄→상승→목표 이동)로
   컵이 실제로 40 mm 이상 들려 따라오는지. 이게 안 되면 정책도 못 한다.

사용: isaaclab.sh -p scripts/probes/probe_fab_task_gates.py --gate g1|g2|g3
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gate", choices=["g1", "g2", "g3"], required=True)
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402

TASK = "open-grip_l_grasp_sensor_fab"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset()

robot = env.scene["robot"]
obj = env.scene["object"]
_pidx = robot.body_names.index(P.GRIPPER_BASE_BODY)
_off = torch.tensor([0.0, 0.0, P.TCP_OFFSET_IN_BASE_Z], device=env.device).repeat(
    env.num_envs, 1
)
arm_ids = [robot.joint_names.index(n) for n in P.LEFT_ARM_JOINT_NAMES]
n_act = env.action_manager.total_action_dim
assert n_act == 7, f"액션 차원 {n_act} (기대 7)"


def tcp() -> torch.Tensor:
    return (
        robot.data.body_pos_w[:, _pidx, :]
        + quat_apply(robot.data.body_quat_w[:, _pidx, :], _off)
    ) - env.scene.env_origins


_BOX_LO = torch.tensor([P.PALM_BOX_X[0], P.PALM_BOX_Y[0], P.PALM_BOX_Z[0]], device=env.device)
_BOX_HI = torch.tensor([P.PALM_BOX_X[1], P.PALM_BOX_Y[1], P.PALM_BOX_Z[1]], device=env.device)


def act_for(pos_env: torch.Tensor, grip: float) -> torch.Tensor:
    """env-local 목표 위치 → 정규화 액션 (회전 0 = 기준 파지 자세).

    ★목표가 박스 밖이면 액션이 클램프돼 **다른 곳**을 지령하게 된다 — 여기서 밟았다
    (홈 TCP x 0.239 < 박스 하한 0.28 → "제자리 유지"가 41 mm 이동 지령). 밖이면 즉사시킨다.
    """
    if bool((pos_env < _BOX_LO).any() or (pos_env > _BOX_HI).any()):
        raise ValueError(f"목표가 PALM_BOX 밖이다: {pos_env[0].tolist()}")
    a = torch.zeros(env.num_envs, n_act, device=env.device)
    a[:, :3] = (pos_env - 0.5 * (_BOX_LO + _BOX_HI)) / (0.5 * (_BOX_HI - _BOX_LO))
    a[:, -1] = grip
    return a


def park_cup() -> None:
    """컵을 멀리 치운다 — fabric·팔 계측에서 접촉을 배제.

    ★z 를 **명시적으로 안전 높이에 고정**해야 한다. 현재 z 를 보존하면 스텝마다
    자유낙하분(~2 mm)이 누적돼 ~18 스텝 뒤 object_dropping 종료가 발화하고,
    그 리셋이 fabric_q 를 홈으로 되돌려 **모든 수렴 계측을 톱니로 오염**시킨다
    (실제로 이 오염이 기준 자세를 두 번 갈아치우게 만들었다).
    """
    s = obj.data.root_state_w.clone()
    s[:, 0] = 5.0
    s[:, 1] = 5.0
    s[:, 2] = 0.5
    s[:, 7:] = 0.0
    obj.write_root_state_to_sim(s)


def limit_margin() -> tuple[float, str]:
    q = robot.data.joint_pos[:, arm_ids]
    lo = robot.data.soft_joint_pos_limits[:, arm_ids, 0]
    hi = robot.data.soft_joint_pos_limits[:, arm_ids, 1]
    m = torch.minimum(q - lo, hi - q)
    v, flat = m.min(), m.min(dim=0).values
    j = int(flat.argmin())
    return float(v), P.LEFT_ARM_JOINT_NAMES[j]


# ══════════════════════════════════════════════════════════════════
if args.gate == "g1":
    # 스폰 접근점 4 + 목표 박스 8 꼭짓점 + 박스 중심
    pts = [
        (P.CUP_SPAWN_X_CENTER - 0.02, P.CUP_SPAWN_Y_CENTER - 0.02, P.CUP_SPAWN_Z),
        (P.CUP_SPAWN_X_CENTER - 0.02, P.CUP_SPAWN_Y_CENTER + 0.02, P.CUP_SPAWN_Z),
        (P.CUP_SPAWN_X_CENTER + 0.02, P.CUP_SPAWN_Y_CENTER - 0.02, P.CUP_SPAWN_Z),
        (P.CUP_SPAWN_X_CENTER + 0.02, P.CUP_SPAWN_Y_CENTER + 0.02, P.CUP_SPAWN_Z),
    ] + [
        (x, y, z)
        for x in P.GOAL_POS_X for y in P.GOAL_POS_Y for z in P.GOAL_POS_Z
    ] + [P.GOAL_POINT]
    print(f"\n=== G1 도달성 · {len(pts)} 지점 × 150 스텝 수렴 ===")
    worst = 0.0
    for x, y, z in pts:
        env.reset()
        target = torch.tensor([x, y, z], device=env.device).repeat(env.num_envs, 1)
        a = act_for(target, grip=1.0)
        for _ in range(150):
            park_cup()
            env.step(a)
        err = float((tcp() - target).norm(dim=-1).mean()) * 1e3
        ft = env.action_manager._terms["arm_action"]
        l1 = float((ft._fabric.get_palm_pose(ft._fabric_q.detach(), "quaternion")[:, :3]
                    - target).norm(dim=-1).mean()) * 1e3
        mg, jn = limit_margin()
        worst = max(worst, l1)
        flag = "  ←" if l1 > 20.0 or mg < 0.05 else ""
        print(f"  ({x:+.3f},{y:+.3f},{z:+.3f})  L1 {l1:7.1f} mm · TCP {err:6.1f} mm(처짐 포함) "
              f"· 한계여유 {mg:.3f} rad ({jn}){flag}")
    # ★판정은 fabric 층(L1). TCP 오차에는 PD 중력 처짐(~34 mm)이 섞이고 그건 정책이
    #   절대 목표 보정으로 흡수하는 몫이다(test17 이 관절오차 19~22° 를 안고 성공).
    print(f"\n  판정: L1 최악 {worst:.1f} mm — {'PASS (<20)' if worst < 20.0 else 'FAIL: 박스/자세 재검토'}")

elif args.gate == "g2":
    print("\n=== G2 정착 정확도 + hold + z 계단 ===")
    # ★"리셋 직후 TCP" 를 기준으로 잡으면 안 된다 — 액션 0 = 박스 중심 지령이라 팔이
    #   이미 이동 중이고, 그 순간을 hold 목표로 삼으면 이동분이 오차로 섞인다(밟았다).
    #   올바른 절차: 목표를 정해 **정착시킨 뒤** 그 상태에서 hold·계단을 잰다.
    fab_term = env.action_manager._terms["arm_action"]

    def settle_err(target, steps=150):
        a = act_for(target, grip=1.0)
        for _ in range(steps):
            park_cup()
            env.step(a)
        l1 = float((fab_term._fabric.get_palm_pose(fab_term._fabric_q.detach(), "quaternion")[:, :3]
                    - target).norm(dim=-1).mean()) * 1e3
        tcp_err = float((tcp() - target).norm(dim=-1).mean()) * 1e3
        return l1, tcp_err

    env.reset()
    T0 = torch.tensor(P.GOAL_POINT, device=env.device).repeat(env.num_envs, 1)
    l1, e0 = settle_err(T0)
    print(f"  정착(목표점): L1 {l1:6.2f} mm · TCP 오차 {e0:6.2f} mm"
          f"   {'PASS' if l1 < 5.0 else 'FAIL(L1)'} / sag {e0 - l1:.1f} mm")

    # hold: 같은 지령 유지, 정착점에서의 드리프트.
    # ★80 스텝만 잰다 — 정착 150 + 100 이면 250 에서 **에피소드 타임아웃 리셋**이 fabric 을
    #   홈으로 되돌려 최대 드리프트에 130 mm 스파이크가 찍힌다(실측, 계측 오염).
    settled = tcp().clone()
    a_hold = act_for(T0, grip=1.0)
    drift = []
    for _ in range(80):
        park_cup()
        env.step(a_hold)
        drift.append(float((tcp() - settled).norm(dim=-1).mean()) * 1e3)
    print(f"  hold 80스텝 드리프트: 평균 {sum(drift)/len(drift):5.2f} mm · 최대 {max(drift):5.2f} mm"
          f"   {'PASS (<3)' if max(drift) < 3.0 else 'FAIL'}")
    env.reset()  # 계단은 새 에피소드에서 — 타임아웃이 중간에 안 끼게

    # 계단: 새 에피소드에서 T0 로 재정착 후 +5cm z
    a0 = act_for(T0, grip=1.0)
    for _ in range(150):
        park_cup()
        env.step(a0)
    settled = tcp().clone()
    T1 = T0.clone(); T1[:, 2] += 0.05
    a_step = act_for(T1, grip=1.0)
    zs, fzs = [], []
    for _ in range(200):
        park_cup()
        env.step(a_step)
        zs.append(float(tcp()[:, 2].mean()))
        fzs.append(float(fab_term._fabric.get_palm_pose(
            fab_term._fabric_q.detach(), "quaternion")[:, 2].mean()))
    z0 = float(settled[:, 2].mean()); zf = sum(zs[-20:]) / 20
    span = zf - z0; peak = max(zs)
    overshoot = ((peak - z0) / max(span, 1e-9) - 1.0) * 100.0
    t90 = next((i for i, v in enumerate(zs) if (v - z0) / max(span, 1e-9) >= 0.9), None)
    sse = (float(T1[0, 2]) - zf) * 1e3
    l1s = float((fab_term._fabric.get_palm_pose(fab_term._fabric_q.detach(), "quaternion")[:, :3]
                 - T1).norm(dim=-1).mean()) * 1e3
    fz0 = fzs[0]; fzf = sum(fzs[-20:]) / 20
    fspan = fzf - fz0
    fovershoot = ((max(fzs) - fz0) / max(fspan, 1e-9) - 1.0) * 100.0
    print(f"  └ 층 분해: fabric z 오버슈트 {fovershoot:5.1f}% (TCP 오버슈트에서 이만큼이 fabric 몫)")
    # ★판정은 층별로: fabric(L1)·오버슈트가 게이트다. TCP sse 에는 PD 중력 처짐이 섞이는데
    #   그건 정책이 절대 목표를 보정해 흡수하는 몫이라(관절공간 test17 이 관절오차 19~22° 를
    #   안고도 성공) 게이트가 아니라 **보고 항목**이다.
    # 게이트 = fabric 층(L1 < 5 mm, fabric 오버슈트 < 15%). TCP 오버슈트·sse 는 PD 과도·
    # 중력 처짐이 섞인 참고 항목이다 — 정지 품질의 실체는 hold 드리프트(0.03 mm)이고,
    # RL 의 실제 지령은 5 cm 계단이 아니라 미소 보정이라 과도 오버슈트는 접근 1회성이다.
    print(f"  z+5cm 계단: 90% 상승 {t90} 스텝 · fabric 오버슈트 {fovershoot:5.1f}% · L1 {l1s:.2f} mm"
          f" · [참고] TCP 오버슈트 {overshoot:5.1f}% · TCP sse {sse:6.2f} mm"
          f"   {'PASS' if (fovershoot < 15.0 and l1s < 5.0) else 'FAIL'}")

elif args.gate == "g3":
    print("\n=== G3 스크립트 파지-리프트-이송 (처짐 폐루프 보정) ===")
    # ★PD 중력 처짐(~34 mm) 때문에 목표를 그대로 지령하면 TCP 가 어긋난 채 허공을 닫는다
    #   (보정 없는 1차 시도: 컵 상승 0.0 mm). 정책이 학습으로 하는 절대 목표 보정을
    #   스크립트에서는 **측정 오차를 지령에 되먹이는 폐루프**로 재현한다.
    env.reset()
    cup0 = (obj.data.root_pos_w - env.scene.env_origins).clone()

    def goto(target_env, grip, iters=3, steps=50):
        """target 으로 이동하되, 매 iter 마다 TCP 오차만큼 지령을 보정한다."""
        cmd = target_env.clone()
        for _ in range(iters):
            a = act_for(cmd.clamp(_BOX_LO, _BOX_HI), grip=grip)
            for _ in range(steps):
                env.step(a)
            cmd = cmd + (target_env - tcp())
        return float((tcp() - target_env).norm(dim=-1).mean()) * 1e3

    goal = torch.tensor(P.GOAL_POINT, device=env.device).expand_as(cup0)

    # 접근은 2단: pregrasp(컵 뒤 10 cm) → 진입. 홈→컵 직행은 경로가 컵을 스칠 수 있다.
    # ★진입은 **접근축 방향**으로 후퇴한 pregrasp 에서 축을 따라 들어간다.
    #   순수 +X 로 밀어넣으면(1·2차 시도) 홈 자세의 접근축이 (+0.94,+0.26,−0.24) 대각이라
    #   손가락 몸체가 컵 옆벽을 쓸며 +Y 로 83 mm 밀렸다(env별 1~155 = 스폰 산포에 따라
    #   걸리거나 안 걸리거나). 축 방향 진입이면 손가락이 쓸고 지나갈 측면적이 없다.
    _w, _x, _y, _z = P.PALM_REF_QUAT_WXYZ
    approach = torch.tensor([2*(_x*_z + _w*_y), 2*(_y*_z - _w*_x), 1 - 2*(_x*_x + _y*_y)],
                            device=env.device)
    approach = approach / approach.norm()
    _gp0 = cup0.clone(); _gp0[:, 2] += P.CUP_ORIGIN_TO_GRASP_Z
    e0 = goto((_gp0 - 0.10 * approach).clamp(_BOX_LO, _BOX_HI), grip=+1.0)
    for t in (0.06, 0.03):
        goto((_gp0 - t * approach).clamp(_BOX_LO, _BOX_HI), grip=+1.0, iters=2)
    # ★★진입 목표는 컵 **원점**이 아니라 **파지 대역**이다. 원점은 상면 +92 mm 인데 통과
    #   대역은 +10~85 mm 라, 원점을 겨냥하면 턱이 컵의 가장 넓은 단(88 mm > 개구 84.5 mm)에
    #   부딪혀 밀려난다(실측: TCP 오차 100.2 mm, 벡터 −31,+93,+15).
    grasp_pt = cup0.clone()
    grasp_pt[:, 2] += P.CUP_ORIGIN_TO_GRASP_Z
    e1 = goto(grasp_pt, grip=+1.0, iters=4)
    err_vec = (tcp() - grasp_pt).mean(dim=0) * 1e3
    per_env = ((tcp() - grasp_pt).norm(dim=-1) * 1e3)
    r1 = float((obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - cup0[:, 2]).mean()) * 1e3
    print(f"  pregrasp        TCP 오차 {e0:6.1f} mm")
    print(f"  접근(컵 원점)   TCP 오차 {e1:6.1f} mm · 벡터 ({float(err_vec[0]):+.0f},{float(err_vec[1]):+.0f},{float(err_vec[2]):+.0f}) "
          f"· env별 min/max {float(per_env.min()):.0f}/{float(per_env.max()):.0f} · 컵 상승 {r1:+6.1f} mm")

    a_close = act_for((cup0 + (cup0 - cup0)).clamp(_BOX_LO, _BOX_HI), grip=-1.0)
    # 폐쇄는 현재 지령 유지 + 그리퍼만 닫기 — goto 의 마지막 보정 지령을 재사용한다.
    a_close = act_for((tcp() + (grasp_pt - tcp())).clamp(_BOX_LO, _BOX_HI), grip=-1.0)
    for _ in range(60):
        env.step(a_close)
    print(f"  폐쇄            그리퍼 닫음")

    lift_t = grasp_pt.clone(); lift_t[:, 2] += 0.08
    e2 = goto(lift_t, grip=-1.0)
    cup = obj.data.root_pos_w - env.scene.env_origins
    r2 = float((cup[:, 2] - cup0[:, 2]).mean()) * 1e3
    lifted = (cup[:, 2] - cup0[:, 2]) > 0.04
    print(f"  상승 +80mm      TCP 오차 {e2:6.1f} mm · 컵 상승 {r2:+6.1f} mm · 40mm 이상 {int(lifted.sum())}/{env.num_envs}")

    e3 = goto(goal, grip=-1.0)
    cup = obj.data.root_pos_w - env.scene.env_origins
    d3 = float((cup - goal).norm(dim=-1).mean()) * 1e3
    print(f"  목표 이동        TCP 오차 {e3:6.1f} mm · 컵-목표 {d3:6.1f} mm")

    a_hold = act_for((tcp() + (goal - tcp())).clamp(_BOX_LO, _BOX_HI), grip=-1.0)
    vels = []
    for _ in range(80):
        env.step(a_hold)
        vels.append(float(obj.data.root_lin_vel_w.norm(dim=-1).mean()))
    cup = obj.data.root_pos_w - env.scene.env_origins
    rise = float((cup[:, 2] - cup0[:, 2]).mean()) * 1e3
    d = float((cup - goal).norm(dim=-1).mean()) * 1e3
    v_tail = sum(vels[-20:]) / 20
    held = (cup[:, 2] - cup0[:, 2]) > 0.04
    print(f"  정지 유지        컵 상승 {rise:+6.1f} mm · 컵-목표 {d:6.1f} mm · 잔류 속도 {v_tail:.3f} m/s · 유지 {int(held.sum())}/{env.num_envs}")
    ok = int(held.sum()) >= env.num_envs * 3 // 4 and d < 100.0
    print(f"\n  판정: {'PASS' if ok else 'FAIL'} (기준: 3/4 이상 40mm 유지 & 컵-목표<100mm)")

env.close()
app.close()
