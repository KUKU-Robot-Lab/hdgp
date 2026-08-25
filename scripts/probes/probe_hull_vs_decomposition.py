"""convexHull vs convexDecomposition — 유령접촉·처리량·손가락 교차를 한 번에 잰다.

왜: KUKA DEXTRAH 는 `convexHull + self-collision ON + body_repulsion 13쌍(손↔팔뚝)` 으로
돈다. 우리는 `convexDecomposition + self-collision ON + repulsion 전부 OFF` 다.
hull 로 바꾸면 검출 비용이 크게 주는데, 과거 `urdf/tools/build_usd.py` 가 hull 을
되돌린 근거가 **64 정점 hull 팽창으로 인한 유령접촉**(9.6mm 간격에서 427kN)이었다.
그 근거가 지금도 유효한지, 그리고 repulsion 13쌍이 그걸 덮는지를 조건별로 측정한다.

측정 3종:
  ①유령접촉 — 홈 자세 **정지 명령**에서 관절이 튀는가. 유령접촉이 있으면 PD 가
    막을 수 없는 외력이 들어와 관절 속도·토크가 폭주한다. 접촉센서 없이 잡힌다.
  ②처리량 — 물리 스텝 fps. hull 의 실익이 여기 있다.
  ③손가락 교차 — 강제 폐합에서 서로 다른 손가락 링크 간 최소 중심거리.
    마디 반경 9mm 이므로 18mm 가 물리적 접촉, 그보다 작으면 **겹친 것**이다.

사용:
    isaaclab.sh -p scripts/probes/probe_hull_vs_decomposition.py
    (조건은 --conds 로 골라 돌린다)
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--settle", type=int, default=200,
                    help="정지 명령 유지 스텝. ★fabric 수렴(과도응답)과 유령접촉을 "
                         "구분하려면 충분히 길어야 한다.")
parser.add_argument("--close", type=int, default=120, help="강제 폐합 스텝")
parser.add_argument("--fps_steps", type=int, default=120)
# ★Isaac Sim 은 한 프로세스에서 env 를 두 번 만들 수 없다(SimulationContext 싱글톤).
#   조건마다 **별도 프로세스**로 돌리고 셸이 결과를 모은다.
parser.add_argument("--cond", default="A", choices=list("ABCD"),
                    help="A=decomp/selcol · B=hull/selcol · C=hull/selcol/repulsion13 "
                         "· D=hull/selcol OFF")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import time                                        # noqa: E402

import gymnasium as gym                            # noqa: E402
import torch                                       # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg     # noqa: E402

import openarm.tasks                               # noqa: E402,F401
from openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg import (  # noqa: E402
    resolve_cfg,
)

HULL = "robot/openarm_tesollo_bi_s_rl_hull/openarm_tesollo_bi_s_rl.usd"

# (라벨, usd_override, self_collisions, body_repulsion_pairs)
# ★★현행 프로필 자산은 이미 **armhull**(팔·몸통 20개 hull + 손 54개 decomposition)이다.
#   08.23 자매 트랙 A/B 실측: 팔 hull 로 처리량 +13.7%, 접촉력·감쌈 불변.
#   같은 실측이 "손까지 hull 로 하면 접촉력 4배(133N)" 라고 기록했다 — B/D 는 그
#   기각 근거를 **우리 트랙에서** 재확인하는 조건이다.
#   따라서 KUKA 와 남은 진짜 차이는 `body_repulsion 13쌍`(C) 하나다.
CONDS = {
    "A": ("armhull + selcol ON  (현행 기준선)", "", True, False),
    "B": ("full HULL + selcol ON  (손까지 hull)", HULL, True, False),
    "C": ("armhull + selcol ON + repulsion 13쌍  ★KUKA 차이", "", True, True),
    "D": ("full HULL + selcol ON + repulsion 13쌍  ★사용자 지정", HULL, True, True),
}


def build(usd_override: str, selcol: bool, rep_pairs: bool):
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # ★스위치는 cfg 필드로만 — 파생 객체 직접 수정은 resolve_cfg 가 조용히 되돌린다(08.22).
    cfg.robot_usd_override = usd_override
    cfg.enable_self_collisions = bool(selcol)
    cfg.use_body_repulsion_pairs = bool(rep_pairs)
    cfg.enable_gravity = False
    cfg.episode_length_s = 1.0e6        # ★리셋 오염 차단(프로브 필수 규약)
    resolve_cfg(cfg)
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    return env


def finger_min_dist(env) -> torch.Tensor:
    """서로 다른 손가락의 wrap 마디 간 **최소 중심거리** [m] (env 별)."""
    idx = env._wrap_t                                   # (F, P) 손가락×마디
    pos = env.robot.data.body_pos_w[:, idx.reshape(-1)]  # (N, F*P, 3)
    n, fp, _ = pos.shape
    f = idx.shape[0]
    d = torch.cdist(pos, pos)                            # (N, F*P, F*P)
    # 같은 손가락 쌍은 제외
    fid = torch.arange(f, device=d.device).repeat_interleave(idx.shape[1])
    same = (fid[:, None] == fid[None, :])
    d = d.masked_fill(same.unsqueeze(0), float("inf"))
    return d.reshape(n, -1).min(dim=1).values


def run(key: str) -> dict:
    label, usd, selcol, rep = CONDS[key]
    env = build(usd, selcol, rep)
    dev, N = env.device, args.num_envs
    zero = torch.zeros(N, env.cfg.action_space, device=dev)

    # ---- ① 유령접촉: 홈 자세 정지 명령 --------------------------------------
    for _ in range(args.settle):
        env.step(zero)
    qd = env.robot.data.joint_vel.abs()
    tau = env.robot.data.applied_torque.abs()
    ghost = dict(qd_max=float(qd.max()), qd_mean=float(qd.mean()),
                 tau_max=float(tau.max()))

    # ---- ② 처리량 ------------------------------------------------------------
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.fps_steps):
        env.step(zero)
    torch.cuda.synchronize()
    fps = args.fps_steps * N / (time.time() - t0)

    # ---- ③ 손가락 교차 + 접촉력: 컵을 손에 넣고 강제 폐합 ---------------------
    # ★★컵을 파지중심으로 옮기지 않으면 손이 허공을 쥐어 **접촉력이 전부 0** 이 된다.
    #   손 hull 기각 근거가 "접촉력 4배" 였으므로 그걸 못 재면 측정이 무의미하다.
    #   (probe_penetration.py 가 같은 함정을 이미 겪고 같은 방식으로 고쳤다.)
    close = zero.clone()
    close[:, 6:] = -1.0
    # ★★world 기준으로 계산한다. `_palm_frame` 은 **env-local** 을 돌려주는데
    #   `write_root_state_to_sim` 은 world 를 받는다 — 섞으면 env_origins 만큼
    #   어긋나 컵이 손에서 16.6cm 떨어진 채로 "접촉력 0" 이 나온다(1차 실행이 그랬다).
    _palm_w = env.robot.data.body_pos_w[:, env.palm_idx]
    _tips_w = env.robot.data.body_pos_w[:, env._tip_t].mean(dim=1)
    _gc = 0.5 * (_palm_w + _tips_w)
    for i in range(args.close):
        # 컵을 매 스텝 파지중심에 **다시 놓는다** — 손이 밀어내면 빠져나가 접촉이 사라진다.
        if i % 10 == 0:
            _root = torch.zeros(N, 13, device=dev)
            _root[:, :3] = _gc
            _root[:, 3] = 1.0
            env.object.write_root_state_to_sim(_root)
        env.step(close)
    # ★손 hull 기각 근거가 **접촉력 4배**였다 — 반드시 함께 잰다.
    _f, _, _, _ = env._contact()
    # ★★진단: 힘이 정확히 0 이면 "안 닿았다" 와 "센서가 죽었다" 를 갈라야 한다(08.22 선례).
    _raw = 0.0
    for _fg in env._fingers:
        for _role in ("tip", "wrap"):
            for _s in env._sensors[_fg][_role]:
                _raw = max(_raw, float(_s.data.force_matrix_w.abs().max()))
    _tip_w = env.robot.data.body_pos_w[:, env._tip_t]
    _obj = env.object.data.root_pos_w
    _gap = (_tip_w - _obj[:, None, :]).norm(dim=-1).min(dim=1).values
    force = dict(max_N=float(_f.max()), mean_N=float(_f.mean()),
                 raw_max=_raw, tip_obj_gap_mm=float(_gap.mean()) * 1e3)
    dmin = finger_min_dist(env)
    cross = dict(min_mm=float(dmin.min()) * 1e3,
                 mean_mm=float(dmin.mean()) * 1e3,
                 overlap_frac=float((dmin < 0.018).float().mean()))
    qd2 = env.robot.data.joint_vel.abs()

    env.close()
    return dict(label=label, ghost=ghost, fps=fps, cross=cross, force=force,
                qd_closed_max=float(qd2.max()))


r = run(args.cond)
print("\n" + "=" * 96)
print(f"RESULT\t{args.cond}\t{r['label']}\t"
      f"qd_max={r['ghost']['qd_max']:.3f}\ttau_max={r['ghost']['tau_max']:.2f}\t"
      f"fps={r['fps']:.0f}\tclose_min_mm={r['cross']['min_mm']:.1f}\t"
      f"overlap={r['cross']['overlap_frac']*100:.1f}%\t"
      f"force_max={r['force']['max_N']:.1f}N\traw_max={r['force']['raw_max']:.2f}\t"
      f"tip_obj_gap={r['force']['tip_obj_gap_mm']:.1f}mm\t"
      f"qd_closed={r['qd_closed_max']:.3f}")
print("=" * 96)
app.close()
