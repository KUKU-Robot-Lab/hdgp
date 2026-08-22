"""손가락이 컵을 **뚫었는지** 기하로 판정한다.

접촉력 스파이크(전형 13~20N vs 최대 7218N)가 관통 신호인지 확인.
컵을 원기둥으로 근사해, 손끝/마디가 그 **내부**에 들어간 깊이를 잰다.
표면 접촉이면 깊이 ≈ 0, 관통이면 음수(=반경보다 안쪽)로 크게 나온다.

    isaaclab.sh -p .../probe_penetration.py --checkpoint <path>
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--checkpoint", default=None, help="없으면 손을 강제 폐합")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--self_collisions", action="store_true")
parser.add_argument("--gravity", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

from isaaclab.utils.math import quat_apply_inverse   # noqa: E402
import openarm.tasks         # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
# ★스위치는 cfg 필드로만(파생 객체 직접 수정은 resolve_cfg 가 조용히 되돌린다, 08.22)
from openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg import resolve_cfg
env_cfg.enable_self_collisions = bool(args.self_collisions)
env_cfg.enable_gravity = bool(args.gravity)
resolve_cfg(env_cfg)
_env = gym.make(args.task, cfg=env_cfg)
env = _env.unwrapped
env.reset()
zero = torch.zeros(args.num_envs, env.cfg.action_space, device=env.device)
env.step(zero)

# ★컵 기하 — **원뿔** 모델. bbox 반경 45mm 는 손잡이 포함 최대치라 원기둥으로 쓰면
#   관통을 15~27mm 과대평가한다(1차 실행이 pinky 31.7mm 로 그랬다).
#   실측(08.19, 그리퍼 파지대역 측정): cup_big 몸통은 원뿔형 지름 35(바닥)~60(상단)mm.
R_BOT, R_TOP = 0.0175, 0.030
OFF, H = 0.0773, 0.1776
spec = env.bank.specs[0]
sc = float(spec.scale[2])
R_BOT, R_TOP, OFF, H = R_BOT * sc, R_TOP * sc, OFF * sc, H * sc


def cup_radius(z_from_bottom: torch.Tensor) -> torch.Tensor:
    """높이별 몸통 반경 (선형 보간)."""
    t = (z_from_bottom / H).clamp(0.0, 1.0)
    return R_BOT + t * (R_TOP - R_BOT)


print(f"\n컵 근사(원뿔): 반경 {R_BOT*1000:.1f}→{R_TOP*1000:.1f}mm · 높이 {H*1000:.1f}mm"
      f" · 원점은 바닥+{OFF*1000:.1f}mm")
print("판정: 반경거리 < 그 높이의 몸통 반경 이면 관통. (손잡이는 모델 밖 — 위양성 가능)\n")

# ★컵을 손 안으로 옮긴다. 안 그러면 손이 컵에서 15cm 떨어진 홈 자세를 재게 되어
#   "관통 없음" 이라는 무의미한 결과가 나온다(1차 실행이 그랬다).
_palm = env.robot.data.body_pos_w[:, env.palm_idx]
_tips = env.robot.data.body_pos_w[:, env._tip_t].mean(dim=1)
_root = torch.zeros(args.num_envs, 13, device=env.device)
_root[:, :3] = 0.5 * (_palm + _tips)
_root[:, 3] = 1.0
env.object.write_root_state_to_sim(_root)

if args.checkpoint:
    # ★실제 정책 파지 상태에서 잰다 — 강제 폐합은 정책이 만드는 파지 기하와 다르다.
    #   (컵 텔레포트도 하지 않는다: 정책이 스스로 접근·파지하게 둔다.)
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
    from rl_games.torch_runner import Runner
    from rl_games.common import env_configurations, vecenv
    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    wenv = RlGamesVecEnvWrapper(_env, args.device,
                                agent_cfg["params"]["env"].get("clip_observations", 5.0),
                                agent_cfg["params"]["env"].get("clip_actions", 1.0))
    vecenv.register("IsaacRlgWrapper", lambda cn, ne, **kw: wenv)
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wenv})
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
        return o.get("policy", o.get("obs")) if isinstance(o, dict) else o

    # 컵 텔레포트를 되돌린다(정책 롤아웃은 정상 리셋 상태에서)
    env.reset()
    obs = _t(wenv.reset())
    SKIP = args.steps // 3
    max_depth = None
    gate_steps = 0
    for i in range(args.steps):
        with torch.no_grad():
            a = agent.get_action(obs, is_deterministic=True)
        obs = _t(wenv.step(a)[0])
        if i < SKIP:
            continue
        _obj = env.object.data.root_pos_w - env.scene.env_origins
        # 아래 공통 판정과 같은 기하 — 여기선 스텝별 최대만 누적
        _names, _ids = [], []
        if max_depth is None:
            for f in env.profile.fingers:
                for b in (env.profile.finger_tip_bodies[f]
                          + env.profile.finger_wrap_bodies.get(f, ())):
                    bid, _ = env.robot.find_bodies(b)
                    _names.append(b); _ids.append(bid[0])
            _bt = torch.tensor(_ids, device=env.device)
            max_depth = torch.zeros(len(_ids), device=env.device)
            roll_names = _names
        _pos = env.robot.data.body_pos_w[:, _bt] - env.scene.env_origins[:, None, :]
        _rel = _pos - _obj[:, None, :]
        # ★컵 **로컬** 프레임으로 회전 — world xy 로 재면 넘어진 컵에서 가상의 수직
        #   원뿔에 대한 위양성이 난다(middle_4 "25mm" = 링크 중심이 축 위라는 모순).
        _q = env.object.data.root_quat_w
        _rel = quat_apply_inverse(_q.unsqueeze(1).expand(-1, _rel.shape[1], -1), _rel)
        _rad = _rel[:, :, :2].norm(dim=-1)
        _zb = _rel[:, :, 2] + OFF                      # 바닥 기준 높이
        _inz = (_zb > 0) & (_zb < H)
        _d = ((cup_radius(_zb) - _rad) * _inz.float()).clamp(min=0)
        max_depth = torch.maximum(max_depth, _d.max(dim=0).values)
        f_, _ = env._contact()
        gate_steps += int((f_.max(dim=1).values > 1.0).any())
    print(f"\n=== 정책 롤아웃 관통 (마지막 {args.steps-SKIP} 스텝 · 전 env 최대) ===")
    print(f"  접촉 발생 스텝: {gate_steps}/{args.steps-SKIP}")
    for k, n in enumerate(roll_names):
        flag = " ★" if max_depth[k] > 0.010 else ("  ·" if max_depth[k] > 0.002 else "")
        print(f"  {n:20s} 최대 관통 {max_depth[k]*1000:6.1f}mm{flag}")
    print(f"  전체 최대: {max_depth.max()*1000:.1f}mm"
          f"  → {'★관통' if max_depth.max()>0.010 else ('경미' if max_depth.max()>0.002 else '없음')}")

act = torch.zeros(args.num_envs, env.cfg.action_space, device=env.device)
env.reset(); env.step(act)
_palm = env.robot.data.body_pos_w[:, env.palm_idx]
_tips = env.robot.data.body_pos_w[:, env._tip_t].mean(dim=1)
_root2 = torch.zeros(args.num_envs, 13, device=env.device)
_root2[:, :3] = 0.5 * (_palm + _tips); _root2[:, 3] = 1.0
env.object.write_root_state_to_sim(_root2)
for i in range(args.steps):
    act[:, 6:] = min(1.0, i / (args.steps * 0.3))     # 손만 서서히 폐합
    env.step(act)

obj = env.object.data.root_pos_w - env.scene.env_origins
# 손끝 + 감쌈 마디 전부
names, ids = [], []
for f in env.profile.fingers:
    for b in env.profile.finger_tip_bodies[f] + env.profile.finger_wrap_bodies.get(f, ()):
        bid, _ = env.robot.find_bodies(b)
        names.append(b); ids.append(bid[0])
bt = torch.tensor(ids, device=env.device)
pos = env.robot.data.body_pos_w[:, bt] - env.scene.env_origins[:, None, :]

rel = pos - obj[:, None, :]
_q2 = env.object.data.root_quat_w
rel = quat_apply_inverse(_q2.unsqueeze(1).expand(-1, rel.shape[1], -1), rel)
radial = rel[:, :, :2].norm(dim=-1)                    # 컵 축까지 수평거리(컵 로컬)
zb = rel[:, :, 2] + OFF                                # 바닥 기준 높이
inside_z = (zb > 0) & (zb < H)
depth = (cup_radius(zb) - radial) * inside_z.float()   # 양수 = 몸통 표면 안쪽

print(f"{'body':22s} {'반경거리':>9s} {'관통깊이':>9s} {'관통 env%':>10s}")
worst = []
for k, n in enumerate(names):
    r_ = radial[:, k].mean().item()
    d_ = depth[:, k].clamp(min=0)
    frac = (d_ > 0.002).float().mean().item() * 100
    print(f"  {n:20s} {r_*1000:8.1f}mm {d_.max().item()*1000:8.1f}mm {frac:9.1f}%")
    worst.append((d_.max().item(), n))
worst.sort(reverse=True)

# ── 손가락 상호 관통 ────────────────────────────────────────────────
# self-collision 을 껐으므로(Fabrics 가 팔만 제어, 손은 직접 PD) 손가락끼리
# 막는 것이 아무것도 없다. 다른 손가락 링크 사이 최소거리로 판정한다.
print("\n=== 손가락 상호 관통 ===")
fing_of = []
for f in env.profile.fingers:
    for _ in env.profile.finger_tip_bodies[f] + env.profile.finger_wrap_bodies.get(f, ()):
        fing_of.append(f)
fi = torch.tensor([env.profile.fingers.index(x) for x in fing_of], device=env.device)
D = torch.cdist(pos, pos)                                   # (N, L, L)
diff = fi[:, None] != fi[None, :]                           # 다른 손가락 쌍만
D = D.masked_fill(~diff.unsqueeze(0), float("inf"))
mind, idx = D.view(D.shape[0], -1).min(dim=1)
LINK_R = 0.010          # 마디 반경 근사 [m] — 두 마디 중심거리가 2R 보다 작으면 겹침
print(f"  마디 반경 근사 {LINK_R*1000:.0f}mm → 중심거리 < {2*LINK_R*1000:.0f}mm 이면 겹침")
print(f"  다른 손가락 링크 간 **최소** 거리: 평균 {mind.mean()*1000:.1f}mm"
      f" · 최소 {mind.min()*1000:.1f}mm")
for thr in (0.010, 0.015, 0.020):
    print(f"    < {thr*1000:.0f}mm 인 env 비율: {(mind < thr).float().mean()*100:5.1f}%")
_worst_env = int(mind.argmin())
_a, _b = divmod(int(idx[_worst_env]), len(fing_of))
print(f"  최악 쌍: {names[_a]} ↔ {names[_b]}  ({mind.min()*1000:.1f}mm)")

print("\n" + "=" * 60)
mx, who = worst[0]
if mx > 0.010:
    print(f"★관통 확인 — {who} 이 컵 표면 안쪽 {mx*1000:.1f}mm 까지 들어갔다.")
    print("  콜라이더(convexHull vs SDF)·depenetration·접촉 offset 을 확인할 것.")
elif mx > 0.002:
    print(f"경미한 침투 {mx*1000:.1f}mm ({who}) — 접촉 해석상 정상 범위일 수 있다.")
else:
    print(f"관통 없음 (최대 {mx*1000:.1f}mm).")
env.close()
app.close()
