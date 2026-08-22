"""어느 손가락이 파지에 참여하지 않는지, 그리고 컵 기울기의 원인을 가른다.

fab_test9 실측 배경: `grip_frac` 이 2800 에폭 내내 0.44(≈2.2/5 손가락)에서 **전혀
움직이지 않았다**. 집계 지표는 손가락별로 나뉘어 있지 않아 어느 손가락이 빠지는지
알 수 없다(grasp_v2 에서도 같은 측정 공백이 3지 국소최적 진단을 막았다).

또 `obj/tilt_deg` 가 8°↔14° 로 진동하며 총보상과 역상관인데, env/grip 과는
상관이 없다. 기울기가 (a)참여 손가락 수 (b)리프트 높이 (c)팔 지터 중
무엇과 붙어 있는지 같은 롤아웃 안에서 잰다.

    isaaclab.sh -p .../probe_finger_participation.py --checkpoint <path> --steps 400
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry   # noqa: E402
from rl_games.torch_runner import Runner                                 # noqa: E402
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper                     # noqa: E402

import openarm.tasks         # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
env = gym.make(args.task, cfg=env_cfg)
raw = env.unwrapped
env = RlGamesVecEnvWrapper(env, args.device,
                           agent_cfg["params"]["env"].get("clip_observations", 5.0),
                           # ★params.config.clip_actions 는 rl_games 내부 플래그(False)다. 래퍼가 쓰는 것은
                           #   params.env.clip_actions(1.0). 잘못 읽으면 Box(0,0) 이 되어 **모든 액션이 0**
                           #   → 지표가 전부 정확히 0.0000 으로 나온다(play.py 와 같은 키를 쓸 것).
                           agent_cfg["params"]["env"].get("clip_actions", 1.0))
from rl_games.common import env_configurations, vecenv   # noqa: E402
vecenv.register("IsaacRlgWrapper", lambda cn, ne, **kw: env)
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
runner = Runner()
agent_cfg["params"]["config"]["num_actors"] = args.num_envs
runner.load(agent_cfg)
agent = runner.create_player()
agent.restore(args.checkpoint)
# ★rl_games BasePlayer 는 기본이 단일 env(has_batch_dimension=False)라 obs 를
#   unsqueeze 해 (1, N*obs) 로 만든다. 벡터 env 를 쓰므로 명시적으로 켠다.
agent.has_batch_dimension = True
agent.batch_size = args.num_envs
agent.reset()

def _t(o):
    """obs 가 dict 로 오는 래퍼 버전이 있다 — policy 텐서만 뽑는다."""
    if isinstance(o, tuple):
        o = o[0]
    if isinstance(o, dict):
        for k in ("obs", "policy"):
            if k in o:
                return o[k]
        return next(iter(o.values()))
    return o


obs = _t(env.reset())

F = raw._fingers
thr = float(raw.cfg.contact_force_threshold)
tipC = torch.zeros(len(F), device=raw.device)     # tip 접촉 스텝수
wrapC = torch.zeros(len(F), device=raw.device)    # wrap 접촉 스텝수
anyC = torch.zeros(len(F), device=raw.device)
fsum = torch.zeros(len(F), device=raw.device)
n = 0
rows = []          # (n_fingers, tilt, dz, ajit) 스텝별 env 평균
prev = None
SKIP = args.steps // 4     # 초기 접근 구간은 제외

for i in range(args.steps):
    with torch.no_grad():
        act = agent.get_action(obs, is_deterministic=True)
    obs = _t(env.step(act)[0])

    force, wrapped = raw._contact()                       # (N,F), (N,F)
    # tip 만의 힘을 따로 재구성
    tipf = torch.zeros_like(force)
    for j, f in enumerate(F):
        t = torch.zeros(raw.num_envs, device=raw.device)
        for s in raw._sensors[f]["tip"]:
            t = t + s.data.force_matrix_w.view(raw.num_envs, -1, 3).sum(1).norm(dim=-1)
        tipf[:, j] = t

    if i >= SKIP:
        tipC += (tipf > thr).float().mean(0)
        wrapC += wrapped.mean(0)
        anyC += (force > thr).float().mean(0)
        fsum += force.mean(0)
        n += 1
        nf = (force > thr).float().sum(1)                 # (N,)
        tilt = raw._object_tilt_deg()
        dz = raw._local(raw.object.data.root_pos_w)[:, 2] - raw.object_spawn_pos[:, 2]
        aj = (raw.actions[:, :6] - prev[:, :6]).abs().mean(1) if prev is not None else torch.zeros_like(nf)
        # ★기울기의 귀속: 팔이 기울어 있나(정책의 접근 자세), 컵만 미끄러졌나(파지 기하)
        from isaaclab.utils.math import quat_apply
        pq = raw.robot.data.body_quat_w[:, raw.palm_idx]
        oq = raw.object.data.root_quat_w
        e = torch.zeros(raw.num_envs, 3, device=raw.device)
        wz = e.clone(); wz[:, 2] = 1.0
        cup_ax = quat_apply(oq, wz)                       # 컵 대칭축(로컬 +z)
        deg = lambda a, b: torch.rad2deg(torch.acos(
            (a * b).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)))
        best = None
        for k in range(3):                                # palm 의 세 축 중 컵과 정렬된 것
            ax = e.clone(); ax[:, k] = 1.0
            pa = quat_apply(pq, ax)
            d = torch.minimum(deg(pa, cup_ax), 180.0 - deg(pa, cup_ax))
            if best is None or d.mean() < best[1].mean():
                best = (k, d, pa)
        palm_vs_world = torch.minimum(deg(best[2], wz), 180.0 - deg(best[2], wz))
        rows.append(torch.stack([nf, tilt, dz, aj, palm_vs_world, best[1]], 1))
        _axis_k = best[0]
    prev = raw.actions.clone()

print("\n" + "=" * 72)
print(f"손가락별 참여 (마지막 {n} 스텝 · env {args.num_envs} · 임계 {thr}N)")
print("=" * 72)
print(f"{'손가락':<8s}{'접촉(any)':>10s}{'감쌈(wrap)':>11s}{'손끝(tip)':>10s}{'평균힘N':>10s}")
for j, f in enumerate(F):
    print(f"{f:<8s}{anyC[j]/n:10.3f}{wrapC[j]/n:11.3f}{tipC[j]/n:10.3f}{fsum[j]/n:10.3f}")
print(f"{'합계':<8s}{anyC.sum()/n:10.3f}{wrapC.sum()/n:11.3f}{tipC.sum()/n:10.3f}")
print(f"  → grip_frac {anyC.sum()/n/len(F):.3f} · envelope_frac {wrapC.sum()/n/len(F):.3f}")

D = torch.cat(rows, 0)
nf, tilt, dz, aj = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
pw, pc = D[:, 4], D[:, 5]
print("\n" + "=" * 72)
print("기울기의 귀속")
print("=" * 72)
print(f"  컵 축 ↔ world z (=obj tilt)     {tilt.mean():6.2f}°")
print(f"  palm 정렬축(idx {_axis_k}) ↔ world z  {pw.mean():6.2f}°   ← 팔이 기울인 각")
print(f"  palm 정렬축 ↔ 컵 축              {pc.mean():6.2f}°   ← 컵이 손 안에서 미끄러진 각")
print("  ★palm↔world 가 크면 **정책의 접근 자세**, palm↔컵 이 크면 **파지 미끄러짐**")
print("\n" + "=" * 72)
print("컵 기울기의 원인 — 같은 롤아웃 안에서 조건부 평균")
print("=" * 72)
print(f"{'참여손가락':<10s}{'표본비율':>9s}{'기울기°':>9s}{'상승m':>9s}{'팔지터':>9s}")
for k in range(0, 6):
    m = nf == k
    if m.sum() < 10:
        continue
    print(f"{k:<10d}{m.float().mean():9.3f}{tilt[m].mean():9.2f}{dz[m].mean():9.4f}{aj[m].mean():9.3f}")

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).mean() / (a.std() * b.std() + 1e-9)
print(f"\n  상관 tilt↔참여손가락 {corr(tilt, nf):+.3f}   tilt↔상승 {corr(tilt, dz):+.3f}   "
      f"tilt↔팔지터 {corr(tilt, aj):+.3f}")
print(f"  기울기 분포: 중앙값 {tilt.median():.2f}°  p90 {tilt.quantile(0.9):.2f}°  "
      f">10° 비율 {(tilt>10).float().mean():.3f}  >30° 비율 {(tilt>30).float().mean():.3f}")
print("\n  ★해석: >30° 비율이 크면 upright_quality 가 0 에 붙어 gradient 가 없다.")
env.close()
app.close()
