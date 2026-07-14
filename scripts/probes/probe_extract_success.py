"""LEFT 정책이 실제로 물체를 드는 자세를 그대로 추출한다.

스크립트 probe 로는 top-down 파지가 어떤 조건에서도 5cm 를 못 넘었다
(dx·dz·폐쇄방식·basis 를 전부 스윕했다. 최대 4.2cm).
그런데 LEFT 정책은 object_height 0.119(11.9cm)를 실제로 든다.

이 간극은 "내가 가정한 파지 자세가 틀렸다"는 뜻이다. 그러니 상상하지 말고
성공한 정책이 무엇을 하는지 직접 본다.

체크포인트를 로드해 롤아웃하고, **물체가 실제로 들린 순간**(object_height > 5cm)의
상태를 모아 평균/분포를 낸다:
  - palm ← 물체 상대 위치, palm 법선/손가락 방향
  - 손 관절 20개
  - action 16개 (palm 6 + 시너지 5 + abduction 5)
  - 손끝 5개 ← 물체 상대 위치

사용:
  ./isaaclab.sh -p scripts/probes/probe_extract_success.py \
      --task open-tesol_l_grasp_v2-lstm --checkpoint <path.pth>
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_l_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--lift_thresh", type=float, default=0.05, help="이 이상 들리면 '성공 순간'")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

from rl_games.torch_runner import Runner  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

_OUT = open("/tmp/probe_extract_success.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")

env = gym.make(args.task, cfg=env_cfg)
raw = env.unwrapped

clip_obs = agent_cfg["params"]["env"].get("clip_observations", math_inf := float("inf"))
clip_act = agent_cfg["params"]["env"].get("clip_actions", math_inf)
env = RlGamesVecEnvWrapper(env, args.device, clip_obs, clip_act)

import gym as _gym  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402

vecenv.register("IsaacRlgWrapper", lambda cfg_name, nenv, **kw: env)
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

agent_cfg["params"]["config"]["num_actors"] = raw.num_envs
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = args.checkpoint

runner = Runner()
runner.load(agent_cfg)
agent = runner.create_player()
agent.restore(args.checkpoint)
agent.reset()

n = raw.num_envs
D = raw.device
FING = ("thumb", "index", "middle", "ring", "pinky")

obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

# 성공 순간 누적 버퍼
acc = {k: [] for k in ("d_palm", "normal", "finger", "hand_q", "act", "d_tips", "obj_h")}
is_rnn = getattr(agent, "is_rnn", False)
if is_rnn:
    agent.init_rnn()

for t in range(args.steps):
    with torch.no_grad():
        a = agent.get_action(obs, is_deterministic=True)
    obs, _r, dones, _info = env.step(a)
    if isinstance(obs, dict):
        obs = obs["obs"] if "obs" in obs else obs

    oh = raw.object_pos[:, 2] - raw.object_init_pos[:, 2]        # 안착 대비 상승
    hit = oh > args.lift_thresh
    if hit.any():
        idx = hit.nonzero(as_tuple=True)[0]
        obj = raw.object_pos[idx]
        palm = raw.palm_center_pos[idx]
        R = matrix_from_quat(raw.robot.data.body_quat_w[idx, raw.palm_body_index])
        acc["d_palm"].append((palm - obj).cpu())
        acc["normal"].append(R[:, :, 0].cpu())
        acc["finger"].append(R[:, :, 2].cpu())
        acc["hand_q"].append(raw.robot.data.joint_pos[idx][:, raw.hand_dof_indices].cpu())
        acc["act"].append(a[idx].cpu())
        acc["d_tips"].append((raw.fingertip_pos[idx] - obj.unsqueeze(1)).cpu())
        acc["obj_h"].append(oh[idx].cpu())

    if is_rnn and dones.any():
        agent.reset()

N = sum(x.shape[0] for x in acc["obj_h"]) if acc["obj_h"] else 0
print("=" * 88)
print("성공 파지 자세 추출 — %s" % args.task)
print("  체크포인트: %s" % args.checkpoint)
print("  '성공 순간' = 물체가 안착 위치보다 %.0fcm 이상 들린 스텝" % (args.lift_thresh * 100))
print("  수집된 성공 샘플: %d" % N)
print("=" * 88)

if N == 0:
    print("\n  성공 샘플이 없다. lift_thresh 를 낮추거나 체크포인트를 확인하라.")
    _OUT.close(); env.close(); app.close(); raise SystemExit

def cat(k):
    return torch.cat(acc[k], dim=0)

oh = cat("obj_h")
print("\n[리프트 높이]  평균 %.3f m   중앙 %.3f   최대 %.3f" % (oh.mean(), oh.median(), oh.max()))

dp = cat("d_palm").mean(dim=0)
print("\n[palm ← 물체]  (성공 순간)")
print("  d_palm = (%+.4f, %+.4f, %+.4f)   |d| = %.4f" % (*dp, cat("d_palm").norm(dim=-1).mean()))
print("  ※ 내 스크립트 probe 는 (dx, 0, dz) 로 dz=0.02~0.12 만 훑었다. 위 값과 비교하라.")

nm = cat("normal").mean(dim=0)
fg = cat("finger").mean(dim=0)
print("\n[palm 자세]")
print("  손바닥 법선 (+X) = (%+.3f, %+.3f, %+.3f)   ← top-down 이면 z ≈ -1" % (*nm,))
print("  손가락 방향 (+Z) = (%+.3f, %+.3f, %+.3f)" % (*fg,))

dt = cat("d_tips").mean(dim=0)
dd = cat("d_tips").norm(dim=-1).mean(dim=0)
print("\n[손끝 ← 물체]")
for k in range(5):
    print("  %-8s (%+.4f, %+.4f, %+.4f)   거리 %.4f" % (FING[k], *dt[k], dd[k]))

hq = cat("hand_q").mean(dim=0)
print("\n[손 관절 20개]  (성공 순간 평균)")
for k in range(5):
    print("  %-8s " % FING[k] + "  ".join("%+7.3f" % hq[4 * k + j] for j in range(4)))

ac = cat("act").mean(dim=0)
print("\n[action 16개]")
print("  palm 6D   : " + "  ".join("%+6.3f" % ac[i] for i in range(6)))
print("  시너지 5D : " + "  ".join("%+6.3f" % ac[i] for i in range(6, 11)))
print("  abduct 5D : " + "  ".join("%+6.3f" % ac[i] for i in range(11, 16)))

print("\n  → d_palm 과 손 관절이 내 probe 가정과 어디서 갈리는지가 답이다.")

_OUT.close()
env.close()
app.close()
