"""fab 계열(재소환 없음) 체크포인트를 **한 번의 sim 기동으로** 줄세운다.

왜 필요한가: `probe_lift_left_policy_contact.py` 에는 **바닥 긁힘 계측이 없다.**
그래서 fab 정책들이 테이블을 얼마나 긁는지 아무도 잰 적이 없다(v2 프로브에만 있었다).
여기서 v2 와 **같은 정의**로 재서 두 트랙을 직접 비교할 수 있게 한다.

  손끝 최저 높이 = min(턱 두 링크 z, TCP z) − env원점 z − TABLE_SURFACE_Z

같은 env 를 재사용하고 체크포인트만 갈아끼운다 — 기동 비용(수 분)을 한 번만 낸다.

실행:
  ../IsaacLab/isaaclab.sh -p scripts/probes/probe_fab_lift_quality.py \
      --checkpoints log/checkpoints_fab/t65.pth,log/checkpoints_fab/t79.pth
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoints", type=str, required=True, help="쉼표 구분 경로 목록")
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor_fab")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402


def pct(x):
    return f"{float(x) * 100:5.1f}%"


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if hasattr(env_cfg.actions, "arm_action"):
        env_cfg.actions.arm_action.debug_vis = False   # 종료 시 scene 참조 크래시 회피
    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg)
    raw = env.unwrapped
    inf = float("inf")
    wrapped = RlGamesVecEnvWrapper(
        env, args.device,
        agent_cfg["params"]["env"].get("clip_observations", inf),
        agent_cfg["params"]["env"].get("clip_actions", inf))
    vecenv.register("IsaacRlgWrapper", lambda a, b, **kw: RlGamesGpuEnv(a, b, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["config"]["env_info"] = {
        "observation_space": wrapped.observation_space,
        "action_space": wrapped.action_space, "agents": 1}
    hz = int(agent_cfg["params"]["config"].get("horizon_length", 24))
    agent_cfg["params"]["config"]["minibatch_size"] = args.num_envs * hz

    robot = raw.scene["robot"]; obj = raw.scene["object"]; eef = raw.scene["ee_frame"]
    jaw = SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES))
    jaw.resolve(raw.scene)
    base_bi = robot.body_names.index(P.GRIPPER_BASE_BODY)
    dev, N = args.device, args.num_envs

    rows = []
    for ck in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        name = os.path.basename(ck).replace(".pth", "")
        runner = Runner(); runner.load(agent_cfg)
        agent = runner.create_player()
        try:
            agent.restore(ck)
        except Exception as e:                       # 구조가 다른 세대는 건너뛴다
            print(f"[fab] {name}: 로드 실패 — {str(e)[:70]}", flush=True)
            continue
        agent.reset()
        # ★reset 도 `inference_mode` 안에서 해야 한다. 앞 체크포인트의 롤아웃이
        #   inference_mode 로 돌면 씬 텐서가 inference 텐서가 되고, 그 밖에서
        #   in-place 로 쓰면 RuntimeError 가 난다(체크포인트 2 개째부터 죽는다).
        with torch.inference_mode():
            obs = wrapped.reset()
        obs = obs["obs"] if isinstance(obs, dict) else obs
        agent.get_batch_size(obs, 1)
        if getattr(agent, "is_rnn", False):
            agent.init_rnn()

        tip_min = torch.full((N,), 9.9, device=dev)
        max_rise = torch.zeros(N, device=dev)
        min_cos = torch.ones(N, device=dev)
        ang_sum = torch.zeros(N, device=dev); ang_n = torch.zeros(N, device=dev)
        lock = torch.zeros(N, dtype=torch.bool, device=dev)
        with torch.inference_mode():
            for _ in range(args.steps):
                act = agent.get_action(obs, is_deterministic=True)
                obs, _, dones, _ = wrapped.step(act)
                obs = obs["obs"] if isinstance(obs, dict) else obs
                live = ~lock
                jz = robot.data.body_pos_w[:, jaw.body_ids, 2]
                ez = eef.data.target_pos_w[:, 0, 2]
                low = (torch.minimum(jz.min(dim=1).values, ez)
                       - raw.scene.env_origins[:, 2] - P.TABLE_SURFACE_Z)
                tip_min = torch.where(live, torch.minimum(tip_min, low), tip_min)
                rise = obj.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2] - P.CUP_SPAWN_Z
                max_rise = torch.where(live, torch.maximum(max_rise, rise), max_rise)
                q = obj.data.root_quat_w
                cos = (1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1, 1)
                min_cos = torch.where(live, torch.minimum(min_cos, cos), min_cos)
                gq = robot.data.body_quat_w[:, base_bi, :]
                az = (1.0 - 2.0 * (gq[:, 1] ** 2 + gq[:, 2] ** 2)).clamp(-1, 1)
                pre = live & (rise < 0.01)
                ang_sum += torch.where(pre, torch.rad2deg(torch.acos(az)),
                                       torch.zeros_like(az))
                ang_n += pre.float()
                d = dones.bool() if torch.is_tensor(dones) else torch.as_tensor(dones, device=dev).bool()
                lock |= d.reshape(-1)

        tm = tip_min * 1000.0
        rows.append((name,
                     float((max_rise > 0.01).float().mean()),
                     float(max_rise.median()) * 1000,
                     float(tm.median()), float(tm.min()),
                     float((tm < 20).float().mean()), float((tm < 10).float().mean()),
                     float((min_cos < 0.5).float().mean()),
                     float((ang_sum / ang_n.clamp(min=1)).median()),
                     float(lock.float().mean())))
        print(f"[fab] {name} 완료", flush=True)

    print("\n" + "=" * 104)
    print("fab 계열 비교 — 64 env · 250 step · 결정론 · 재소환 없음(전도=종료)")
    print("=" * 104)
    print(f"{'체크포인트':<12}{'리프트':>8}{'상승중앙':>10}{'손끝중앙':>10}{'손끝최소':>10}"
          f"{'<20mm':>8}{'<10mm':>8}{'전도':>8}{'접근각':>9}{'조기종료':>9}")
    for r in rows:
        print(f"{r[0]:<12}{pct(r[1]):>8}{r[2]:9.1f}mm{r[3]:9.1f}mm{r[4]:9.1f}mm"
              f"{pct(r[5]):>8}{pct(r[6]):>8}{pct(r[7]):>8}{r[8]:8.1f}°{pct(r[9]):>9}")
    print("\n★손끝 = min(턱 두 링크, TCP) 의 판 위 높이 — v2 프로브와 같은 정의")
    print("★접근각 = 그리퍼 +z ∠ world +z, 리프트 전 구간 평균 (90° = 수직)")
    print("★조기종료 = 전도·낙하로 에피소드가 끊긴 비율 (재소환이 없으므로 곧 실패율)")
    env.close()


main()
app.close()
