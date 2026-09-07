#!/usr/bin/env python3
"""pour 정책 결정론 롤아웃 → s2r 재생용 궤적(HDF5 v2) 기록.

**왜 신규인가.** `reinforcement_learning/probes/record_pour_traj.py` 는 컵 포즈와
`joint_pos` 만 담는다. s2r 재생의 진실원천은 측정이 아니라 **지령**(`joint_pos_target`)
이고, grasp 정책이 만든 파지와 pour 초기 파지가 얼마나 어긋나는지 재려면
`cup_in_hand_pose` 가 있어야 한다. 스키마가 다르므로 기존 산출물(회귀 픽스처)을
깨지 않도록 기록기를 새로 둔다.

프레임 정합 규약 — 조용히 어긋나기 쉬운 곳이다:
  · **측정**은 스텝 **전**에 찍는다. `env.step` 이 done env 를 내부에서 reset 하므로
    뒤에서 찍으면 다음 에피소드의 첫 프레임을 이번 에피소드 끝에 붙이게 된다.
  · **지령**은 스텝 **후**에 찍는다. 그 스텝의 액션이 만든 목표값이기 때문이다.
  · done 스텝은 지령이 이미 reset 값이라 **그 한 프레임을 버린다**. 700 프레임 중
    하나를 버리는 대신, 남은 전 프레임에서 (상태, 액션, 지령)이 같은 스텝을 가리킨다.

사용:
  python record_pour_traj_s2r.py \\
      --task open-tesol_r_pour_sensor-play-lstm \\
      --checkpoint <abs .pth> --record_out <out.hdf5> \\
      --record_episodes 8 --eval_steps 6000 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="pour 궤적을 s2r 재생 스키마로 기록한다.")
parser.add_argument("--task", type=str, required=True, help="gym task id (play 계열 권장)")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="체크포인트 절대경로. 태스크 이름 기반 자동탐색은 쓰지 않는다 "
                         "(개명된 런을 재생해야 하므로).")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument(
    "--num_envs", type=int, default=32,
    help="병렬 환경 수. IsaacLab 은 스텝당 고정 오버헤드가 지배적이라 1 env 와 N env 의 "
         "스텝 시간이 거의 같다 — 수집 처리량이 그대로 N 배가 된다. 이 태스크는 "
         "num_envs=2048 · replicate_physics=True 로 학습했으므로 다중 env 가 원래 구성이다.",
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--record_out", type=str, required=True, help="저장할 HDF5 경로")
parser.add_argument("--record_episodes", type=int, default=8,
                    help="저장할 에피소드 수 (bead_frac 상위부터)")
parser.add_argument("--record_collect", type=int, default=0,
                    help="선별 전 수집할 에피소드 수 (0 = record_episodes*3)")
parser.add_argument("--eval_steps", type=int, default=8000, help="롤아웃 상한 스텝")
parser.add_argument(
    "--adr_progress", type=float, default=1.0,
    help="ADR 스케줄러를 이 진행도(0~1)로 **고정**한다. -1 = 손대지 않음(자연 램프). "
         "★play 세션은 step 카운터가 0에서 시작해 ADR 레벨도 0부터 램프한다. pour 의 조준/성공 "
         "기준은 ADR 로 anneal 되므로, 고정하지 않으면 학습 종료 시점(progress 1.0)이 아니라 "
         "학습 **시작** 난이도로 재생된다 — 실측: 고정 없이 bead 0.00, 고정 시 학습 대역 회복.",
)
parser.add_argument(
    "--keep_obs_noise", action="store_true", default=False,
    help="관측 노이즈 ADR 도 켠 채로 둔다. 기본은 꺼서 결정론 궤적을 얻는다 "
         "(노이즈는 난이도가 아니라 재현성의 문제라 조준 ADR 과 분리해 다룬다).",
)
parser.add_argument("--bead_fixed", type=int, default=None, help="bead 수 고정")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--fabric_params", type=str, default=None,
    help="Fabrics params yaml 파일명 재정의. a1 체크포인트를 학습 당시 충돌구 배치로 재생하려면 "
         "openarm_tesollo_pose_params_pre0823.yaml (공유 기본값은 08.23 이후 a2 링크명을 쓴다).",
)
parser.add_argument(
    "--warm_state_paths", type=str, nargs="+", default=None,
    help="warm state 뱅크 경로 재지정. 학습 당시 경로는 그 머신의 절대경로라 여기서 못 찾을 수 "
         "있고, 08.17 격리로 옮겨진 뱅크도 있다. 뱅크 없이 떨어지면 pour 가 컵을 든 자세로 "
         "시작하지 않아 궤적 자체가 무의미해지므로, 조용한 degrade 대신 명시 지정을 받는다.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# ★openarm/tools 경로는 **isaaclab 을 import 하기 전에** 확정한다. `import isaaclab_tasks`
#   가 확장 진입점을 훑으며 `openarm` 을 먼저 import 해 버리면, 그 뒤에 sys.path 를 고쳐도
#   sys.modules 에 남은 다른 트리의 모듈이 이긴다. 과거 커밋을 worktree 로 재현할 때
#   조용히 **현재** 소스로 돌아 버리는 사고가 여기서 난다.
import os  # noqa: E402
from pathlib import Path  # noqa: E402

_HDGP = Path(__file__).resolve().parents[2]
_OPENARM_SRC = str(_HDGP / "source/openarm")
_TOOLS_SRC = str(_HDGP / "scripts/tools")
for _p in (_TOOLS_SRC, _OPENARM_SRC):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
for _name in [m for m in sys.modules if m == "openarm" or m.startswith("openarm.")]:
    del sys.modules[_name]

from datetime import datetime  # noqa: E402
import gymnasium as gym  # noqa: E402
import hashlib  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import openarm  # noqa: E402
_EXPECTED = str((_HDGP / "source/openarm/openarm").resolve())
if not str(Path(openarm.__file__).resolve()).startswith(_EXPECTED + os.sep):
    raise SystemExit(f"openarm 이 저장소 밖에서 왔다: {openarm.__file__} (기대 {_EXPECTED})")
import openarm.tasks  # noqa: F401,E402

import pour_traj_capture as PC  # noqa: E402
import pour_traj_io as PT  # noqa: E402
from run_cfg_restore import restore_run_cfg_if_available  # noqa: E402

#: env 가 노출하는 ADR 스케줄러 속성 이름. 학습 종료 레벨로 고정할 대상이다.
_ADR_ATTRS = ("spill_adr", "noise_adr", "success_adr", "outcome_adr")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_HDGP), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001 — 기록 자체를 막을 이유는 아니다.
        return ""


def _apply_overrides(env_cfg) -> None:
    """기록용 재정의. 재현 가능성이 전제고, 수집량은 env 수로 번다."""
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # ★조준/성공 ADR 플래그는 **끄지 않는다.** 끄면 스케줄러 객체가 None 이 되어 env 가
    #   초기값 상수로 떨어지는데, 그것이 학습 시작 난이도라 정책이 전혀 못 붓는다.
    #   대신 아래 `_pin_adr` 로 학습 종료 레벨에 고정한다. 노이즈만 재현성을 위해 끈다.
    if not args_cli.keep_obs_noise and hasattr(env_cfg, "enable_noise_adr"):
        env_cfg.enable_noise_adr = False
        print("[REC] 관측 노이즈 ADR off (결정론 궤적)")

    if args_cli.fabric_params is not None:
        env_cfg.fabric_params_filename = args_cli.fabric_params
        print(f"[REC] Fabrics params 재정의: {args_cli.fabric_params}")

    if args_cli.warm_state_paths is not None:
        resolved = tuple(str(Path(p).expanduser().resolve()) for p in args_cli.warm_state_paths)
        for path in resolved:
            if not Path(path).is_file():
                raise SystemExit(f"warm state 뱅크가 없다: {path}")
        env_cfg.warm_state_paths = resolved
        env_cfg.warm_state_source = "disk"
        print(f"[REC] warm state 뱅크 재지정: {resolved}")

    if args_cli.bead_fixed is not None:
        if hasattr(env_cfg, "bead_count_min") and hasattr(env_cfg, "bead_count_max"):
            env_cfg.bead_count_min = env_cfg.bead_count_max = args_cli.bead_fixed
        elif hasattr(env_cfg, "bead_count"):
            env_cfg.bead_count = args_cli.bead_fixed
        else:
            raise SystemExit("--bead_fixed 를 받을 설정이 이 env 에 없다")
        print(f"[REC] bead 고정: {args_cli.bead_fixed}")


def _pin_adr(raw_env, progress: float) -> None:
    """ADR 스케줄러를 지정 진행도로 고정한다.

    학습이 끝난 정책은 **그때의 ADR 레벨**에서만 그 성능을 낸다. play 세션은 step 카운터가
    0이라 레벨도 0부터 다시 램프하므로, 고정하지 않으면 앞쪽 에피소드가 통째로 버려진다
    (실측: 고정 없이 bead 0.00 → 0.05 → 0.90 으로 램프하며 회복).
    """
    if progress < 0:
        print("[REC] ADR 고정 안 함 (자연 램프)")
        return
    pinned = []
    for name in _ADR_ATTRS:
        adr = getattr(raw_env, name, None)
        if adr is None:
            continue
        adr.set_increment(int(round(adr.num_increments * progress)))
        pinned.append(f"{name}={adr.progress:.2f}")
    if not pinned:
        raise SystemExit(
            f"ADR 스케줄러를 하나도 못 찾았다 (기대: {_ADR_ATTRS}). "
            "env 가 바뀌었는지 확인할 것 — 조용히 넘기면 학습 시작 난이도로 재생된다."
        )
    print(f"[REC] ADR 고정: {' · '.join(pinned)}")


def _build_agent(env, agent_cfg: dict, resume_path: str) -> BasePlayer:
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()
    return agent


def _assert_contract(agent: BasePlayer, env, layout: PC.TaskLayout) -> None:
    """체크포인트가 **이 env 로** 학습된 것인지 실행 전에 확인한다."""
    obs_shape = agent.model.obs_shape
    obs_dim = int(obs_shape[0] if isinstance(obs_shape, (tuple, list)) else obs_shape)
    env_obs = int(env.unwrapped.num_obs) if hasattr(env.unwrapped, "num_obs") else obs_dim
    action_dim = int(agent.actions_num)
    print(f"[REC] 계약: obs {obs_dim} (env {env_obs}) · action {action_dim} "
          f"· RNN {agent.is_rnn} · 레이아웃 {layout.name}(action {layout.num_actions})")
    if action_dim != layout.num_actions:
        raise SystemExit(
            f"액션 차원 불일치: 체크포인트 {action_dim} != 레이아웃 {layout.num_actions}. "
            "태스크와 체크포인트가 짝이 맞는지 확인할 것."
        )
    if obs_dim != env_obs:
        raise SystemExit(f"관측 차원 불일치: 체크포인트 {obs_dim} != env {env_obs}")


class _EnvBuffer:
    """env 하나의 진행 중인 에피소드. 측정/액션/지령을 스텝 단위로 쌓는다."""

    __slots__ = ("states", "actions", "commands")

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self.states: list[dict] = []
        self.actions: list[np.ndarray] = []
        self.commands: list[dict] = []

    @property
    def n_closed(self) -> int:
        """(상태, 액션, 지령)이 모두 갖춰진 스텝 수."""
        return min(len(self.states), len(self.actions), len(self.commands))


def _rollout(env, agent: BasePlayer, capture: PC.PourTrajCapture, collect: int) -> list[PT.Episode]:
    """결정론 롤아웃. **env 별 버퍼**로 갈라 담고 done 이 뜬 env 만 닫는다.

    ★단일 env 로 묶지 않는다. IsaacLab 은 스텝당 고정 오버헤드가 지배적이라
      1 env 와 N env 의 스텝 시간이 거의 같다(1 env 실측 0.5 s/step · GPU 3%).
      env 마다 done 시점이 다르므로 경계 처리를 여기서 진다.
    """
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    n_envs = capture.num_envs
    buffers = [_EnvBuffer() for _ in range(n_envs)]
    episodes: list[PT.Episode] = []

    for _ in range(args_cli.eval_steps):
        state = capture.state_frame()
        for i, buf in enumerate(buffers):
            buf.states.append({k: v[i] for k, v in state.items()})

        with torch.inference_mode():
            action = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            action_np = action.detach().float().cpu().numpy().copy()
            for i, buf in enumerate(buffers):
                buf.actions.append(action_np[i])
            obs, _, dones, _ = env.step(action)
            done_np = dones.detach().cpu().numpy().astype(bool).reshape(-1)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for hidden in agent.states:
                    hidden[:, dones, :] = 0.0

        command = capture.command_frame()
        bead, spill = capture.outcome()
        for i, buf in enumerate(buffers):
            if not done_np[i]:
                buf.commands.append({k: v[i] for k, v in command.items()})
                continue
            # done 스텝의 지령은 이미 reset 값이다 → 그 스텝의 (상태, 액션)까지 버린다.
            buf.states.pop()
            buf.actions.pop()
            if buf.n_closed > 0:
                episodes.append(_close_episode(buf, float(bead[i]), float(spill[i])))
                print(f"[REC] 에피소드 {len(episodes)}/{collect} env{i} "
                      f"T={episodes[-1].n_steps} bead={bead[i]:.2f} spill={spill[i]:.2f}",
                      flush=True)
            buf.clear()
        if len(episodes) >= collect:
            break

    if not episodes:
        raise SystemExit(f"{args_cli.eval_steps} 스텝 안에 끝난 에피소드가 없다. eval_steps 를 늘릴 것.")
    return episodes


def _close_episode(buf: _EnvBuffer, bead: float, spill: float) -> PT.Episode:
    n = buf.n_closed
    arrays = {k: np.stack([s[k] for s in buf.states[:n]]) for k in buf.states[0]}
    arrays.update({k: np.stack([c[k] for c in buf.commands[:n]]) for k in buf.commands[0]})
    arrays["action"] = np.stack(buf.actions[:n]).astype(np.float32)
    return PT.Episode(
        arrays=arrays,
        bead_frac=float(np.clip(bead, 0.0, 1.0)),
        bead_spill=float(np.clip(spill, 0.0, 1.0)),
        seed=int(args_cli.seed) if args_cli.seed is not None else -1,
        success=None,
    )


def _build_meta(env, capture: PC.PourTrajCapture, layout: PC.TaskLayout,
                resume_path: Path, log_dir: Path) -> PT.TrajMeta:
    raw = env.unwrapped
    while hasattr(raw, "env"):
        raw = raw.env.unwrapped
    env_yaml = log_dir / "params" / "env.yaml"
    robot_usd = ""
    try:
        robot_usd = str(raw.cfg.robot_cfg.spawn.usd_path)
    except Exception:  # noqa: BLE001
        pass
    step_dt = float(getattr(raw, "step_dt", raw.physics_dt * raw.cfg.decimation))
    root_pos = (raw.robot.data.root_pos_w[0] - raw.scene.env_origins[0]).cpu().numpy()
    root_quat = raw.robot.data.root_quat_w[0].cpu().numpy()

    return PT.TrajMeta(
        task_id=args_cli.task.split(":")[-1],
        checkpoint=resume_path.name,
        checkpoint_sha256=_sha256(resume_path),
        git_commit=_git_commit(),
        robot_usd=robot_usd,
        env_yaml_sha256=_sha256(env_yaml) if env_yaml.is_file() else "",
        dt=step_dt,
        decimation=int(raw.cfg.decimation),
        num_beads=int(getattr(raw, "num_beads", getattr(raw.cfg, "bead_count", 0))),
        env_origin=raw.scene.env_origins[0].cpu().numpy(),
        robot_root=np.concatenate([root_pos, root_quat]),
        joint_names=capture.joint_names,
        body_names=capture.body_names,
        right_arm_joint_names=capture.groups.right_arm,
        right_hand_joint_names=capture.groups.right_hand,
        left_arm_joint_names=capture.groups.left_arm,
        left_gripper_joint_names=capture.groups.left_gripper,
        right_palm_body=PC.RIGHT_PALM_BODY,
        left_ee_body=PC.LEFT_EE_BODY,
        recorded_at=datetime.now().isoformat(timespec="seconds"),
        missing_channels=layout.missing_channels,
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    layout = PC.layout_for(args_cli.task)

    resume_path = Path(args_cli.checkpoint).expanduser().resolve()
    if not resume_path.is_file():
        raise SystemExit(f"체크포인트가 없다: {resume_path}")
    log_dir = resume_path.parent.parent

    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg = restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=str(resume_path),
        workspace_root=str(_HDGP.parent),
    )
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["params"]["seed"]
    _apply_overrides(env_cfg)
    env_cfg.log_dir = str(log_dir)

    device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concat_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions, obs_groups, concat_groups)

    agent = _build_agent(env, agent_cfg, str(resume_path))
    _assert_contract(agent, env, layout)

    raw = env.unwrapped
    while hasattr(raw, "env"):
        raw = raw.env.unwrapped
    _pin_adr(raw, args_cli.adr_progress)
    capture = PC.PourTrajCapture(raw, layout)
    print(f"[REC] 병렬 환경 {capture.num_envs}개")

    collect = args_cli.record_collect or (args_cli.record_episodes * 3)
    print(f"[REC] 수집 목표 {collect} 에피소드 → 상위 {args_cli.record_episodes} 저장", flush=True)
    episodes = _rollout(env, agent, capture, collect)

    episodes.sort(key=lambda e: -e.bead_frac)
    best = tuple(episodes[: args_cli.record_episodes])
    meta = _build_meta(env, capture, layout, resume_path, log_dir)

    out = PT.write_traj(args_cli.record_out, meta, best)
    check_meta, check_eps = PT.read_traj(out)
    problems = PT.validate(check_meta, check_eps)
    if problems:
        raise SystemExit("[REC] 저장본이 계약을 어긴다:\n  " + "\n  ".join(problems))

    fracs = [e.bead_frac for e in best]
    lengths = [e.n_steps for e in best]
    print("\n" + "=" * 68)
    print(f"[REC] 저장 {len(best)}/{len(episodes)} 에피소드 → {out}")
    print(f"[REC] bead 평균 {np.mean(fracs):.3f}  범위 [{min(fracs):.2f}, {max(fracs):.2f}]")
    print(f"[REC] 길이 평균 {np.mean(lengths):.0f} step ({np.mean(lengths)*meta.dt:.2f} s)")
    print(f"[REC] missing_channels: {meta.missing_channels or '(없음)'}")
    print("=" * 68)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
