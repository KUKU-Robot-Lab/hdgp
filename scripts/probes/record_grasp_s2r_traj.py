#!/usr/bin/env python3
"""grasp_s2r 정책을 **지정한 컵 소환 (X, Y)** 에서 결정론 롤아웃 → 궤적(HDF5 v1)+영상.

용도: 실기 s2r 이 아직 없는 우팔(`DEPLOY_CONTRACT.md` "아직 없는 것" 1~4)을,
정책 추론 없이 **관절 지령 궤적 재생**으로 돌리기 위한 소스를 뽑는다.
위치 하나당 한 번 실행한다 — 3위치 격자는 `run_grasp_s2r_traj_grid.sh` 가 돌린다.

소환 위치를 바꾸는 방법 (여기가 조용히 어긋나기 쉬운 곳이다):
  `robot_profiles.PROFILES` 의 `object_spawn_center` 를 **gym.make 전에** 교체한다.
  env 는 `_reset_idx` 에서 `self.profile.object_spawn_center` 를, cfg 는
  `finalize_after_overrides` 에서 같은 값을 읽으므로 한 곳만 바꾸면 스폰·목표·
  액션 앵커·부팅 도달성 가드가 전부 같은 값을 본다. env 코드는 건드리지 않는다.

컵 1.0 을 고르는 방법:
  물체 뱅크를 `single_cup` 으로 **바꾸지 않는다**. 바꾸면 `replicate_physics` 가
  True 로, 테이블이 `env_rigid.usd` → `env.usd` 로 돌아가 학습과 물리 구성이
  달라진다. 대신 학습 그대로(`cup_family`, env_id % 8) 두고 **scale 1.00 인 종에
  배정된 env 만 기록**한다. 뱅크를 바꾸고 싶으면 `--object_bank` 로 명시할 것.

사용:
  python record_grasp_s2r_traj.py \\
      --checkpoint <abs .pth> --spawn_y -0.21 \\
      --record_out <out.hdf5> --video --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="grasp_s2r 궤적을 소환 위치별로 기록한다.")
parser.add_argument("--task", type=str, default="open-sens_r_grasp_s2r-play-lstm",
                    help="gym task id. g1 은 LSTM 이라 -play-lstm 이 기본이다.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="체크포인트 절대경로. 이름 기반 자동탐색은 쓰지 않는다.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=16,
                    help="병렬 환경 수. 종 배정이 env_id %% 8 이라 목표 종을 뽑으려면 "
                         "8 이상이어야 한다.")
parser.add_argument("--seed", type=int, default=None)

parser.add_argument("--spawn_x", type=float, default=None,
                    help="컵 소환 중심 x (env-local, m). 생략 = 프로필 기본값.")
parser.add_argument("--spawn_y", type=float, default=None,
                    help="컵 소환 중심 y (env-local, m). 생략 = 프로필 기본값.")
parser.add_argument("--spawn_range", type=float, default=0.0,
                    help="소환 xy 균등 반범위. **기본 0 = 결정론** — 이 스크립트의 산출물은 "
                         "(X, Y) 라벨이 붙은 궤적이라 흔들면 라벨이 거짓이 된다. "
                         "결정론 롤아웃이 실패하면 0.01 정도로 흔들어 재시도할 것.")
parser.add_argument("--object_bank", type=str, default=None,
                    help="물체 뱅크 재정의. 기본 = 런 dump 그대로(g1 은 cup_family).")
parser.add_argument("--object_species", type=str, default="cup_big_s100",
                    help="기록할 물체 종 id. 이 종에 배정된 env 만 남긴다. "
                         "★스케일로 고르면 안 된다 — cup_family 에는 scale 1.0 이 둘이다 "
                         "(cup_big_s100 과 shaker_closed).")

parser.add_argument("--record_out", type=str, required=True, help="저장할 HDF5 경로")
parser.add_argument("--record_episodes", type=int, default=1,
                    help="저장할 에피소드 수 (성공 tail 긴 순).")
parser.add_argument("--eval_steps", type=int, default=1400, help="롤아웃 상한 스텝")
parser.add_argument("--keep_obs_noise", action="store_true", default=False,
                    help="관측 노이즈를 켠 채로 둔다. 기본은 꺼서 결정론 궤적을 얻는다.")
parser.add_argument("--keep_respawn", action="store_true", default=False,
                    help="실패 시 재소환을 켠 채로 둔다. 기본은 끈다 — 컵이 다른 자리로 "
                         "옮겨지면 이 파일의 (X, Y) 라벨이 거짓이 된다.")

parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=600,
                    help="영상 길이(스텝). episode_length_s 10.0 · 60Hz = 600 이 한 에피소드.")
parser.add_argument("--video_out", type=str, default=None,
                    help="영상 폴더. 생략 = <record_out 폴더>/videos")
parser.add_argument("--video_eye", type=float, nargs=3, default=(1.00, -0.88, 0.74),
                    help="뷰어 카메라 위치 (env-local). 기본값은 파지·리프트가 테이블 "
                         "기둥에 안 가리는 전방 우측 시점이다.")
parser.add_argument("--video_lookat", type=float, nargs=3, default=(0.37, -0.16, 0.34),
                    help="뷰어 카메라 목표점 (env-local). 기본값 = 컵 파지 높이.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# ★openarm/tools 경로는 **isaaclab 을 import 하기 전에** 확정한다. `import isaaclab_tasks`
#   가 확장 진입점을 훑으며 `openarm` 을 먼저 import 해 버리면, 그 뒤에 sys.path 를 고쳐도
#   sys.modules 에 남은 다른 트리의 모듈이 이긴다.
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

import re as _re  # noqa: E402


def _pin_gain_branch_from_dump() -> None:
    """게인 분기(`HDGP_S2R_REAL_GAINS`)를 런 dump 에 맞춘다. **openarm import 전에** 부른다.

    `robot_profiles.py` 는 이 변수를 **import 시점**에 한 번 읽어 액추에이터 게인을
    고른다. cfg 의 `use_real_gains` 와 어긋나면 `_assert_gain_branch` 가 부팅을 막는데,
    막히는 것이 옳다 — 재생은 학습과 **같은 게인**에서만 같은 궤적을 낸다.
    진실원천은 코드 기본값이 아니라 dump 다(DEPLOY_CONTRACT: 상수를 손으로 옮기지 말 것).
    """
    env_yaml = Path(args_cli.checkpoint).expanduser().resolve().parent.parent / "params" / "env.yaml"
    if not env_yaml.is_file():
        print(f"[REC] ⚠ 런 dump 가 없다 ({env_yaml}) — 게인 분기를 코드 기본값에 맡긴다")
        return
    # yaml 로더를 쓰지 않는다 — dump 에 `!!python/tuple` 태그가 있어 안전 로더로는 못 읽는다.
    match = _re.search(r"^use_real_gains:\s*(true|false)\s*$",
                       env_yaml.read_text(), flags=_re.MULTILINE)
    if match is None:
        print("[REC] ⚠ dump 에 use_real_gains 가 없다 — 게인 분기를 코드 기본값에 맡긴다")
        return
    real = match.group(1) == "true"
    os.environ["HDGP_S2R_REAL_GAINS"] = "1" if real else "0"
    print(f"[REC] 게인 분기 = {'실측 정합(r2s)' if real else 'KUKA 기본'} (dump 기준)")


_pin_gain_branch_from_dump()

import dataclasses  # noqa: E402
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
from openarm.agnostic.modules import object_bank as _ob  # noqa: E402
from openarm.agnostic.tasks.grasp_s2r import robot_profiles as _rp  # noqa: E402

import grasp_traj_io as GT  # noqa: E402
from run_cfg_restore import restore_run_cfg_if_available  # noqa: E402

PALM_BODY = "r_hl_palm"
#: 관측 노이즈 cfg 필드 — 결정론 롤아웃에서 0 으로 내린다.
_NOISE_FIELDS = ("obs_noise_qpos", "obs_noise_qvel", "obs_noise_body", "obs_noise_object")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_HDGP), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 — 기록 자체를 막을 이유는 아니다.
        return ""


def _patch_spawn_center(profile_name: str) -> tuple[float, float]:
    """프로필의 소환 중심을 CLI 값으로 교체하고 최종 중심을 돌려준다.

    ★cfg 조립(`finalize_after_overrides`)과 env(`_reset_idx`)가 **같은 객체**를 읽으므로
      여기 한 번이면 스폰·목표·액션 앵커·부팅 도달성 가드가 전부 따라온다.
      env 를 만든 뒤에 바꾸면 cfg 쪽(초기 스폰 위치·앵커 검사)이 옛 값으로 남는다.
    """
    base = _rp.PROFILES[profile_name]
    x = float(args_cli.spawn_x) if args_cli.spawn_x is not None else float(base.object_spawn_center[0])
    y = float(args_cli.spawn_y) if args_cli.spawn_y is not None else float(base.object_spawn_center[1])
    _rp.PROFILES[profile_name] = dataclasses.replace(base, object_spawn_center=(x, y))
    print(f"[REC] 컵 소환 중심 = ({x:.4f}, {y:.4f}) "
          f"(프로필 기본 {tuple(round(float(v), 4) for v in base.object_spawn_center)})")
    return x, y


def _apply_overrides(env_cfg) -> None:
    """기록용 재정의. 전부 "라벨을 참으로 유지한다"는 한 가지 이유에서 나온다."""
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.object_bank is not None:
        env_cfg.object_bank = args_cli.object_bank
        print(f"[REC] 물체 뱅크 재정의: {args_cli.object_bank}")

    env_cfg.spawn_range = float(args_cli.spawn_range)
    env_cfg.enable_adr = False          # 레벨 램프가 스폰 범위를 되살리지 못하게.
    print(f"[REC] spawn_range={env_cfg.spawn_range} · enable_adr=False")

    if not args_cli.keep_respawn:
        env_cfg.respawn_on_fail = False
        print("[REC] respawn_on_fail=False (실패 후 컵 이동 금지 — (X,Y) 라벨 보존)")

    if not args_cli.keep_obs_noise:
        for field in _NOISE_FIELDS:
            if hasattr(env_cfg, field):
                setattr(env_cfg, field, 0.0)
        print("[REC] 관측 노이즈 0 (결정론 궤적)")


def _build_agent(env, agent_cfg: dict, resume_path: str) -> BasePlayer:
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
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


def _assert_contract(agent: BasePlayer, env) -> None:
    """체크포인트가 **이 env 로** 학습된 것인지 실행 전에 확인한다."""
    obs_shape = agent.model.obs_shape
    obs_dim = int(obs_shape[0] if isinstance(obs_shape, (tuple, list)) else obs_shape)
    env_obs = int(env.unwrapped.num_obs) if hasattr(env.unwrapped, "num_obs") else obs_dim
    print(f"[REC] 계약: obs {obs_dim} (env {env_obs}) · action {int(agent.actions_num)} "
          f"· RNN {agent.is_rnn}")
    if obs_dim != env_obs:
        raise SystemExit(f"관측 차원 불일치: 체크포인트 {obs_dim} != env {env_obs}")


def _target_envs(bank_name: str, num_envs: int, species: str) -> tuple[list[int], float]:
    """기록할 env 인덱스와 그 종의 스케일. 종이 뱅크에 없으면 이름을 대고 거부한다.

    ★env 를 만들기 **전에** 부른다 — 영상 뷰어의 `env_index` 를 여기서 정해야 하고,
      뷰어 cfg 는 씬 생성 시점에 소비되므로 뒤에서 고치면 안 먹는다.
      배정 규칙은 env 가 쓰는 `bank.assign_indices` 그대로다(env_id % N).
    """
    bank = _ob.get(bank_name)
    slots = [i for i, spec in enumerate(bank.specs) if spec.id == species]
    if not slots:
        raise SystemExit(
            f"뱅크 '{bank.name}' 에 종 '{species}' 가 없다. "
            f"있는 것: {[s.id for s in bank.specs]}")
    assigned = list(bank.assign_indices(num_envs))
    envs = [i for i, sid in enumerate(assigned) if sid == slots[0]]
    if not envs:
        raise SystemExit(
            f"종 '{species}' 에 배정된 env 가 없다 — --num_envs 를 {len(bank.specs)} 이상으로.")
    scale = float(bank.specs[slots[0]].scale[0])
    print(f"[REC] 기록 대상 env {envs} (종 '{species}' scale {scale} · 뱅크 {bank.name})")
    return envs, scale


def _assert_species(raw, targets: list[int], species: str) -> None:
    """env 가 실제로 배정한 종이 우리가 고른 것과 같은지 대조한다(조용한 어긋남 차단)."""
    assigned = raw._species_ids.cpu().numpy()
    names = list(raw._species_names)
    wrong = [i for i in targets if names[int(assigned[i])] != species]
    if wrong:
        raise SystemExit(
            f"종 배정이 예상과 다르다 — env {wrong} 는 '{species}' 가 아니다. "
            "뱅크 정의가 바뀌었는지 확인할 것.")


# ---------------------------------------------------------------------------
# 프레임 캡처
# ---------------------------------------------------------------------------

def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().copy()


class _Capture:
    """env 전 환경에서 프레임 채널을 배치로 뽑는다. 규약은 `grasp_traj_io` 의 것이다."""

    def __init__(self, raw):
        self.env = raw
        self.arm_ids = torch.as_tensor(raw.arm_ids, device=raw.device, dtype=torch.long)
        self.hand_ids = torch.as_tensor(raw.hand_ids, device=raw.device, dtype=torch.long)
        names = list(raw.robot.data.joint_names)
        self.arm_joint_names = tuple(names[i] for i in raw.arm_ids)
        self.hand_joint_names = tuple(names[i] for i in raw.hand_ids)

    def state_frame(self) -> dict[str, np.ndarray]:
        """스텝 **전** 측정. env.step 이 done env 를 내부에서 reset 하므로 앞에서 찍는다."""
        env = self.env
        data = env.robot.data
        origin = env.scene.env_origins
        palm_pos = data.body_pos_w[:, env.palm_idx] - origin
        palm_quat = data.body_quat_w[:, env.palm_idx]
        obj_pos = env.object.data.root_pos_w - origin
        obj_quat = env.object.data.root_quat_w
        frame = {
            "arm_q": data.joint_pos[:, self.arm_ids],
            "arm_qd": data.joint_vel[:, self.arm_ids],
            "hand_q": data.joint_pos[:, self.hand_ids],
            "hand_qd": data.joint_vel[:, self.hand_ids],
            "palm_pose": torch.cat([palm_pos, palm_quat], dim=-1),
            "object_pose": torch.cat([obj_pos, obj_quat], dim=-1),
        }
        return {k: _np(v) for k, v in frame.items()}

    def command_frame(self) -> dict[str, np.ndarray]:
        """스텝 **후** 지령과 그 스텝의 판정. 이 스텝의 액션이 만든 값이다."""
        env = self.env
        target = env.robot.data.joint_pos_target
        obj_pos = env.object.data.root_pos_w - env.scene.env_origins
        frame = {
            "arm_q_cmd": target[:, self.arm_ids],
            "hand_q_cmd": target[:, self.hand_ids],
            "palm_cmd": env.palm_targets[:, :6],
            "success": env._success_now.float(),
            "stay_run": env._stay_run.float(),
            "goal_dist": (obj_pos - env.goal_pos).norm(dim=-1),
            "obj_height_delta": obj_pos[:, 2] - env.object_spawn_pos[:, 2],
        }
        return {k: _np(v) for k, v in frame.items()}


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
        return min(len(self.states), len(self.actions), len(self.commands))


def _tail_run(flags: np.ndarray) -> int:
    """마지막 연속 True 구간의 길이. 0 이면 끝에서 성공하지 못한 것이다."""
    run = 0
    for value in flags[::-1]:
        if value < 0.5:
            break
        run += 1
    return run


def _close_episode(buf: _EnvBuffer, env_index: int) -> GT.GraspEpisode:
    n = buf.n_closed
    arrays = {k: np.stack([s[k] for s in buf.states[:n]]) for k in buf.states[0]}
    arrays.update({k: np.stack([c[k] for c in buf.commands[:n]]) for k in buf.commands[0]})
    arrays["action"] = np.stack(buf.actions[:n]).astype(np.float32)
    success = arrays["success"]
    return GT.GraspEpisode(
        arrays=arrays,
        success_steps=int((success >= 0.5).sum()),
        success_tail=_tail_run(success),
        goal_dist_final=float(arrays["goal_dist"][-1]),
        lift_max=float(arrays["obj_height_delta"].max()),
        env_index=env_index,
        seed=int(args_cli.seed) if args_cli.seed is not None else -1,
    )


def _rollout(env, agent: BasePlayer, capture: _Capture,
             targets: list[int], want: int) -> list[GT.GraspEpisode]:
    """결정론 롤아웃. env 별 버퍼로 갈라 담고 done 이 뜬 **대상 env** 만 닫는다."""
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    buffers = {i: _EnvBuffer() for i in targets}
    episodes: list[GT.GraspEpisode] = []

    for _ in range(args_cli.eval_steps):
        state = capture.state_frame()
        for i, buf in buffers.items():
            buf.states.append({k: v[i] for k, v in state.items()})

        with torch.inference_mode():
            action = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            action_np = action.detach().float().cpu().numpy().copy()
            for i, buf in buffers.items():
                buf.actions.append(action_np[i])
            obs, _, dones, _ = env.step(action)
            done_np = dones.detach().cpu().numpy().astype(bool).reshape(-1)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for hidden in agent.states:
                    hidden[:, dones, :] = 0.0

        command = capture.command_frame()
        for i, buf in buffers.items():
            if not done_np[i]:
                buf.commands.append({k: v[i] for k, v in command.items()})
                continue
            # done 스텝의 지령은 이미 reset 값이다 → 그 스텝의 (상태, 액션)까지 버린다.
            buf.states.pop()
            buf.actions.pop()
            if buf.n_closed > 0:
                episodes.append(_close_episode(buf, i))
                last = episodes[-1]
                print(f"[REC] 에피소드 {len(episodes)} env{i} T={last.n_steps} "
                      f"success_tail={last.success_tail} "
                      f"goal_dist={last.goal_dist_final:.4f} lift={last.lift_max:.3f}",
                      flush=True)
            buf.clear()
        if len(episodes) >= want:
            break

    if not episodes:
        raise SystemExit(
            f"{args_cli.eval_steps} 스텝 안에 끝난 에피소드가 없다. --eval_steps 를 늘릴 것.")
    return episodes


def _build_meta(raw, capture: _Capture, resume_path: Path, log_dir: Path,
                center: tuple[float, float], species: str,
                scale: float) -> GT.GraspTrajMeta:
    env_yaml = log_dir / "params" / "env.yaml"
    robot_usd = ""
    try:
        robot_usd = str(raw.cfg.robot_cfg.spawn.usd_path)
    except Exception:  # noqa: BLE001
        pass
    step_dt = float(getattr(raw, "step_dt", raw.physics_dt * raw.cfg.decimation))
    root_pos = (raw.robot.data.root_pos_w[0] - raw.scene.env_origins[0]).cpu().numpy()
    root_quat = raw.robot.data.root_quat_w[0].cpu().numpy()

    return GT.GraspTrajMeta(
        task_id=args_cli.task.split(":")[-1],
        checkpoint=resume_path.name,
        checkpoint_sha256=_sha256(resume_path),
        git_commit=_git_commit(),
        robot_usd=robot_usd,
        env_yaml_sha256=_sha256(env_yaml) if env_yaml.is_file() else "",
        dt=step_dt,
        decimation=int(raw.cfg.decimation),
        spawn_center_xy=center,
        spawn_range=float(raw.cfg.spawn_range),
        object_species=species,
        object_scale=scale,
        goal_offset_xyz=tuple(float(v) for v in raw.cfg.goal_offset_xyz),
        env_origin=tuple(float(v) for v in raw.scene.env_origins[0].cpu().numpy()),
        robot_root=tuple(float(v) for v in np.concatenate([root_pos, root_quat])),
        arm_joint_names=capture.arm_joint_names,
        hand_joint_names=capture.hand_joint_names,
        palm_body=PALM_BODY,
        recorded_at=datetime.now().isoformat(timespec="seconds"),
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    resume_path = Path(args_cli.checkpoint).expanduser().resolve()
    if not resume_path.is_file():
        raise SystemExit(f"체크포인트가 없다: {resume_path}")
    log_dir = resume_path.parent.parent
    out_path = Path(args_cli.record_out).expanduser().resolve()

    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    # ★런 dump 복원이 먼저다 — 이 뒤에 우리 재정의를 올려야 덮이지 않는다.
    agent_cfg = restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=str(resume_path), workspace_root=str(_HDGP.parent))
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["params"]["seed"]

    center = _patch_spawn_center(env_cfg.profile_name)
    _apply_overrides(env_cfg)
    env_cfg.log_dir = str(log_dir)
    # 소환 중심 교체는 프로필에만 실렸다 — 파생 구조(초기 스폰 pos·앵커 검사)를 다시 짠다.
    env_cfg.finalize_after_overrides()

    targets, scale = _target_envs(
        env_cfg.object_bank, int(env_cfg.scene.num_envs), args_cli.object_species)

    if args_cli.video:
        # ★영상은 뷰어 env **하나**만 찍는다. 다른 env 의 에피소드를 저장하면 영상과
        #   궤적이 서로 다른 롤아웃이 되어, 영상으로 궤적을 검증할 수 없게 된다.
        #   결정론이라 어느 env 든 거의 같지만 "거의"를 진실원천으로 삼지 않는다.
        targets = targets[:1]
        print(f"[REC] --video → 기록 env 를 뷰어 env {targets[0]} 하나로 제한한다")
        # 뷰어 cfg 는 씬 생성 시점에 소비된다 — 여기서 고정해야 먹는다.
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = targets[0]
        env_cfg.viewer.eye = tuple(float(v) for v in args_cli.video_eye)
        env_cfg.viewer.lookat = tuple(float(v) for v in args_cli.video_lookat)

    device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concat_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        folder = args_cli.video_out or str(out_path.parent / "videos")
        env = gym.wrappers.RecordVideo(
            env, video_folder=folder, step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length, disable_logger=True,
            name_prefix=out_path.stem)
        print(f"[REC] 영상 → {folder}/{out_path.stem}-step-0.mp4")
    env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions, obs_groups, concat_groups)

    agent = _build_agent(env, agent_cfg, str(resume_path))
    _assert_contract(agent, env)

    raw = env.unwrapped
    while hasattr(raw, "env"):
        raw = raw.env.unwrapped
    _assert_species(raw, targets, args_cli.object_species)

    capture = _Capture(raw)
    want = max(args_cli.record_episodes, len(targets))
    episodes = _rollout(env, agent, capture, targets, want)

    # 성공 tail 이 긴 것 우선, 같으면 목표에 가까운 것.
    episodes.sort(key=lambda e: (-e.success_tail, e.goal_dist_final))
    best = tuple(episodes[: args_cli.record_episodes])
    meta = _build_meta(raw, capture, resume_path, log_dir, center,
                       args_cli.object_species, scale)

    saved = GT.write_traj(out_path, meta, best)
    check_meta, check_eps = GT.read_traj(saved)
    problems = GT.validate(check_meta, check_eps)
    if problems:
        raise SystemExit("[REC] 저장본이 계약을 어긴다:\n  " + "\n  ".join(problems))
    csv_path = GT.write_command_csv(saved.with_suffix(".csv"), check_meta, check_eps[0])

    print("\n" + "=" * 68)
    print(f"[REC] 소환 중심 ({center[0]:.3f}, {center[1]:.3f}) · "
          f"종 {args_cli.object_species} scale {scale}")
    print(f"[REC] 저장 {len(best)}/{len(episodes)} 에피소드 → {saved}")
    print(f"[REC] 재생용 지령 CSV → {csv_path}")
    for ep in best:
        verdict = "성공" if ep.success else "**실패**"
        print(f"[REC]   env{ep.env_index} T={ep.n_steps} ({ep.n_steps * meta.dt:.2f}s) "
              f"{verdict} tail={ep.success_tail} goal_dist={ep.goal_dist_final:.4f} "
              f"lift_max={ep.lift_max:.3f}")
    print("=" * 68)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
