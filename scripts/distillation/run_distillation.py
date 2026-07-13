# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""teacher(state) → student(vision) DAgger 증류.

Dagger 가 DDP 위에서 돌기 때문에 단일 GPU 라도 torchrun 으로 기동해야 한다:

    torchrun --standalone --nproc_per_node=1 \
        scripts/distillation/run_distillation.py \
        --task open-tesol_r_grasp_v2-distill \
        --teacher log/rl_games/open-tesol/right/grasp-v2/test6/nn/last.pth \
        --num_envs 256 --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Distill a teacher policy into a vision student.")
parser.add_argument("--task", type=str, required=True, help="Distillation task id (…-distill).")
parser.add_argument("--teacher", type=str, required=True, help="Teacher checkpoint (.pth).")
parser.add_argument("--student", type=str, default=None, help="Student checkpoint to resume from.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--label", type=str, default="distill", help="Run label (log subdirectory).")
parser.add_argument(
    "--play_policy", action="store_true", default=False,
    help="Roll out the student without training (evaluation).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 증류는 카메라가 본질이다 — 켜지 않으면 TiledCamera 가 렌더되지 않아 student 가
# 빈 이미지를 본다(조용히 학습이 진행되므로 더 나쁘다). DEXTRAH 는 이걸 CLI 로
# 넘기지만, 여기선 -distill 태스크가 항상 카메라를 요구하므로 강제한다.
args_cli.enable_cameras = True
# torchrun 이 rank/local_rank 를 넣어준다. AppLauncher 가 이를 읽어 rank 별
# device 를 잡게 한다 (Dagger 자체가 DDP 를 요구하므로 항상 분산 모드).
args_cli.distributed = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import pathlib  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch.distributed as dist  # noqa: E402
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

_HDGP_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDGP_ROOT / "source" / "openarm"))

import openarm.tasks  # noqa: F401,E402
from openarm.distillation.dagger import Dagger  # noqa: E402

_RL_DEVICE = "cuda:0"
_CLIP_OBS = 5.0
_CLIP_ACTIONS = 1.0


def _resolve_student_cfg(task: str) -> str:
    """gym registry 의 student_cfg_entry_point("<pkg>:<file>.yaml") → 절대 경로."""
    spec = gym.spec(task)
    entry_point = spec.kwargs.get("student_cfg_entry_point")
    if entry_point is None:
        raise ValueError(
            f"task '{task}' 에 student_cfg_entry_point 가 없다. "
            "-distill 로 등록된 태스크인지 확인할 것."
        )
    module_name, file_name = entry_point.split(":")
    module = __import__(module_name, fromlist=["__file__"])
    return os.path.join(os.path.dirname(module.__file__), file_name)


def _resolve_teacher_cfg(task: str) -> str:
    spec = gym.spec(task)
    module_name, file_name = spec.kwargs["rl_games_cfg_entry_point"].split(":")
    module = __import__(module_name, fromlist=["__file__"])
    return os.path.join(os.path.dirname(module.__file__), file_name)


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Dagger 는 DDP 를 요구한다. 단일 GPU 라도 torchrun 으로 기동할 것:\n"
            "  torchrun --standalone --nproc_per_node=1 "
            "scripts/distillation/run_distillation.py …"
        )
    dist.init_process_group(
        "nccl",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    local_rank = int(os.environ["LOCAL_RANK"])

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.sim.device = f"cuda:{local_rank}"

    if not env_cfg.distillation:
        raise ValueError(
            f"env_cfg.distillation=False — '{args_cli.task}' 는 증류용 태스크가 아니다. "
            "카메라도, student obs 도 생성되지 않는다."
        )

    teacher_ckpt = os.path.abspath(args_cli.teacher)
    if not os.path.isfile(teacher_ckpt):
        raise FileNotFoundError(f"teacher 체크포인트가 없다: {teacher_ckpt}")

    log_dir = _HDGP_ROOT / "log" / "distillation" / args_cli.task / args_cli.label
    (log_dir / "nn").mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(env, _RL_DEVICE, _CLIP_OBS, _CLIP_ACTIONS)

    dagger = Dagger(
        env,
        config={
            "student": {
                "cfg": _resolve_student_cfg(args_cli.task),
                "ckpt": os.path.abspath(args_cli.student) if args_cli.student else None,
                "obs_type": "policy",
                "data_aug": True,
            },
            "teacher": {
                "cfg": _resolve_teacher_cfg(args_cli.task),
                "ckpt": teacher_ckpt,
                "obs_type": "expert_policy",
            },
            "play_policy": args_cli.play_policy,
        },
        summaries_dir=str(log_dir / "summaries"),
        nn_dir=str(log_dir / "nn"),
    )
    dagger.distill()

    env.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
    simulation_app.close()
