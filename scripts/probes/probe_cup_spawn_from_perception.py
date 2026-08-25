"""인지가 준 컵 pose 로 sim 에 컵을 소환하고, **실제로 거기 놓였는지** 되읽는다.

정책이 필요 없다. 이음매의 Isaac 쪽 반쪽만 잰다:
  · 요청한 pose 대로 컵이 놓이는가 (요청 vs 되읽은 pose)
  · 물리가 돌아도 그 자리에 머무는가 (테이블에 안착하는가, 빠지거나 튀지 않는가)
  · 학습 스폰 상자 안인가 (밖이면 정책이 분포 밖에서 도는 것이다)

env 는 건드리지 않는다 — 씬 객체에 직접 쓴다. 스폰은 이벤트가 무작위로 하므로 리셋
직후에 덮어쓰고, 리셋이 또 나면 다시 덮어쓴다.

실행:
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_cup_spawn_from_perception.py \
        --cup_pose ~/rl_ws/sim2real/logs/shadow/cup_pose.json
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="open-grip_l_grasp_sensor_fab-play")
parser.add_argument("--cup_pose", type=Path, required=True)
parser.add_argument("--settle", type=int, default=120, help="물리 안착 대기 스텝")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--fabrics_src", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402

_HDGP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDGP / "source/openarm"))
sys.path.insert(0, str(_HDGP.parent / "sim2real/scripts"))
if args.fabrics_src is not None:
    sys.path.insert(0, str(args.fabrics_src.resolve()))

from cup_pose_capture import load_capture, spawn_box_from_preset, verdict  # noqa: E402
import openarm.tasks                                                       # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg                             # noqa: E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P       # noqa: E402


def main() -> int:
    pose = load_capture(args.cup_pose, expect_frame="base_link")
    report = verdict(pose, spawn_box_from_preset(P))
    print("[CUP] " + report.describe().replace("\n", "\n[CUP] "), flush=True)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 1e6
    if hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()

    cup = env.scene["object"]
    want = torch.tensor([*pose.position, *pose.orientation_wxyz],
                        device=env.device, dtype=torch.float32)
    root = want.unsqueeze(0).repeat(env.num_envs, 1).clone()
    root[:, :3] += env.scene.env_origins
    cup.write_root_pose_to_sim(root)
    cup.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))

    def cup_pos() -> np.ndarray:
        return (cup.data.root_pos_w - env.scene.env_origins)[0].detach().cpu().numpy()

    placed = cup_pos()
    # ★`env.step` 을 쓰면 안 된다. 이 태스크의 액션은 **절대 palm** 이라 a=0 이 "정지"가
    #   아니라 PALM_BOX **중심으로 이동**이다. 팔이 움직여 컵을 쓸어내고, 그걸 "컵이
    #   안 놓인다"로 읽게 된다(실측: 175.9 mm 이탈, z −103.7 mm). 여기서 묻는 것은
    #   "그 높이에 놓으면 테이블에 얹혀 있는가" 뿐이므로 **물리만** 돌린다.
    for _ in range(args.settle):
        env.sim.step(render=False)
        env.scene.update(env.sim.get_physics_dt())
    settled = cup_pos()

    want_np = np.asarray(pose.position)
    print(f"\n요청  {np.round(want_np, 4).tolist()}")
    print(f"배치  {np.round(placed, 4).tolist()}   오차 "
          f"{np.linalg.norm(placed - want_np) * 1000:.2f} mm")
    print(f"안착  {np.round(settled, 4).tolist()}   요청 대비 "
          f"{np.linalg.norm(settled - want_np) * 1000:.2f} mm "
          f"(물리 {args.settle} 스텝 뒤)")
    drop = float(want_np[2] - settled[2]) * 1000.0
    print(f"      z 변화 {drop:+.2f} mm — 양수면 가라앉음(테이블 위가 아닐 수 있다)")
    print(f"\n판정: {'분포 안' if report.inside else '★분포 밖'}")
    return 0 if report.inside else 1


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
