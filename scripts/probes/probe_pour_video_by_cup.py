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

"""지정한 **컵 종류**의 한 에피소드를 영상으로 뽑는다.

왜 별도 스크립트인가. `play.py --video` 는 글로벌 뷰포트(env0)만 찍는다. 컵은
`env_id % N` 로 배정되므로 보고 싶은 컵이 env0 이 아니면 못 찍는다. 여기서는
env 별 카메라를 붙여 **원하는 env 만** 프레임을 모은다.

★ADR 은 반드시 학습 레벨로 고정한다 — play 세션은 카운터가 0 부터라 고정하지 않으면
  학습 시작 난이도로 재생된다.

실행:
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_pour_video_by_cup.py \\
        --checkpoint <ckpt> --cups shaker_closed,cup_big_s130 \\
        --adr "success=1.0,outcome=0.625,noise=0.15" --out /tmp/vid --headless
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_pour_sensor-play-lstm")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--cups", type=str, required=True, help="촬영할 컵 id, 쉼표 구분")
parser.add_argument("--adr", type=str, default="", help="'success=1.0,outcome=0.625,noise=0.15'")
parser.add_argument("--out", type=str, required=True, help="출력 mp4 접두사")
parser.add_argument("--max_steps", type=int, default=900)
parser.add_argument("--every", type=int, default=2, help="이 스텝마다 1 프레임 (60Hz→30fps)")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402
from pathlib import Path  # noqa: E402

_HDGP = Path(__file__).resolve().parents[2]
for _p in (str(_HDGP / "scripts/tools"), str(_HDGP / "source/openarm")):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
for _n in [m for m in sys.modules if m == "openarm" or m.startswith("openarm.")]:
    del sys.modules[_n]

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import openarm  # noqa: E402
_EXPECTED = str((_HDGP / "source/openarm/openarm").resolve())
if not str(Path(openarm.__file__).resolve()).startswith(_EXPECTED + os.sep):
    raise SystemExit(f"openarm 이 저장소 밖에서 왔다: {openarm.__file__}")
import openarm.tasks  # noqa: F401,E402

from run_cfg_restore import restore_run_cfg_if_available  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    resume = Path(args_cli.checkpoint).expanduser().resolve()
    if not resume.is_file():
        raise SystemExit(f"체크포인트가 없다: {resume}")
    agent_cfg = restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=str(resume), workspace_root=str(_HDGP.parent))
    env_cfg.seed = agent_cfg["params"]["seed"]

    from openarm.agnostic.modules import object_bank as _ob
    bank = _ob.get(env_cfg.object_bank)
    names = list(bank.ids)
    want = [c.strip() for c in args_cli.cups.split(",")]
    for c in want:
        if c not in names:
            raise SystemExit(f"컵 '{c}' 이 뱅크에 없다. 보유: {names}")
    # env_id % N 배정이므로 env 수를 뱅크 종수로 두면 env i = 컵 i 다.
    env_cfg.scene.num_envs = len(names)
    env_cfg.scene.env_spacing = 4.0
    env_cfg.scene.shot_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/shot_cam", update_period=0.0,
        height=args_cli.height, width=args_cli.width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=26.0, clipping_range=(0.02, 20.0)),
    )

    device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_act,
                               agent_cfg["params"]["env"].get("obs_groups"),
                               agent_cfg["params"]["env"].get("concate_obs_groups", True))
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(resume)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(str(resume))
    agent.reset()

    raw = env.unwrapped
    while hasattr(raw, "env"):
        raw = raw.env.unwrapped

    if args_cli.adr:
        pinned = []
        for kv in args_cli.adr.split(","):
            k, v = kv.split("=")
            adr = getattr(raw, f"{k.strip()}_adr", None)
            if adr is None:
                raise SystemExit(f"ADR '{k}_adr' 없음")
            adr.set_increment(int(round(adr.num_increments * float(v))))
            pinned.append(f"{k}={adr.progress:.3f}")
        print(f"[VID] ADR 고정: {' · '.join(pinned)}", flush=True)
    else:
        print("[VID] ⚠ADR 미고정 — 학습 시작 난이도로 재생된다", flush=True)

    targets = [names.index(c) for c in want]
    print(f"[VID] 촬영 대상 env: {list(zip(want, targets))}", flush=True)

    # ---- 좌팔 계측 준비 (좌팔 모드일 때만) ----
    #   "좌팔이 제대로 컵을 따라 움직이는가"는 영상만으로는 애매하다. 세 층을 같이 잰다:
    #     ① 정책이 TCP 목표를 rest 에서 얼마나 옮기는가(=채널을 쓰는가)
    #     ② 실제 왼손이 그 목표를 따라오는가(=IK 가 도는가)
    #     ③ 그 결과 받는 컵이 얼마나 움직이는가(=의미 있는 변위인가)
    _left_on = getattr(raw, "_left_ik", None) is not None
    _lt = {"cmd": [], "track": [], "cup": [], "d_cup": [], "contacts": []}
    if _left_on:
        from isaaclab.utils.math import subtract_frame_transforms as _sft
        _rest = raw._left_tcp_rest_pos_b.clone()
        _cup0 = None   # ★기준선은 **reset 이후** 첫 스텝에서 잡는다 — reset 전 값을 쓰면
                       #   리셋 이동량이 통째로 섞여 변위가 부풀려진다(09.02에 한 번 속았다).
        print(f"[VID] 좌팔 모드 ON — rest(base) "
              f"{[round(float(v), 4) for v in _rest[0]]} · "
              f"워크스페이스 half-extent {tuple(raw.cfg.left_tcp_workspace_range)}", flush=True)
    else:
        print("[VID] 좌팔 모드 OFF — 좌팔 계측 생략", flush=True)

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    cam = raw.scene["shot_cam"]
    frames = {i: [] for i in targets}
    done_once = {i: False for i in targets}

    for step in range(args_cli.max_steps):
        with torch.inference_mode():
            action = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, _ = env.step(action)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for h in agent.states:
                    h[:, dones, :] = 0.0

        if step % args_cli.every == 0 and not all(done_once.values()):
            # 소스컵과 받는컵 중간을 본다 — 붓는 순간이 화면에 담기게.
            mid = 0.5 * (raw.cup.data.root_pos_w + raw.left_target_cup.data.root_pos_w)
            eye = mid.clone()
            eye[:, 0] += 0.62
            eye[:, 1] += 0.52
            eye[:, 2] += 0.34
            cam.set_world_poses_from_view(eye, mid)
            raw.sim.render()
            cam.update(dt=0.0)
            rgb = cam.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
            for i in targets:
                if not done_once[i]:
                    frames[i].append(rgb[i])

        if _left_on:
            _hand_b, _ = _sft(raw.robot.data.root_pos_w, raw.robot.data.root_quat_w,
                              raw.robot.data.body_pos_w[:, raw._left_hand_body_index],
                              raw.robot.data.body_quat_w[:, raw._left_hand_body_index])
            _lt["cmd"].append((raw.left_tcp_target_pos_b - _rest).norm(dim=-1))
            _lt["track"].append((raw.left_tcp_target_pos_b - _hand_b).norm(dim=-1))
            if _cup0 is None:
                _cup0 = raw.left_target_cup.data.root_pos_w.clone()
            _lt["cup"].append((raw.left_target_cup.data.root_pos_w - _cup0).norm(dim=-1))
            # ★근접도: 받는 컵이 소스 컵/손에 얼마나 붙는가. 좌팔이 과하게 다가와
            #   파지를 밀어내는지(= contacts 하락의 원인인지) 보려면 이게 필요하다.
            _lt["d_cup"].append(
                (raw.left_target_cup.data.root_pos_w - raw.cup.data.root_pos_w).norm(dim=-1))
            _lt["contacts"].append(raw.grip_contacts_log.clone()
                                   if hasattr(raw, "grip_contacts_log") else
                                   torch.zeros(raw.num_envs, device=raw.device))

        d = dones.to(raw.device).bool().reshape(-1)
        for i in targets:
            if bool(d[i]) and not done_once[i] and len(frames[i]) > 30:
                done_once[i] = True
                b = float(raw._last_done_bead[i]); s = float(raw._last_done_spill[i])
                print(f"[VID] {names[i]:16s} 에피소드 종료 step {step} · "
                      f"{len(frames[i])} 프레임 · bead {b:.3f} spill {s:.3f} "
                      f"잔량 {max(0.0, 1-b-s):.3f}", flush=True)
        if all(done_once.values()):
            break

    if _left_on and _lt["cmd"]:
        _c = torch.stack(_lt["cmd"]); _t = torch.stack(_lt["track"]); _u = torch.stack(_lt["cup"])
        print(f"\n[VID] 좌팔 계측 ({_c.shape[0]} 스텝 · {_c.shape[1]} env)", flush=True)
        print(f"  ① TCP 목표 이탈(rest 대비)  평균 {_c.mean()*1000:6.1f}mm · "
              f"최대 {_c.max()*1000:6.1f}mm   (0 이면 정책이 채널을 안 쓴다)", flush=True)
        print(f"  ② 추종 오차(목표−실제 왼손)  평균 {_t.mean()*1000:6.1f}mm · "
              f"최대 {_t.max()*1000:6.1f}mm   (크면 IK 가 못 따라간다)", flush=True)
        print(f"  ③ 받는 컵 변위(시작 대비)    평균 {_u.mean()*1000:6.1f}mm · "
              f"최대 {_u.max()*1000:6.1f}mm", flush=True)
        _d = torch.stack(_lt["d_cup"])
        print(f"  ④ 컵-컵 거리(받는↔소스)     평균 {_d.mean()*1000:6.1f}mm · "
              f"**최소 {_d.min()*1000:6.1f}mm**  (너무 작으면 좌팔이 파지를 밀어낸다)",
              flush=True)
        for i in targets:
            print(f"     {names[i]:16s} 목표이탈 {_c[:, i].max()*1000:6.1f}mm · "
                  f"추종오차 {_t[:, i].mean()*1000:6.1f}mm · "
                  f"컵변위 {_u[:, i].max()*1000:6.1f}mm · "
                  f"컵-컵 최소 {_d[:, i].min()*1000:6.1f}mm", flush=True)

    import imageio.v2 as iio
    for i in targets:
        if not frames[i]:
            print(f"[VID] {names[i]}: 프레임 없음 — 건너뜀", flush=True)
            continue
        path = f"{args_cli.out}_{names[i]}.mp4"
        iio.mimwrite(path, frames[i], fps=30, quality=8)
        print(f"  저장 {path}  ({len(frames[i])} 프레임)", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
