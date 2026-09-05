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

"""pour 가 warm 파지를 **실제로 유지하는지** 본다. 정책 없이 제로 액션으로.

왜 필요한가. 09.01 사고: 뱅크의 **측정** 자세를 hold 목표로 줘서 PD 오차가 0 이 되어
파지력이 사라졌다. 손가락이 최대 80° 벌어진 채 컵이 사이에 끼워지기만 했는데,
기존 가드 둘(`dropped_by_force`=손가락 힘의 **max**, `grasp_broken`=컵 상대 드리프트)이
**둘 다** 그 상태를 통과시켜 4,575 epoch 동안 안 잡혔다.

그래서 여기서는 지표를 **개수와 평균**으로 본다:
  ① 접촉 손가락 **수** (max 아님 — 한 손가락만 닿아도 max 는 멀쩡해 보인다)
  ② 팁힘 **평균**
  ③ 컵-손 상대 드리프트 (미끄러짐)
컵 종류마다 다르므로 env 별로 나눠 찍고, 마지막에 컵별 사진도 남긴다.

실행:
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_pour_grip_hold.py \\
        --num_envs 8 --steps 240 --out /tmp/grip --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_pour_sensor-play-lstm")
parser.add_argument("--num_envs", type=int, default=8, help="뱅크 물체 수와 같게 둘 것")
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--out", type=str, default="", help="주면 컵별 PNG 도 저장")
parser.add_argument("--cam_offset", type=str, default="0.34,0.30,0.16")
# ★이 프로브는 `parse_env_cfg` 를 쓰므로 hydra `env.x=` 오버라이드가 **안 먹는다**.
#   (09.02: `env.arm_gain_profile=r2s env.warm_state_paths=[...]` 를 줬는데 조용히
#    무시되고 옛 뱅크·옛 게인으로 돌았다.) 바꿀 것은 명시 인자로 받는다.
parser.add_argument("--bank", type=str, default="", help="warm 뱅크 HDF5 경로(비우면 cfg 기본)")
parser.add_argument("--arm_gains", type=str, default="", help="우팔 게인 프로필 kuka|r2s")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.out:
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

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm  # noqa: E402
_EXPECTED = str((_HDGP / "source/openarm/openarm").resolve())
if not str(Path(openarm.__file__).resolve()).startswith(_EXPECTED + os.sep):
    raise SystemExit(f"openarm 이 저장소 밖에서 왔다: {openarm.__file__}")
import openarm.tasks  # noqa: F401,E402

CONTACT_N = 0.1  # N — 팁 접촉 판정(수집기와 같은 문턱)


def main() -> None:
    # ★`parse_env_cfg` 를 쓴다 — play.py 처럼 run dump 를 복원하면 **지금 고친 cfg** 가
    #   아니라 옛 런의 설정을 검사하게 된다(09.01에 한 번 속았다).
    cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    if args_cli.bank:
        cfg.warm_state_paths = (str(Path(args_cli.bank).expanduser().resolve()),)
    if args_cli.arm_gains:
        cfg.arm_gain_profile = args_cli.arm_gains
    # ★파생 구조(게인 재조립·뱅크 경로)를 다시 태운다 — cfg 필드만 바꾸면 반영 안 된다.
    cfg.finalize_after_overrides()
    if args_cli.out:
        cfg.scene.shot_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/shot_cam", update_period=0.0,
            height=960, width=1280, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=32.0,
                                             clipping_range=(0.02, 20.0)),
        )
    import openarm.tesollo.right.pour_sensor.pour_right_env_cfg as _pc
    print(f"[GRIP] cfg 모듈 {_pc.__file__}", flush=True)
    print(f"[GRIP] cfg.warm_state_paths = {tuple(cfg.warm_state_paths)}", flush=True)
    _src = open(_pc.__file__).read()
    print(f"[GRIP] 모듈 파일 안의 'n2048_maxgrip' 등장 횟수 = {_src.count('n2048_maxgrip')} "
          f"(0 이면 파일이 옛것, >0 인데 위 경로가 n256 이면 .pyc 캐시 문제)", flush=True)
    _f = _pc.PourRightEnvCfg.__dataclass_fields__["warm_state_paths"]
    print(f"[GRIP] 클래스 기본값 = {_f.default if _f.default is not None else _f.default_factory}", flush=True)
    env = gym.make(args_cli.task, cfg=cfg).unwrapped

    from openarm.agnostic.modules import object_bank as _ob
    bank = _ob.get(env.cfg.object_bank)
    names = [bank.specs[k].id for k in bank.assign_indices(env.num_envs)]

    print(f"[GRIP] 게인 프로필 '{env.cfg.arm_gain_profile}' — cfg 액추에이터:", flush=True)
    for key, act in env.cfg.robot_cfg.actuators.items():
        if "hand" in key or "right_arm" in key:
            print(f"        {key:26s} k={act.stiffness} d={act.damping} "
                  f"effort_sim={getattr(act, 'effort_limit_sim', None)}", flush=True)

    env.reset()
    zero = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)

    # ★지령이 실제로 실렸는지 — "측정을 목표로 준" 사고의 지문은 둘이 **같은 것**이다.
    _meas = env.robot.data.joint_pos[0, env.hand_dof_indices]
    _cmd = env.hand_joint_targets[0]
    _gap = (_cmd - _meas).abs()
    print(f"[GRIP] 리셋 직후 env0 손 — |지령−측정| 평균 {float(_gap.mean()):.4f} "
          f"최대 {float(_gap.max()):.4f} rad · 지령이 0벡터인가: "
          f"{bool((_cmd.abs() < 1e-6).all())}", flush=True)
    print(f"[GRIP]   측정 {[round(float(v),3) for v in _meas[:10]]} …", flush=True)
    print(f"[GRIP]   지령 {[round(float(v),3) for v in _cmd[:10]]} …", flush=True)
    # ★★복원이 **파지 기하 자체**를 보존했는가 — 뱅크의 컵-손 거리와 대조한다.
    #   손·컵을 각각 제자리에 넣어도 둘의 상대가 어긋나면 접촉이 성립하지 않는다.
    #   뱅크 실측(d3): 컵종별 39.7 ~ 53.9 mm.
    from isaaclab.utils.math import quat_apply as _qa2
    _bi2 = env.palm_body_index
    _ee_all = env.robot.data.body_pos_w[:, _bi2] + _qa2(
        env.robot.data.body_quat_w[:, _bi2],
        env._palm_ee_offset_local.unsqueeze(0).expand(env.num_envs, -1))
    _d_cup = (env.cup.data.root_pos_w - _ee_all).norm(dim=-1) * 1e3
    print(f"[GRIP] 리셋 직후 컵−palm_ee 거리 (mm): "
          f"{[round(float(v),1) for v in _d_cup]}", flush=True)
    print(f"[GRIP]   뱅크 기대치 39.7~53.9 mm — 벗어나면 복원이 파지 기하를 깬 것", flush=True)
    _cup0 = float(env.cup.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2])
    print(f"[GRIP]   컵 z(env-local) {_cup0:.4f} · 스폰고 {env.cfg.object_spawn_z:.4f}", flush=True)

    def snapshot(tag: str) -> None:
        nc = (env.contact_force_raw > CONTACT_N).sum(dim=-1)      # (N,) 접촉 손가락 수
        tf = env.contact_force_raw.mean(dim=-1)                   # (N,) 팁힘 평균
        drift = env._cup_rel_drift_deg                            # (N,) 컵 상대 회전
        print(f"[GRIP] {tag:>10s} · 접촉수 평균 {nc.float().mean():.2f} "
              f"(최소 {int(nc.min())}) · 팁힘평균 {float(tf.mean()):.3f}N · "
              f"드리프트 최대 {float(drift.max()):.1f}°", flush=True)
        return nc, tf, drift

    snapshot("리셋직후")
    # ★초반 20스텝을 스텝별로 본다 — 드리프트가 60스텝에 이미 171°라 붕괴는 여기서 난다.
    #   팔이 리셋 직후 움직이면(팜 어트랙터 목표 ≠ 현재 팜) 컵이 그대로 튕긴다.
    print("[GRIP] step | 컵z(env)  드리프트°  팔지령오차rad  팜목표−실제mm  접촉수", flush=True)
    for step in range(args_cli.steps):
        if step < 20 or step in (24, 30, 40, 50):
            cz = float(env.cup.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2])
            dr = float(env._cup_rel_drift_deg[0])
            aq = env.robot.data.joint_pos[0, env.arm_dof_indices]
            ae = float((env.fabric_q[0, :7] - aq).abs().max())
            _bi = env.palm_body_index
            _o = env.scene.env_origins[0]
            _link = env.robot.data.body_pos_w[0, _bi] - _o                 # r_hl_palm
            from isaaclab.utils.math import quat_apply as _qa
            _ee = _link + _qa(env.robot.data.body_quat_w[0, _bi].unsqueeze(0),
                              env._palm_ee_offset_local.unsqueeze(0))[0]   # palm_ee
            _tgt = env.palm_pose_targets[0, :3]
            pe = float((_tgt - env.palm_center_pos[0]).norm()) * 1e3
            if step == 2:
                print(f"[GRIP] ★프레임 대조 (step2, env0, env-local mm)", flush=True)
                print(f"[GRIP]   목표 palm_pose_targets {[round(float(v)*1e3,1) for v in _tgt]}", flush=True)
                print(f"[GRIP]   실제 r_hl_palm         {[round(float(v)*1e3,1) for v in _link]}", flush=True)
                print(f"[GRIP]   실제 palm_ee           {[round(float(v)*1e3,1) for v in _ee]}", flush=True)
                print(f"[GRIP]   palm_center_pos        {[round(float(v)*1e3,1) for v in env.palm_center_pos[0]]}", flush=True)
                print(f"[GRIP]   |목표−link| {float((_tgt-_link).norm())*1e3:.1f} mm · "
                      f"|목표−ee| {float((_tgt-_ee).norm())*1e3:.1f} mm", flush=True)
            nc = int((env.contact_force_raw[0] > CONTACT_N).sum())
            print(f"[GRIP] {step:4d} | {cz:8.4f} {dr:9.1f} {ae:14.4f} {pe:14.1f} {nc:7d}",
                  flush=True)
        env.step(zero)
        if (step + 1) % 60 == 0:
            snapshot(f"step {step+1}")

    nc, tf, drift = snapshot("최종")
    print("\n[GRIP] 컵별 최종:", flush=True)
    print(f"  {'컵':16s}{'접촉수':>7s}{'팁힘평균N':>11s}{'드리프트°':>11s}")
    for i in range(env.num_envs):
        print(f"  {names[i]:16s}{int(nc[i]):7d}{float(tf[i]):11.3f}{float(drift[i]):11.1f}",
              flush=True)   # ★flush 필수 — 종료 시 stdout 버퍼가 통째로 유실된다(09.02)

    if args_cli.out:
        cam = env.scene["shot_cam"]
        cup_w = env.cup.data.root_pos_w.clone()
        off = torch.tensor([float(v) for v in args_cli.cam_offset.split(",")],
                           device=env.device, dtype=cup_w.dtype)
        cam.set_world_poses_from_view(cup_w + off.unsqueeze(0), cup_w)
        for _ in range(6):
            env.sim.render()
            cam.update(dt=0.0)
        rgb = cam.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
        try:
            from PIL import Image
            save = lambda a, p: Image.fromarray(a).save(p)  # noqa: E731
        except ImportError:
            import imageio.v3 as iio
            save = lambda a, p: iio.imwrite(p, a)  # noqa: E731
        for i in range(env.num_envs):
            path = f"{args_cli.out}_{i}_{names[i]}.png"
            save(rgb[i], path)
            print(f"  저장 {path}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
