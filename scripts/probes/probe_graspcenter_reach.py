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

"""파지중심-물체 거리(`d_gc`)가 물리적으로 어디까지 줄어드는가.

왜 필요한가: `agnostic/grasp_sensor` 보상의 연속 부분이 전부 `G` 하나에 곱해진다.

    near_q = exp(−d_gc / stage_grasp_near_tau)      τ = 0.04 (40mm)
    G      = five_frac · near_q
    grasp = G · w  ·  lift = G·H·U  ·  transfer = G·H·U·T  ·  stay = G·H·U·T·S

학습 실측(lstm_test14 정점 ep1102)은 d_gc = 81~106mm 에서 안 내려갔다.
그러면 near_q ≈ 0.10~0.13 이라 **연속 보상 전체가 정격의 10% 로 눌린다**.
남는 것은 가중 20.0 의 **이진** success 항뿐이고, 실제로 두 런 모두 그 항이
켜진 직후에 붕괴했다.

여기서 가르는 것은 하나다 — **40mm 가 도달 가능한 값인가.**
  · 도달 가능하면 → 정책이 못 간 것 → 접근 보상 문제
  · 도달 불가능하면 → τ=0.04 가 애초에 잘못된 척도 → near_q 가 구조적으로
    0.1 을 못 넘고, 연속 보상이 영영 10% 에 갇힌다 → 이진 항이 지배할 수밖에 없다

측정 방법: 정책을 쓰지 않는다. **파지중심이 컵 원점에 오도록 palm 을 폐루프로
지령**하고(`palm_target = palm + (cup − gc)`), 손 폐쇄도를 env 마다 다르게 고정해
한 번의 롤아웃으로 전 구간을 훑는다. 폐쇄도 −1(완전 개방)이 **기구학 천장**이다.

★`episode_length_s` 를 크게 잡아 리셋을 배제한다 — 안 그러면 측정 창에 리셋이
  끼어들어 손이 홈으로 돌아간다(이 저장소가 같은 함정에 두 번 빠졌다).
★컵 밀림을 같이 잰다. d_gc 가 줄어도 컵을 밀어내서 준 것이면 무의미하다.

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_graspcenter_reach.py
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-sens_r_grasp_sensor")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--approach_steps", type=int, default=250, help="손을 연 채 palm 접근")
parser.add_argument("--close_steps", type=int, default=350, help="지정 폐쇄도 유지")
parser.add_argument("--seed", type=int, default=12345)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.tasks  # noqa: E402,F401

# 폐쇄도(액션 값). env 내부에서 cmd = 0.5·(a+1) 인 **절대 폐쇄도**로 쓰인다.
LEVELS = [-1.0, -0.6, -0.2, 0.2, 0.5, 0.7, 0.85, 1.0]


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # ★리셋 배제. 측정 창 안에서 손이 홈으로 돌아가면 전부 무효다.
    env_cfg.episode_length_s = 1.0e6

    env = gym.make(args_cli.task, cfg=env_cfg)
    core = env.unwrapped
    torch.manual_seed(args_cli.seed)
    core.reset()

    n = core.num_envs
    dev = core.device
    per = max(1, n // len(LEVELS))
    lvl = torch.zeros(n, device=dev)
    for i, v in enumerate(LEVELS):
        lvl[i * per:(i + 1) * per if i < len(LEVELS) - 1 else n] = v

    lo, hi = core._palm_lo, core._palm_hi
    span = (hi - lo).clamp(min=1e-6)
    n_act = int(core.cfg.action_space)   # ★.shape[0] 은 num_envs 다
    n_hand = n_act - 6

    def grasp_center() -> torch.Tensor:
        """env 의 보상 경로와 **같은 식**으로 파지중심을 낸다(사본 금지)."""
        qa = core.robot.data.joint_pos[:, core._fab_t].contiguous()
        o, R = core._tip_palm_frame(qa)
        return o + torch.einsum("bij,j->bi", R, core._gc_local) + core._fab_to_env

    cup0 = None
    snap: dict[str, torch.Tensor] = {}
    act = torch.zeros(n, n_act, device=dev)
    act[:, 6:] = -1.0                                   # 손을 연 채 시작

    total = args_cli.approach_steps + args_cli.close_steps
    for t in range(total):
        with torch.inference_mode():
            gc = grasp_center()
            palm = core._env_local(core.robot.data.body_pos_w[:, core.palm_idx])
            obj = core._env_local(core.object.data.root_pos_w)
            if cup0 is None and t == 2:                  # 리셋 직후 버퍼는 stale
                cup0 = obj.clone()
            # 폐루프 P 제어: 파지중심 오차를 palm 위치 지령에 그대로 싣는다.
            tgt = palm + (obj - gc)
            a_pos = (2.0 * (tgt - lo[:3]) / span[:3] - 1.0).clamp(-1.0, 1.0)
            act[:, :3] = a_pos
            act[:, 3:6] = 0.0                            # 자세는 박스 중앙 고정
            if t >= args_cli.approach_steps:
                act[:, 6:] = lvl.unsqueeze(1).expand(-1, n_hand)
            core.step(act)
            # ★접근 단계(손 연 채)와 폐쇄 단계를 분리해 기록한다. 합쳐 놓으면
            #   "손가락이 컵을 밀었나" 와 "팔이 컵을 쓸었나" 를 못 가른다.
            if t == args_cli.approach_steps - 1:
                snap["d_gc"] = torch.norm(grasp_center() - obj, dim=-1).clone()
                snap["drift"] = torch.norm(obj - cup0, dim=-1).clone()
                snap["tilt"] = core._tilt_deg_buf.clone()

    with torch.inference_mode():
        gc = grasp_center()
        obj = core._env_local(core.object.data.root_pos_w)
        d_gc = torch.norm(gc - obj, dim=-1)
        drift = torch.norm(obj - cup0, dim=-1) if cup0 is not None else torch.zeros(n, device=dev)
        # `_contact_forces_split` 은 (mid, dist) 2-tuple 이다. env 의 five_c 도
        # 이 둘만 본다(팁 단독 접촉은 감쌈으로 안 친다) — 같은 규약을 쓴다.
        mid_f, dist_f = core._contact_forces_split()
        thr = float(core.cfg.stage_contact_threshold)
        five = ((mid_f > thr) | (dist_f > thr)).float().mean(dim=-1)
        fmax = torch.maximum(mid_f, dist_f).max(dim=-1).values
        tilt = core._tilt_deg_buf

    tau = float(core.cfg.stage_grasp_near_tau)
    print("\n" + "=" * 86, flush=True)
    print(f"파지중심 palm-local {[round(float(v)*1000) for v in core._gc_local]}mm · "
          f"τ = {tau*1000:.0f}mm · 접촉임계 {thr}N · env {n} · "
          f"{args_cli.approach_steps}+{args_cli.close_steps} 스텝", flush=True)
    print("-" * 86, flush=True)
    print(f"{'폐쇄도':>7}{'d_gc(mm)':>11}{'near_q':>9}{'five_frac':>11}"
          f"{'힘max(N)':>11}{'컵밀림(mm)':>12}{'컵기울기°':>11}", flush=True)
    for i, v in enumerate(LEVELS):
        s = slice(i * per, (i + 1) * per if i < len(LEVELS) - 1 else n)
        d = float(d_gc[s].mean())
        print(f"{v:>7.2f}{d*1000:>11.1f}{torch.exp(torch.tensor(-d/tau)):>9.3f}"
              f"{float(five[s].mean()):>11.3f}{float(fmax[s].mean()):>11.2f}"
              f"{float(drift[s].mean())*1000:>12.1f}{float(tilt[s].mean()):>11.2f}", flush=True)
    print("-" * 86, flush=True)
    if snap:
        print(f"[접근 종료 시점 · 손 완전 개방 · 전 env 공통]  "
              f"d_gc {float(snap['d_gc'].mean())*1000:.1f}mm · "
              f"컵밀림 {float(snap['drift'].mean())*1000:.1f}mm · "
              f"컵기울기 {float(snap['tilt'].mean()):.2f}°", flush=True)
        _ok = snap["drift"] < 0.02
        print(f"  └ 컵을 20mm 미만으로 민 env {int(_ok.sum())}/{n} — "
              f"그 중 d_gc 평균 "
              f"{(float(snap['d_gc'][_ok].mean())*1000 if bool(_ok.any()) else float('nan')):.1f}mm",
              flush=True)
    _best = float(d_gc.min())
    print(f"전 env 최소 d_gc = {_best*1000:.1f}mm  →  near_q 상한 "
          f"{float(torch.exp(torch.tensor(-_best/tau))):.3f}", flush=True)
    print(f"τ={tau*1000:.0f}mm 를 near_q≥0.5 로 만들려면 d_gc ≤ "
          f"{tau*0.6931*1000:.0f}mm 가 필요하다.", flush=True)
    print("=" * 86, flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
