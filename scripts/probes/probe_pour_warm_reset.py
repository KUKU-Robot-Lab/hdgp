#!/usr/bin/env python3
"""pour warm 리셋이 실제로 **먹었는지** 본다 — 다물체 전환의 판정 프로브.

왜 필요한가. 다물체는 `replicate_physics=False` 를 강제하고, 그때 `clone_environments`
전에 씬에 등록된 자산 뷰는 **갱신되지 않는다**. 증상은 부팅에서 안 보이고 리셋에서만
난다 — grasp_s2r 08.29 실측: `write_joint_state_to_sim` 이 반영 안 돼 관절 편차
18.7 rad · 속도 2,973 rad/s · `episode_lengths` 260 → 1.2(무한 리셋).

그래서 정책 없이 **제로 액션**으로 돌리며 세 가지만 본다:
  ① 리셋 직후 관절이 warm 뱅크 자세로 실제로 이동했는가 (지령 ↔ 측정)
  ② 관절 속도가 폭발하지 않는가
  ③ 에피소드가 즉시 다시 리셋되지 않는가
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="pour warm 리셋 판정 프로브")
parser.add_argument("--task", type=str, default="open-tesol_r_pour_sensor-play-lstm")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
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
import torch  # noqa: E402

import openarm  # noqa: E402
_EXPECTED = str((_HDGP / "source/openarm/openarm").resolve())
if not str(Path(openarm.__file__).resolve()).startswith(_EXPECTED + os.sep):
    raise SystemExit(f"openarm 이 저장소 밖에서 왔다: {openarm.__file__}")
import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    # ★런 cfg 를 복원하지 **않는다** — 이 프로브는 구 체크포인트가 아니라 **현재 cfg** 를
    #   시험한다. play.py 로 하면 구 런의 warm 경로·자산이 되살아나 새 설정을 못 본다.
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped
    while hasattr(raw, "env"):
        raw = raw.env.unwrapped

    print(f"[PROBE] task={args_cli.task} envs={raw.num_envs} "
          f"replicate_physics={raw.scene.cfg.replicate_physics} "
          f"object_bank={getattr(raw.cfg, 'object_bank', '?')}", flush=True)

    env.reset()
    arm = raw.arm_dof_indices
    q0 = raw.robot.data.joint_pos[:, arm].clone()
    want = raw._warmstart_arm_pos
    # 리셋 직후 자세가 뱅크 어느 상태와도 안 맞으면 write_joint_state_to_sim 이 안 먹은 것.
    nn_err = torch.cdist(q0, want).min(dim=1).values
    print(f"[PROBE] 리셋 직후 우팔자세 ↔ 뱅크 최근접 거리: "
          f"평균 {nn_err.mean():.5f} 최대 {nn_err.max():.5f} rad "
          f"(먹었으면 ~0, 안 먹었으면 rad 단위로 벌어진다)", flush=True)

    # ---- 붓기 기하 검사 (컵을 바꾸면 여기가 조용히 어긋난다) --------------------
    from isaaclab.utils.math import quat_apply  # noqa: PLC0415

    src_p, src_q = raw.cup.data.root_pos_w, raw.cup.data.root_quat_w
    tgt_p, tgt_q = raw.left_target_cup.data.root_pos_w, raw.left_target_cup.data.root_quat_w
    pour_pt = src_p + quat_apply(src_q, raw._src_pour_offset())
    open_pt = tgt_p + quat_apply(tgt_q, raw._target_cup_opening_pos_b.unsqueeze(0)
                                 .expand(raw.num_envs, -1))
    d = open_pt - pour_pt
    xy = torch.norm(d[:, :2], dim=-1)
    print(f"[PROBE] 붓기점→입구  xy {xy.mean()*1e3:6.1f} mm (범위 "
          f"{xy.min()*1e3:.1f}~{xy.max()*1e3:.1f}) · "
          f"z여유 {d[:, 2].mean()*1e3:+6.1f} mm (범위 {d[:, 2].min()*1e3:+.1f}~"
          f"{d[:, 2].max()*1e3:+.1f})", flush=True)
    print("[PROBE]   ★z여유는 **음수**여야 소스가 입구 위다 (양수면 컵 아래에서 붓는다)",
          flush=True)
    # 컵 최저점이 테이블을 뚫지 않는가
    off = raw._src_rim_offset[:, 2]
    print(f"[PROBE] 소스 림 오프셋 {off.min():.4f}~{off.max():.4f} m · "
          f"컵 z {src_p[:, 2].min()-raw.scene.env_origins[:, 2].min():.4f}~"
          f"{src_p[:, 2].max()-raw.scene.env_origins[:, 2].max():.4f}", flush=True)

    zero = torch.zeros(raw.num_envs, raw.cfg.num_actions, device=raw.device)
    # ★리셋 직후 몇 프레임은 **텔레포트 프레임**이라 속도가 의미 없다(수십 rad/s 도약).
    #   그 구간을 빼고 재야 "물리가 폭발했나"를 본다.
    SETTLE = 8
    since = torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device)
    vmax_raw, vmax_settled, resets = 0.0, 0.0, 0
    for step in range(args_cli.steps):
        with torch.inference_mode():
            _, _, term, trunc, _ = env.step(zero)
        done = (term | trunc).bool().reshape(-1)
        v = raw.robot.data.joint_vel.abs().amax(dim=1)
        vmax_raw = max(vmax_raw, float(v.max()))
        settled = since >= SETTLE
        if settled.any():
            vmax_settled = max(vmax_settled, float(v[settled].max()))
        since = torch.where(done, torch.zeros_like(since), since + 1)
        resets += int(done.sum())
        if step % 60 == 0:
            ep = raw.episode_length_buf.float().mean().item()
            print(f"[PROBE] step {step:>4} · ep_len 평균 {ep:7.1f} · 누적리셋 {resets:>4} · "
                  f"|v|max 전체 {vmax_raw:8.2f} · 정착후 {vmax_settled:8.2f} rad/s", flush=True)

    ep = raw.episode_length_buf.float().mean().item()
    # ★판정 기준은 C 의 실패 서명(grasp_s2r 08.29 실측)이다:
    #   관절편차 18.7 rad · 속도 2,973 rad/s · ep_len 260→1.2.
    #   제로 액션이라 에피소드가 끝나는 것 자체는 정상 — 리셋 **횟수**로 판정하지 않는다.
    reset_ok = nn_err.max() < 0.05
    phys_ok = vmax_settled < 500.0
    live_ok = ep > 5.0
    print("\n" + "=" * 62)
    print(f"[PROBE] 최종 ep_len 평균 {ep:.1f} · 누적리셋 {resets}")
    print(f"[PROBE] |joint_vel|max  전체 {vmax_raw:.2f} · 정착후 {vmax_settled:.2f} rad/s")
    print(f"[PROBE] ① 리셋 반영(<0.05 rad) : {'OK' if reset_ok else 'FAIL'} "
          f"({float(nn_err.max()):.5f})")
    print(f"[PROBE] ② 물리 폭발 없음(<500)  : {'OK' if phys_ok else 'FAIL'} ({vmax_settled:.1f})")
    print(f"[PROBE] ③ 무한리셋 아님(ep>5)   : {'OK' if live_ok else 'FAIL'} ({ep:.1f})")
    print(f"[PROBE] 판정: "
          f"{'통과 — C(clone 순서) 불필요' if (reset_ok and phys_ok and live_ok) else 'C 필요'}")
    print("=" * 62)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
