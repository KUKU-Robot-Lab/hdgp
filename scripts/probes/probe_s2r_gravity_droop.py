#!/usr/bin/env python3
"""중력 × 게인 4조합의 **정적 처짐** 실측 — grasp_s2r 학습 조건 확정용.

왜 재는가 (09.01)
-----------------
두 가지를 동시에 판정해야 한다.

  1. **로봇 중력을 켤 것인가.** 실기는 중력보상을 켠 채로 운용하므로 PD 가 보는 것은
     중력이 상쇄된 잔차뿐이다. sim 의 `disable_gravity=True` 가 바로 그 상태이고,
     중력을 켜면 **중력을 두 번 세는 것**이 된다.
  2. **r2s 정합 게인으로 재학습할 것인가.** 그 게인은 손목 kp 를 50→10 으로 낮추는데,
     이는 중력보상을 전제로 동정된 값이다(r2s collect 가 `gravity_comp_node.py` 를
     필수로 요구한다).

실기 실측이 판정선을 준다(08.31 우팔):

    무보상 처짐 **12.76°**  →  중력보상 후 **2.05°**

★09.01 실측 결과 (손목 최대 지령↔실제 오차 [deg], 4 env · 300 스텝):

  | 게인 | 중력 OFF | 중력 ON | 중력 기여분 |
  |---|---|---|---|
  | KUKA 기본  | 1.889 | 4.791 | +2.90 |
  | r2s 정합   | 1.162 | 5.668 | **+4.51** |

읽는 법 두 가지를 **틀리지 말 것**:
  · OFF 의 1~2° 는 처짐이 아니라 **fabric 정상상태 추종 오차**다(무부하).
    중력 성분은 두 열의 **차분**이다.
  · 실기 12.76°/2.05° 와 직접 비교하면 안 된다 — 저쪽은 정적 유지 측정이고
    이쪽은 제어 루프가 능동적으로 버티는 중의 오차다.

판정: 중력을 켜면 실기(보상 ON, 잔차 2.05°)에 **없는** 2.9~4.5° 오차가 더해진다.
정합 게인에서 그 폭이 1.6배로 커지는 것이 그 게인이 **중력보상 전제**로 동정됐다는
방증이다. ⇒ **로봇 중력은 끈 채로 학습한다.**

★기각된 사전 예측 — "손 1.763 kg × 레버 ≈2.1 N·m 를 kp 10 으로 받으면 0.21 rad =
  12°" 라고 예측했으나 실측은 4.5° 였다. 정적 처짐 공식을 **능동 제어 중인 계**에
  적용한 것이 오류다(fabric+PD 가 버티고 부하가 관절들에 분산된다). 결론의 방향은
  같지만 크기는 예측의 1/3 이다.

무엇을 재는가
-------------
액션 0(= 홈 자세 지령)으로 정착시킨 뒤 **지령 대비 실제** 관절각 차이를 읽는다.
`disable_gravity` 는 USD spawn 속성이라 cfg 로만 바꿀 수 있어 조합마다 env 를
새로 만든다(느리지만 정직하다).

★★**조합당 한 프로세스**다. 이유 둘:
  · 게인 분기는 `robot_profiles.py` 가 **import 시점**에 `HDGP_S2R_REAL_GAINS` 로 한다.
  · Isaac 은 한 프로세스에서 env 를 두 번 만들면 죽는다(09.01 실측 — 첫 조합 뒤
    두 번째 `GraspS2REnv(...)` 에서 조용히 종료됐다).

사용
----
    for G in off on; do
      python scripts/probes/probe_s2r_gravity_droop.py --gravity $G          # KUKA
      HDGP_S2R_REAL_GAINS=1 python scripts/probes/probe_s2r_gravity_droop.py --gravity $G
    done

한 줄 요약이 `[RESULT]` 로 나오므로 네 번 돌린 뒤 그 줄만 모아 비교한다.

옵션: --settle 300 (정착 스텝) --envs 4
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--settle", type=int, default=300,
                    help="a=0 유지 후 정착까지 스텝 수 (60 Hz 기준 5초)")
parser.add_argument("--envs", type=int, default=4, help="평균낼 env 수")
parser.add_argument("--gravity", choices=("on", "off"), default="off",
                    help="로봇 중력. ★조합당 한 프로세스 — 두 번 돌려 비교한다")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from openarm.agnostic.tasks.grasp_s2r.grasp_s2r_env import GraspS2REnv  # noqa: E402
from openarm.agnostic.tasks.grasp_s2r.grasp_s2r_env_cfg import (  # noqa: E402
    GraspS2RTesolloRightEnvCfg,
)

_REAL = os.environ.get("HDGP_S2R_REAL_GAINS") == "1"
_GAIN_TAG = "r2s 정합" if _REAL else "KUKA 기본"


def _run(gravity_on: bool) -> dict:
    """한 조합을 돌려 관절별 처짐[rad]을 돌려준다."""
    cfg = GraspS2RTesolloRightEnvCfg()
    cfg.scene.num_envs = int(args.envs)
    cfg.object_bank = "single_cup"
    cfg.enable_events = False
    cfg.enable_adr = False
    cfg.enable_self_collisions = False
    cfg.episode_length_s = 10_000.0          # 프로브 중 리셋 오염 방지(관례)
    # ★중력은 USD spawn 속성이라 여기서만 바꿀 수 있다. finalize 가 robot_cfg 를
    #   다시 만드므로 그 **뒤에** 덮어야 한다.
    cfg.finalize_after_overrides()
    cfg.robot_cfg.spawn.rigid_props.disable_gravity = not gravity_on

    env = GraspS2REnv(cfg, render_mode=None)
    u = env.unwrapped
    dev = u.device
    zero = torch.zeros(u.num_envs, u.cfg.action_space, device=dev)
    for _ in range(int(args.settle)):
        env.step(zero)

    # 지령 대비 실제 — env 가 이미 같은 뜻의 헬퍼를 갖고 있다(이름 리터럴 금지).
    err = u._joint_pos_err().abs().mean(dim=0)          # (n_hand,) 손
    arm_ids = u._arm_ids_t
    q = u.robot.data.joint_pos[:, arm_ids]
    tgt = u.robot.data.joint_pos_target[:, arm_ids]
    arm_err = (tgt - q).abs().mean(dim=0)               # (7,)
    palm_z = u._env_local(u.robot.data.body_pos_w[:, u.palm_idx])[:, 2].mean()
    out = {
        "arm": arm_err.detach().cpu(),
        "hand_max": float(err.max()),
        "palm_z": float(palm_z),
    }
    env.close()
    return out


def main() -> None:
    _g_on = args.gravity == "on"
    tag = "중력 ON " if _g_on else "중력 OFF"
    print(f"\n[probe] 게인 = {_GAIN_TAG}  (HDGP_S2R_REAL_GAINS="
          f"{os.environ.get('HDGP_S2R_REAL_GAINS', '<unset>')})", flush=True)
    print(f"[probe] {tag} · 정착 {args.settle} 스텝 · env {args.envs}개 평균\n",
          flush=True)

    r = _run(_g_on)
    deg = [float(v) * 180.0 / 3.141592653589793 for v in r["arm"]]
    print(f"[{tag}] 팔 관절 지령↔실제 오차 [deg]", flush=True)
    for i, d in enumerate(deg, start=1):
        _mark = "  ←★손목" if i >= 5 else ""
        print(f"    j{i}: {d:7.3f}{_mark}", flush=True)
    print(f"    손목 최대 {max(deg[4:]):.3f}° · 팔 전체 최대 {max(deg):.3f}° · "
          f"palm z {r['palm_z']:.4f} m · 손 최대 {r['hand_max']:.4f} rad", flush=True)
    # ★한 줄 요약 — 조합당 프로세스가 다르므로 이 줄만 모아서 비교한다.
    print(f"\n[RESULT] gain={_GAIN_TAG} gravity={args.gravity} "
          f"wrist_max_deg={max(deg[4:]):.3f} arm_max_deg={max(deg):.3f} "
          f"palm_z={r['palm_z']:.4f}", flush=True)
    print("  기준선: 실기 무보상 12.76° · 중력보상 후 2.05°", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
