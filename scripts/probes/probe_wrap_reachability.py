"""grasp_s2r W1 — 감쌈 **도달성** 실측. 원위 링크가 닿는 palm 자세가 존재하는가.

08.27 H1 판정(`s2r_diag1`, 2048env): 원위(`_4`) 접촉률이 컵-필터 0.0099 · 무필터 0.0101
로 **같고**, 임계를 1.0N → 0.1N 으로 내려도 0.0144 에 머문다. 원위에 걸리는 힘 평균은
0.053 N. 즉 센서 결함도 임계 미달도 아니고 **진짜로 안 닿는다**. 그런데 같은 시점
`hand_blocked_frac` 이 0.5267 로, 가동 관절의 절반 이상이 관절 한계가 아니라 **외부 물체**
에 막혀 있다. 닿긴 닿는데 원위만 못 닿는 것이다.

→ `wrap = mid ∧ distal` 이 이 손·이 컵에서 **원리적으로 성립 불가**인지 판정한다.

이 probe 는 정책을 쓰지 않는다. palm 을 컵 주변 **격자**에 직접 놓고(env 하나당 한 자세)
손을 완전 폐쇄시켜 마디별(중간·원위·팁·손바닥) 접촉을 실측한다.

  · 어느 자세에서도 원위가 안 닿는다 → 정의를 폐기해야 한다(형상 조합 게이트 불가)
  · 닿는 자세가 있다 → 정의는 살리고 **접근 자세**를 고치면 된다

★리셋 오염 방지로 `episode_length_s` 를 무한대로 둔다(이 저장소에서 세 번 당한 함정).
★닫기 게이트를 끈다 — probe 는 폐쇄를 직접 지시한다.
★접촉은 학습과 **같은 판독 경로**(`force_matrix_w`)와 **무필터**(`net_forces_w`)를 나란히
  본다. 둘이 갈리면 그 자체가 결과다.

사용: isaaclab.sh -p scripts/probes/probe_wrap_reachability.py [--closure 1.0]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--closure", type=float, default=1.0, help="최종 폐쇄도 [0,1]")
parser.add_argument("--settle_steps", type=int, default=260, help="palm 이동·정착")
parser.add_argument("--close_steps", type=int, default=340, help="폐쇄 램프")
parser.add_argument("--capture_dir", type=str, default="",
                    help="자세별 RGB 저장 경로. ★수치만으로는 자세를 확인할 수 없다.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.capture_dir:
    args.enable_cameras = True          # TiledCamera 는 이 플래그 없이는 조용히 죽는다
app = AppLauncher(args).app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.tasks  # noqa: F401,E402

TASK = "open-sens_r_grasp_s2r"
# ★★격자는 **절대 palm 위치**(env-local)로 짠다. 컵 상대 오프셋으로 짜면 좌표계를
#   추측해야 하는데, 08.27 에 그걸 세 판 연속 틀렸다:
#     1차 월드 z 부호 반대 → palm 이 컵을 위에서 압착(palm 40~150N)
#     2차 월드 격자        → 25 중 20 자세가 명령 박스 밖
#     3차 palm 프레임 격자 → **100% 박스 밖**(축 대응이 또 달랐다)
#   절대 위치면 검증된 명령 집합 안인지 산술로 확인된다.
#
#   유효 명령 집합 = 홈 ± palm_delta_xyz ∩ 프로필 박스. 홈 (0.280, −0.3801, 0.4178) ·
#   delta (0.15, 0.35, 0.15) → x[0.20,0.430] · y[−0.550,−0.030] · z[**0.2678**,0.568].
#   ★z 바닥 0.2678 은 컵 원점 0.2773 보다 **9.5 mm 아래**뿐이다 — palm 은 컵보다
#     거의 못 내려간다. 이 자체가 H2 의 핵심 실측이다.
#   컵 (0.362, −0.160, 0.2773) · 파지중심 z 0.3073.
PALM_X = 0.362                                          # 컵 x 에 정렬
PALM_Y = (-0.280, -0.250, -0.220, -0.190, -0.160)       # 접근 여유(작을수록 밀착)
PALM_Z = (0.270, 0.290, 0.310, 0.330, 0.350)            # 높이(파지중심 0.3073)


def _save_captures(cam, tag: str) -> None:
    """타일별 RGB 를 PNG 로. 자세 번호 = 표의 `#` 와 같다."""
    import numpy as np
    from PIL import Image
    os.makedirs(args.capture_dir, exist_ok=True)
    rgb = cam.data.output["rgb"]                      # (N, H, W, 3|4) uint8
    arr = rgb[..., :3].detach().cpu().numpy().astype(np.uint8)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i]).save(
            os.path.join(args.capture_dir, f"pose{i:02d}_{tag}.png"))
    print(f"[capture] {arr.shape[0]}장 저장 → {args.capture_dir} ({tag})", flush=True)


def main() -> None:
    n_env = len(PALM_Y) * len(PALM_Z)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=n_env)
    cfg.close_gate_enabled = False          # probe 가 폐쇄를 직접 지시한다
    cfg.debug_camera = bool(args.capture_dir)   # ★센서는 clone 전에 만들어져야 한다
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.cfg.episode_length_s = 1.0e6        # ★리셋 오염 차단
    env.reset()
    dev = env.device
    cam = env.scene.sensors["debug_cam"] if args.capture_dir else None
    if cam is not None:
        # ★쿼터니언을 손으로 적지 않는다 — 08.27 에 추측했다가 컵이 화면 밖으로 나갔다.
        #   `set_world_poses_from_view` 가 eye→target 에서 회전을 **계산**한다.
        _eye = torch.tensor([0.66, -0.44, 0.50], device=dev)
        _tgt = torch.tensor([0.362, -0.160, 0.300], device=dev)   # 컵 파지중심 부근
        cam.set_world_poses_from_view(
            env.scene.env_origins + _eye, env.scene.env_origins + _tgt)

    palm_goal = torch.tensor([[PALM_X, gy, gz] for gz in PALM_Z for gy in PALM_Y],
                             device=dev)
    # 액션 역함수: palm = 홈 + delta(a) · delta = 0.5(a+1)(hi−lo) + lo
    span = (env._delta_hi - env._delta_lo)[:3].clamp(min=1e-9)
    a_xyz = (2.0 * (palm_goal - env._home_palm[:3].unsqueeze(0)
                    - env._delta_lo[:3].unsqueeze(0)) / span.unsqueeze(0) - 1.0)
    clipped = (a_xyz.abs() > 1.0).any(dim=-1)
    # ★액션은 6D 다 — 회전 3열을 빼면 손 액션이 밀려 조용히 어긋난다.
    #   회전 델타 박스는 ±palm_delta_rot_deg 로 대칭이라 a=0 이 정확히 홈 자세다.
    a_palm = torch.cat([a_xyz.clamp(-1.0, 1.0),
                        torch.zeros(n_env, 3, device=dev)], dim=-1)

    n_hand = int(env.cfg.action_space) - 6
    peak = {k: torch.zeros(n_env, len(env._finger_names), device=dev)
            for k in ("mid", "dist", "tip", "mid_net", "dist_net", "tip_net")}
    peak_palm = torch.zeros(n_env, device=dev)

    total = args.settle_steps + args.close_steps
    for t in range(total):
        if t < args.settle_steps:
            close = 0.0
        else:
            close = args.closure * min(
                1.0, (t - args.settle_steps) / max(args.close_steps * 0.6, 1.0))
        act = torch.cat([a_palm,
                         torch.full((n_env, n_hand), 2.0 * close - 1.0, device=dev)],
                        dim=-1)
        env.step(act)
        if cam is not None and t == args.settle_steps - 1:
            _save_captures(cam, "open")      # 폐쇄 직전 = 접근 자세
        if t < args.settle_steps:
            continue
        m, d, ti = env._finger_link_forces(env._mag_filtered)
        mn, dn, tn = env._finger_link_forces(env._mag_net)
        for k, v in (("mid", m), ("dist", d), ("tip", ti),
                     ("mid_net", mn), ("dist_net", dn), ("tip_net", tn)):
            peak[k] = torch.maximum(peak[k], v)
        peak_palm = torch.maximum(peak_palm, env._palm_contact_force())

    if cam is not None:
        _save_captures(cam, "closed")        # 완전 폐쇄 = 파지 결과
    palm_now = env._env_local(env.robot.data.body_pos_w[:, env.palm_idx])
    cup_now = env._env_local(env.object.data.root_pos_w)
    # ★지령이 아니라 **달성** 자세로 보고한다 — 팔이 컵에 막히거나 박스에 잘리면
    #   지령 격자는 실제로 탐색된 자세가 아니다(1~3차에서 오차 최대 0.35 m).
    _report(env, palm_now - cup_now, palm_goal, palm_now, clipped, peak, peak_palm)
    env.close()


def _report(env, offs, goal, now, clipped, peak, peak_palm) -> None:
    thr = float(env.cfg.contact_force_threshold)
    lo = float(env.cfg.diag_contact_threshold_lo)
    print("\n" + "=" * 100)
    print("W1 감쌈 도달성 — palm 자세 격자 × 완전 폐쇄. 마디별 **최대** 접촉력(N)")
    print(f"임계 {thr}N · 낮은임계 {lo}N · 손가락 {env._finger_names}")
    print("=" * 100)
    _clip = float(clipped.float().mean())
    if _clip > 0.2:
        print(f"⚠ 지령잘림 {_clip:.0%} — 격자가 명령 도달 범위(홈±delta) 밖이다. "
              "이 표는 격자를 재는 것이 아니라 팔이 갈 수 있던 곳을 잰다. 격자를 좁혀라.")
    print(f"{'#':>3} {'palm−컵 달성(mm)':>22} {'지령잘림':>8} {'palm오차':>8} "
          f"{'mid':>7} {'dist':>7} {'tip':>7} {'palm':>7} {'dist_net':>9} "
          f"{'≥thr 마디':>22}")
    n_dist_ok = 0
    for i in range(offs.shape[0]):
        o = (offs[i] * 1000).tolist()
        err = float((goal[i] - now[i]).norm())
        mx = {k: float(peak[k][i].max()) for k in peak}
        hit = [k for k in ("mid", "dist", "tip") if mx[k] > thr]
        if mx["dist"] > thr:
            n_dist_ok += 1
        print(f"{i:>3} ({o[0]:>6.1f},{o[1]:>5.1f},{o[2]:>6.1f}) "
              f"{'YES' if bool(clipped[i]) else '-':>8} {err:>8.3f} "
              f"{mx['mid']:>7.2f} {mx['dist']:>7.2f} {mx['tip']:>7.2f} "
              f"{float(peak_palm[i]):>7.2f} {mx['dist_net']:>9.2f} "
              f"{','.join(hit) if hit else '(없음)':>22}")

    print("-" * 100)
    print(f"원위가 임계를 넘은 자세: {n_dist_ok} / {offs.shape[0]}")
    # 손가락별 — 어느 손가락이 원위를 대는가.
    print("\n손가락별 원위 최대력(N) — 전 자세 통틀어:")
    for j, f in enumerate(env._finger_names):
        print(f"  {f:8s} filtered={float(peak['dist'][:, j].max()):6.2f}  "
              f"net={float(peak['dist_net'][:, j].max()):6.2f}  "
              f"(mid {float(peak['mid'][:, j].max()):5.2f} · "
              f"tip {float(peak['tip'][:, j].max()):5.2f})")
    _best = int(peak["dist"].max(dim=1).values.argmax())
    print(f"\n원위가 가장 강하게 닿은 자세 = #{_best} "
          f"offset(mm) {[round(v, 1) for v in (offs[_best] * 1000).tolist()]}")
    print("★판정: 원위 0/25 이면 `wrap = mid ∧ distal` 은 이 손·이 컵에서 성립 불가 —")
    print("       지표가 아니라 **정의**를 폐기해야 한다.")
    print("=" * 100 + "\n", flush=True)


main()
