"""팔이 실제로 접근·폐쇄·**리프트**하는 스크립트 롤아웃 — 파지 성립 지도.

## 왜 이걸 만들었나 (앞선 프로브 2종이 무효였다)

`probe_ua_cage_fit` / `probe_ua_place_scan` 은 컵을 **고정한 채** 손을 닫고 접촉
링크를 셌다. 09.03 에 이 방식이 두 층으로 무효임이 드러났다:

  ① **관통 에너지** — 닫히는 손가락 안에 컵을 붙들면 침투가 쌓이고, 놓는 순간
     방출된다. 실측에서 컵이 **2,085mm 위로 발사**됐고(낙하 −1897mm), 같은
     격자점이 실행마다 다른 답을 냈다(‘잡힘’ 1/64 → 재실행 0/64).
  ② **기하가 과제와 다르다** — 컵을 palm 기준 오프셋으로 **공중에** 띄웠다.
     실제 과제에서 컵은 테이블(z 0.200)에 서 있고 홈 손바닥은 z 0.419 로
     **165mm 위**다. 손이 내려와야 한다. 띄운 배치는 도달 가능성조차 없다.

그래서 여기서는 과제 그대로 잰다 — 컵은 자유(고정 없음·중력 있음), 팔은 env 의
액션·fabric 으로 실제 이동, 마지막에 **들어올려** 컵이 따라오는지 본다.
테이블은 손을 따라올 수 없으므로 ‘지지’와 ‘파지’가 이진으로 갈린다.

## 사용

    cd hdgp && ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_ua_scripted_grasp.py
"""

from __future__ import annotations

import argparse
import itertools

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-rh_r_grasp_ua-play-lstm")
parser.add_argument("--family", default="cup", choices=["shaker", "cup"])
parser.add_argument("--scale", type=float, default=0.58)
parser.add_argument("--close", type=float, default=0.85, help="폐쇄 목표 [0,1]")
# ★겨냥점을 **palm 프레임**에서 훑는다(열0 법선 · 열1 측방 · 열2 손가락방향).
#   월드축 격자는 손 방향과 무관해 "닫힘 수렴점 쪽"이라는 방향을 못 찍는다.
#   기준: 열린손 케이지 (34.9, 14.0, 71.2) · 폐쇄 0.5 엄지–검지 중점 (49.2, 34.8, 62.4).
parser.add_argument("--an", default="0.035,0.045,0.055", help="법선 [m]")
parser.add_argument("--al", default="0.014,0.024,0.034,0.044", help="측방 [m]")
parser.add_argument("--af", default="0.050,0.062,0.072", help="손가락방향 [m]")
parser.add_argument("--n_approach", type=int, default=220)
parser.add_argument("--n_close", type=int, default=140)
parser.add_argument("--n_lift", type=int, default=220)
parser.add_argument("--lift_m", type=float, default=0.08)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.grasp_ua.config  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> int:
    from openarm.agnostic.modules import object_bank as _ob

    grid = list(itertools.product(
        [float(x) for x in args_cli.an.split(",")],
        [float(x) for x in args_cli.al.split(",")],
        [float(x) for x in args_cli.af.split(",")]))
    n = len(grid)

    mk = _ob._cup if args_cli.family == "cup" else _ob._shaker
    _ob.BANKS["_script_scan"] = _ob.ObjectBank(
        name="_script_scan", specs=(mk(args_cli.scale),),
        note="probe_ua_scripted_grasp 임시 뱅크(비영속)")

    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=n)
    cfg.scene.num_envs = n
    cfg.object_bank = "_script_scan"
    cfg.enable_events = False
    cfg.enable_adr = False
    # ★★재소환도 종료도 **둘 다** 없애야 한다. 09.03 에 한쪽씩만 껐다가 두 번 속았다:
    #   · 재소환 OFF → 컵이 쓰러지면 **에피소드 종료** → 팔이 홈으로 → 도달오차 210mm
    #   · 재소환 ON  → 컵이 밀려나면 **순간이동 재소환** → 접촉 0 인데 "80mm 상승"
    #   그래서 재소환을 끄고, 종료 트리거(범위 이탈·낙하·전도)를 전부 무력화한다.
    #   컵은 물리가 데려간 자리에 그대로 두고, 끝에서 실제로 딸려 올라오는지만 본다.
    cfg.respawn_on_fail = False
    cfg.object_out_x = (-1e3, 1e3)
    cfg.object_out_y = (-1e3, 1e3)
    cfg.object_min_z = -1e3
    cfg.tilt_reset_deg = 1e9
    # ★임시 뱅크의 정착고 — cfg 기본값은 기본 뱅크(shaker) 기준이라 컵이 뜨거나 파묻힌다.
    cfg.object_origin_offset_z = float(mk(args_cli.scale).base_origin_offset_z) \
        * float(args_cli.scale)
    cfg.episode_length_s = 120.0
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    env.reset()

    dev = u.device
    aim = torch.tensor(grid, device=dev)                          # (n,3) palm 프레임 겨냥점
    obj0 = u._env_local(u.object.data.root_pos_w).clone()          # (n,3)
    # ★★목표는 **케이지**가 컵에 가는 것이지 손바닥 원점이 아니다. 이 손의 파지 지점은
    #   손바닥에서 `_cage_offset_palm` (34.9, 14.0, 71.2)mm 떨어져 있다. 09.03 에 손바닥
    #   원점을 컵으로 보냈다가 48점 전부 접촉 0 이 나왔다 — 손가락이 컵에서 80mm 밖에서
    #   닫히고 있었다(밀림 0.5mm 가 "건드리지도 못했다"는 증거였다).
    _aim_w = torch.einsum("nij,nj->ni", u._palm_ee_R(), aim)
    tgt_pos = obj0 - _aim_w                                        # 팔 목표(=palm) env-local

    anchor = u._palm_anchor()                                      # (n,6)
    lo, hi = u._delta_lo, u._delta_hi

    def act(target_pos: torch.Tensor, close: float) -> torch.Tensor:
        """palm 6D 목표 + 폐쇄도 → 액션. `_pre_physics_step` 의 역함수."""
        d = torch.zeros(n, 6, device=dev)
        d[:, :3] = target_pos - anchor[:, :3]                      # 회전 델타 = 0(프리셋 유지)
        a = (2.0 * (d - lo) / (hi - lo).clamp(min=1e-9) - 1.0).clamp(-1.0, 1.0)
        h = torch.full((n, u.cfg.action_space - 6), 2.0 * close - 1.0, device=dev)
        return torch.cat([a, h], dim=1)

    n_term = torch.zeros(n, device=dev)

    def run(steps: int, target_pos: torch.Tensor, close_from: float, close_to: float):
        nonlocal n_term
        for i in range(steps):
            c = close_from + (close_to - close_from) * min(1.0, i / max(1, steps - 40))
            _, _, _te, _tr, _ = env.step(act(target_pos, c))
            n_term += (_te | _tr).float()

    # ---- A 접근(손 열림) → B 폐쇄 → C 리프트 -----------------------------------------
    # ★★접근 구간에만 컵을 고정한다. 09.03 실측: 홈→목표 **직선 접근이 컵을 쓸어**
    #   테이블 밖으로 떨어뜨렸다(밀림 222.7mm = 낙하 200mm + 수평, 36개 env 동일).
    #   그 상태로 손을 닫으니 컵은 이미 바닥에 있어 접촉 0 이었다. 학습 정책은 이 문제가
    #   없다(rh_e1 `cup_disp` 2.8mm) — 직선으로 밀고 들어간 건 이 스크립트다.
    #   ⚠폐쇄 중에는 고정하지 않는다 — 닫히는 손가락 안에 붙들면 관통 에너지가 쌓여
    #   놓는 순간 컵이 발사된다(실측 2,085mm).
    _obj_hold = u.object.data.root_state_w.clone()

    def _freeze_obj() -> None:
        u.object.write_root_state_to_sim(_obj_hold.clone())

    for _i in range(args_cli.n_approach):
        _freeze_obj()
        env.step(act(tgt_pos, 0.0))
    # 해제 후 정착 — 손이 잡은 자리가 컵과 겹치면 여기서 튄다(그건 실패로 셀 것).
    for _i in range(40):
        env.step(act(tgt_pos, 0.0))
    _obj_rel = u._env_local(u.object.data.root_pos_w).clone()
    palm_a = u._env_local(u.robot.data.body_pos_w[:, u.palm_idx]).clone()
    reach_mm = (palm_a - tgt_pos).norm(dim=-1) * 1000.0
    _aim_a = palm_a + torch.einsum("nij,nj->ni", u._palm_ee_R(), aim)
    aim_err_mm = (_aim_a - obj0).norm(dim=-1) * 1000.0
    obj_a = _obj_rel
    knock_mm = (obj_a - obj0).norm(dim=-1) * 1000.0

    run(args_cli.n_close, tgt_pos, 0.0, float(args_cli.close))
    m1, d1 = u._contact_forces_split(); t1 = u._tip_contact_forces()
    thr = float(cfg.contact_force_threshold)
    z_b = u.object.data.root_pos_w[:, 2].clone()

    tgt_lift = tgt_pos.clone(); tgt_lift[:, 2] += float(args_cli.lift_m)
    run(args_cli.n_lift, tgt_lift, float(args_cli.close), float(args_cli.close))
    rise_mm = (u.object.data.root_pos_w[:, 2] - z_b) * 1000.0
    palm_rise_mm = (u._env_local(u.robot.data.body_pos_w[:, u.palm_idx])[:, 2]
                    - palm_a[:, 2]) * 1000.0
    tilt = u._tilt_deg.clone()

    fingers = list(u.profile.finger_sensor_bodies.keys())
    print("\n" + "=" * 104, flush=True)
    print(f"[script] {args_cli.family} scale {args_cli.scale} · 폐쇄 {args_cli.close} · "
          f"리프트 지령 {args_cli.lift_m*1000:.0f}mm · 격자 {n}점 "
          f"(컵 기준 팔목표 오프셋 dx,dy,dz [mm])", flush=True)
    rows = []
    for i in range(n):
        links = int((m1[i] > thr).sum() + (d1[i] > thr).sum() + (t1[i] > thr).sum())
        pat = " ".join(
            f"{fingers[k][:5]}:"
            f"{'M' if m1[i, k] > thr else '.'}"
            f"{'D' if d1[i, k] > thr else '.'}"
            f"{'T' if t1[i, k] > thr else '.'}" for k in range(len(fingers)))
        # ★리셋이 한 번이라도 났으면 그 env 는 판정 불가(팔이 홈으로 돌아갔다).
        reset = int(n_term[i]) > 0
        # ★접촉 필수 — 접촉 0 인데 컵이 올라갔다면 그건 파지가 아니라 계측 오염이다.
        held = (not reset) and links > 0 and float(rise_mm[i]) > 30.0 \
            and float(rise_mm[i]) > 0.5 * float(palm_rise_mm[i])
        rows.append((held, float(rise_mm[i]), links, grid[i], float(reach_mm[i]),
                     float(knock_mm[i]), float(tilt[i]), pat, reset,
                     float(aim_err_mm[i])))
    rows.sort(key=lambda r: (not r[0], r[8], -r[1]))
    for held, rise, links, g, rc, kn, tl, pat, rs, ae in rows[:16]:
        print(f"[script] 겨냥({g[0]*1000:5.0f},{g[1]*1000:5.0f},{g[2]*1000:5.0f}) "
              f"{'★파지' if held else ('리셋  ' if rs else '  실패')} 컵상승{rise:7.1f}mm "
              f"겨냥오차{ae:6.1f}mm 밀림{kn:6.1f}mm 기울{tl:5.1f}° "
              f"링크{links:2d} {pat}", flush=True)
    print(f"[script] ★리셋 발생 env {sum(1 for r in rows if r[8])}/{n} "
          f"— 이 env 들은 판정 불가", flush=True)
    print(f"[script] 손바닥 실제 상승 {palm_rise_mm.mean():.1f}mm "
          f"(지령 {args_cli.lift_m*1000:.0f}mm)", flush=True)
    print(f"[script] ★파지 성립 {sum(1 for r in rows if r[0])}/{n}", flush=True)
    print("=" * 104, flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    import traceback
    try:
        _rc = main()
    except BaseException:
        traceback.print_exc()
        _rc = 3
    _app.close()
    raise SystemExit(_rc)
