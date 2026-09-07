#!/usr/bin/env python3
"""양팔 파지를 **실기 배포와 동형의 폐루프**로 한 씬에서 보인다.

실기 구조(사용자 확정 09.02): 정책이 (로봇 상태 + FD++ 컵 pose)를 관측하고,
액션이 로봇을 직접 제어한다. 여기서 '현실' = 통합 pour 씬. 정책 사슬은
`bimanual_chain.py` 가 각 학습 env 의 원본 코드로 세운다.

순서:  정착(preset=정책 홈) → [Enter] ① 좌팔 v2B25 폐루프 — shaker 를 접촉으로
       파지·리프트 → [Enter] ② 유지 → (물리 dt 100→120Hz 전환) → [Enter]
       ③ 우팔 E1 폐루프 — cup_big_s100 → [Enter] ④ 양팔 유지

미러/재생과 다른 점: 텔레포트·컵 root 고정·순간 부착이 **없다**. 초기 배치
1회 뒤에는 PD 와 접촉만 있다. 여기서 파지가 실패하면 실기도 실패한다는 뜻이다.

    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_bimanual_closedloop.py \\
        --auto --render /tmp/bi_cl --verify           # 영상 + 재현 대조
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_bimanual_closedloop.py --gui
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_SR = Path("/home/user/rl_ws/sim2real")
parser.add_argument("--left-checkpoint", type=Path,
                    default=_SR / "logs/policy/left_v2B25/nn/v2B25_tip30_ep2150.pth")
parser.add_argument("--right-checkpoint", type=Path,
                    default=_SR / "logs/policy/right_g1/nn/g1_ep17000.pth",
                    help="우팔 grasp_s2r (09.02 사용자 지정 g1 — use_real_gains 런은 "
                         "HDGP_S2R_REAL_GAINS=1 로 실행)")
parser.add_argument("--left-stream", type=Path,
                    default=_SR / "logs/shadow/pour_entry/stream_left_v2b25.npz",
                    help="goal·컵 스폰·(--verify 시) 대조 기준")
parser.add_argument("--right-stream", type=Path,
                    default=_SR / "logs/shadow/pour_entry/stream_right_g1.npz")
parser.add_argument("--left-cup-json", type=Path, default=None,
                    help="FD++ capture JSON (base_link) — shaker 스폰을 실측으로 대체")
parser.add_argument("--right-cup-json", type=Path, default=None,
                    help="FD++ capture JSON — cup_big 스폰을 실측으로 대체")
parser.add_argument("--force-spawn", action="store_true",
                    help="실측 스폰이 학습 분포 상자 밖이어도 진행 (기본은 거부+재배치 안내)")
parser.add_argument("--stream", default=None,
                    help="HOST:PORT — 정책 스텝마다 관절지령 35f(<Id35f>: 좌팔7+우팔7+"
                         "좌그립1+우손20, 미사용 채널 NaN=실기 홀드) 송출")
parser.add_argument("--echo-port", type=int, default=None,
                    help="실기 실측 35f(magic 0x5A2B12) 수신 — console 유휴 동안 "
                         "sim 로봇에 그대로 써 GUI=실기 미러")
parser.add_argument("--console", type=Path, default=None,
                    help="명령 파일 경로 — reset|preset|left|right|attention|abort|"
                         "status|quit 를 써 넣으면 재부팅 없이 라운드를 반복한다 "
                         "(Step 3 콘솔 모드, auto 강제)")
parser.add_argument("--real", action="store_true",
                    help="실기 모드 — 차렷(전관절 0) 부팅·유휴는 echo 가 로봇 소유·"
                         "정책/전환 중 실측 스트림·reset 은 로봇 무접촉(컵만 재소환)")
parser.add_argument("--table-z", type=float, default=0.200,
                    help="실물 테이블 상면 z(로봇 베이스=ENV 원점 기준). 09.02 최종 실측 "
                         "0.200~0.205 — sim 과 사실상 정합이라 기본 0.200 유지. "
                         "(비전 컵 정착 Δ−21mm 는 z 편향 허수였다) 정밀 실측 시 조정용")
parser.add_argument("--bag-dir", type=Path,
                    default=_SR / "logs/shadow/reset_both",
                    help="차렷↔preset 전환 bag(npz) 디렉토리")
parser.add_argument("--transit-stride", type=int, default=3,
                    help="전환 bag 재생 배속(프레임 스트라이드) — 3이면 3배속 "
                         "0.75 rad/s. 실기 JTC --max-vel 이 이를 감당해야 한다")
parser.add_argument("--cmd-vel-limit", type=float, default=0.0,
                    help="정책 구간 팔 지령 속도상한[rad/s] — 초과하면 그만큼 sim 을 "
                         "**늦춰** 보낸다(시간 스케일링). 브리지의 위치 클램프와 달리 "
                         "경로를 왜곡하지 않는다. 0=끔. sim 요구치는 3.65 rad/s, 실기 "
                         "관절한계는 2.0 — 그 사이/이하로 잡는다")
parser.add_argument("--shadow-pace", type=float, default=0.0,
                    help="정책 스텝마다 실기가 이 값(rad) 안으로 따라올 때까지 sim 을 "
                         "**멈춰** 기다린다(0=끔). 액션은 16Hz·PD 는 고주파라, 다음 "
                         "액션 전에 추종을 끝낼 시간을 주는 것. 물리 스텝을 안 밟으므로 "
                         "상태·궤적은 불변이고 실행 속도만 느려진다. 권장 0.15")
parser.add_argument("--pace-timeout", type=float, default=2.0,
                    help="한 스텝당 추종 대기 상한[s] — 초과하면 경고 후 진행")
parser.add_argument("--live-follow", type=int, default=None,
                    help="UDP 포트 — FD++ pose 스트림(<Bfff: side,x,y,z)을 받아 spawn-only "
                         "동안 물체가 실측 x·y 를 따라다닌다 (z 는 정착고 유지)")
parser.add_argument("--spawn-only", action="store_true",
                    help="씬+FD++ 스폰+settle 까지만 하고 GUI 를 유지한다 "
                         "(정책 로드 없음 — Step1 실물↔sim 대조용)")
parser.add_argument("--left-steps", type=int, default=300)
parser.add_argument("--right-steps", type=int, default=420)
parser.add_argument("--stop-lift", type=float, default=0.08)
parser.add_argument("--lift-hold", type=int, default=40)
parser.add_argument("--settle", type=int, default=120)
parser.add_argument("--hold", type=int, default=120)
parser.add_argument("--final-hold", type=int, default=300)
parser.add_argument("--pour-checkpoint", type=Path,
                    default=_SR / "logs/policy/pour_e1/nn/e1_pour1_ep6500.pth",
                    help="e1_pour1 최종본 (ep6500 — 학습이 그 시점에 종료됨, md5 6e3366d5)")
parser.add_argument("--pour-steps", type=int, default=900)
parser.add_argument("--pour-mode", choices=("follow", "policy"), default="follow",
                    help="follow=네이티브 성공 에피소드의 관절 궤적 추종(사용자 지시 09.02) · "
                         "policy=폐루프 (현재 β=0 미해결)")
parser.add_argument("--pour-traj", type=Path,
                    default=_SR / "logs/shadow/pour_entry/pour_traj_receiver_live_ep6500.npz",
                    help="기본 = ep6500 · 실측 받는점(0.265,0.045,0.296) 기준 성공 궤적 20/20")
parser.add_argument("--skip-pour", action="store_true", help="파지 4국면까지만")
parser.add_argument("--receiver-up-contract", action="store_true",
                    help="pour obs 의 tgt_up 을 훈련 상수로 고정 — pour 훈련에서 받는컵"
                         " 자세는 kinematic 상수 계약이었다 (실컵 기울기 편차는 계약 밖)")
parser.add_argument("--diag-spoof-left-obs", action="store_true",
                    help="진단: pour obs 의 좌팔 18D 를 훈련 상수(REST·qd0)로 교체 — 좌팔 원인 분리")
parser.add_argument("--carry-vel", type=float, default=0.25,
                    help="전환 램프 관절속도 상한 [rad/s]")
parser.add_argument("--pour-entry-joints", default="0.512,0.414,-0.487,0.243,0.084,0.546,1.168",
                    help="⑤′ 우팔 pour 세팅 관절 7 — 기본 = E1 뱅크 mean (n=2107)")
parser.add_argument("--auto", action="store_true")
parser.add_argument("--verify", action="store_true",
                    help="기록 에피소드와 같은 스폰·goal 로 돌고 궤적 편차를 보고")
parser.add_argument("--render", type=Path, default=None)
parser.add_argument("--render-every", type=int, default=4)
parser.add_argument("--gui", action="store_true")

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "source" / "openarm"), str(_REPO / "scripts" / "tools"),
           str(_REPO / "scripts" / "probes"), str(_SR / "scripts")):
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher                              # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args.headless = not args.gui
args.enable_cameras = args.render is not None
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym                                           # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
import yaml                                                       # noqa: E402

import openarm  # noqa: E402,F401
import openarm.tasks  # noqa: E402,F401
from isaaclab.sim.utils import find_matching_prim_paths           # noqa: E402
from isaaclab_tasks.utils.parse_cfg import (                      # noqa: E402
    load_cfg_from_registry, parse_env_cfg)
from rl_games.torch_runner import Runner                          # noqa: E402
from run_cfg_restore import restore_run_cfg_if_available          # noqa: E402

from bimanual_chain import (                                      # noqa: E402
    LEFT9, LeftChain, RightChainShim, align_pour_cfg, make_show_env)
from openarm.tesollo.right.pour_sensor.pour_right_env import PourRightEnv  # noqa: E402

POUR_TASK = "open-tesol_r_pour_sensor-play-lstm"
LEFT_TASK = "open-grip_l_grasp_sensor_v2-play"
RIGHT_TASK = "open-sens_r_grasp_s2r-play-lstm"
DEVICE = "cuda:0"


def make_player(agent_cfg: dict, ckpt: Path, obs_dim: int, act_dim: int):
    """rl_games 플레이어를 env 없이 만든다 — 배포 노드가 하는 그대로."""
    cfg = agent_cfg["params"]
    cfg["config"]["env_info"] = {
        "observation_space": gym.spaces.Box(-np.inf, np.inf, (obs_dim,)),
        "action_space": gym.spaces.Box(-1.0, 1.0, (act_dim,)),
        "agents": 1,
    }
    cfg["config"]["num_actors"] = 1
    cfg["load_checkpoint"] = True
    cfg["load_path"] = str(ckpt)
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(str(ckpt))
    player.has_batch_dimension = True
    player.batch_size = 1
    return player


def main() -> int:
    # ── cfg 3벌: pour(씬) · 좌(사슬) · 우(사슬) — 전부 런 dump 로 복원 ────────
    left_cfg = parse_env_cfg(LEFT_TASK, device=DEVICE, num_envs=1)
    left_agent = load_cfg_from_registry(LEFT_TASK, "rl_games_cfg_entry_point")
    left_agent = restore_run_cfg_if_available(
        left_cfg, left_agent, resume_path=str(args.left_checkpoint),
        workspace_root=str(_REPO.parent))
    right_cfg = parse_env_cfg(RIGHT_TASK, device=DEVICE, num_envs=1)
    right_agent = load_cfg_from_registry(RIGHT_TASK, "rl_games_cfg_entry_point")
    right_agent = restore_run_cfg_if_available(
        right_cfg, right_agent, resume_path=str(args.right_checkpoint),
        workspace_root=str(_REPO.parent))
    right_cfg.finalize_after_overrides()
    _tz_off = float(args.table_z) - float(right_cfg.table_surface_z)
    if abs(_tz_off) > 1e-6:
        right_cfg.table_surface_z = float(args.table_z)
        print(f"[table-z] 우 사슬 표면 {args.table_z:.3f} (Δ{_tz_off:+.3f}) — "
              f"fabric 장애물·정착고·이송목표 일괄 파생", flush=True)


    zl = np.load(args.left_stream, allow_pickle=True)
    zr = np.load(args.right_stream, allow_pickle=True)
    left_goal = zl["goal"][0].astype(np.float32)              # 7D (root 프레임)
    right_goal = zr["goal"][0].astype(np.float32)             # 3D (env-local)
    left_spawn = zl["meta_cup_spawn"].astype(np.float32)
    right_spawn = zr["meta_cup_spawn"].astype(np.float32)

    # FD++ capture JSON 스폰 (Step 1) — base_link ≡ env-local (로봇 베이스가 원점).
    # 분포 상자 가드: 실물 배치가 학습 분포 밖이면 정책이 조용히 이상해진다 — 거부한다.
    from cup_pose_capture import load_capture, spawn_box_for_side, verdict  # noqa: PLC0415
    for _side, _jpath, _tag in (("left", args.left_cup_json, "shaker"),
                                ("right", args.right_cup_json, "cup_big")):
        if _jpath is None:
            continue
        _cp = load_capture(_jpath, expect_frame="base_link")
        _vd = verdict(_cp, spawn_box_for_side(_side))
        print(f"[FD++ 스폰] {_tag} ← {_jpath.name}: "
              f"{[round(v, 4) for v in _cp.position]} · "
              f"{'분포 안 ✅' if _vd.inside else '분포 밖 ❌'}", flush=True)
        if not _vd.inside and not args.force_spawn:
            raise SystemExit(
                f"[FD++ 스폰] {_tag} 이 학습 분포 밖이다:\n{_vd.describe()}\n"
                "실물 컵을 상자 안으로 재배치하거나 --force-spawn 으로 강행하라.")
        if _side == "left":
            left_spawn = np.array(_cp.position, dtype=np.float32)
            left_spawn[2] = args.table_z + 0.102   # 상면 + shaker 원점(바닥+0.0921, shaker_closed_rl) + 1cm
        else:
            right_spawn = np.array(_cp.position, dtype=np.float32)
            right_spawn[2] = args.table_z + 0.076  # 상면 + cup_big 원점 0.0657 + 1cm
            right_goal = right_spawn + np.array([0.0, 0.0, 0.12], dtype=np.float32)
            print("[FD++ 스폰] 우 goal = 스폰 + (0,0,0.12) (E1/g1 liftonly 규약)", flush=True)

    left_yaml = yaml.unsafe_load(
        (args.left_checkpoint.parent.parent / "params" / "env.yaml").read_text())
    right_yaml = yaml.unsafe_load(
        (args.right_checkpoint.parent.parent / "params" / "env.yaml").read_text())
    # ★skip-pour 여도 pour dump 는 항상 읽는다 — 씬(테이블 env_rigid·물리 상수)이
    #   이 복원에 딸려 있어, 건너뛰면 검증된 E2E 와 다른 테이블에서 파지가 돈다.
    pour_yaml = yaml.unsafe_load(
        (args.pour_checkpoint.parent.parent / "params" / "env.yaml").read_text())

    pour_cfg = parse_env_cfg(POUR_TASK, device=DEVICE, num_envs=1)
    pour_agent = load_cfg_from_registry(POUR_TASK, "rl_games_cfg_entry_point")
    # ★e1_pour1 런 cfg 복원 — 씬 필드는 뒤의 align 이 다시 정리하고, 사슬이 읽는
    #   런타임 파라미터(리미터·게이트·보상 상수·freeze·구슬 수)가 학습본으로 잠긴다.
    #   skip-pour 여도 복원한다(씬 동일성) — 국면 실행만 생략.
    pour_agent = restore_run_cfg_if_available(
        pour_cfg, pour_agent, resume_path=str(args.pour_checkpoint),
        workspace_root=str(_REPO.parent))
    lgrip = str(_REPO / "assets/robot/openarm_tesollo_sensor_rl_lgrip"
                / "openarm_tesollo_sensor_rl.usd")
    for line in align_pour_cfg(pour_cfg, left_scene=left_yaml["scene"],
                               right_actuators=right_yaml["robot_cfg"]["actuators"],
                               lgrip_usd=lgrip, left_spawn=left_spawn,
                               physics_dt=float(left_cfg.sim.dt)):
        print(f"[정합] {line}")
    _tz_off2 = float(args.table_z) - 0.200
    if abs(_tz_off2) > 1e-6 and hasattr(pour_cfg, "table_cfg"):
        _tp = list(pour_cfg.table_cfg.init_state.pos)
        _tp[2] += _tz_off2
        pour_cfg.table_cfg.init_state.pos = tuple(_tp)
        print(f"[table-z] 씬 테이블(env_rigid) z {_tz_off2:+.3f} — 실측 상면 "
              f"{args.table_z:.3f} 정합", flush=True)


    # ── 씬 ──────────────────────────────────────────────────────────────────
    env = make_show_env(PourRightEnv)(cfg=pour_cfg, render_mode=None)
    env.reset()
    robot, scene = env.robot, env.scene
    if not find_matching_prim_paths("/World/envs/env_0/Cup/baseLink"):
        raise RuntimeError("컵 필터 프림이 없다 — 센서 force_matrix 가 무증상 0 이 된다")

    dt_box = [float(left_cfg.sim.dt)]                    # 국면별 물리 dt (좌 먼저)

    def T(a, shape=None):
        t = torch.tensor(np.asarray(a, dtype=np.float32), device=env.device)
        return t if shape is None else t.reshape(shape)

    def place(obj, pos_env_local):
        pose = torch.cat([scene.env_origins[0] + T(pos_env_local),
                          T([1.0, 0, 0, 0])]).unsqueeze(0)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))

    place(env.cup, right_spawn)
    place(env.left_target_cup, left_spawn)
    if hasattr(env, "_hide_beads"):
        env._hide_beads(torch.arange(1, device=env.device))

    # 정책 홈 주입 — default_joint_pos 를 (좌 v2B25 · 우 E1) 로. 이것이 preset 이다.
    # ★★각 dump 는 **반대팔의 주차 자세도** 담고 있다 (우 dump 의 l_aj_2=-0.671 등).
    #   필터 없이 합치면 나중 소스가 앞 팔의 홈을 덮어써 TCP 가 11cm 어긋난다 —
    #   그런데 obs joint 항은 자기 default 상대라 0 으로 보여 **조용히** 틀린다
    #   (09.02 diagL2 실측, obs0 대조가 잡음). 각 소스는 자기 팔 접두사만 낸다.
    dq = robot.data.default_joint_pos
    injected = 0
    for prefix, src in (("l_", left_yaml["scene"]["robot"]["init_state"]["joint_pos"]),
                        ("r_", _right_home(right_cfg))):
        for pat, val in src.items():
            if not pat.startswith(prefix):
                continue
            ids, _ = robot.find_joints(pat)
            for i in ids:
                dq[:, i] = float(val)
                injected += 1
    print(f"[홈] 관절 {injected}개 주입 (좌 v2B25 · 우 E1)")
    robot.write_joint_state_to_sim(dq.clone(), torch.zeros_like(dq))
    robot.set_joint_position_target(dq.clone())

    # ── 우 사슬 (부팅 게이트 3종이 여기서 돈다) + 좌 사슬 ────────────────────
    right = RightChainShim(env, right_cfg, env.bi_finger_sensors, env.bi_palm_sensor)
    right.zero_obs_noise()
    right.goal_pos[:] = T(right_goal).unsqueeze(0)
    left = LeftChain(env, left_cfg,
                     step_dt=float(left_cfg.sim.dt) * int(left_cfg.decimation))
    left.reset()

    if args.real:
        # 실기 규약 — 부팅 = 차렷: 팔 0 · 좌그립 OPEN · ★우손 = bag 차렷 손자세(주먹형).
        # 팔0 에서 손목이 상판(z0.19~0.2, x≥0.07)까지 6cm 뿐이라 손 형상이 전부다:
        # 폄(0)은 손끝이 바닥 관통, preset 파지형은 손가락이 +x 로 뻗어 상판 모서리
        # 관통 — 둘 다 즉발 470 rad/s (09.02 FK+실측). bag grip[0] 주먹형은
        # 손끝 x≤0.051·z≥0.117 로 관통권 0 — 실물 zero 주먹과도 동형.
        from openarm.gripper.left.grasp_sensor import grasp_left_preset as _P  # noqa: PLC0415
        _jn0 = robot.joint_names
        q_att = robot.data.joint_pos.clone()
        for _n in ([f"l_aj_{i}" for i in range(1, 8)]
                   + [f"r_aj_{i}" for i in range(1, 8)]):
            q_att[0, _jn0.index(_n)] = 0.0
        for _n in ("l_hj_gripper_1", "l_hj_gripper_2"):
            q_att[0, _jn0.index(_n)] = float(_P.GRIPPER_OPEN_POS)
        _fist = np.load(args.bag_dir / "reset_right_safe.npz", allow_pickle=True)
        for _n, _v in zip(_fist["meta_grip_names"],
                          _fist["grip_cmd"].reshape(len(_fist["grip_cmd"]), -1)[0]):
            q_att[0, _jn0.index(str(_n))] = float(_v)
        robot.write_joint_state_to_sim(q_att, torch.zeros_like(q_att))
        robot.set_joint_position_target(q_att)
        print("[real] 차렷 부팅 — 유휴=echo 소유 · 전환/정책=sim 소유+실측 스트림",
              flush=True)

    # ── 실기 스트림/에코 (Step 3) ───────────────────────────────────────────
    STREAM_MAGIC_CMD = 0x5A2B11
    ECHO_MAGIC_MEAS = 0x5A2B12
    _jn = robot.joint_names
    s3_l_arm = [_jn.index(f"l_aj_{i}") for i in range(1, 8)]
    s3_r_arm = [_jn.index(f"r_aj_{i}") for i in range(1, 8)]
    s3_l_grip = [_jn.index(n) for n in ("l_hj_gripper_1", "l_hj_gripper_2")]
    s3_r_hand = [_jn.index(f"r_hj_{f}_{j}") for f in
                 ("thumb", "index", "middle", "ring", "pinky") for j in range(1, 5)]
    import math as _s3math  # noqa: PLC0415
    import socket as _s3sock  # noqa: PLC0415
    import struct as _s3struct  # noqa: PLC0415
    import time as _s3time  # noqa: PLC0415
    _S3FMT = "<Id35f"
    stream_sock = stream_addr = None
    if args.stream is not None:
        _h, _pp = args.stream.rsplit(":", 1)
        stream_addr = (_h, int(_pp))
        stream_sock = _s3sock.socket(_s3sock.AF_INET, _s3sock.SOCK_DGRAM)
        print(f"[stream] 지령 35f → {stream_addr}", flush=True)
    echo_sock = None
    if args.echo_port is not None:
        echo_sock = _s3sock.socket(_s3sock.AF_INET, _s3sock.SOCK_DGRAM)
        echo_sock.bind(("0.0.0.0", args.echo_port))
        echo_sock.setblocking(False)
        print(f"[echo] 실측 35f 수신 :{args.echo_port} — 유휴 중 GUI=실기 미러", flush=True)

    def stream_meas() -> None:
        """sim **실측**을 전 35채널 송출 — 정책·전환 공용 신호.

        ★왜 fabric 목표가 아니라 실측인가 (09.02 실증). sim 좌팔은 kp10 손목에
          중력보상이 **없어** 목표에 못 닿고 처진 채 움직인다. 실기는 중력보상이
          있어 목표에 **닿는다**. 그래서 목표를 보내면 실기가 sim 보다 더 꺾인다
          (j7 과도 굴곡 — 사용자 관찰, pace 괴리 0.42 rad 이 51스텝 중 44회 타임아웃).
          두 PD 의 추종오차가 다르기 때문이다. sim 이 **실제로 도달한** 자세를 보내면
          중력보상 덕에 실기가 그 자세에 정확히 도달한다 → sim 실측 = 실기 실측.
        ★단 두 조건이 붙는다: (i) **물리스텝마다** 보낼 것 — 정책 주기로 솎으면
          4~6° 씩 점프해 도착한다. (ii) 속도는 **시간 스케일링**으로 제한할 것 —
          브리지 위치 클램프로 자르면 잘린 몫이 누적돼 가드가 끊는다(0.65 rad).
        """
        if stream_sock is None:
            return
        q = robot.data.joint_pos[0]
        _vel_gate([float(q[i]) for i in s3_l_arm] + [float(q[i]) for i in s3_r_arm])
        pay = ([float(q[i]) for i in s3_l_arm] + [float(q[i]) for i in s3_r_arm]
               + [float(q[s3_l_grip[0]])] + [float(q[i]) for i in s3_r_hand])
        stream_sock.sendto(
            _s3struct.pack(_S3FMT, STREAM_MAGIC_CMD, _s3time.time(), *pay), stream_addr)

    _tgt_box: list = [None, 0.0]      # [직전 팔 지령(14ch), 그 송출 시각]
    # live-follow 수신부 — 유휴엔 물체를 옮기고(lf_poll), 정책 중엔 **기록만** 한다.
    # (FP++ 좌표 vs sim 물체 좌표를 같은 시간축에 남겨 인식 오차·지연을 본다)
    lf_sock_box: list = [None]
    lf_live: dict = {}                # side → (x, y, z, 수신시각)
    if args.live_follow is not None:
        _ls = _s3sock.socket(_s3sock.AF_INET, _s3sock.SOCK_DGRAM)
        _ls.bind(("0.0.0.0", args.live_follow))
        _ls.setblocking(False)
        lf_sock_box[0] = _ls
        print(f"[live-follow] UDP :{args.live_follow} — 유휴 중 컵 x·y 추종 "
              f"(정책 후 잠금, reset 이 해제) · 정책 중엔 기록만", flush=True)

    def lf_drain() -> None:
        sk = lf_sock_box[0]
        if sk is None:
            return
        try:
            while True:
                _pkt, _ = sk.recvfrom(64)
                if len(_pkt) == 13:
                    _sd, _x, _y, _z = _s3struct.unpack("<Bfff", _pkt)
                    lf_live[_sd] = (_x, _y, _z, _s3time.time())
        except BlockingIOError:
            pass
    # ★제어 사슬 4신호 기록 — 액션 / fabric 관절목표 / sim 실측 / 실기 실측.
    #   "제어가 맞는데 궤적이 이상하다"를 판별하려면 이 넷을 같은 시간축에서 봐야 한다
    #   (09.02: 정책 첫 프레임에 지령이 홈보다 30° 앞서 있었는데, 그게 액션 탓인지
    #   fabric 탓인지 스트림 탓인지 bag 만으로는 구분되지 않았다).
    trace_box: dict = {"on": False, "side": "l", "rows": [], "acts": []}

    def _trace_substep() -> None:
        if not trace_box["on"]:
            return
        ids = l_arm_ids7 if trace_box["side"] == "l" else r_arm_ids7
        lo = 0 if trace_box["side"] == "l" else 7
        echo_drain()
        m = echo_latest[0]
        tq = robot.data.joint_pos_target[0, ids]
        mq = robot.data.joint_pos[0, ids]
        lf_drain()
        obj = env.left_target_cup if trace_box["side"] == "l" else env.cup
        op = (obj.data.root_pos_w[0] - scene.env_origins[0]).tolist()
        side_id = 1 if trace_box["side"] == "l" else 0
        fp = lf_live.get(side_id)
        trace_box["rows"].append(
            [_s3time.time(), float(len(trace_box["acts"]))]
            + [float(x) for x in tq] + [float(x) for x in mq]
            + ([float(m[lo + i]) for i in range(7)] if m is not None
               else [float("nan")] * 7)
            + [float(v) for v in op]
            + ([fp[0], fp[1], fp[2], _s3time.time() - fp[3]] if fp is not None
               else [float("nan")] * 4))

    def _trace_dump(nm: str) -> None:
        if not trace_box["rows"]:
            return
        out = _SR / f"logs/shadow/policy_trace_{nm}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out,
                 rows=np.array(trace_box["rows"], dtype=np.float64),
                 acts=np.array(trace_box["acts"], dtype=np.float64),
                 cols=np.array(["t", "step"] + [f"tgt{i}" for i in range(1, 8)]
                               + [f"sim{i}" for i in range(1, 8)]
                               + [f"real{i}" for i in range(1, 8)]
                               + ["obj_x", "obj_y", "obj_z"]
                               + ["fpp_x", "fpp_y", "fpp_z", "fpp_age"]))
        print(f"[trace] {out.name} · {len(trace_box['rows'])}행 · "
              f"액션 {len(trace_box['acts'])}개 저장", flush=True)
        trace_box["rows"], trace_box["acts"] = [], []

    def _vel_gate(arm_now: list) -> None:
        """★속도 제한은 **시간 스케일링**으로 — 위치 클램프(브리지 --max-vel)는 신호를
        잘라 경로를 왜곡하고 잘린 몫이 누적된다(09.02: 0.65 rad → 가드 중단).
        여기서는 지령이 너무 빨리 변하면 sim 을 그만큼 늦춘다. 경로는 글자 그대로
        보존되고 실행 속도만 내려간다(위치 제어라 정적으로 동등)."""
        if args.cmd_vel_limit <= 0:
            return
        prev, t_prev = _tgt_box[0], _tgt_box[1]
        if prev is not None:
            dmax = max(abs(a - b) for a, b in zip(arm_now, prev))
            wait = dmax / args.cmd_vel_limit - (_s3time.time() - t_prev)
            if wait > 0:
                _s3time.sleep(wait)
        _tgt_box[0], _tgt_box[1] = arm_now, _s3time.time()

    def stream_target() -> None:
        """정책 구간 송출 = **fabric 이 만든 관절목표**(= 학습 때 sim PD 가 받은 입력).

        구조(좌·우 동일):  action → fabric IK → 관절목표 → PD → 실측
        실기도 같은 지점에서 갈라져야 한다. 관절목표를 보내면 실기 PD 가 sim PD 와
        같은 입력을 받아 같은 움직임을 낸다. 실측(PD **출력**)을 보내면 실기가
        '이미 한 번 뒤처진 신호'를 다시 뒤처져 쫓아 지연이 2중으로 쌓인다.

        ★08.31 의 "지령 말고 실측을 보내라" 규약은 **중력 처짐** 때문이었다 — 지령이
          앞서가는데 팔이 4.2° 처져 테이블을 긁었다. 09.02 좌팔 중력보상으로 그 전제가
          사라졌으므로 목표 스트림이 맞다. (전환 구간은 검증된 실측 스트림 유지)
        ★fabric 은 물리스텝마다 목표를 재생성하므로 서브스텝 훅으로 불린다 — 정책
          주기로 솎으면 실기엔 4~6° 씩 점프해 도착한다(09.02 bag 실측).
        """
        if stream_sock is None:
            return
        q = robot.data.joint_pos_target[0]
        _vel_gate([float(q[i]) for i in s3_l_arm] + [float(q[i]) for i in s3_r_arm])
        pay = ([float(q[i]) for i in s3_l_arm] + [float(q[i]) for i in s3_r_arm]
               + [float(q[s3_l_grip[0]])] + [float(q[i]) for i in s3_r_hand])
        stream_sock.sendto(
            _s3struct.pack(_S3FMT, STREAM_MAGIC_CMD, _s3time.time(), *pay), stream_addr)

    # 소유권 상태기계 — idle: echo 가 sim 로봇 소유 / transit·policy: sim 이 소유하고
    # 실측을 스트림, echo 는 괴리 감시로만 쓴다. 동시 소유가 지난 폭발의 원인 후보.
    mode_box = ["idle"]
    echo_latest: list = [None]
    att_hand_box: list = [None]    # 부팅 실측 어텐션 손 자세(20ch) — 전환 손 목표
    guard_box = [0, 0.0]           # [연속 초과 프레임, 최대 괴리]

    def echo_drain() -> bool:
        """소켓의 최신 실측 패킷을 버퍼로 — 적용 여부는 호출자가 결정."""
        if echo_sock is None:
            return False
        got = False
        try:
            while True:
                _pkt, _ = echo_sock.recvfrom(256)
                if len(_pkt) == _s3struct.calcsize(_S3FMT):
                    _v = _s3struct.unpack(_S3FMT, _pkt)
                    if _v[0] == ECHO_MAGIC_MEAS:
                        echo_latest[0] = _v[2:]
                        got = True
                        if att_hand_box[0] is None and not _s3math.isnan(_v[2 + 15]):
                            att_hand_box[0] = [float(x) for x in _v[2 + 15:2 + 35]]
                            print("[real] 어텐션 손 자세 기록(실측 20ch) — 전환 손 목표",
                                  flush=True)
        except BlockingIOError:
            pass
        return got

    def shadow_guard() -> None:
        """전환·정책 중 실기 괴리 감시(팔 14ch) — 0.5 rad 초과가 0.5초 지속되면 중단."""
        if echo_sock is None:
            return
        echo_drain()
        m = echo_latest[0]
        if m is None:
            return
        q = robot.data.joint_pos[0]
        d = 0.0
        for ids, seg in ((s3_l_arm, m[0:7]), (s3_r_arm, m[7:14])):
            for k, v in zip(ids, seg):
                if not _s3math.isnan(v):
                    d = max(d, abs(float(q[k]) - float(v)))
        guard_box[1] = max(guard_box[1], d)
        guard_box[0] = guard_box[0] + 1 if d > 0.5 else 0
        if guard_box[0] >= 25:
            raise RuntimeError(f"실기 괴리 {d:.2f} rad 이 0.5초 지속 — 그림자 이탈")

    pace_box = [0.0, 0.0, 0]     # [누적 대기 s, 최대 괴리 rad, 타임아웃 횟수]

    def pace_shadow(ids, seg_lo: int) -> None:
        """★저Hz 액션 · 고Hz PD — 다음 액션 전에 실기가 따라잡도록 sim 을 멈춰 기다린다.

        09.02 실측: 정책 지령이 62ms 마다 4.19°(중앙 1.13 rad/s)씩 오는데 브리지 상한이
        1.0 rad/s 라 매 프레임 뒤처졌고, 잘린 몫 **0.650 rad** 이 선형으로 쌓여 3.1초 만에
        가드(0.66 rad)가 끊었다. 빠르게 쫓게 만드는 대신 **천천히 실행**한다.

        sim 을 '멈춘다' = 물리 스텝을 안 밟는다. 상태가 안 변하니 정책은 같은 상태에서
        같은 액션을 내고, 궤적은 그대로다 — 바뀌는 건 벽시계 속도뿐이다. 대기 중에도
        stream_meas 를 계속 보내야 브리지 CMD_TIMEOUT(1.0s)에 걸리지 않는다.
        """
        if args.shadow_pace <= 0 or echo_sock is None:
            return
        t0 = _s3time.time()
        d = 0.0
        while _s3time.time() - t0 < args.pace_timeout:
            echo_drain()
            m = echo_latest[0]
            if m is None:
                return
            q = robot.data.joint_pos[0]
            d = max((abs(float(q[k]) - float(m[seg_lo + i]))
                     for i, k in enumerate(ids)
                     if not _s3math.isnan(m[seg_lo + i])), default=0.0)
            pace_box[1] = max(pace_box[1], d)
            if d <= args.shadow_pace:
                break
            stream_meas()            # 대기 중에도 같은 신호를 유지 송출
            _s3time.sleep(0.02)
        else:
            pace_box[2] += 1
            print(f"[pace] 추종 대기 {args.pace_timeout:.1f}s 초과 — 괴리 {d:.3f} rad "
                  f"(상한 {args.shadow_pace:.2f})", flush=True)
        pace_box[0] += _s3time.time() - t0

    def pace_report(nm: str, steps: int) -> None:
        if args.shadow_pace <= 0:
            return
        print(f"[pace:{nm}] 누적 대기 {pace_box[0]:.1f}s / {steps}스텝 · 최대 괴리 "
              f"{pace_box[1]:.3f} rad · 타임아웃 {pace_box[2]}회", flush=True)

    def check_abort() -> bool:
        """콘솔 파일의 abort 를 소비 — 전환·정책 루프가 주기적으로 부른다."""
        if args.console is None:
            return False
        try:
            c = args.console.read_text().strip()
        except OSError:
            return False
        if c == "abort":
            args.console.write_text("")
            print("[console] ← abort — 즉시 중단(스트림 정지, 실기는 현 자세 유지)",
                  flush=True)
            return True
        return False

    def echo_apply() -> bool:
        """(유휴 전용) 실측 패킷을 sim 로봇에 그대로 쓴다(상태+목표). NaN 채널은 유지."""
        if not echo_drain():
            return False
        latest = echo_latest[0]
        q = robot.data.joint_pos.clone()
        for ids, seg in ((s3_l_arm, latest[0:7]), (s3_r_arm, latest[7:14]),
                         (s3_r_hand, latest[15:35])):
            for k, v in zip(ids, seg):
                if not _s3math.isnan(v):
                    q[0, k] = float(v)
        if not _s3math.isnan(latest[14]):
            for k in s3_l_grip:
                q[0, k] = float(latest[14])
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        return True

    # ── 렌더 ────────────────────────────────────────────────────────────────
    shots = [0]

    def shot(tag: str) -> None:
        return

    if args.render is not None:
        import omni.replicator.core as rep  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        args.render.mkdir(parents=True, exist_ok=True)
        c = scene.env_origins[0].cpu().numpy() + np.array([0.34, 0.0, 0.32])
        cam = rep.create.camera(position=tuple(c + np.array([1.20, -0.70, 0.55])),
                                look_at=tuple(float(v) for v in c))
        rp = rep.create.render_product(cam, (1280, 800))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp])

        def shot(tag: str) -> None:  # noqa: F811
            env.sim.render()
            arr = np.asarray(annot.get_data())
            if arr.size:
                Image.fromarray(arr[:, :, :3]).save(
                    args.render / f"{shots[0]:04d}_{tag}.png")
                shots[0] += 1

    # ── 감시 (전 스텝) — 폭발·NaN 을 수치로 잡는다. 미러 사고의 재발 방지 ────
    worst = {"qd": 0.0, "cupv": 0.0}

    def watch() -> None:
        qd_all = robot.data.joint_vel[0].abs()
        qd = float(qd_all.max())
        cv = max(float(env.cup.data.root_lin_vel_w.norm(dim=-1).max()),
                 float(env.left_target_cup.data.root_lin_vel_w.norm(dim=-1).max()))
        worst["qd"] = max(worst["qd"], qd)
        worst["cupv"] = max(worst["cupv"], cv)
        if not torch.isfinite(robot.data.joint_pos).all():
            raise RuntimeError("관절 NaN — 물리 폭발")
        if qd > 12.0 or cv > 3.0:
            top = torch.topk(qd_all, min(5, qd_all.numel()))
            who = " ".join(f"{robot.joint_names[i]}={v:.0f}"
                           for i, v in zip(top.indices.tolist(), top.values.tolist()))
            raise RuntimeError(f"폭발 감지: |q̇|max {qd:.1f} rad/s · 컵 {cv:.2f} m/s"
                               f" · 상위 [{who}]")

    def cups_z() -> str:
        s = float(env.cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        r = float(env.left_target_cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        return f"cup_big z {s:.3f} · shaker z {r:.3f}"

    def passive(tag: str, n: int) -> None:
        for f in range(n):
            scene.write_data_to_sim()
            env.sim.step(render=args.gui)
            scene.update(dt_box[0])
            watch()
            if mode_box[0] != "idle":
                stream_meas()          # 유지 구간도 정책과 같은 신호
            if f % args.render_every == 0:
                shot(tag)
        print(f"[{tag}] {n}스텝 · {cups_z()}")

    def gate(msg: str) -> None:
        print(f"\n▶ {msg}")
        if not args.auto:
            input("  [Enter] …")

    passive("0settle", args.settle)
    settle_rz = settle_lz = 0.0
    for _nm, _obj, _rec in (("cup_big", env.cup, right_spawn),
                            ("shaker", env.left_target_cup, left_spawn)):
        _z = float(_obj.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        if _nm == "cup_big":
            settle_rz = _z
        else:
            settle_lz = _z
        print(f"[정착] {_nm} z {_z:.4f} (기록 {float(_rec[2]):.4f} · "
              f"Δ {_z - float(_rec[2]):+.4f})", flush=True)
    # 앵커 스냅샷 — env 리셋 정착 스냅샷의 대응물 (anchor_mode=spawn 이 이걸 쓴다)
    right.object_spawn_pos[:] = (env.cup.data.root_pos_w - scene.env_origins)[0:1]

    if args.spawn_only:
        print("[spawn-only] 정책 로드 생략 — GUI 유지 (Ctrl+C 종료)", flush=True)
        sock = None
        if args.live_follow is not None:
            import socket as _socket  # noqa: PLC0415
            import struct as _struct  # noqa: PLC0415
            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", args.live_follow))
            sock.setblocking(False)
            # z 는 정착고를 유지한다 — 실물 테이블과 sim 테이블 높이가 달라(cup Δ−21mm)
            # 실측 z 로 박으면 뜨거나 파고든다. x·y 만 실측을 따른다.
            _objs = {0: env.cup, 1: env.left_target_cup}
            _settle_z = {k: float(o.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
                         for k, o in _objs.items()}
            _live: dict[int, tuple[float, float]] = {}
            _rx = [0]
            print(f"[live-follow] UDP :{args.live_follow} — FD++ x·y 추종 시작", flush=True)
        try:
            while simulation_app.is_running():
                for _f in range(600):
                    if sock is not None:
                        try:
                            while True:
                                _pkt, _ = sock.recvfrom(64)
                                if len(_pkt) == 13:
                                    _sd, _x, _y, _z = _struct.unpack("<Bfff", _pkt)
                                    if _sd in _objs:
                                        _live[_sd] = (_x, _y)
                                        _rx[0] += 1
                        except BlockingIOError:
                            pass
                        for _sd, _xy in _live.items():
                            place(_objs[_sd], (_xy[0], _xy[1], _settle_z[_sd]))
                    scene.write_data_to_sim()
                    env.sim.step(render=args.gui)
                    scene.update(dt_box[0])
                    watch()
                if sock is not None:
                    print(f"[live-follow] 수신 {_rx[0]} · {cups_z()}", flush=True)
                else:
                    print(f"[spawn_hold] {cups_z()}", flush=True)
        except KeyboardInterrupt:
            pass
        return 0

    # ── 플레이어 2벌 ────────────────────────────────────────────────────────
    from left_obs_builder import ACTOR_OBS_DIM as LEFT_OBS_DIM  # noqa: PLC0415
    from left_obs_builder import NUM_ACTIONS as LEFT_ACTS  # noqa: PLC0415
    player_l = make_player(left_agent, args.left_checkpoint, LEFT_OBS_DIM, LEFT_ACTS)
    player_r = make_player(right_agent, args.right_checkpoint,
                           int(right_cfg.observation_space),
                           int(right_cfg.action_space))

    trace: dict[str, list] = {k: [] for k in
                              ("l_palm", "l_armt", "r_palm", "r_armt", "r_latch")}

    def _policy_start_check(side: str) -> None:
        """정책 시작 직전 sim(preset)↔실기 정합 — 어긋난 채 시작하면 그림자가 튄다."""
        echo_drain()
        m = echo_latest[0]
        if m is None:
            return
        q = robot.data.joint_pos[0]
        ids, seg = (s3_l_arm, m[0:7]) if side == "left" else (s3_r_arm, m[7:14])
        ds = [abs(float(q[k]) - float(v))
              for k, v in zip(ids, seg) if not _s3math.isnan(v)]
        mx = max(ds) if ds else float("nan")
        print(f"[정합] {side} 정책 시작 sim↔실기 |Δq|max {mx:.3f} rad"
              + (" ⚠ 0.1 초과 — preset 정렬을 확인하라" if mx > 0.1 else " ✓"),
              flush=True)

    def do_left() -> int:
        mode_box[0] = "policy"
        guard_box[0], guard_box[1] = 0, 0.0
        _policy_start_check("left")
        trace_box.update(on=True, side="l", rows=[], acts=[])
        if player_l.is_rnn:
            player_l.init_rnn()
        # ── ① 좌팔 폐루프 ───────────────────────────────────────────────────────
        from bimanual_obs import left_actor_obs  # noqa: PLC0415
        q0_left = np.array([float(dq[0, robot.joint_names.index(n)]) for n in LEFT9])
        gate("① 좌팔(v2B25) 폐루프 — shaker 파지")
        la = np.zeros(LEFT_ACTS, dtype=np.float32)
        z0 = float(env.left_target_cup.data.root_pos_w[0, 2])
        streak, l_steps = 0, 0
        with torch.inference_mode():
            for step in range(args.left_steps):
                obs = left_actor_obs(env, left, goal7=left_goal, last_action=la,
                                     q_default=q0_left)
                if step < 8:
                    # obs 결백 검사 — 초기 스텝의 세그먼트별 편차. 접촉 전 발산의 근원을
                    # 짚는다(어긋난 세그먼트 = 조립 버그 또는 물리 부정합).
                    from left_obs_builder import SEGMENTS as _LSEG  # noqa: PLC0415
                    _rec0, _i, _tops = zl["obs"][step], 0, []
                    for _nm, _dd in _LSEG:
                        _dm = float(np.abs(obs[_i:_i + _dd] - _rec0[_i:_i + _dd]).max())
                        if _dm > (0.02 if step == 0 else 0.05):
                            _tops.append(f"{_nm} {_dm:.3f}")
                        _i += _dd
                    if _tops:
                        print(f"  [obs{step} Δ] " + " · ".join(_tops), flush=True)
                act = player_l.get_action(T(obs, (1, -1)), is_deterministic=True)
                act = act.reshape(1, LEFT_ACTS)   # raw — 텀이 내부 클램프(학습 동형)
                # ★fabric 관절목표를 물리스텝마다 송출 (action→fabric→PD 체인 유지)
                trace_box["acts"].append(act[0].detach().cpu().numpy().tolist())
                left.step_policy(act, render=args.gui,
                                 on_substep=lambda: (stream_meas(), _trace_substep()))
                pace_shadow(s3_l_arm, 0)      # 실기가 따라올 때까지 sim 정지
                watch()
                shadow_guard()
                if step % 10 == 9 and check_abort():
                    break
                la = act[0].detach().cpu().numpy()
                trace["l_palm"].append(
                    left.arm.processed_actions[0].detach().cpu().numpy().copy())
                trace["l_armt"].append(robot.data.joint_pos_target[
                    0, left.arm._arm_joint_ids].detach().cpu().numpy().copy())
                l_steps = step + 1
                if step % 20 == 0 or step < 24 and step % 2 == 0:
                    sp = (env.left_target_cup.data.root_pos_w[0]
                          - scene.env_origins[0]).cpu().numpy()
                    print(f"  [좌{step:3d}] gate {int(left.gate_open[0])} · shaker "
                          f"({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f}) · palm_cmd "
                          f"{np.round(trace['l_palm'][-1][:3], 3).tolist()}", flush=True)
                if step % args.render_every == 0:
                    shot("1left")
                dz = float(env.left_target_cup.data.root_pos_w[0, 2]) - z0
                streak = streak + 1 if dz > args.stop_lift else 0
                if streak >= args.lift_hold:
                    print(f"[좌] step {step}: Δz {dz:+.3f} 를 {args.lift_hold}스텝 유지 — 파지 성립")
                    break
            left.freeze_targets()

            gate("② 좌팔 유지")
            passive("2holdL", args.hold)
        mode_box[0] = "idle"
        pace_report("좌", l_steps)
        trace_box["on"] = False
        _trace_dump("left")
        return l_steps

    def do_right() -> int:
        mode_box[0] = "policy"
        guard_box[0], guard_box[1] = 0, 0.0
        _policy_start_check("right")
        trace_box.update(on=True, side="r", rows=[], acts=[])
        with torch.inference_mode():
            # ── 물리 dt 전환 100 → 120 Hz (우 국면은 E1 학습 주기) ───────────────
            rdt = float(right_cfg.sim.dt)
            env.sim.set_simulation_dt(physics_dt=rdt)
            dt_box[0] = rdt
            right.physics_dt = rdt
            print(f"[dt] 물리 {1/rdt:.0f} Hz 로 전환 (우 국면)")

            # ── ③ 우팔 폐루프 ───────────────────────────────────────────────────
            gate("③ 우팔(E1) 폐루프 — cup_big_s100 파지")
            if player_r.is_rnn:
                player_r.init_rnn()
            obs_r = right.observe()
            z0r = float(env.cup.data.root_pos_w[0, 2])
            streak, r_steps = 0, 0
            for step in range(args.right_steps):
                act = player_r.get_action(obs_r.reshape(1, -1), is_deterministic=True)
                act = act.reshape(1, -1)          # raw — _pre_physics_step 이 클램프
                # ★좌와 동일 규약 — fabric 관절목표를 물리스텝마다 송출
                trace_box["acts"].append(act[0].detach().cpu().numpy().tolist())
                obs_r = right.step_policy(
                    act, render=args.gui,
                    on_substep=lambda: (stream_meas(), _trace_substep()))
                pace_shadow(s3_r_arm, 7)
                watch()
                shadow_guard()
                if step % 10 == 9 and check_abort():
                    break
                trace["r_palm"].append(right.palm_targets[0].detach().cpu().numpy().copy())
                trace["r_armt"].append(robot.data.joint_pos_target[
                    0, right.arm_ids].detach().cpu().numpy().copy())
                trace["r_latch"].append(bool(right._latched[0]))
                r_steps = step + 1
                if step % args.render_every == 0:
                    shot("3right")
                dz = float(env.cup.data.root_pos_w[0, 2]) - z0r
                streak = streak + 1 if dz > args.stop_lift else 0
                if streak >= args.lift_hold:
                    print(f"[우] step {step}: Δz {dz:+.3f} 를 {args.lift_hold}스텝 유지 — 파지 성립")
                    break
            right.freeze_targets()
        mode_box[0] = "idle"
        pace_report("우", r_steps)
        trace_box["on"] = False
        _trace_dump("right")
        return r_steps

    # ── 차렷↔preset 전환 (검증 bag 재생 + 브리지 램프) ──────────────────────
    # Phase0 실측: 우 reset_right_v2 는 g1 홈과 0.0° 정합. 좌 reset_left 끝은
    # v2B25 홈과 j4 21°·j7 28.6° 어긋남 → 브리지 램프로 잇는다. 전 bag 최대
    # 속도 0.25 rad/s — 브리지도 같은 상한(0.499 rad / 2.5s = 0.2 rad/s).
    _bags: dict[str, dict] = {}
    BRIDGE_FRAMES = 250            # stride 나눗셈 후에도 ~1.7초 (0.3 rad/s)
    RELEASE_FRAMES = 300           # 손 램프 — stride 후 ~2초
    l_arm_ids7 = [robot.joint_names.index(f"l_aj_{i}") for i in range(1, 8)]
    l_home7 = [float(dq[0, i]) for i in l_arm_ids7]
    r_arm_ids7 = [robot.joint_names.index(f"r_aj_{i}") for i in range(1, 8)]
    r_home7 = [float(dq[0, i]) for i in r_arm_ids7]

    def att_hand(fallback) -> list:
        return att_hand_box[0] if att_hand_box[0] is not None else list(fallback)

    def load_bag(name: str) -> dict:
        if name not in _bags:
            d = np.load(args.bag_dir / f"{name}.npz", allow_pickle=True)
            _bags[name] = dict(
                arm=d["arm_target"].reshape(len(d["arm_target"]), -1),
                grip=d["grip_cmd"].reshape(len(d["grip_cmd"]), -1),
                dt=float(d["meta_step_dt"][0]),
                arm_ids=[robot.joint_names.index(str(n))
                         for n in d["meta_joint_names"]],
                grip_ids=[robot.joint_names.index(str(n))
                          for n in d["meta_grip_names"]])
        return _bags[name]

    def _transit_frame(pairs, spf: int) -> None:
        tq = robot.data.joint_pos_target.clone()
        for ids, vec in pairs:
            for k, v in zip(ids, vec):
                tq[0, k] = float(v)
        robot.set_joint_position_target(tq)
        for _k in range(spf):
            scene.write_data_to_sim()
            env.sim.step(render=args.gui and _k == spf - 1)
            scene.update(dt_box[0])
        stream_meas()
        watch()
        shadow_guard()

    def play_bag(name: str, label: str, *, grip: bool = True) -> bool:
        b = load_bag(name)
        spf = max(1, round(b["dt"] / dt_box[0]))
        stride = max(1, int(args.transit_stride))
        print(f"[transit] {label} — {name} {len(b['arm'])}f "
              f"(물리 {spf}스텝/프레임 · {stride}배속 · 손 {'재생' if grip else '고정'})",
              flush=True)
        with torch.inference_mode():
            for f in range(0, len(b["arm"]), stride):
                pairs = [(b["arm_ids"], b["arm"][f])]
                if grip:
                    pairs.append((b["grip_ids"], b["grip"][f]))
                _transit_frame(pairs, spf)
                if (f // stride) % 25 == 24 and check_abort():
                    return False
        return True

    def ramp_to(ids, goal, frames: int, label: str, settle: int = 0) -> bool:
        start = robot.data.joint_pos_target[0, ids].clone()
        g = torch.tensor([float(v) for v in goal], device=env.device)
        spf = max(1, round(0.02 / dt_box[0]))
        frames = max(1, frames // max(1, int(args.transit_stride)))
        print(f"[transit] {label} — 램프 {frames}f"
              + (f" +정착 {settle}f" if settle else ""), flush=True)
        with torch.inference_mode():
            for f in range(frames):
                a = (f + 1) / frames
                _transit_frame([(ids, (start * (1 - a) + g * a).tolist())], spf)
                if f % 25 == 24 and check_abort():
                    return False
            # 정착 — 목표 유지하며 PD 수렴 대기 (3배속 램프는 목표에 못 닿고 끝난다,
            # 09.02 실측: 브리지 끝 좌 j7 11.5° 미달)
            gl = g.tolist()
            for f in range(settle):
                _transit_frame([(ids, gl)], spf)
                if f % 25 == 24 and check_abort():
                    return False
        return True

    _stiff_save: dict = {}

    def _hand_stiff(hi: bool) -> None:
        """전환 중 sim 팔·손 강성 상향 — 두 이유. (손) 기본 kp5 는 3배속 하강 중
        주먹을 못 쥐어 손가락이 상판 관통(thumb_3 16 rad/s). (팔) HDGP_S2R_REAL_GAINS
        의 약한 r2s 게인은 브리지 램프 목표를 못 따라가 홈에 j7 11.5° 미달 —
        그 미달한 sim 실측이 stream 으로 실기에 가서 실기도 미달했다(09.02 확정).
        전환 중에만 세게, 정책 전 원복(정책은 s2r 게인이어야 학습 동형)."""
        import os as _os  # noqa: PLC0415
        arm_ids = s3_l_arm + s3_r_arm
        if hi and _os.environ.get("HDGP_TRANSIT_ARM_STIFF", "1") == "1":
            if "arm_k" not in _stiff_save:
                _stiff_save["arm_k"] = robot.data.joint_stiffness[0, arm_ids].clone()
                _stiff_save["arm_d"] = robot.data.joint_damping[0, arm_ids].clone()
            ak = torch.full((1, len(arm_ids)), 400.0, device=env.device)
            robot.write_joint_stiffness_to_sim(ak, joint_ids=arm_ids)
            robot.write_joint_damping_to_sim(torch.full_like(ak, 40.0), joint_ids=arm_ids)
        elif "arm_k" in _stiff_save:
            robot.write_joint_stiffness_to_sim(
                _stiff_save["arm_k"].unsqueeze(0), joint_ids=arm_ids)
            robot.write_joint_damping_to_sim(
                _stiff_save["arm_d"].unsqueeze(0), joint_ids=arm_ids)
        k = 50.0 if hi else 5.0
        dmp = 2.5 if hi else 2.0
        kt = torch.full((1, len(s3_r_hand)), k, device=env.device)
        robot.write_joint_stiffness_to_sim(kt, joint_ids=s3_r_hand)
        robot.write_joint_damping_to_sim(torch.full_like(kt, dmp),
                                         joint_ids=s3_r_hand)

    def _transit_dt() -> None:
        # bag 은 50Hz(0.02) — 좌 물리 dt(100Hz) 기준으로 복원해 재생 주기를 맞춘다
        ldt = float(left_cfg.sim.dt)
        env.sim.set_simulation_dt(physics_dt=ldt)
        dt_box[0] = ldt
        right.physics_dt = ldt

    def do_preset(side: str = "both") -> bool:
        """차렷 → preset. 우팔 먼저(강제 규약) → 좌팔 → 좌 브리지(bag 끝→v2B25 홈).

        bag 프레임0 의 팔 target 은 차렷(0) — 시작자세가 차렷이 아니면 점프가
        생기므로 거부한다. 손 채널은 bag 시작값이 비 0 이라 사전정렬 램프로 잇는다.

        ★side 로 편측 실행(09.02). 좌/우 정책을 따로 돌리므로 쓰지 않는 팔을 preset 에
        세워둘 이유가 없다 — 우 j7 은 preset 자세에서 3.17 N·m(한계 7)를 물어 18분 만에
        과열 고장났다(차렷은 0.03). 대기하는 팔은 차렷에 둔다.
        """
        br0, bl0 = load_bag("reset_right_safe"), load_bag("reset_left_safe")
        do_l, do_r = side in ("both", "left"), side in ("both", "right")
        cur = robot.data.joint_pos_target[0]
        _bs = ([br0] if do_r else []) + ([bl0] if do_l else [])
        d0 = max(abs(float(cur[i]) - float(v))
                 for b in _bs for i, v in zip(b["arm_ids"], b["arm"][0]))
        if d0 > 0.3:
            print(f"[transit] preset({side}) 거부 — 시작자세가 차렷이 아니다 "
                  f"(|Δ|max {d0:.2f} rad). attention 으로 먼저 복귀하라", flush=True)
            return False
        mode_box[0] = "transit"
        guard_box[0], guard_box[1] = 0, 0.0
        _transit_dt()
        _hand_stiff(True)
        # ★전환 중 손은 주먹(안전형) 고정 — 3배속에선 무른 sim 손(kp5)이 bag 손
        #   궤적을 못 따라가 팔이 내려간 시점에 덜 접힌 손이 바닥/상판을 관통했다
        #   (attention 중 thumb_3 16 rad/s 폭발, 09.02 재현 2회). 손 형상 변경은
        #   팔이 위(도착 후)에 있을 때만 한다.
        # 순서는 both 검증본과 동일하게 유지하고, 해당 팔의 단계만 건너뛴다.
        steps = (
            [(do_r, lambda: ramp_to(br0["grip_ids"], att_hand(br0["grip"][0]),
                                    RELEASE_FRAMES, "우손 어텐션 손자세 정렬")),
             (do_l, lambda: ramp_to(bl0["grip_ids"], bl0["grip"][0], RELEASE_FRAMES,
                                    "좌 그립 정렬")),
             (do_r, lambda: play_bag("reset_right_safe", "우 차렷→preset(safe·j2상승)",
                                     grip=False)),
             (do_l, lambda: play_bag("reset_left_safe", "좌 차렷→preset(safe·j4창)",
                                     grip=False)),
             (do_l, lambda: ramp_to(l_arm_ids7, l_home7, BRIDGE_FRAMES,
                                    "좌 브리지 bag끝→홈", settle=60)),
             (do_r, lambda: ramp_to(r_arm_ids7, r_home7, BRIDGE_FRAMES,
                                    "우 브리지 safe끝(j2↑)→홈", settle=60)),
             (do_r, lambda: ramp_to(br0["grip_ids"], br0["grip"][-1], RELEASE_FRAMES,
                                    "우손 preset 손자세")),
             (do_l, lambda: ramp_to(bl0["grip_ids"], bl0["grip"][-1], RELEASE_FRAMES,
                                    "좌 그립 preset"))])
        ok = True
        for want, fn in steps:
            if want and ok:
                ok = fn()
        _hand_stiff(False)
        mode_box[0] = "idle"
        print(f"[transit] preset({side}) {'완료' if ok else '중단'} · 최대 괴리 "
              f"{guard_box[1]:.3f} rad · {cups_z()}", flush=True)
        for _nm, _want, _ids, _home, _off in (("좌팔", do_l, l_arm_ids7, l_home7, 0),
                                              ("우팔", do_r, r_arm_ids7, r_home7, 7)):
            if _want:
                _preset_check(_nm, _ids, _home, _off)
        return ok

    def _preset_check(nm: str, ids, home, echo_off: int) -> None:
        """도달 자세를 sim·실기 두 기준으로 대조 — 한쪽만 보면 속는다.

        09.02: sim 0.4° 인데 실기는 j7 4.2° 처져 테이블을 긁었다(손목 kp10 + 중력보상
        없음). 게다가 미러(joint_states_to_udp)가 죽어 그 사실이 화면에 없었다.
        그래서 sim−홈·target−홈·실기−홈 세 줄을 **항상 같이** 찍는다.
        """
        _q = robot.data.joint_pos[0, ids]
        _t = robot.data.joint_pos_target[0, ids]
        _deg = [round(float(np.degrees(float(_q[i]) - home[i])), 1) for i in range(7)]
        _terr = [round(float(np.degrees(float(_t[i]) - home[i])), 1) for i in range(7)]
        _pt = [round(float(np.degrees(float(_q[i]) - float(_t[i]))), 1) for i in range(7)]
        print(f"[preset검증2:{nm}] target−홈(deg) {_terr} · pos−target(deg) {_pt}",
              flush=True)
        print(f"[preset검증:{nm}] sim 도달−홈(deg) {_deg} · |max| "
              f"{max(abs(d) for d in _deg):.1f}° (0 이어야 정책 시작자세)", flush=True)
        echo_drain()
        _m = echo_latest[0]
        if _m is None:
            print(f"[preset검증3:{nm}] 실측無 — joint_states_to_udp 가 죽었다"
                  " (미러·괴리감시 모두 무동작)", flush=True)
            return
        _rd = [round(float(np.degrees(float(_m[echo_off + i]) - home[i])), 1)
               if not _s3math.isnan(_m[echo_off + i]) else float("nan") for i in range(7)]
        _fin = [abs(d) for d in _rd if not _s3math.isnan(d)]
        print(f"[preset검증3:{nm}] 실기−홈(deg) {_rd} · |max| "
              f"{max(_fin) if _fin else float('nan'):.1f}° (≤1.5° 합격)", flush=True)

    def do_home(side: str = "left") -> bool:
        """정책 후 자세 → 정책 홈 **직접 램프** (차렷 경유 없이 재시도).

        정책이 끝난 팔은 홈에서 수십 도 떨어진 임의 자세다. 차렷까지 갔다가 preset 을
        다시 밟으면 1분이 걸리는데, 두 자세가 가까우면 관절 직선보간이 훨씬 싸다.
        ★경로 안전은 자동으로 보장되지 않는다 — 관절 직선보간이 TCP 를 판 쪽으로
        스치게 할 수 있다. 09.02 실측에선 정책 직후 자세(홈 대비 |max| 34.5°)에서
        경로 전체 TCP 가 판 위 80~95 mm 였다. 크게 벗어난 자세면 attention 을 써라.
        """
        do_l, do_r = side in ("both", "left"), side in ("both", "right")
        bl0, br0 = load_bag("reset_left_safe"), load_bag("reset_right_safe")
        mode_box[0] = "transit"
        guard_box[0], guard_box[1] = 0, 0.0
        _transit_dt()
        _hand_stiff(True)
        steps = (
            [(do_l, lambda: ramp_to(l_arm_ids7, l_home7, BRIDGE_FRAMES * 2,
                                    "좌 현자세→홈 직접램프", settle=60)),
             (do_r, lambda: ramp_to(r_arm_ids7, r_home7, BRIDGE_FRAMES * 2,
                                    "우 현자세→홈 직접램프", settle=60)),
             (do_l, lambda: ramp_to(bl0["grip_ids"], bl0["grip"][-1], RELEASE_FRAMES,
                                    "좌 그립 preset")),
             (do_r, lambda: ramp_to(br0["grip_ids"], br0["grip"][-1], RELEASE_FRAMES,
                                    "우손 preset 손자세"))])
        ok = True
        for want, fn in steps:
            if want and ok:
                ok = fn()
        _hand_stiff(False)
        mode_box[0] = "idle"
        print(f"[transit] home({side}) {'완료' if ok else '중단'} · 최대 괴리 "
              f"{guard_box[1]:.3f} rad · {cups_z()}", flush=True)
        for _nm, _want, _ids, _home, _off in (("좌팔", do_l, l_arm_ids7, l_home7, 0),
                                              ("우팔", do_r, r_arm_ids7, r_home7, 7)):
            if _want:
                _preset_check(_nm, _ids, _home, _off)
        return ok

    def do_droop(side: str = "left") -> bool:
        """★sim 의 **정적 처짐** 측정 — 중력보상이 실기에 필요한지 판별하는 유일한 실험.

        v2E29 는 중력 ON + 벤더게인(kp10 손목)으로 학습했다. 그러면 sim 팔도 목표에
        못 닿고 처진 채 학습된 것이고, 같은 게인·같은 중력의 실기도 **같은 만큼** 처져야
        한다 — 그렇다면 중력보상은 불필요하고 fabric 목표를 그대로 보내는 게 정답이다.
        판별식: sim 정착 `pos−target` vs 실기 무보상 처짐(09.02 실측 j7 4.2°, 모델 4.36°).
        같으면 목표 스트림 + 무보상, 다르면 sim 자산 질량/관성이 실기와 다르다는 뜻.

        전환용 강성(kp400)을 **쓰지 않고** s2r 게인 그대로 홈 목표를 걸어 정착시킨다.
        스트림을 내지 않으므로(mode=idle) 실기는 움직이지 않는다.
        """
        ids = l_arm_ids7 if side == "left" else r_arm_ids7
        home = l_home7 if side == "left" else r_home7
        nm = "좌팔" if side == "left" else "우팔"
        print(f"[droop] {nm} — s2r 게인 그대로 홈 목표 유지 {args.settle*3}스텝 정착", flush=True)
        with torch.inference_mode():
            tq = robot.data.joint_pos_target.clone()
            for k, v in zip(ids, home):
                tq[0, k] = float(v)
            robot.set_joint_position_target(tq)
            for _ in range(int(args.settle) * 3):
                scene.write_data_to_sim()
                env.sim.step(render=args.gui)
                scene.update(dt_box[0])
            q = robot.data.joint_pos[0, ids]
            t = robot.data.joint_pos_target[0, ids]
            k_ = robot.data.joint_stiffness[0, ids]
            d_ = np.degrees([float(q[i]) - float(t[i]) for i in range(7)])
        print(f"[droop:{nm}] pos−target(deg) {np.round(d_, 2).tolist()} · |max| "
              f"{np.abs(d_).max():.2f}°", flush=True)
        print(f"[droop:{nm}] 적용 중인 kp {np.round(k_.tolist(), 1).tolist()}", flush=True)
        print(f"[droop:{nm}] ★실기 무보상 처짐(09.02) j7 4.2° / 모델 4.36° 와 비교하라",
              flush=True)
        return True

    def do_mass() -> bool:
        """★sim 이 실제로 쓰는 링크 질량을 PhysX 에서 직접 읽어 URDF 와 대조.

        `mass_props: null` 이라 sim 은 USD 에 적힌 값을 그대로 쓴다. 중력보상·기구학
        모델은 `urdf/generated/rl/*.urdf` 를 쓰므로 둘이 다르면 정적 처짐이 달라진다
        (09.02: sim j7 11.07° vs 실기 4.2°). 어디서 갈렸는지 이걸로 확정한다.
        """
        import xml.etree.ElementTree as _ET  # noqa: PLC0415
        try:
            m = robot.root_physx_view.get_masses()[0].cpu().numpy()
        except Exception as e:                # noqa: BLE001
            print(f"[mass] PhysX 질량 읽기 실패: {e}", flush=True)
            return False
        names = robot.body_names
        _r = _ET.parse("/home/user/rl_ws/urdf/generated/rl/"
                       "openarm_tesollo_sensor_rl.urdf").getroot()
        u = {lk.get("name"): float(lk.find("inertial").find("mass").get("value"))
             for lk in _r.findall("link") if lk.find("inertial") is not None}
        print(f"{'링크':<26}{'sim(USD)':>10}{'URDF':>10}{'차이':>10}", flush=True)
        # ★URDF 에 <inertial> 이 없는 링크(순수 좌표 프레임)는 Isaac 임포트에서
        #   PhysX 기본 **1.0 kg** 이 붙는다 — 09.02 좌 `l_hl_gripper_tcp` 가 그 사례다.
        #   손끝에 붙으면 중력 모멘트를 통째로 바꿔 sim 처짐이 실기의 2.6배가 된다.
        ghost = []
        for pre, lbl in (("l_", "좌"), ("r_", "우")):
            tot_s = tot_u = 0.0
            for i, n in enumerate(names):
                if not n.startswith(pre):
                    continue
                uu = u.get(n)
                tot_s += float(m[i])
                tot_u += uu or 0.0
                mark = ""
                if uu is None:
                    ghost.append((n, float(m[i])))
                    mark = "  ← URDF 무질량(유령)"
                elif abs(float(m[i]) - uu) > 1e-3:
                    mark = "  ← 불일치"
                print(f"{n:<26}{float(m[i]):10.3f}"
                      f"{(uu if uu is not None else float('nan')):10.3f}"
                      f"{(float(m[i]) - uu if uu is not None else float('nan')):10.3f}"
                      f"{mark}", flush=True)
            print(f"{lbl + '팔 합계':<26}{tot_s:10.3f}{tot_u:10.3f}{tot_s-tot_u:10.3f}",
                  flush=True)
        print(f"[mass] ★유령 질량 {len(ghost)}개 · 합 "
              f"{sum(g[1] for g in ghost):.3f} kg — {[g[0] for g in ghost]}", flush=True)
        return True

    def do_attention(side: str = "both") -> bool:
        """preset(파지 후 포함) → 차렷. 손 릴리즈 → 팔 사전정렬 램프 → 좌·우 복귀.

        파지 후 팔은 정책이 움직인 임의 자세 — reverse bag 프레임0(홈)과 다르므로
        램프로 먼저 잇는다(ramp_to 는 현재 target 에서 출발해 점프가 없다).
        ★side 로 편측 실행 — 근거는 do_preset 주석(우 j7 과열).
        """
        do_l, do_r = side in ("both", "left"), side in ("both", "right")
        mode_box[0] = "transit"
        guard_box[0], guard_box[1] = 0, 0.0
        _transit_dt()
        _hand_stiff(True)
        bl = load_bag("reset_left_safe_reverse")
        br = load_bag("reset_right_safe_reverse")
        # 릴리즈는 2단: ①펴기(reverse[0]=preset 손 — 컵을 놓는다) ②주먹(안전형,
        #   =reverse[-1]) — 그 뒤 팔 이동은 손 고정(주먹)으로 관통 원천 차단.
        steps = (
            [(do_l, lambda: ramp_to(bl["grip_ids"], bl["grip"][0], RELEASE_FRAMES,
                                    "좌 그립 릴리즈")),
             (do_r, lambda: ramp_to(br["grip_ids"], br["grip"][0], RELEASE_FRAMES,
                                    "우손 릴리즈(펴기)")),
             (do_r, lambda: ramp_to(br["grip_ids"], att_hand(br["grip"][-1]),
                                    RELEASE_FRAMES, "우손 어텐션 손자세")),
             (do_l, lambda: ramp_to(l_arm_ids7, bl["arm"][0], BRIDGE_FRAMES,
                                    "좌 사전정렬 →bag시작")),
             (do_r, lambda: ramp_to(br["arm_ids"], br["arm"][0], BRIDGE_FRAMES,
                                    "우 사전정렬 →bag시작")),
             (do_l, lambda: play_bag("reset_left_safe_reverse", "좌 preset→차렷(safe)",
                                     grip=False)),
             (do_r, lambda: play_bag("reset_right_safe_reverse", "우 preset→차렷(safe)",
                                     grip=False))])
        ok = True
        for want, fn in steps:
            if want and ok:
                ok = fn()
        _hand_stiff(False)
        mode_box[0] = "idle"
        print(f"[transit] attention({side}) {'완료' if ok else '중단'} · 최대 괴리 "
              f"{guard_box[1]:.3f} rad", flush=True)
        return ok

    def console_reset() -> None:
        nonlocal left_spawn, right_spawn, right_goal, settle_lz, settle_rz
        from cup_pose_capture import load_capture  # noqa: PLC0415
        if args.left_cup_json is not None and args.left_cup_json.exists():
            left_spawn = np.array(load_capture(
                args.left_cup_json, expect_frame="base_link").position, dtype=np.float32)
            left_spawn[2] = args.table_z + 0.102
        if args.right_cup_json is not None and args.right_cup_json.exists():
            right_spawn = np.array(load_capture(
                args.right_cup_json, expect_frame="base_link").position, dtype=np.float32)
            right_spawn[2] = args.table_z + 0.076
            right_goal = right_spawn + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        ldt = float(left_cfg.sim.dt)
        env.sim.set_simulation_dt(physics_dt=ldt)
        dt_box[0] = ldt
        right.physics_dt = ldt
        if not args.real:
            place(env.cup, right_spawn)
            place(env.left_target_cup, left_spawn)
            robot.write_joint_state_to_sim(dq.clone(), torch.zeros_like(dq))
            robot.set_joint_position_target(dq.clone())
            right._init_task_state()
        else:
            # 실기 — 로봇은 echo 소유: 텔레포트 금지. _init_task_state 의 preset
            # 잔상(2스텝)은 즉시 현 자세 복원으로 지운다(유휴라 스트림 무송출 —
            # 실기 무영향). 컵은 로봇 복원 **후** 재소환해야 잔상 접촉이 없다.
            q_keep = robot.data.joint_pos.clone()
            right._init_task_state()
            robot.write_joint_state_to_sim(q_keep, torch.zeros_like(q_keep))
            robot.set_joint_position_target(q_keep)
            place(env.cup, right_spawn)
            place(env.left_target_cup, left_spawn)
        right.zero_obs_noise()
        right.goal_pos[:] = T(right_goal).unsqueeze(0)
        right.episode_length_buf.zero_()
        left.reset()
        with torch.inference_mode():
            passive("reset_settle", args.settle)
        settle_rz = float(env.cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        settle_lz = float(env.left_target_cup.data.root_pos_w[0, 2]
                          - scene.env_origins[0, 2])
        right.object_spawn_pos[:] = (env.cup.data.root_pos_w - scene.env_origins)[0:1]
        print(f"[console] reset 완료 — 스폰 좌 {np.round(left_spawn, 3).tolist()} · "
              f"우 {np.round(right_spawn, 3).tolist()} · {cups_z()}", flush=True)

    def _judge(nm: str, obj, base_z: float, steps_n: int) -> None:
        z = float(obj.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        ok = z > base_z + 0.05
        print(f"[판정:{nm}] {'✅' if ok else '❌'} z {z:.3f} "
              f"(정착 {base_z:.3f} · {steps_n}스텝)", flush=True)

    def _console_loop() -> None:
        cmd_path = args.console
        cmd_path.parent.mkdir(parents=True, exist_ok=True)
        cmd_path.write_text("")
        lf_objs = {0: env.cup, 1: env.left_target_cup}
        lf_armed = [True]          # 정책 시작 후엔 잠금 — 들린 컵을 끌어내리지 않게

        def lf_poll() -> None:
            """유휴 중에만 물체를 옮긴다 — 수신 자체는 공용 lf_drain 이 한다."""
            lf_drain()
            if not lf_armed[0]:
                return
            for _sd, _xy in lf_live.items():
                if _sd not in lf_objs:
                    continue
                _z = settle_rz if _sd == 0 else settle_lz
                place(lf_objs[_sd], (_xy[0], _xy[1], _z))

        def _dispatch(cmd: str) -> None:
            parts = cmd.replace("_", " ").split()
            base = parts[0] if parts else ""
            arg = parts[1] if len(parts) > 1 else "both"
            if cmd == "reset":
                console_reset()
                lf_live.clear()
                lf_armed[0] = True
            elif base == "mass":
                do_mass()
            elif base == "droop":
                do_droop("left" if arg == "both" else arg)
            elif base == "home":
                # 정책 후 → 홈 직접 램프(차렷 경유 생략). 인자 없으면 좌팔.
                do_home("left" if arg == "both" else arg)
            elif base in ("preset", "attention"):
                # 편측 실행: "preset left" / "preset_left" / "attention right" …
                # 쓰지 않는 팔은 차렷에 둔다 — 우 j7 은 preset 자세에서 3.17 N·m 를
                # 물어 18분 만에 과열 고장났다(차렷 0.03). 09.02 bag 실측.
                if arg not in ("both", "left", "right"):
                    print(f"[console] preset/attention 인자는 left|right|both — 받은 "
                          f"{arg!r}", flush=True)
                elif base == "preset":
                    do_preset(arg)
                else:
                    do_attention(arg)
            elif cmd == "left":
                lf_armed[0] = False
                _judge("좌", env.left_target_cup, settle_lz, do_left())
            elif cmd == "right":
                lf_armed[0] = False
                _judge("우", env.cup, settle_rz, do_right())
            elif cmd == "abort":
                print("[console] 유휴 상태 — abort 는 전환·정책 중에만 유효", flush=True)
            elif cmd == "status":
                print(f"[console] mode {mode_box[0]} · {cups_z()} · "
                      f"dt {dt_box[0]:.4f} · |q̇|max {worst['qd']:.2f} · "
                      f"echo {'수신' if echo_latest[0] is not None else '없음'} · "
                      f"괴리max {guard_box[1]:.3f}", flush=True)
                m = echo_latest[0]
                if m is not None:
                    q = robot.data.joint_pos[0]
                    segs = (("좌팔", s3_l_arm, m[0:7]), ("우팔", s3_r_arm, m[7:14]),
                            ("우손", s3_r_hand, m[15:35]))
                    for nm, ids, seg in segs:
                        ds = [abs(float(q[k]) - float(v))
                              for k, v in zip(ids, seg) if not _s3math.isnan(v)]
                        print(f"[미러] {nm} |sim−real| max "
                              f"{max(ds) if ds else float('nan'):.4f} rad "
                              f"(유효 {len(ds)}ch)", flush=True)
            else:
                print(f"[console] 모르는 명령: {cmd}", flush=True)

        print(f"[console] 명령 파일 {cmd_path} — "
              f"reset|preset[ left|right]|home[ left|right]|droop[ left|right]|"
              f"mass|left|right|attention[ left|right]|"
              f"abort|status|quit", flush=True)
        while simulation_app.is_running():
            with torch.inference_mode():
                for _ in range(60):
                    echo_apply()
                    lf_poll()
                    scene.write_data_to_sim()
                    env.sim.step(render=args.gui)
                    scene.update(dt_box[0])
            try:
                cmd = cmd_path.read_text().strip()
            except OSError:
                cmd = ""
            if not cmd:
                continue
            cmd_path.write_text("")
            print(f"[console] ← {cmd}", flush=True)
            try:
                if cmd == "quit":
                    break
                with torch.inference_mode():
                    _dispatch(cmd)
            except RuntimeError as e:
                mode_box[0] = "idle"
                _hand_stiff(False)
                print(f"[console] ⚠ {e} — reset 으로 복구하라", flush=True)

    if args.console is not None:
        args.auto = True
        _console_loop()
        return 0

    l_steps = do_left()
    r_steps = do_right()
    with torch.inference_mode():
        gate("④ 양팔 유지 — 두 컵")
        passive("4final", args.final_hold)
        # 파지 판정은 여기서 캐시 — pour 가 컵을 내리면 최종값으로는 못 잰다
        grasp_lz = float(env.left_target_cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        grasp_rz = float(env.cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])

        pour_report = None
        if not args.skip_pour:
            # ── ⑤ 좌팔 전환 — pour 받는 자세(학습 REST)로, 쥔 채 관절 램프 ──────
            gate("⑤ 좌팔 전환 — 받는 자세로 (쥔 채)")
            rest7 = env.left_arm_zero_pos[0, :7].clone()      # 학습 상수 (덮어쓰기 전)
            lids = list(env.left_arm_dof_indices[:7])
            cur = robot.data.joint_pos_target[0, lids].clone()
            # ★2단 램프 — 한 번에 lerp 하면 경로가 낮게 스쳐 컵 바닥이 테이블면 아래로
            #   ~16mm 파고든다(09.02 사용자 관찰 + z 추적 실측). 1단: j1(베이스 요)만
            #   돌려 수평 이동 → 2단: 나머지 6관절로 자세 조정(목표 지점 위에서 수직).
            via = cur.clone()
            via[0] = rest7[0]
            steps = 0
            for seg_from, seg_to in ((cur, via), (via, rest7)):
                seg_n = max(int(float((seg_to - seg_from).abs().max())
                               / (args.carry_vel * dt_box[0])), 1)
                for f in range(seg_n):
                    a = (f + 1) / seg_n
                    robot.set_joint_position_target(
                        (seg_from * (1 - a) + seg_to * a).unsqueeze(0), joint_ids=lids)
                    scene.write_data_to_sim()
                    env.sim.step(render=args.gui)
                    scene.update(dt_box[0])
                    watch()
                    if steps % 40 == 0:
                        _sz = (env.left_target_cup.data.root_pos_w[0]
                               - scene.env_origins[0]).cpu().numpy()
                        print(f"  [전환{steps:3d}] shaker ({_sz[0]:.3f},{_sz[1]:.3f},"
                              f"{_sz[2]:.3f})", flush=True)
                    if steps % (args.render_every * 3) == 0:
                        shot("5carry")
                    steps += 1
            for f in range(90):                                # 정착
                scene.write_data_to_sim()
                env.sim.step(render=args.gui)
                scene.update(dt_box[0])
                watch()
                if f % args.render_every == 0:
                    shot("5carry")
            # 받는점 잔차 — **서보는 3방식 모두 발산해 폐기**(09.02: fabric v1 euler
            #   클램프 / v2 palm 프레임 불일치 / v3 자코비안 노이즈 지배). 대신 pour
            #   궤적을 실측 받는점 기준으로 재추출한다(probe_pour_native_check
            #   --receiver-pos) — 궤적이 실컵을 노리므로 좌팔은 REST 도착이면 충분하다.
            _ref_src = pour_cfg.left_target_cup_pos_env_local
            if args.pour_mode == "follow" and args.pour_traj.exists():
                _zt0 = np.load(args.pour_traj)
                if "meta_receiver" in _zt0:
                    _ref_src = tuple(float(v) for v in _zt0["meta_receiver"])
            ref = T(_ref_src)
            lz2 = float(env.left_target_cup.data.root_pos_w[0, 2]
                        - scene.env_origins[0, 2])
            res = float((ref - (env.left_target_cup.data.root_pos_w[0]
                                - scene.env_origins[0])).norm())
            print(f"[전환] {steps}+90스텝 · shaker z {lz2:.3f} · 받는점 잔차 "
                  f"{res * 1000:.0f}mm {'✅' if res < 0.05 and lz2 > 0.28 else '⚠'}",
                  flush=True)

            # ── ⑤′ 우팔 FK 세팅 램프 — pour-sensor 세팅 자세로 (사용자 지시) ─────
            # ★교차 0 의 판명 원인: E1 이 transfer 목표까지 이송한 진입 palm 이 뱅크
            #   분포 밖(y −2.6cm·z −3.4cm). 뱅크 mean 관절로 쥔 채 램프하면 컵도 뱅크
            #   mean 위치(0.363,−0.159,0.400) 부근으로 따라와 분포 정중앙에서 시작한다.
            rest9_orig = env.left_arm_zero_pos[0].clone()      # 스푸핑 진단용 (덮어쓰기 전)
            gate("⑤′ 우팔 세팅 — pour 진입 자세로 (쥔 채)")
            entry7 = T([float(v) for v in args.pour_entry_joints.split(",")])
            rids = list(right.arm_ids)
            cur_r = robot.data.joint_pos_target[0, rids].clone()
            right.freeze_targets()                     # 속도 FF 잔재 제거
            steps_r = max(int(float((entry7 - cur_r).abs().max())
                              / (args.carry_vel * dt_box[0])), 1)
            for f in range(steps_r):
                a = (f + 1) / steps_r
                robot.set_joint_position_target(
                    (cur_r * (1 - a) + entry7 * a).unsqueeze(0), joint_ids=rids)
                scene.write_data_to_sim()
                env.sim.step(render=args.gui)
                scene.update(dt_box[0])
                watch()
                if f % (args.render_every * 3) == 0:
                    shot("5entry")
            for f in range(90):
                scene.write_data_to_sim()
                env.sim.step(render=args.gui)
                scene.update(dt_box[0])
                watch()
                if f % args.render_every == 0:
                    shot("5entry")
            cpos = (env.cup.data.root_pos_w[0] - scene.env_origins[0]).cpu().numpy()
            print(f"[우세팅] {steps_r}+90스텝 · 컵 ({cpos[0]:.3f},{cpos[1]:.3f},{cpos[2]:.3f})"
                  f" (뱅크 mean 0.363,-0.159,0.400) · "
                  f"{'✅쥔 채' if cpos[2] > 0.33 else '❌이탈'}", flush=True)

            # ── ⑥ pour 초기화 — warm 텔레포트의 라이브 대응 ────────────────────
            gate("⑥ pour 초기화 (구슬 자동 소환 예약)")
            from bimanual_chain import (  # noqa: PLC0415
                disarm_receiver_pin, init_pour_from_live, refresh_receiver_buffer)
            disarm_receiver_pin(env)
            init_pour_from_live(env, right, pour_yaml["robot_cfg"]["actuators"])
            player_p = make_player(pour_agent, args.pour_checkpoint,
                                   int(pour_cfg.observation_space),
                                   int(pour_cfg.action_space))

            if args.pour_mode == "follow":
                # ── ⑦ pour 궤적 추종 — 네이티브 성공 에피소드의 실측 관절을 따라간다
                #   (사용자 지시 09.02). 실측 궤적은 동역학적으로 실현 가능하고,
                #   속도 FF 를 함께 주면 PD 지연이 사라진다(vel_ff 교훈 그대로).
                gate("⑦ pour — 성공 궤적 추종 (우팔)")
                ztr = np.load(args.pour_traj)
                tq = torch.tensor(ztr["arm_q"], dtype=torch.float32, device=env.device)
                tqd = torch.tensor(ztr["arm_qd"], dtype=torch.float32, device=env.device)
                print(f"[추종] 궤적 {tq.shape[0]}스텝 · 원본 교차 "
                      f"{int(ztr['meta_final_cross'])}/20", flush=True)
                rids2 = [robot.joint_names.index(f"r_aj_{i}") for i in range(1, 8)]
                cur2 = robot.data.joint_pos_target[0, rids2].clone()
                st = max(int(float((tq[0] - cur2).abs().max())
                             / (args.carry_vel * dt_box[0])), 1)
                for f in range(st + 60):                      # 궤적 시작점으로 램프+정착
                    a = min((f + 1) / st, 1.0)
                    robot.set_joint_position_target(
                        (cur2 * (1 - a) + tq[0] * a).unsqueeze(0), joint_ids=rids2)
                    scene.write_data_to_sim()
                    env.sim.step(render=args.gui)
                    scene.update(dt_box[0])
                    watch()
                    if f % (args.render_every * 3) == 0:
                        shot("6pour")
                # 구슬 주입 — env 자체 샘플러로 쥔 컵 안에 (원본 리셋과 같은 규약)
                cup_pose_now = torch.cat([env.cup.data.root_pos_w,
                                          env.cup.data.root_quat_w], dim=-1)
                bead_state = env._sample_bead_states_inside_cup(cup_pose_now)
                env.beads.write_object_state_to_sim(
                    bead_state, env_ids=torch.arange(1, device=env.device))
                env._beads_spawned[:] = True
                for f in range(60):                            # 구슬 정착
                    scene.write_data_to_sim()
                    env.sim.step(render=args.gui)
                    scene.update(dt_box[0])
                    watch()
                    if f % args.render_every == 0:
                        shot("6pour")
                print("[추종] 구슬 주입 완료 — 추종 시작", flush=True)
                for i in range(tq.shape[0]):
                    robot.set_joint_position_target(tq[i:i + 1], joint_ids=rids2)
                    robot.set_joint_velocity_target(tqd[i:i + 1], joint_ids=rids2)
                    for _ in range(int(pour_cfg.decimation)):
                        scene.write_data_to_sim()
                        env.sim.step(render=args.gui)
                        scene.update(dt_box[0])
                    refresh_receiver_buffer(env)
                    env._get_rewards()                         # 교차 계수 갱신
                    watch()
                    if i % 30 == 0:
                        _ins = int(env._bead_in_source[0].sum())
                        _int = int(env._bead_in_target[0].sum())
                        _bz = env.beads.data.object_pos_w[0, :, 2]
                        _floor = int((_bz < scene.env_origins[0, 2] + 0.25).sum())
                        print(f"  [추종{i:3d}] 교차 {int(env._bead_cross_count[0])}"
                              f"/{int(env.num_beads)} · 소스안 {_ins} · 받는안 {_int} · "
                              f"바닥 {_floor} · 소스컵 z "
                              f"{float(env.cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2]):.3f}",
                              flush=True)
                    if i % args.render_every == 0:
                        shot("6pour")
                robot.set_joint_velocity_target(
                    torch.zeros(1, 7, device=env.device), joint_ids=rids2)
                # 받는컵 안 구슬 = 직접 계산 (env 카운터는 이 실행 경로에서 stale)
                _rc = env.left_target_cup.data.root_pos_w[0]
                _bp = env.beads.data.object_pos_w[0]
                _inb = int((((_bp[:, :2] - _rc[:2]).norm(dim=-1) < 0.043)
                            & (_bp[:, 2] > _rc[2] - 0.10)
                            & (_bp[:, 2] < _rc[2] + 0.12)).sum())
                pour_report = {"steps": int(tq.shape[0]),
                               "crossed": _inb,
                               "beads": int(env.num_beads),
                               "success": _inb >= 10}
                gate("⑧ 마무리 유지")
                passive("7end", args.hold)
            if args.pour_mode == "policy":
                    # ── ⑦ pour 폐루프 (e1_pour1) ────────────────────────────────────
                    gate("⑦ pour — 우팔이 왼손 shaker 에 붓는다")
                    if player_p.is_rnn:
                        player_p.init_rnn()
                    obs_p = env._get_observations()["policy"]
                    _nat_p = _SR / "logs/shadow/pour_entry/pour_obs0_native.npz"
                    if _nat_p.exists():
                        # obs0 결백 검사 — native 리셋 직후 표본의 min/max 범위 밖 세그먼트를 짚는다
                        _nat = np.load(_nat_p)["obs"]
                        _segs = (("arm_q", 7), ("arm_qd", 7), ("grasp_prog", 5), ("l_q", 9),
                                 ("l_qd", 9), ("pp_to_open", 3), ("pour_axis", 3),
                                 ("src_up", 3), ("tgt_up", 3), ("last_act", 6))
                        _mine = obs_p.reshape(-1).detach().cpu().numpy()
                        _i = 0
                        for _nm, _dd in _segs:
                            _lo = _nat[:, _i:_i + _dd].min(0) - 0.05
                            _hi = _nat[:, _i:_i + _dd].max(0) + 0.05
                            _out = float(np.maximum(_lo - _mine[_i:_i + _dd],
                                                    _mine[_i:_i + _dd] - _hi).max())
                            if _out > 0:
                                print(f"  [pour obs0 밖] {_nm}: 이탈 {_out:.3f} · "
                                      f"내 {np.round(_mine[_i:_i + _dd], 3).tolist()}", flush=True)
                            _i += _dd
                    succ_streak, p_steps = 0, 0
                    tgt_up_const = None
                    if args.receiver_up_contract:
                        # 훈련 받는컵 up = R(cfg quat)·ẑ — cfg 상수에서 직접 파생
                        from isaaclab.utils.math import quat_apply  # noqa: PLC0415
                        _q = T(pour_cfg.left_target_cup_quat_wxyz).unsqueeze(0)
                        tgt_up_const = quat_apply(_q, T([0.0, 0.0, 1.0]).unsqueeze(0))[0]
                        print(f"[pour] tgt_up 계약 상수 고정: "
                              f"{[round(float(v), 3) for v in tgt_up_const]}", flush=True)
                    for step in range(args.pour_steps):
                        if tgt_up_const is not None:
                            obs_p = obs_p.clone()
                            obs_p.view(-1)[46:49] = tgt_up_const
                        if args.diag_spoof_left_obs:
                            obs_p = obs_p.clone()
                            obs_p.view(-1)[19:28] = rest9_orig
                            obs_p.view(-1)[28:37] = 0.0
                        act = player_p.get_action(obs_p.reshape(1, -1), is_deterministic=True)
                        env._pre_physics_step(act.reshape(1, -1))
                        for _ in range(int(pour_cfg.decimation)):
                            env._apply_action()                # 컵 고정핀은 ⑥에서 무장해제됨
                            scene.write_data_to_sim()
                            env.sim.step(render=args.gui)
                            scene.update(dt_box[0])
                        refresh_receiver_buffer(env)           # obs 가 읽는 받는컵 버퍼 = 라이브
                        env.episode_length_buf += 1
                        env._get_rewards()                     # 상태·계수 갱신 (상태쓰기 없음 검증)
                        env._get_dones()
                        obs_p = env._get_observations()["policy"]
                        watch()
                        p_steps = step + 1
                        if step % 30 == 0:
                            _su = env._source_up_axis_w[0]
                            print(f"  [pour{step:3d}] mouth_xy {float(env._mouth_xy_distance[0]):.3f} · "
                                  f"gate {float(env._action_tilt_gate[0]):.2f} · "
                                  f"β {float(env._beta_cmd[0]):.2f} · "
                                  f"src_up_z {float(_su[2]):+.2f} · "
                                  f"교차 {int(env._bead_cross_count[0])}"
                                  f"/{int(env.num_beads)} · 성공 {bool(env.episode_success_buf[0])}"
                                  f" · 소스컵 z {float(env.cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2]):.3f}",
                                  flush=True)
                        if step % args.render_every == 0:
                            shot("6pour")
                        if bool(env.episode_success_buf[0]):
                            succ_streak += 1
                            if succ_streak >= 60:
                                print(f"[pour] step {step}: 성공 판정 60스텝 유지 — 종료", flush=True)
                                break
                        else:
                            succ_streak = 0
                    robot.set_joint_velocity_target(
                        torch.zeros(1, len(env.arm_dof_indices), device=env.device),
                        joint_ids=list(env.arm_dof_indices))
                    pour_report = {
                        "steps": p_steps,
                        "crossed": int(env._bead_cross_count[0]),
                        "beads": int(env.num_beads),
                        "success": bool(env.episode_success_buf[0]),
                    }
                    gate("⑧ 마무리 유지")
                    passive("7end", args.hold)

    # ── 보고 ────────────────────────────────────────────────────────────────
    lz = float(env.left_target_cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    rz = float(env.cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    print(f"\n[결과] 좌 {l_steps}스텝 · 우 {r_steps}스텝 · {cups_z()}")
    print(f"[감시] |q̇|max {worst['qd']:.2f} rad/s · 컵속도 max {worst['cupv']:.2f} m/s "
          f"(임계 12 / 3 — 미러는 여기서 죽었다)")
    # ★정착 z 기준 — JSON 기록 z 는 인지 편향(경량 노드 +3cm)이 섞여 허수 판정을 낸다
    ok_l = grasp_lz > settle_lz + 0.05
    ok_r = grasp_rz > settle_rz + 0.05
    print(f"[판정] 좌 파지 {'✅' if ok_l else '❌'} (파지시 z {grasp_lz:.3f} · 최종 {lz:.3f}) · "
          f"우 파지 {'✅' if ok_r else '❌'} (파지시 z {grasp_rz:.3f} · 최종 {rz:.3f})")
    if pour_report is not None:
        pr = pour_report
        print(f"[판정] pour {'✅' if pr['success'] else '진행중/미성공'} — "
              f"{pr['steps']}스텝 · 구슬 교차 {pr['crossed']}/{pr['beads']} "
              f"(e1_pour1 최종본 ep6500)")

    if args.verify:
        _verify(trace, zl, zr)

    if args.render is not None:
        print(f"[렌더] {shots[0]}장 → {args.render}")
    if args.gui and not args.auto:
        print("\n창 유지 — Ctrl-C 로 닫는다")
        try:
            with torch.inference_mode():
                while simulation_app.is_running():
                    passive("live", 1)
        except KeyboardInterrupt:
            pass
    return 0 if (ok_l and ok_r) else 1


def _right_home(right_cfg) -> dict:
    jp = right_cfg.robot_cfg.init_state.joint_pos
    return dict(jp)


def _verify(trace: dict, zl, zr) -> None:
    """기록 에피소드 대비 편차 — 보고용(단정 아님: 접촉 카오스는 발산이 정상).

    잡으려는 것은 사슬 배선 오류다: 스케일·순서·앵커가 틀리면 **첫 스텝부터**
    수십 cm/rad 로 어긋난다. 물리 발산은 뒤로 갈수록 서서히 커진다 — 둘은
    시작 구간(초기 20스텝)의 편차로 구분된다.
    """
    def rms(a, b, n=None):
        n = n or min(len(a), len(b))
        if n == 0:
            return float("nan"), 0
        d = np.asarray(a[:n]) - np.asarray(b[:n])
        return float(np.sqrt((d ** 2).mean())), n

    print("\n[재현 대조] (같은 스폰·goal·노이즈0 — 초기 20스텝이 배선 판정)")
    for key, rec, name in (("l_palm", zl.get("palm_targets"), "좌 palm 지령 6D"),
                           ("l_armt", zl.get("arm_target"), "좌 팔 관절지령 7D"),
                           ("r_palm", zr.get("palm_targets"), "우 palm 지령 6D"),
                           ("r_armt", zr.get("arm_target"), "우 팔 관절지령 7D")):
        if rec is None or not trace[key]:
            continue
        e20, _ = rms(trace[key], rec, min(20, len(trace[key]), len(rec)))
        efull, n = rms(trace[key], rec)
        print(f"  {name}: 초기20 RMS {e20:.4f} · 전체({n}) RMS {efull:.4f}")
    if trace["r_latch"] and "latched" in zr:
        mine = next((i for i, v in enumerate(trace["r_latch"]) if v), -1)
        rec = next((i for i, v in enumerate(zr["latched"]) if v > 0.5), -1)
        print(f"  우 latch 프레임: 러너 {mine} vs 기록 {rec}")


if __name__ == "__main__":
    code = main()
    # ★simulation_app.close() 가 replicator attach 상태에서 행 — 프로세스가 좀비로
    #   남아 GPU 를 물고 결과 출력이 버퍼에 갇힌다(09.02 실측, 47분 좀비 2개).
    #   결과를 밀어내고 즉시 죽는다 — 프로세스 종료가 GPU 를 돌려준다.
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(int(code or 0))
