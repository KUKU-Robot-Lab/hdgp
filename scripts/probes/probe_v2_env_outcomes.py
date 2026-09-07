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

"""env **하나하나의 결말**을 세어 분포로 보여준다 — 평균이 숨기는 것을 드러낸다.

★★왜 필요한가 (08.29). `diag_r_lift` 0.68 · `atgoal` 0.148 인데 사용자가 영상에서
  본 것은 "컵을 잡지도 못하고 쓰러뜨린다" 였다. 두 진술이 동시에 참일 수 있다 —
  TFEvents 는 **스텝·env 평균**이라 "80% 는 멀쩡하고 20% 가 컵을 넘어뜨린다" 를
  하나의 중간값으로 뭉갠다. 실제로 같은 구간에서
      `p_upright`(smoothstep 평균) 0.787   vs   `cup_upright`(cos 평균) 0.881
  이 어긋나는데, smoothstep(0.881) = 0 이므로 **두 값이 같은 분포에서 나올 수 없다**.
  분포가 이봉이라는 뜻이다. 평균을 아무리 봐도 이건 안 보인다.

  ⇒ 이 프로브는 env 별로 **한 번이라도 일어났는가**를 누적해 결말을 분류한다.
    play 영상 1 env 와 TFEvents 평균 사이의 판정을 이걸로 한다.

결말 분류 (배타적, 위에서부터 먼저 맞는 것):
    ⑤ 성공        — 도달 + 정지 + 직립 + 파지 유지를 동시에 만족한 적이 있다
    ④ 도달        — 목표 반경 안에 든 적이 있다
    ③ 리프트      — 컵을 리프트 임계 위로 올린 적이 있다
    ② 파지만      — `grasp_ok` 는 성립했으나 못 들었다
    ① 실패        — 파지조차 못 했다
  그리고 결말과 **독립적으로**: 컵을 넘어뜨린 env 비율(직립 cos 임계 이하).

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_v2_env_outcomes.py \
        --checkpoint <path.pth> --num_envs 256 --steps 250
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor_v2")
parser.add_argument("--adr_level", type=int, default=None,
                    help="ADR 사다리 레벨을 강제한다(0~4). 생략하면 레벨 0 = 현행 과제. "
                         "★학습 중인 정책은 레벨 0 이 아닌 난이도에서 굴러가므로, "
                         "훈련 난이도에서의 성능을 재려면 반드시 지정할 것.")
parser.add_argument("--adr_off", type=str, default="",
                    help="레벨을 적용한 뒤 되돌릴 노브(쉼표): bias,mass,spawn,goal. "
                         "★노브 분리 실험용 — 어느 노브가 성능을 깎는지 가른다.")
parser.add_argument("--spawn_box", type=str, default="",
                    help="스폰 상자를 **절대 좌표**로 강제: x_lo,x_hi,y_lo,y_hi (m). "
                         "이벤트의 pose_range 는 스폰 중심 기준 오프셋이라 여기서 변환한다. "
                         "★봉투 실측용 — IK 프로브가 오라클로 무효라 정책으로 잰다.")
parser.add_argument("--goal_box", type=str, default="",
                    help="목표 상자를 **절대 좌표**로 강제: x_lo,x_hi,y_lo,y_hi,z_lo,z_hi (m).")
parser.add_argument("--dump_map", type=str, default="",
                    help="스폰 위치로 결말을 비닝해 지도로 출력: nx,ny (격자 칸 수). "
                         "★셀마다 프로브를 따로 돌리면 2 시간이 넘는다 — 넓은 상자에 "
                         "한 번 뿌리고 **초기 컵 위치로 결과를 나눠** 한 번에 잰다.")
parser.add_argument("--no_obs_noise", action="store_true",
                    help="컵 관측의 **스텝 잡음**(±3 mm)을 0 으로. ★진동 원인 분리용 — "
                         "절대 위치 지령이라 관측이 떨면 지령이 떨고 fabric 이 그대로 따라간다.")
parser.add_argument("--stochastic", action="store_true",
                    help="★σ 를 켜고 샘플링한다 — 학습 로그와 같은 조건. "
                         "결정론과 갈리면 정책이 탐색 노이즈에 기대고 있다는 뜻이다.")
parser.add_argument("--cup_z_bias", type=float, default=0.0,
                    help="컵 **인지** z 에 상수 편향(m)을 주입한다 — FP++ 원점 규약 불일치 "
                         "모사. 두 obs 경로(`object_position`·`goal_minus_cup`)에 같이 "
                         "먹인다(실기에서는 같은 인지값에서 파생되므로). 보상·판정은 "
                         "ground truth 그대로라 '정책이 속았을 때 무슨 일이 나는가'를 잰다.")
parser.add_argument("--cup_z_bias_obs_only", action="store_true",
                    help="편향을 `object_position` 에만 준다 — 그 축이 정말 무시되는지 분리 측정.")
parser.add_argument("--tip_cos", type=float, default=0.50,
                    help="이 코사인 아래면 '넘어뜨렸다'(0.50 = 60°)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from openarm.gripper.left.grasp_sensor_v2 import v2_preset as P  # noqa: E402
from openarm.gripper.left.grasp_sensor_v2 import v2_stages as S  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

TASK = args.task


def _bar(frac: float, width: int = 34) -> str:
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")

    env = gym.make(TASK, cfg=env_cfg)
    raw = env.unwrapped

    # ── ★ADR 난이도 강제 ──────────────────────────────────────────────
    #   커리큘럼 상태는 체크포인트에 없다. 프로브가 환경을 새로 띄우면 `ADRLadder`
    #   가 항상 **레벨 0**(가장 쉬운 분포)으로 초기화되므로, 그대로 재면 "훈련
    #   난이도에서의 성능"이 아니라 "쉬운 판에서의 성능"을 재게 된다.
    #   여기서 레벨을 직접 세우고 `_apply` 를 불러 목표·스폰·질량·obs bias 를 그
    #   레벨 값으로 덮어쓴다(사다리가 승급 때 하는 일과 동일한 경로).
    if args.adr_level is not None:
        cm = getattr(raw, "curriculum_manager", None)
        names = list(getattr(cm, "active_terms", []) or [])
        if cm is None or "adr" not in names:
            raise SystemExit("[probe] ADR 항이 없다 — HDGP_V2_DR=1 로 실행할 것"
                             f" (현재 curriculum 항: {names})")
        term = cm._term_cfgs[names.index("adr")].func
        term._level = int(args.adr_level)
        term._apply(raw)
        off = {k.strip() for k in args.adr_off.split(",") if k.strip()}
        if off:
            # 레벨 0 값으로 되돌린다 — `_apply` 가 건드리는 네 곳과 정확히 대응한다.
            if "bias" in off:
                raw.event_manager.get_term_cfg("dr_obs_bias").params["bias_range"] = 0.0
            if "mass" in off:
                raw.event_manager.get_term_cfg("dr_cup_mass").params[
                    "mass_distribution_params"] = (1.0, 1.0)
            if "spawn" in off:
                pr = raw.event_manager.get_term_cfg("reset_object_position").params["pose_range"]
                pr["x"] = (-P.CUP_SPAWN_X_RANGE, P.CUP_SPAWN_X_RANGE)
                pr["y"] = (-P.CUP_SPAWN_Y_RANGE, P.CUP_SPAWN_Y_RANGE)
            if "goal" in off:
                rg = raw.command_manager.get_term("object_pose").cfg.ranges
                cx = 0.5 * (rg.pos_x[0] + rg.pos_x[1])
                cy = 0.5 * (rg.pos_y[0] + rg.pos_y[1])
                cz = 0.5 * (rg.pos_z[0] + rg.pos_z[1])
                jx, jy, jz = P.GOAL_JITTER_V2
                rg.pos_x = (cx - jx, cx + jx)
                rg.pos_y = (cy - jy, cy + jy)
                rg.pos_z = (cz - jz, cz + jz)

            print(f"[probe] 노브 되돌림: {sorted(off)}", flush=True)
        print(f"[probe] ADR 레벨 {args.adr_level} 강제 적용", flush=True)
    # ── ★절대 범위 강제 (봉투 실측) ───────────────────────────────────
    #   ADR 레벨 강제보다 **뒤에** 둔다 — 레벨을 세운 뒤 특정 축만 덮어쓸 수 있게.
    #   ⚠ `reset_object_position` 의 pose_range 는 **스폰 중심 기준 오프셋**이다.
    #     절대 좌표를 그대로 넣으면 컵이 테이블 밖으로 날아간다(부호 함정).
    # ── ★스텝 잡음 차단 (진동 원인 분리) ──────────────────────────────
    if args.cup_z_bias != 0.0:
        # ★인지 편향 주입. `object_position` 은 기존 에피소드 bias 버퍼를 그대로 쓰고
        #   (리셋마다 재샘플되므로 매 스텝 덮어써야 한다), `goal_minus_cup` 은 sim 에서
        #   ground truth 를 쓰므로 함수를 감싸서 같은 양을 뺀다(목표−컵 이므로 부호 반대).
        import openarm.gripper.left.grasp_sensor_v2.v2_observations as _obs
        _bz = float(args.cup_z_bias)
        _terms = raw.observation_manager._group_obs_term_cfgs["policy"]
        _names = raw.observation_manager._group_obs_term_names["policy"]
        for nm, cfg in zip(_names, _terms):
            if "step_noise" in getattr(cfg, "params", {}):
                _orig0 = cfg.func
                def _w0(env, *a, __o=_orig0, __b=_bz, **k):
                    out = __o(env, *a, **k)
                    out = out.clone(); out[:, 2] += __b
                    return out
                cfg.func = _w0
                print(f"[probe] {nm} z 에 {_bz*1000:+.1f} mm", flush=True)
        if not args.cup_z_bias_obs_only:
            for nm, cfg in zip(_names, _terms):
                if getattr(cfg.func, "__name__", "") == "goal_minus_cup":
                    _orig = cfg.func
                    def _wrapped(env, *a, __o=_orig, __b=_bz, **k):
                        out = __o(env, *a, **k)
                        out = out.clone(); out[:, 2] -= __b
                        return out
                    cfg.func = _wrapped
                    print(f"[probe] goal_minus_cup z 에 {-_bz*1000:+.1f} mm", flush=True)
        print(f"[probe] 컵 인지 z 편향 {_bz*1000:+.1f} mm"
              f"{' (object_position 만)' if args.cup_z_bias_obs_only else ''}", flush=True)

    if args.no_obs_noise:
        t = raw.observation_manager._group_obs_term_cfgs["policy"]
        names = raw.observation_manager._group_obs_term_names["policy"]
        hit = False
        for nm, cfg in zip(names, t):
            if "step_noise" in getattr(cfg, "params", {}):
                cfg.params["step_noise"] = 0.0; hit = True
                print(f"[probe] obs 스텝 잡음 0 으로 ({nm})", flush=True)
        if not hit:
            raise SystemExit("[probe] step_noise 파라미터를 못 찾았다 — HDGP_V2_DR=1 인가")

    if args.spawn_box:
        xl, xh, yl, yh = (float(v) for v in args.spawn_box.split(","))
        pr = raw.event_manager.get_term_cfg("reset_object_position").params["pose_range"]
        pr["x"] = (xl - P.CUP_SPAWN_X_CENTER, xh - P.CUP_SPAWN_X_CENTER)
        pr["y"] = (yl - P.CUP_SPAWN_Y_CENTER, yh - P.CUP_SPAWN_Y_CENTER)
        print(f"[probe] 스폰 강제 x[{xl:.3f},{xh:.3f}] y[{yl:.3f},{yh:.3f}]"
              f"  (offset x{pr['x']} y{pr['y']})", flush=True)
    if args.goal_box:
        xl, xh, yl, yh, zl, zh = (float(v) for v in args.goal_box.split(","))
        rg = raw.command_manager.get_term("object_pose").cfg.ranges
        rg.pos_x, rg.pos_y, rg.pos_z = (xl, xh), (yl, yh), (zl, zh)
        print(f"[probe] 목표 강제 x[{xl:.3f},{xh:.3f}] y[{yl:.3f},{yh:.3f}]"
              f" z[{zl:.3f},{zh:.3f}]", flush=True)
    inf = float("inf")
    wrapped = RlGamesVecEnvWrapper(
        env, args.device,
        agent_cfg["params"]["env"].get("clip_observations", inf),
        agent_cfg["params"]["env"].get("clip_actions", inf))

    vecenv.register("IsaacRlgWrapper", lambda cfg_name, n, **kw: RlGamesGpuEnv(cfg_name, n, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["config"]["env_info"] = {
        "observation_space": wrapped.observation_space,
        "action_space": wrapped.action_space,
        "agents": 1,
    }
    # ⚠ rl_games 는 `batch_size % minibatch_size == 0` 을 어서션한다. 학습 설정의
    #   minibatch(24576 = 1024×24)를 그대로 두면 num_envs 를 바꾸는 순간 죽는다.
    #   플레이어는 minibatch 를 쓰지 않으므로 여기서 정합시켜 준다.
    hz = int(agent_cfg["params"]["config"].get("horizon_length", 24))
    agent_cfg["params"]["config"]["minibatch_size"] = args.num_envs * hz

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args.checkpoint)
    agent.reset()

    N = args.num_envs
    dev = args.device
    obj = raw.scene["object"]
    robot_cfg = SceneEntityCfg("robot"); robot_cfg.resolve(raw.scene)
    jaw_cfg = SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES))
    jaw_cfg.resolve(raw.scene)
    ee_cfg = SceneEntityCfg("ee_frame"); ee_cfg.resolve(raw.scene)
    obj_cfg = SceneEntityCfg("object"); obj_cfg.resolve(raw.scene)

    # ★★TFEvents 대조용 — 스텝·env 평균. 학습 로그와 **같은 정의**라 직접 비교된다.
    #   이 값이 로그와 맞으면 프로브 환경이 학습과 같다는 뜻이고, 결말 분포가 진실이다.
    #   어긋나면 환경이 다른 것이므로 결말 분포부터 의심해야 한다.
    acc = {k: torch.zeros((), device=dev) for k in
           ("r_close", "r_lift", "r_transport", "at_goal", "upright", "grasp_ok")}
    n_acc = 0

    z = lambda: torch.zeros(N, dtype=torch.bool, device=dev)   # noqa: E731
    ever_grasp, ever_lift, ever_goal, ever_succ, ever_tip = z(), z(), z(), z(), z()
    # ★★⑤ 정지가 막히는 이유를 조건별로 가른다. 성공은 네 조건의 **동시** 만족이라
    #   어느 하나가 병목인지 따로 세지 않으면 처방을 못 고른다.
    ok_gu, ok_gs, ok_gr = z(), z(), z()          # 도달+직립 / 도달+정지 / 도달+파지
    min_spd_at = torch.full((N,), 9.9, device=dev)   # 목표 안에 있을 때의 최저 속도
    max_cos_at = torch.zeros(N, device=dev)          # 목표 안에 있을 때의 최고 직립
    # ★★08.31 — **최저 속도로 정지를 재면 진동을 통과시킨다.** 진동체는 매 사이클
    #   반환점에서 속도가 0 을 지나므로 `min_spd_at` 이 낮게 나온다(사용자 영상 관찰로
    #   발각: "목표 위치에서 계속 손을 떨고 있음"). 라운드 1 의 순변위 판정도 같은
    #   이유로 실패했다(제자리 진동의 순변위 = 0). ⇒ **평균 속도**와 **최장 연속 정지
    #   구간**을 함께 잰다. 이 둘은 진동에 속지 않는다.
    sum_spd_at = torch.zeros(N, device=dev)          # 목표 안 속도 누적
    cnt_at = torch.zeros(N, device=dev)              # 목표 안 스텝 수
    run_ok = torch.zeros(N, device=dev)              # 현재 연속 합격 스텝
    best_run = torch.zeros(N, device=dev)            # 최장 연속 합격 스텝
    min_cos = torch.ones(N, device=dev)
    min_dist = torch.full((N,), 9.9, device=dev)
    max_z = torch.zeros(N, device=dev)
    # ★09.03 — **실제 파지 높이**. 대역은 범위일 뿐이고 정책이 그 안 어디서 잡는지는
    #   따로 재야 한다. "대역 중앙에서 잡는다"는 가정이 H2/H3 에서 깨졌다
    #   (대역을 60mm 올렸는데 손끝 최저가 14mm 그대로).
    #   첫 grasp_ok 순간의 TCP 높이(판 위)와 턱 최저 높이를 기록한다.
    grasp_tcp_z = torch.full((N,), float("nan"), device=dev)
    grasp_tip_z = torch.full((N,), float("nan"), device=dev)
    # 리셋된 env 는 새 에피소드다 — 누적을 이어 붙이면 결말이 섞인다.
    # 첫 에피소드만 세기 위해 종료된 env 를 잠근다.
    done_lock = z()

    def _t(o):
        return o["obs"] if isinstance(o, dict) else o

    obs = _t(wrapped.reset())
    # ★스폰 지도용 — 리셋 직후의 컵 위치(= 이 에피소드의 스폰). env 원점을 빼
    #   로봇 root 기준 절대 좌표로 되돌린다(스폰 상자와 같은 좌표계).
    spawn_xy = (obj.data.root_pos_w[:, :2] - raw.scene.env_origins[:, :2]).clone()
    # ★★08.31 — **테이블 긁힘 실측**. 손끝(두 턱 링크)과 TCP 중 가장 낮은 점의
    #   에피소드 최저 높이(판 위, m). "정책이 실제로 판을 긁는가"에 직접 답한다.
    tip_min = torch.full((raw.num_envs,), 9.9, device=raw.device)
    # ★★08.31 사용자 지적 — "jaw 끝단이 바닥에 닿을 듯 가다가 잡을 때 j7 을 든다".
    #   **파지 전(접근 구간)** 만 따로 본다. 파지 순간의 낮은 높이는 컵 형상이 정한
    #   값이라(파지 대역 10~85 mm · 패드가 강체보다 31.9 mm 아래) 어쩔 수 없지만,
    #   **옆으로 이동하는 동안** 낮으면 그건 쓸고 가는 것이다.
    #   접근 각도 = 그리퍼 +z 축과 world +z 의 사잇각. 90° = 수평, >90° = 아래로 기울어짐.
    pre_tip_min = torch.full((raw.num_envs,), 9.9, device=raw.device)
    pre_ang_max = torch.zeros(raw.num_envs, device=raw.device)
    pre_ang_sum = torch.zeros(raw.num_envs, device=raw.device)
    pre_steps = torch.zeros(raw.num_envs, device=raw.device)
    base_bi = raw.scene["robot"].body_names.index(P.GRIPPER_BASE_BODY)
    # ★play.py 와 같은 준비 절차. 없으면 player 가 배치를 1 로 보고
    #   (1, num_envs*obs_dim) 로 flatten 해 행렬곱이 깨진다.
    agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    with torch.inference_mode():
        for _ in range(args.steps):
            act = agent.get_action(obs, is_deterministic=not args.stochastic)
            obs, _, dones, _ = wrapped.step(act)
            obs = _t(obs)

            live = ~done_lock
            r_close = S.stage_close(raw, jaw_cfg, obj_cfg)
            cup_z = obj.data.root_pos_w[:, 2]
            dist = S.cup_goal_distance(raw, "object_pose", robot_cfg, obj_cfg)
            spd = torch.norm(obj.data.root_lin_vel_w, dim=1)
            cos = S._cup_upright_cos(raw, obj_cfg)
            succ = S.success_ok(dist, spd, cos, r_close) > 0.5

            # `r_close == 1.0` ⟺ `grasp_ok` (그 외에는 ≤ 0.5) — v2_stages 계약 참조
            ever_grasp |= live & (r_close > 0.5)
            # ★★09.03 — `ever_lift` 에 **파지 조건을 AND** 한다. 높이만 보면 잡지 않고
            #   튕겨 날아간 컵과 재소환 텔레포트 순간까지 "들었다"로 세어, 결말 분류가
            #   배타적이지 않게 된다(합 117% 로 관측). 이 저장소가 이미 기록해 둔 함정:
            #   "판정 ✅ 는 날아간 물체도 센다".
            ever_lift |= live & (cup_z > P.MINIMAL_LIFT_HEIGHT) & (r_close > 0.5)
            ever_goal |= live & (dist < P.SETTLE_RADIUS)
            ever_succ |= live & succ
            ever_tip |= live & (cos < args.tip_cos)
            r_lift_t = r_close * ((cup_z - P.LIFT_RAMP_ZERO_Z)
                                  / (P.MINIMAL_LIFT_HEIGHT - P.LIFT_RAMP_ZERO_Z)).clamp(0, 1)
            acc["r_close"] += r_close.mean(); acc["r_lift"] += r_lift_t.mean()
            acc["r_transport"] += (r_lift_t * S.d_shape(dist, P.TRANSPORT_S,
                                                        P.TRANSPORT_TAU)).mean()
            acc["at_goal"] += (dist < P.SETTLE_RADIUS).float().mean()
            acc["upright"] += cos.mean(); acc["grasp_ok"] += (r_close > 0.5).float().mean()
            _jz = raw.scene["robot"].data.body_pos_w[:, jaw_cfg.body_ids, 2]
            _ez = raw.scene["ee_frame"].data.target_pos_w[:, 0, 2]
            _low = (torch.minimum(_jz.min(dim=1).values, _ez)
                    - raw.scene.env_origins[:, 2] - P.TABLE_SURFACE_Z)
            tip_min = torch.where(live, torch.minimum(tip_min, _low), tip_min)
            _newly = live & (r_close > 0.5) & torch.isnan(grasp_tcp_z)
            grasp_tcp_z = torch.where(_newly, _ez - raw.scene.env_origins[:, 2]
                                      - P.TABLE_SURFACE_Z, grasp_tcp_z)
            grasp_tip_z = torch.where(_newly, _low, grasp_tip_z)
            _q = raw.scene["robot"].data.body_quat_w[:, base_bi, :]
            _azc = (1.0 - 2.0 * (_q[:, 1] ** 2 + _q[:, 2] ** 2)).clamp(-1.0, 1.0)
            _ang = torch.rad2deg(torch.acos(_azc))
            _pre = live & (~ever_grasp)
            pre_tip_min = torch.where(_pre, torch.minimum(pre_tip_min, _low), pre_tip_min)
            pre_ang_max = torch.where(_pre, torch.maximum(pre_ang_max, _ang), pre_ang_max)
            pre_ang_sum = pre_ang_sum + torch.where(_pre, _ang, torch.zeros_like(_ang))
            pre_steps = pre_steps + _pre.float()
            n_acc += 1

            at = live & (dist < P.SETTLE_RADIUS)
            ok_gr |= at & (r_close > 0.5)
            ok_gs |= at & (spd < P.STAGE3_SPEED_MAX)
            ok_gu |= at & (cos > P.STAGE3_UPRIGHT_MIN)
            min_spd_at = torch.where(at, torch.minimum(min_spd_at, spd), min_spd_at)
            max_cos_at = torch.where(at, torch.maximum(max_cos_at, cos), max_cos_at)
            sum_spd_at += torch.where(at, spd, torch.zeros_like(spd))
            cnt_at += at.float()
            run_ok = torch.where(live & succ, run_ok + 1.0, torch.zeros_like(run_ok))
            best_run = torch.maximum(best_run, run_ok)

            min_cos = torch.where(live, torch.minimum(min_cos, cos), min_cos)
            min_dist = torch.where(live, torch.minimum(min_dist, dist), min_dist)
            max_z = torch.where(live, torch.maximum(max_z, cup_z), max_z)

            d = dones.bool() if torch.is_tensor(dones) else torch.as_tensor(dones, device=dev).bool()
            done_lock |= d.reshape(-1)

    # ── 결말 분류 (배타적) ────────────────────────────────────────────
    c5 = ever_succ
    c4 = ever_goal & ~c5
    c3 = ever_lift & ~ever_goal
    c2 = ever_grasp & ~ever_lift
    c1 = ~ever_grasp
    # ★분류가 배타적인지 자체 검산한다 — 합이 100% 가 아니면 정의가 겹친 것이다.
    _sum = float((c1 | c2 | c3 | c4 | c5).float().mean())
    _tot = sum(float(c.float().mean()) for c in (c1, c2, c3, c4, c5))
    if abs(_tot - 1.0) > 1e-3 or abs(_sum - 1.0) > 1e-3:
        print(f"[probe] ⚠ 결말 분류가 배타적이 아니다 — 합 {_tot*100:.1f}% "
              f"(합집합 {_sum*100:.1f}%). 정의가 겹쳤다.", flush=True)

    rows = [("⑤ 성공 (도달+정지+직립+파지)", c5),
            ("④ 도달 (반경 50mm 진입)", c4),
            ("③ 리프트 (들었으나 미도달)", c3),
            ("② 파지만 (못 들었다)", c2),
            ("① 실패 (파지조차 못 함)", c1)]

    if args.dump_map:
        nx, ny = (int(v) for v in args.dump_map.split(","))
        pr = raw.event_manager.get_term_cfg("reset_object_position").params["pose_range"]
        xlo, xhi = pr["x"][0] + P.CUP_SPAWN_X_CENTER, pr["x"][1] + P.CUP_SPAWN_X_CENTER
        ylo, yhi = pr["y"][0] + P.CUP_SPAWN_Y_CENTER, pr["y"][1] + P.CUP_SPAWN_Y_CENTER
        ix = ((spawn_xy[:, 0] - xlo) / max(xhi - xlo, 1e-9) * nx).long().clamp(0, nx - 1)
        iy = ((spawn_xy[:, 1] - ylo) / max(yhi - ylo, 1e-9) * ny).long().clamp(0, ny - 1)
        print("\n" + "=" * 78)
        print(f"★스폰 봉투 지도 — x[{xlo:.3f},{xhi:.3f}] × y[{ylo:.3f},{yhi:.3f}] "
              f"격자 {nx}×{ny} · env {N}")
        print("  각 칸: ⑤성공% / ①파지실패% (표본수).  봉투 = ① ≤ 5%")
        print("=" * 78)
        hdr = "  y\\x  " + "".join(
            f"{xlo + (i + 0.5) * (xhi - xlo) / nx:>15.3f}" for i in range(nx))
        print(hdr)
        for j in range(ny - 1, -1, -1):
            yc = ylo + (j + 0.5) * (yhi - ylo) / ny
            cells = []
            for i in range(nx):
                m = (ix == i) & (iy == j)
                k = int(m.sum())
                if k == 0:
                    cells.append(f"{'-':>15}")
                    continue
                s5 = float(c5[m].float().mean()) * 100.0
                s1 = float(c1[m].float().mean()) * 100.0
                cells.append(f"{s5:6.0f}/{s1:<4.0f}({k:3d})")
            print(f"{yc:6.3f} " + "".join(cells))
        print("=" * 78)

    print("\n" + "=" * 78)
    print("★TFEvents 대조 — 스텝·env 평균 (학습 로그와 같은 정의)")
    print("=" * 78)
    ref = {"r_lift": 0.679, "r_transport": 0.343, "at_goal": 0.148, "upright": 0.881}
    for k in ("grasp_ok", "r_close", "r_lift", "r_transport", "at_goal", "upright"):
        v = (acc[k] / max(1, n_acc)).item()
        r = ref.get(k)
        tag = "" if r is None else (
            f"   로그 {r:.3f}  {'일치' if abs(v-r) < 0.08 else '★어긋남'}")
        print(f"  {k:<14} {v:7.4f}{tag}")
    print("  ⚠ `r_close` 는 grasp_ok 면 1.0, 아니면 ≤ 0.5 다 —")
    print("     따라서 `r_lift` > 0.5 는 grasp_ok 가 자주 성립해야만 가능하다.")

    print("\n" + "=" * 78)
    _mode = "★확률적(σ 포함, 학습과 동일)" if args.stochastic else "결정론(σ 배제)"
    print(f"env 결말 분포 — {N} env · {args.steps} step · {_mode} · 첫 에피소드만")
    print(f"  체크포인트: {args.checkpoint}")
    print("=" * 78)
    tot = 0
    for lab, m in rows:
        f = m.float().mean().item(); tot += f
        print(f"  {lab:<30} {f:6.1%}  {_bar(f)}")
    print(f"  {'합계 (배타적이므로 1.00)':<30} {tot:6.1%}")

    print("\n" + "-" * 78)
    # ★⑤ 가 0 인 판에서도 아래 손끝·각도 통계는 찍혀야 한다. 예전에는 `qq` 가
    #   "목표 도달 env 있음" 분기 안에서만 정의돼, 전패 판이 NameError 로 죽으면서
    #   정작 실패 원인을 보여줄 통계가 통째로 사라졌다(L22b ep1250 에서 실측).
    qq = lambda t, p: torch.quantile(t.float(), p).item()   # noqa: E731
    print("★⑤ 성공을 막는 조건은 무엇인가 (목표 반경 안에 든 env 기준)")
    print("-" * 78)
    ng = ever_goal.float().sum().clamp(min=1)
    for lab, m in (("도달(dist<50mm)", ever_goal),
                   ("  + 파지 유지", ok_gr),
                   ("  + 직립 cos>0.99", ok_gu),
                   ("  + 정지 speed<0.05", ok_gs),
                   ("  = 네 조건 동시(성공)", ever_succ)):
        f = m.float().sum().item() / ng.item()
        print(f"  {lab:<26} 도달 env 대비 {f:6.1%}  {_bar(f)}")
    inr = ever_goal
    if inr.any():
        qq = lambda t, p: torch.quantile(t.float(), p).item()   # noqa: E731
        mean_at = torch.where(cnt_at > 0, sum_spd_at / cnt_at.clamp(min=1.0),
                              torch.full_like(sum_spd_at, float("nan")))
        ms = mean_at[inr]; br = best_run[inr]
        print(f"\n  ★목표 안 **평균** 컵 속도  p10 {qq(ms,0.1):.4f} · 중앙 {qq(ms,0.5):.4f}"
              f" · p90 {qq(ms,0.9):.4f} m/s   (합격 {P.STAGE3_SPEED_MAX})")
        print(f"     — 최저값이 아니라 평균이다. 진동체는 반환점마다 속도 0 을 지나"
              f" 최저값이 낮게 나온다(진동을 못 거른다).")
        print(f"  ★**최장 연속 합격 스텝**  p50 {qq(br,0.5):.0f} · p90 {qq(br,0.9):.0f}"
              f" · 최대 {float(br.max()):.0f}  (250 스텝 중 · 30 스텝이면 hold 만점)")
        sp = min_spd_at[inr]; cs = max_cos_at[inr]
        qq = lambda t, p: torch.quantile(t.float(), p).item()   # noqa: E731
        print(f"\n  목표 안에서의 **최저 컵 속도**  p10 {qq(sp,0.1):.4f} · 중앙 {qq(sp,0.5):.4f}"
              f" · p90 {qq(sp,0.9):.4f} m/s   (합격 {P.STAGE3_SPEED_MAX})")
        print(f"  목표 안에서의 **최고 직립 cos** p10 {qq(cs,0.1):.4f} · 중앙 {qq(cs,0.5):.4f}"
              f" · p90 {qq(cs,0.9):.4f}        (합격 {P.STAGE3_UPRIGHT_MIN})")

    print("\n" + "-" * 78)
    print("컵을 넘어뜨렸는가 (결말과 독립)")
    # ★★테이블 긁힘 실측 — 에피소드 중 손끝·TCP 가 내려간 **최저 높이**(판 위, mm).
    #   음수면 상면 아래로 파고든 것이다. 보정 문서 §2-2 권고는 여유 ≥20~30 mm.
    tm = tip_min * 1e3
    print("-" * 78)
    print(f"  ★**손끝 최저 높이**(판 위, mm)  p10 {qq(tm,0.1):.1f} · 중앙 {qq(tm,0.5):.1f}"
          f" · p90 {qq(tm,0.9):.1f} · **최소 {tm.min().item():.1f}**")
    for th in (30.0, 20.0, 10.0, 5.0, 0.0):
        f = (tm < th).float().mean().item()
        print(f"    판 위 {th:4.0f} mm 아래로 내려간 env   {f:6.1%}")
    # ★파지 **전** 구간만 — "쓸고 가는가"에 직접 답한다.
    ok = pre_steps > 0
    if ok.any():
        ptm = pre_tip_min[ok] * 1e3
        pam = pre_ang_max[ok]
        pav = (pre_ang_sum[ok] / pre_steps[ok])
        print(f"\n  ★**파지 전(접근) 구간만** — env {int(ok.sum())} · 평균 {pre_steps[ok].mean():.0f} 스텝")
        print(f"    손끝 최저 높이  p10 {qq(ptm,0.1):.1f} · 중앙 {qq(ptm,0.5):.1f}"
              f" · p90 {qq(ptm,0.9):.1f} · 최소 {ptm.min().item():.1f} mm")
        for th in (30.0, 20.0, 10.0, 0.0):
            print(f"      판 위 {th:4.0f} mm 아래로 내려간 env   {(ptm < th).float().mean().item():6.1%}")
        _g = grasp_tcp_z[~torch.isnan(grasp_tcp_z)] * 1000.0
        _t = grasp_tip_z[~torch.isnan(grasp_tip_z)] * 1000.0
        if _g.numel() > 0:
            print(f"  ★**실제 파지 높이** (첫 grasp_ok 순간 · 판 위 mm · n={_g.numel()})")
            print(f"    TCP    p10 {torch.quantile(_g,0.1):.1f} · 중앙 {_g.median():.1f} · p90 {torch.quantile(_g,0.9):.1f}"
                  f"   (대역 {P.GRASP_HEIGHT_BAND[0]*1000:.0f}~{P.GRASP_HEIGHT_BAND[1]*1000:.0f} · 중앙 {(P.GRASP_HEIGHT_BAND[0]+P.GRASP_HEIGHT_BAND[1])*500:.0f})")
            print(f"    턱최저 p10 {torch.quantile(_t,0.1):.1f} · 중앙 {_t.median():.1f} · p90 {torch.quantile(_t,0.9):.1f}"
                  f"   → 파지 순간 TCP−턱최저 = {(_g.median()-_t.median()):.1f} mm")
            print(f"    접근 각도(그리퍼 +z ∠ world +z)  평균 중앙 {qq(pav,0.5):.1f}°"
              f" · 최대 중앙 {qq(pam,0.5):.1f}° · 최대 p90 {qq(pam,0.9):.1f}°")
        for th in (90.0, 100.0, 110.0, 120.0):
            print(f"      한 번이라도 {th:5.0f}° 초과(아래로 기욺) env   "
                  f"{(pam > th).float().mean().item():6.1%}")
    ft = ever_tip.float().mean().item()
    print(f"  직립 cos < {args.tip_cos:.2f} ({math.degrees(math.acos(args.tip_cos)):.0f}°) 인 적이 있는 env"
          f"   {ft:6.1%}  {_bar(ft)}")
    for th in (0.99, 0.95, 0.90, 0.70, 0.50, 0.20):
        f = (min_cos < th).float().mean().item()
        print(f"    최저 직립 cos < {th:.2f} ({math.degrees(math.acos(th)):4.0f}°)   {f:6.1%}")

    print("\n" + "-" * 78)
    print("분포 요약 (평균이 숨긴 것)")
    print("-" * 78)
    q = lambda t, p: torch.quantile(t.float(), p).item()   # noqa: E731
    for lab, t, sc in (("최저 컵–목표 거리 (mm)", min_dist, 1000.0),
                       ("최고 컵 높이 (mm)", max_z, 1000.0),
                       ("최저 직립 cos", min_cos, 1.0)):
        print(f"  {lab:<24} p10 {q(t,0.1)*sc:8.1f} · 중앙 {q(t,0.5)*sc:8.1f} · p90 {q(t,0.9)*sc:8.1f}"
              f" · 평균 {t.mean().item()*sc:8.1f}")
    print(f"  {'리프트 임계':<24} {P.MINIMAL_LIFT_HEIGHT*1000:.1f} mm ·"
          f" 스폰 {P.CUP_SPAWN_Z*1000:.1f} mm · 합격 반경 {P.SETTLE_RADIUS*1000:.0f} mm")
    print("=" * 78)
    print("PROBE_OUTCOMES_DONE")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
