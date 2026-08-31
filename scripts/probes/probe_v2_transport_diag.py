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

"""grasp_sensor_v2 이송 국면 진단 — "왕복이 보상되는가" vs "목표가 도달 불가인가".

08.29 사용자 육안 관찰 두 건을 정량화한다:
  · best(ep780)  — "특정 자세에서 멈췄다가 급격히 목표로 갔다가 **돌아온다**"
  · ep1700       — "잘 잡은 뒤 목표 근처에서 컵을 **왔다갔다**"

두 해석이 완전히 다른 처방을 부른다. 이 프로브가 그 갈림길을 가른다:
  A) 보상이 왕복을 보상한다 → `still` 이 **순간 속도**라 왕복의 반환점(v=0)에서
     최대가 된다. 증거 = `at_goal` **연속 체류 길이가 짧고 많다** + 반환점 정렬.
  B) 목표가 도달 불가다 → min dist 가 상자 크기와 무관하게 벽에 걸린다.
     증거 = env 별 최소 거리 분포가 목표 오프셋 크기와 상관 없이 바닥에 눌린다.

⚠ 저장소 이력: "순간속도 0.07 m/s = 서브밀리미터 솔버 버즈(순변위 2.3 mm/s)" —
  순간 속도는 정지 판정에 못 쓴다. 여기서 **순변위 속도**를 함께 재서 그걸 확인한다.

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_v2_transport_diag.py \
        --checkpoint <path.pth> --num_envs 64 --steps 300
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor_v2")
parser.add_argument("--net_window", type=int, default=20,
                    help="순변위 속도 창(스텝). 순간속도의 솔버 버즈를 걷어낸다.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import openarm.tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from openarm.gripper.left.grasp_sensor_v2 import v2_preset as P
from openarm.gripper.left.grasp_sensor_v2 import v2_stages as S
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

TASK = args.task


def _pct(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float(), q))


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")

    env = gym.make(TASK, cfg=env_cfg)
    raw = env.unwrapped
    inf = float("inf")
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", inf)
    wrapped = RlGamesVecEnvWrapper(env, args.device, clip_obs, clip_act)

    vecenv.register("IsaacRlgWrapper", lambda cfg_name, n, **kw: RlGamesGpuEnv(cfg_name, n, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["config"]["env_info"] = {
        "observation_space": wrapped.observation_space,
        "action_space": wrapped.action_space,
        "agents": 1,
    }
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args.checkpoint)
    agent.reset()

    dev = args.device
    N = args.num_envs
    obj = raw.scene["object"]
    ee = raw.scene["ee_frame"]
    robot_cfg = SceneEntityCfg("robot")
    robot_cfg.resolve(raw.scene)
    jaw_cfg = SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES))
    jaw_cfg.resolve(raw.scene)
    ee_cfg = SceneEntityCfg("ee_frame"); ee_cfg.resolve(raw.scene)
    obj_cfg = SceneEntityCfg("object"); obj_cfg.resolve(raw.scene)

    dt = raw.step_dt

    # ★액션 term 핸들 — 지령(processed_actions)과 리미터 소모량을 직접 읽는다.
    #   "리미터 0.02 m/step 인데 왜 빠른가"는 지령 시계열 없이는 답할 수 없다.
    _aterm = None
    for _n, _tm in raw.action_manager._terms.items():
        if hasattr(_tm, "processed_actions") and hasattr(_tm, "cmd_step_norm"):
            _aterm = _tm
            print(f"[probe] action term = {_n}  ({type(_tm).__name__})")
            break

    # 기록 버퍼
    rec_dist, rec_speed, rec_pos = [], [], []
    rec_stage, rec_close, rec_still, rec_atgoal = [], [], [], []
    rec_cupmtcp, rec_epbuf = [], []
    rec_cmd, rec_cmdstep, rec_tcp, rec_raw, rec_angvel = [], [], [], [], []

    def _t(o):
        return o["obs"] if isinstance(o, dict) else o

    obs = _t(wrapped.reset())
    # ★play.py 와 같은 준비 절차. 없으면 player 가 배치를 1 개로 보고
    #   (1, num_envs*obs_dim) 로 flatten 해 행렬곱이 깨진다.
    agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()
    with torch.inference_mode():
        for _ in range(args.steps):
            act = agent.get_action(obs, is_deterministic=True)
            obs, _, _, _ = wrapped.step(act)
            obs = _t(obs)

            # ★`all_stages` 는 `(pos, val)` 을 돌려준다. 단계 **판정은 pos** 로 —
            #   학습 경로(`Staircase`)와 같은 정의여야 stage 비율이 대조 가능하다.
            _pos, _val = S.all_stages(
                raw, "object_pose", robot_cfg, jaw_cfg, ee_cfg, obj_cfg)
            r_grasp, r_lift, r_transport, r_settle = _pos
            idx = torch.zeros_like(r_grasp)
            idx = torch.where(r_lift > P.STAGE_THRESHOLD, torch.ones_like(idx), idx)
            idx = torch.where(r_transport > P.STAGE_THRESHOLD, torch.full_like(idx, 2.0), idx)
            idx = torch.where(r_settle > P.STAGE_THRESHOLD, torch.full_like(idx, 3.0), idx)

            d = S.cup_goal_distance(raw, "object_pose", robot_cfg, obj_cfg)
            v = torch.norm(obj.data.root_lin_vel_w, dim=1)
            still = S.d_shape(v, P.SETTLE_VEL_S)
            tcp = ee.data.target_pos_w[..., 0, :]

            rec_dist.append(d.clone())
            rec_speed.append(v.clone())
            rec_pos.append(obj.data.root_pos_w.clone())
            rec_stage.append(idx.clone())
            rec_close.append(S.stage_close(raw, jaw_cfg, obj_cfg).clone())
            rec_still.append(still.clone())
            rec_atgoal.append((d < P.SETTLE_RADIUS).float().clone())
            rec_cupmtcp.append((obj.data.root_pos_w - tcp).clone())
            rec_epbuf.append(raw.episode_length_buf.clone())
            rec_tcp.append(tcp.clone())
            rec_angvel.append(torch.norm(obj.data.root_ang_vel_w, dim=1).clone())
            if _aterm is not None:
                rec_cmd.append(_aterm.processed_actions.clone())      # (N,6) [xyz, ez,ey,ex]
                rec_cmdstep.append(_aterm.cmd_step_norm.clone())      # (N,) 리미터 적용 후 실이동
                rec_raw.append(_aterm.raw_actions.clone())

    D = torch.stack(rec_dist)            # (T, N)
    V = torch.stack(rec_speed)
    Pw = torch.stack(rec_pos)            # (T, N, 3)
    ST = torch.stack(rec_stage)
    CL = torch.stack(rec_close)
    SL = torch.stack(rec_still)
    AG = torch.stack(rec_atgoal)
    OF = torch.stack(rec_cupmtcp)
    EB = torch.stack(rec_epbuf)
    TCP = torch.stack(rec_tcp)               # (T,N,3)
    AV = torch.stack(rec_angvel)             # (T,N)
    CMD = torch.stack(rec_cmd) if rec_cmd else None       # (T,N,6)
    CS = torch.stack(rec_cmdstep) if rec_cmdstep else None
    RA = torch.stack(rec_raw) if rec_raw else None
    T = D.shape[0]

    # 에피소드 경계(리셋) — run-length 를 여기서 끊는다
    reset_mask = torch.zeros_like(AG, dtype=torch.bool)
    reset_mask[1:] = EB[1:] <= EB[:-1]

    print("\n" + "=" * 78)
    print(f"v2 이송 진단 — {args.checkpoint.split('/')[-1]}   T={T}  N={N}")
    print("=" * 78)

    # ── ① 도달성: 컵–목표 거리 ────────────────────────────────
    per_env_min = D.min(dim=0).values
    tail = D[T // 2:]
    print("\n[①  도달성 — 컵↔목표 거리 (mm)]")
    print(f"  전체 평균            {float(D.mean())*1e3:7.1f}")
    print(f"  후반 절반 평균       {float(tail.mean())*1e3:7.1f}")
    print(f"  env 별 **최소** 거리  p10 {_pct(per_env_min,0.1)*1e3:6.1f} | "
          f"중앙 {_pct(per_env_min,0.5)*1e3:6.1f} | p90 {_pct(per_env_min,0.9)*1e3:6.1f}")
    print(f"  50 mm 안에 한 번이라도 든 env  {int((per_env_min < P.SETTLE_RADIUS).sum())}/{N}")
    print(f"  ★ 도달 불가 판정: env 최소거리 p10 이 50 mm 를 크게 넘으면 상자가 문제")

    # ── ② 체류 vs 스치기: at_goal run-length ──────────────────
    runs = []
    cur = torch.zeros(N, dtype=torch.long, device=AG.device)
    for t in range(T):
        broke = (AG[t] < 0.5) | reset_mask[t]
        ended = broke & (cur > 0)
        if bool(ended.any()):
            runs += [int(v) for v in cur[ended].tolist()]
        cur = torch.where(AG[t] > 0.5, cur + 1, torch.zeros_like(cur))
        cur = torch.where(broke, torch.zeros_like(cur), cur)
    runs += [int(c) for c in cur.tolist() if c > 0]
    rt = torch.tensor(runs, dtype=torch.float) if runs else torch.zeros(1)
    print("\n[②  목표 반경 안 **연속 체류 길이** (스텝)]  ★진동 해킹의 결정적 지표")
    print(f"  체류 구간 수         {len(runs)}")
    print(f"  평균 길이            {float(rt.mean()):7.2f}   (1 스텝 = {dt*1e3:.0f} ms)")
    print(f"  중앙 / p90 / 최대    {_pct(rt,0.5):.0f} / {_pct(rt,0.9):.0f} / {float(rt.max()):.0f}")
    print(f"  at_goal 총 비율      {float(AG.mean()):.4f}")
    print("  ★ 평균 체류가 짧고(≲10) 구간 수가 많으면 = 스치기(왕복). 길면 = 머물기")

    # ── ③ 순간속도 vs 순변위 ─────────────────────────────────
    W = args.net_window
    if T > W:
        net = torch.norm(Pw[W:] - Pw[:-W], dim=-1) / (W * dt)     # (T-W, N)
        inst = V[W:]
        print(f"\n[③  순간속도 vs **순변위 속도** (창 {W} 스텝 = {W*dt*1e3:.0f} ms)]")
        print(f"  순간속도  평균 {float(inst.mean())*1e3:6.1f} mm/s   p90 {_pct(inst,0.9)*1e3:6.1f}")
        print(f"  순변위속도 평균 {float(net.mean())*1e3:6.1f} mm/s   p90 {_pct(net,0.9)*1e3:6.1f}")
        r = float(net.mean()) / max(float(inst.mean()), 1e-9)
        print(f"  순변위/순간 비율 {r:.3f}")
        print("  ★ 비율이 낮으면 = 제자리 진동(움직이지만 안 간다). 1 에 가까우면 = 직진")

    # ── ③b ★목표 안에서의 순변위 속도 (합격 판정 지표) ────────
    if T > W:
        ag_w = AG[W:] > 0.5
        if bool(ag_w.any()):
            print(f"\n[③b **목표 안에서의 순변위 속도** — ★합격 기준 < 50 mm/s]")
            print(f"  평균 {float(net[ag_w].mean())*1e3:6.1f} mm/s   "
                  f"중앙 {_pct(net[ag_w],0.5)*1e3:6.1f}   p90 {_pct(net[ag_w],0.9)*1e3:6.1f}")
            print(f"  < 50 mm/s 인 비율 {float((net[ag_w] < 0.05).float().mean()):.3f}")
            print(f"  (참고) 목표 안 순간속도 평균 {float(inst[ag_w].mean())*1e3:6.1f} mm/s "
                  f"— 순변위와의 차이가 곧 **진동 성분**이다")

    # ── ④ still 이 언제 커지는가 ─────────────────────────────
    ag = AG > 0.5
    if ag.any():
        print("\n[④  `still` 항 — 목표 안에 있을 때]")
        print(f"  still 평균(목표 안)  {float(SL[ag].mean()):.3f}")
        print(f"  순간속도(목표 안)    평균 {float(V[ag].mean())*1e3:6.1f} mm/s   "
              f"p10 {_pct(V[ag],0.1)*1e3:5.1f}  p90 {_pct(V[ag],0.9)*1e3:6.1f}")
        print(f"  still>0.8 인 비율    {float((SL[ag] > 0.8).float().mean()):.3f}")
        print("  ★ 목표 안에서 순간속도 p10 이 아주 낮은데 평균은 높으면 = **반환점 수확**")

    # ── ⑤ 단계 안정성 ────────────────────────────────────────
    drop = (ST[1:] < ST[:-1]) & (~reset_mask[1:])
    print("\n[⑤  단계 안정성]")
    for k, nm in enumerate(("grasp", "lift", "transport", "settle")):
        print(f"  stage=={k} ({nm:9s}) 비율  {float((ST == k).float().mean()):.4f}")
    print(f"  단계 **하락** 이벤트/스텝   {float(drop.float().mean()):.4f}")
    print(f"  r_close 평균 {float(CL.mean()):.3f}   r_close<0.5 비율 {float((CL<0.5).float().mean()):.4f}")
    print("  ★ 목표로 갈 때 파지가 깨져 단계가 떨어지면 = 돌진·후퇴가 합리적 전략")

    # ── ⑥ 파지 오프셋 실측 (GRASP_OFFSET_ROOT 검증) ──────────
    lifted = CL > 0.5
    if lifted.any():
        off = OF[lifted]
        print("\n[⑥  파지 오프셋 실측 — 컵 − TCP (world, m)]  preset 값과 대조")
        print(f"  실측 평균 ({float(off[:,0].mean()):+.4f}, {float(off[:,1].mean()):+.4f}, "
              f"{float(off[:,2].mean()):+.4f})   ‖·‖ {float(torch.norm(off,dim=1).mean())*1e3:.1f} mm")
        print(f"  preset GRASP_OFFSET_ROOT {P.GRASP_OFFSET_ROOT}  "
              f"‖·‖ {sum(c*c for c in P.GRASP_OFFSET_ROOT)**0.5*1e3:.1f} mm")
        print("  ★ 크게 다르면 목표 상자 평행이동이 어긋나 채점점이 다시 틀어진 것")

    # ── ⑦ ★"리미터 0.02 m/step 인데 왜 빠른가" ─────────────────
    print("\n[⑦  지령 계층 분해 — 리미터·클램프가 정말 속도를 묶고 있는가]")
    # ★A1 이 켜져 있으면 리프트 래치 후 상한이 FINE 으로 갈린다. 기본 상수로 포화를
    #   재면 run B 가 항상 "포화 0" 으로 나와 거꾸로 읽힌다.
    _fine = getattr(_aterm.cfg, "fine_cmd_rate_limit", None) if _aterm is not None else None
    _lim_eff = float(_fine) if _fine is not None else P.PALM_CMD_RATE_LIMIT
    print(f"  PALM_CMD_RATE_LIMIT {P.PALM_CMD_RATE_LIMIT} m/step  ÷ dt {dt:.3f} s "
          f"= **{P.PALM_CMD_RATE_LIMIT/dt*1e3:.0f} mm/s 상한**")
    if _fine is not None:
        print(f"  ★A1 ON — 리프트 래치 후 FINE {_fine} m/step = "
              f"**{_fine/dt*1e3:.0f} mm/s**. 아래 포화는 이 FINE 기준이다.")
        if hasattr(_aterm, "_fine_latched"):
            print(f"    래치된 env {int(_aterm._fine_latched.sum())}/{N} (측정 종료 시점)")
    print(f"  PALM_EULER_RATE_LIMIT {P.PALM_EULER_RATE_LIMIT} rad/step "
          f"= {P.PALM_EULER_RATE_LIMIT/dt:.1f} rad/s")
    if CMD is not None:
        cp = CMD[..., :3]
        dcmd = cp[1:] - cp[:-1]
        dn = torch.norm(dcmd, dim=-1)
        ok = ~reset_mask[1:]
        dn_ok = dn[ok]
        sat = (dn_ok > 0.98 * _lim_eff).float().mean()
        print(f"\n  · 지령 스텝 이동량  평균 {float(dn_ok.mean())*1e3:6.2f} mm "
              f"(= {float(dn_ok.mean())/dt*1e3:6.0f} mm/s)  p90 {_pct(dn_ok,0.9)*1e3:6.2f} mm")
        print(f"    **리미터 포화 비율 {float(sat):.3f}**  ← 1 에 가까우면 정책이 상한을 상시 소모")
        # 방향 반전: 연속 두 지령 변화의 내적
        a, b = dcmd[1:], dcmd[:-1]
        vld = ok[1:] & (torch.norm(a, dim=-1) > 1e-4) & (torch.norm(b, dim=-1) > 1e-4)
        if bool(vld.any()):
            cosang = (a * b).sum(-1) / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1)).clamp(min=1e-9)
            print(f"  · 지령 방향 **반전** 비율(cos<0) {float((cosang[vld] < 0).float().mean()):.3f}   "
                  f"cos 평균 {float(cosang[vld].mean()):+.3f}")
            print("    ★ 반전이 잦고 cos 평균이 0 근처면 = 지령 자체가 **떨고 있다**")
        W2 = args.net_window
        if T > W2 + 1:
            net_cmd = torch.norm(cp[W2:] - cp[:-W2], dim=-1) / (W2 * dt)
            path = dn.unfold(0, W2, 1).sum(-1) / (W2 * dt) if dn.shape[0] >= W2 else None
            print(f"  · 지령 순변위속도 {float(net_cmd.mean())*1e3:6.0f} mm/s")
            if path is not None:
                m = min(net_cmd.shape[0], path.shape[0])
                eff = float(net_cmd[:m].mean()) / max(float(path[:m].mean()), 1e-9)
                print(f"    지령 경로길이속도 {float(path.mean())*1e3:6.0f} mm/s   "
                      f"**직진 효율 {eff:.3f}**  ← 낮을수록 제자리 왕복")
        de = CMD[1:, :, 3:6] - CMD[:-1, :, 3:6]
        dea = de.abs()[ok]
        rsat = (dea > 0.98 * P.PALM_EULER_RATE_LIMIT).float().mean()
        print(f"  · 회전 지령 스텝 변화 평균 {float(dea.mean()):.4f} rad "
              f"(축별 최대 {float(dea.max()):.4f})   회전 리미터 포화 {float(rsat):.3f}")
        print(f"    palm→턱 레버 0.14 m 기준 회전 기여 ≈ "
              f"{float(dea.mean())*0.14/dt*1e3:.0f} mm/s")
    if RA is not None:
        dra = (RA[1:, :, :3] - RA[:-1, :, :3]).abs()[~reset_mask[1:]]
        print(f"  · **raw 액션** 스텝 차분 |Δa| 평균 {float(dra.mean()):.3f} "
              f"(축별 p90 {_pct(dra,0.9):.3f})   ← 1.0 이면 박스 폭 절반을 한 스텝에 요구")
        clamped = (RA[..., :3].abs() > 0.999).float().mean()
        print(f"    |a|>0.999 (클램프) 비율 {float(clamped):.3f}")

    # ── ⑧ 속도 계층: 지령 → TCP → 컵 ─────────────────────────
    dtcp = torch.norm(TCP[1:] - TCP[:-1], dim=-1) / dt
    ok1 = ~reset_mask[1:]
    print("\n[⑧  속도 계층 — 지령 → TCP → 컵]")
    if CMD is not None:
        print(f"  지령 속도  {float(dn[ok1].mean())/dt*1e3:6.0f} mm/s")
    print(f"  TCP  속도  {float(dtcp[ok1].mean())*1e3:6.0f} mm/s")
    print(f"  컵   속도  {float(V[1:][ok1].mean())*1e3:6.0f} mm/s   "
          f"(각속도 {float(AV[1:][ok1].mean()):.2f} rad/s)")
    lever = torch.norm(OF, dim=-1)
    print(f"  컵−TCP 레버 {float(lever.mean())*1e3:.1f} mm  ⇒ 회전 기여 ≈ "
          f"{float((AV*lever).mean())*1e3:.0f} mm/s")
    print("  ★ 컵 속도가 지령 속도보다 크면 = 회전·관성이 증폭한 것(리미터 밖 채널)")

    print("\n" + "=" * 78 + "\n")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
