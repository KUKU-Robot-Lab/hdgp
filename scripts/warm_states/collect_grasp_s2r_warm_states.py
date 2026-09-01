#!/usr/bin/env python3
"""grasp_s2r(b1) 성공 파지 → pour warmstart 뱅크(HDF5) 수집기.

`PourWarmStateBank`(tesollo/right/pour_sensor/warm_state_bank.py)가 그대로 읽는 형식으로
쓴다. 기존 수집기들은 grasp_v1/v5/v11 전용이라 grasp_s2r 의 프로필 기반 구조를 모른다.

★★**palm 원점 변환이 이 스크립트의 존재 이유다.**
  grasp_s2r 의 palm 기준은 프로필 `palm_body`(= `r_hl_palm`) **원점**이다
  (`grasp_s2r_env.py` 주석 "기준 body 는 palm_body 의 원점이다(palm_ee 가 아니다)").
  `palm_targets` 도 변환 없이 fabric 으로 바로 간다.
  반면 pour 는 **palm_ee** 규약이다 — `palm_center_pos = r_hl_palm + R·(0.028,0,0.04)`
  이고, fabric 직전에만 palm_link 로 역변환한다.
  RL URDF 실측: `r_hj_palm_ee: r_hl_palm_alias -> r_hl_palm_ee xyz=0.028 0 0.04` (48.8 mm).
  변환 없이 뱅크를 쓰면 **차원이 같아 로더는 통과하는데 48.8 mm 어긋난** 초기 파지가 된다
  — 2026-08-17 DG-5FS 사고와 같은 실패 모드다. 그래서 여기서 palm_ee 로 옮겨 적고,
  원본 palm_link 포즈는 진단용 별도 채널로 남긴다.

★자산 출처(`robot_usd`)를 반드시 기록한다. 구 캐시에 이 필드가 없어서 08-17 사고를
  자동 검출할 수 없었다(quarantine README 가 재수집 시 넣으라고 명시).

사용:
  python collect_grasp_s2r_warm_states.py \\
      --task open-sens_r_grasp_s2r-play-lstm --checkpoint <b1.pth> \\
      --out data/grasp_warm_s2r_b1.hdf5 --target_count 2048 --num_envs 32 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="grasp_s2r 성공 파지 → pour warm 뱅크")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="체크포인트 절대경로")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--out", type=str, required=True, help="출력 HDF5")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--target_count", type=int, default=2048, help="모을 성공 상태 수")
parser.add_argument("--max_steps", type=int, default=20000)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--hold_steps", type=int, default=10,
    help="파지+리프트 조건을 연속 이만큼 유지해야 캡처. 순간적으로 스치는 상태를 거른다.",
)
parser.add_argument("--min_grip_fingers", type=int, default=2)
parser.add_argument("--contact_threshold", type=float, default=0.1, help="N — 팁 접촉 판정")
parser.add_argument(
    "--max_object_speed", type=float, default=0.10,
    help="m/s — 이 이하라야 '멈춰 있다'로 본다. 리프트 직후 흔들리는 상태를 거른다.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# ★경로는 isaaclab import 보다 먼저 — `import isaaclab_tasks` 가 확장 진입점을 훑으며
#   openarm 을 먼저 import 하면 다른 트리가 sys.modules 를 선점한다.
import os  # noqa: E402
from pathlib import Path  # noqa: E402

_HDGP = Path(__file__).resolve().parents[2]
for _p in (str(_HDGP / "scripts/tools"), str(_HDGP / "source/openarm")):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
for _name in [m for m in sys.modules if m == "openarm" or m.startswith("openarm.")]:
    del sys.modules[_name]

from datetime import datetime  # noqa: E402
import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import hashlib  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import openarm  # noqa: E402
_EXPECTED = str((_HDGP / "source/openarm/openarm").resolve())
if not str(Path(openarm.__file__).resolve()).startswith(_EXPECTED + os.sep):
    raise SystemExit(f"openarm 이 저장소 밖에서 왔다: {openarm.__file__}")
import openarm.tasks  # noqa: F401,E402

from run_cfg_restore import restore_run_cfg_if_available  # noqa: E402

#: RL URDF `r_hj_palm_ee` 실측 — r_hl_palm 프레임 기준 palm_ee 로컬 오프셋 [m].
#: pour 의 `_palm_ee_offset_local` 과 같은 값이어야 한다(양쪽이 갈리면 뱅크가 어긋난다).
PALM_EE_OFFSET_LOCAL = (0.028, 0.0, 0.04)

#: pour 로더가 요구하는 데이터셋. 하나라도 빠지면 로드가 거부된다.
BANK_DATASETS = (
    "arm_joint_pos", "hand_joint_pos", "palm_pose_quat_xyzw",
    "palm_pose_euler_zyx", "cup_pos_local", "cup_quat_wxyz", "num_contacts",
)


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_HDGP), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


class WarmCollector:
    """성공 파지 시점 상태를 env 별로 모은다. **palm_ee 변환은 여기 한 곳에서만** 한다."""

    def __init__(self, env, cfg_ns, hold_steps: int, min_grip: int, thr: float,
                 max_speed: float):
        self.env = env
        self.cfg = cfg_ns
        self.hold_steps = hold_steps
        self.min_grip = min_grip
        self.thr = thr
        self.max_speed = max_speed
        n = env.num_envs
        self._hold = torch.zeros(n, dtype=torch.long, device=env.device)
        # ★에피소드당 1회만 담는다(collect_pour_fab 패턴). 한 파지에서 거의 같은 상태를
        #   수십 개 뽑으면 뱅크 다양성이 죽는다.
        self._captured_ep = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._offset = torch.tensor(PALM_EE_OFFSET_LOCAL, device=env.device).reshape(1, 3)
        # ★물체 배정은 env 와 **같은 함수**로 구한다(`env_id % N` 로 추측하지 않는다).
        #   pour 는 warm state 를 복원할 때 그 상태가 **어느 컵**에서 나왔는지 알아야
        #   같은 컵을 스폰한다 — 크기가 다르면 손은 벌어져 있는데 컵이 안 맞는다
        #   (실측: s085 45.8mm ↔ s130 61.9mm).
        from openarm.agnostic.modules import object_bank as _ob
        _bank = _ob.get(cfg_ns.object_bank)
        self.spec_ids = tuple(sp.id for sp in _bank.specs)
        self._spec_idx = torch.tensor(_bank.assign_indices(n), device=env.device,
                                      dtype=torch.long)
        self.rows: dict[str, list] = {}
        # 에피소드 최고 후보 (env → 행). 점수는 접촉 수 우선.
        self._best_score = torch.full((n,), -1.0, device=env.device)
        self._best_row: dict[int, dict] = {}

    # -- 기하 -------------------------------------------------------------

    def _palm_ee_pose_w(self):
        d = self.env.robot.data
        quat = d.body_quat_w[:, self.env.palm_idx]
        pos = d.body_pos_w[:, self.env.palm_idx]
        return pos + quat_apply(quat, self._offset.expand_as(pos)), quat, pos

    def _tip_forces(self) -> torch.Tensor:
        return self.env._tip_contact_forces()

    # -- 판정 -------------------------------------------------------------

    def terms(self) -> dict[str, torch.Tensor]:
        """캡처 조건을 **항목별로** 돌려준다. AND 하나만 보면 무엇이 막는지 알 수 없다.

        ★조건은 "**컵을 들고 멈춰 있는 상태**" 다 — 이송·goal 도달은 요구하지 않는다.
          pour warm start 가 필요한 것은 그 자세뿐이고, goal 을 요구하면 b1 의 이송
          단계까지 성공해야 해서 수집량이 급감한다(08.31 실측: 8000스텝 0건).
        """
        obj = self.env.object.data
        height = obj.root_pos_w[:, 2] - self.env.object_spawn_pos[:, 2]
        speed = obj.root_lin_vel_w.norm(dim=-1)
        tips = self._tip_forces()
        return {
            "lifted": height >= float(self.cfg.lift_success_height),
            "still": speed <= self.max_speed,
            "grip": (tips > self.thr).sum(dim=-1) >= self.min_grip,
            "_height": height,
            "_speed": speed,
            "_tipmax": tips.max(dim=-1).values,
            "_ncontact": (tips > self.thr).sum(dim=-1).float(),
        }

    def eligible(self) -> torch.Tensor:
        t = self.terms()
        return t["lifted"] & t["still"] & t["grip"]

    def on_done(self, done: torch.Tensor) -> int:
        """에피소드 종료 시 그 env 의 **최고 후보를 확정**하고 카운터를 푼다."""
        n = self.commit(torch.nonzero(done).reshape(-1).tolist())
        self._captured_ep &= ~done
        self._hold[done] = 0
        return n

    def step(self) -> int:
        """에피소드마다 **가장 잘 잡은 순간**을 하나 고른다.

        ★왜 "첫 적격 프레임"이 아니라 "최고 접촉"인가. pour 은 붓는 동안 손가락을
          **전 구간 freeze** 한다 — 뱅크에 담긴 파지 품질이 그대로 붓기 내내의 상한이
          된다. 접촉 2개짜리를 담으면 깊은 tilt 에서 컵이 빠지고, 그때는 되잡을 수단이
          없다. 그래서 접촉 수(동률이면 팁힘 합)가 가장 큰 프레임을 고른다.
        ★후보는 계속 갱신하고 **에피소드 종료 시 확정**한다(`on_done`).
        """
        ok = self.eligible()
        self._hold = torch.where(ok, self._hold + 1, torch.zeros_like(self._hold))
        ready = (self._hold >= self.hold_steps) & (~self._captured_ep)
        ids = torch.nonzero(ready).reshape(-1)
        if ids.numel():
            tips = self._tip_forces()[ids]
            n_c = (tips > self.thr).sum(dim=-1).float()
            # 점수 = 접촉 수 + 팁힘 합의 미세 가중(동률 깨기). 접촉 수가 항상 우선한다.
            score = n_c + 0.001 * tips.sum(dim=-1).clamp(max=100.0)
            better = score > self._best_score[ids]
            upd = ids[better]
            if upd.numel():
                self._best_score[upd] = score[better]
                self._stash(upd)
        return 0

    def _stash(self, ids: torch.Tensor) -> None:
        """후보 갱신 — env 별로 **한 줄만** 들고 있는다(확정은 on_done)."""
        rows = self._make_rows(ids)
        for k, i in enumerate(ids.tolist()):
            self._best_row[i] = {name: arr[k] for name, arr in rows.items()}

    def commit(self, ids) -> int:
        """보유 중인 최고 후보를 뱅크에 확정한다."""
        n = 0
        for i in ids:
            row = self._best_row.get(int(i))
            if row is None:
                continue
            for name, v in row.items():
                self.rows.setdefault(name, []).append(v[None, ...])
            self._best_row.pop(int(i), None)
            self._best_score[int(i)] = -1.0
            n += 1
        return n

    def flush_all(self) -> int:
        """수집 종료 시 아직 안 끝난 에피소드의 후보도 거둔다."""
        return self.commit(list(self._best_row.keys()))

    def _make_rows(self, ids: torch.Tensor) -> dict:
        env, d = self.env, self.env.robot.data
        ee_pos_w, palm_quat, link_pos_w = self._palm_ee_pose_w()
        origin = env.scene.env_origins
        ee_local = (ee_pos_w - origin)[ids]
        quat_wxyz = palm_quat[ids]
        roll, pitch, yaw = euler_xyz_from_quat(quat_wxyz)

        cup_pos = (env.object.data.root_pos_w - origin)[ids]
        cup_quat = env.object.data.root_quat_w[ids]
        tips = self._tip_forces()[ids]
        contact = (tips > self.thr).float()

        # 컵을 palm_ee 프레임에서 본 상대자세 — grasp 인계 공차의 진실원천.
        from pour_traj_capture import relative_pose  # 같은 규약을 한 곳에서만 정의한다
        cup_in_hand = relative_pose(ee_pos_w[ids], quat_wxyz, env.object.data.root_pos_w[ids], cup_quat)

        out: dict = {}

        def add(k, v):
            out[k] = v.detach().float().cpu().numpy()

        add("arm_joint_pos", d.joint_pos[ids][:, env.arm_ids])
        add("hand_joint_pos", d.joint_pos[ids][:, env.hand_ids])
        # ★지령 목표 — 파지력 = kp·(target − q). 측정치만 저장하면 복원 시 파지가 풀린다
        #   (collect_pour_fab_warm_states.py 의 교훈). pour 스키마엔 자리가 없어 진단
        #   채널로 남기되, 복원이 헐거우면 여기부터 본다.
        add("arm_joint_pos_target", d.joint_pos_target[ids][:, env.arm_ids])
        add("hand_joint_pos_target", d.joint_pos_target[ids][:, env.hand_ids])
        add("palm_pose_quat_xyzw",
            torch.cat([ee_local, quat_wxyz[:, [1, 2, 3, 0]]], dim=-1))       # ★wxyz → xyzw
        add("palm_pose_euler_zyx",
            torch.cat([ee_local, torch.stack([yaw, pitch, roll], dim=1)], dim=-1))
        add("cup_pos_local", cup_pos)
        add("cup_quat_wxyz", cup_quat)
        add("num_contacts", contact.sum(dim=-1))
        # --- 진단 전용 (로더는 안 읽는다) ---
        add("palm_link_pos_local", (link_pos_w - origin)[ids])               # 변환 전 원본
        add("cup_in_hand_pose", cup_in_hand)
        add("per_finger_contact", contact)
        add("tip_force", tips)
        add("object_spec_idx", self._spec_idx[ids].float())
        add("env_id", ids.float())
        return out

    # -- 산출 -------------------------------------------------------------

    def stacked(self) -> dict[str, np.ndarray]:
        return {k: np.concatenate(v, axis=0) for k, v in self.rows.items()}

    @property
    def count(self) -> int:
        return sum(len(v) for v in self.rows.get("env_id", []))


def _object_spec_ids(cfg) -> list[str]:
    try:
        from openarm.agnostic.modules import object_bank as ob
        return [s.id for s in ob.get(cfg.object_bank).specs]
    except Exception:  # noqa: BLE001
        return []


def _write_bank(path: Path, data: dict, meta: dict, spec_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    missing = [k for k in BANK_DATASETS if k not in data]
    if missing:
        raise SystemExit(f"뱅크 필수 데이터셋 누락: {missing}")
    try:
        with h5py.File(tmp, "w") as f:
            grp = f.create_group("warm_states")
            for k, v in data.items():
                grp.create_dataset(k, data=v.astype(np.float32), compression="gzip")
            # ★자산 출처 — 로더가 하드 게이트로 쓴다(불일치 시 ValueError).
            f.attrs["robot_usd"] = meta["robot_usd"]
            for k, v in meta.items():
                if k == "robot_usd":
                    continue
                f.attrs[f"meta/{k}" if isinstance(v, (int, float)) else f"prov/{k}"] = v
            if spec_ids:
                f.attrs["prov/object_specs"] = np.array(spec_ids, dtype=h5py.string_dtype())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    resume = Path(args_cli.checkpoint).expanduser().resolve()
    if not resume.is_file():
        raise SystemExit(f"체크포인트가 없다: {resume}")

    agent_cfg = restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=str(resume), workspace_root=str(_HDGP.parent))
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["params"]["seed"]
    # ★복원 **뒤에** 다시 강제 — 덤프의 num_envs 가 되살아난다.
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_act,
                               agent_cfg["params"]["env"].get("obs_groups"),
                               agent_cfg["params"]["env"].get("concate_obs_groups", True))

    vecenv.register("IsaacRlgWrapper",
                    lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
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
    print(f"[WARM] 계약: obs {agent.model.obs_shape} · action {agent.actions_num} · "
          f"RNN {agent.is_rnn}", flush=True)
    print(f"[WARM] palm_ee 변환 {PALM_EE_OFFSET_LOCAL} (r_hl_palm 원점 → palm_ee)", flush=True)

    col = WarmCollector(raw, raw.cfg, args_cli.hold_steps, args_cli.min_grip_fingers,
                        args_cli.contact_threshold, args_cli.max_object_speed)

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    for step in range(args_cli.max_steps):
        with torch.inference_mode():
            action = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, _ = env.step(action)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for h in agent.states:
                    h[:, dones, :] = 0.0
        got = col.on_done(dones.to(raw.device).bool().reshape(-1))   # 에피소드 종료 = 확정
        col.step()                                                    # 후보 갱신만
        if got and col.count % 128 < got:
            print(f"[WARM] {col.count}/{args_cli.target_count} (step {step})", flush=True)
        elif step % 200 == 0:
            # ★0 건일 때도 찍는다. 침묵은 "느린 것"과 "조건이 영영 안 맞는 것"을 구분 못 한다.
            #   ★판별 지표는 **폐쇄지령(syn_close)** 이다 — 정상 재생 0.7 대(b1 ep_10800
            #     실측 0.69~0.76), 죽은 재생 0.26~0.33. 이것 하나로 "정책이 죽었나 /
            #     내 판정이 틀렸나" 가 갈린다.
            t = col.terms()
            n = raw.num_envs
            ex = getattr(raw, "extras", {}) or {}

            def _m(key: str) -> str:
                v = ex.get(key)
                return f"{float(v):.3f}" if v is not None else "—"

            print(f"[WARM] step {step}/{args_cli.max_steps} · 수집 {col.count} | "
                  f"lift {int(t['lifted'].sum())}/{n} · still {int(t['still'].sum())}/{n} · "
                  f"grip {int(t['grip'].sum())}/{n} | "
                  f"높이max {float(t['_height'].max())*1e3:.0f}mm · "
                  f"속도min {float(t['_speed'].min()):.3f} · "
                  f"팁힘max {float(t['_tipmax'].max()):.2f}N | "
                  f"★syn_close {_m('task/syn_close')} · gate {_m('task/close_gate')} · "
                  f"palm_to_cup {_m('task/palm_to_cup')} · cage {_m('task/cage_dist')}",
                  flush=True)
        if col.count >= args_cli.target_count:
            break

    # 아직 안 끝난 에피소드의 후보도 거둔다(끝까지 안 죽은 env 를 버리지 않는다).
    _flushed = col.flush_all()
    if _flushed:
        print(f"[WARM] 미종료 에피소드 후보 {_flushed} 건 확정", flush=True)

    if col.count < args_cli.target_count:
        # ★부분 뱅크를 저장하지 않는다. 반쯤 찬 캐시를 뒤에 학습이 조용히 집어먹는다.
        raise SystemExit(
            f"[WARM] ABORT: {args_cli.max_steps} 스텝에 {col.count}/{args_cli.target_count} — "
            "정책 성공률이 부족하거나 판정 조건이 틀렸다. 뱅크를 저장하지 않는다.")

    data = col.stacked()
    # 프로필은 cfg 의 이름으로 직접 해석한다 — env 속성 이름에 기대지 않는다.
    from openarm.agnostic.tasks.grasp_s2r.robot_profiles import PROFILES
    profile = PROFILES[raw.cfg.profile_name]
    pbox_min = tuple(profile.palm_box_min)
    pbox_max = tuple(profile.palm_box_max)
    meta = {
        "robot_usd": str(raw.cfg.robot_cfg.spawn.usd_path),
        "object_spawn_z": float(raw.cfg.object_spawn_z),
        "palm_min_x": pbox_min[0], "palm_min_y": pbox_min[1], "palm_min_z": pbox_min[2],
        "palm_max_x": pbox_max[0], "palm_max_y": pbox_max[1], "palm_max_z": pbox_max[2],
        "palm_ee_offset_x": PALM_EE_OFFSET_LOCAL[0],
        "palm_ee_offset_y": PALM_EE_OFFSET_LOCAL[1],
        "palm_ee_offset_z": PALM_EE_OFFSET_LOCAL[2],
        "checkpoint": resume.name,
        "checkpoint_sha256": _sha256(resume),
        "task": args_cli.task,
        "git_commit": _git_commit(),
        "object_bank": str(getattr(raw.cfg, "object_bank", "")),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "palm_frame": "palm_ee (r_hl_palm + R*offset)",
    }
    out = Path(args_cli.out).expanduser().resolve()
    _write_bank(out, data, meta, _object_spec_ids(raw.cfg))

    cih = data["cup_in_hand_pose"]
    dist = np.linalg.norm(cih[:, :3], axis=1) * 1e3
    print("\n" + "=" * 68)
    print(f"[WARM] 저장 {len(dist)} 상태 → {out}")
    print(f"[WARM] robot_usd = {meta['robot_usd']}")
    print(f"[WARM] object_spawn_z = {meta['object_spawn_z']:.4f} · bank = {meta['object_bank']}")
    print(f"[WARM] 컵–손(palm_ee) 거리  평균 {dist.mean():.1f} mm · "
          f"범위 [{dist.min():.1f}, {dist.max():.1f}] · 표준편차 {dist.std():.1f}")
    print(f"[WARM] 손가락 접촉 수 평균 {data['num_contacts'].mean():.2f}")
    spec_ids = col.spec_ids
    if spec_ids:
        idx = data["object_spec_idx"].astype(int)
        print(f"\n{'물체':<18}{'개수':>6}{'컵-손 평균(mm)':>16}{'표준편차':>10}")
        for k, sid in enumerate(spec_ids):
            m = idx == k
            if m.any():
                print(f"{sid:<18}{int(m.sum()):>6}{dist[m].mean():>16.1f}{dist[m].std():>10.1f}")
    print("=" * 68)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
