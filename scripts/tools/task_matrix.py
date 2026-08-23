#!/usr/bin/env python3
"""등록된 태스크가 **실제로 부팅 가능한지** 정적으로 판정한다.

Isaac Sim 도 GPU 도 쓰지 않는다 — 프로필 레지스트리 3종이 전부 isaaclab 을 import 하지
않는 순수 데이터이기 때문이다. 그래서 학습이 GPU 를 다 쓰고 있는 중에도, CI 에서도 돈다.

**왜 필요한가.** 태스크가 gym 에 등록됐다는 사실은 부팅 가능하다는 뜻이 아니다. 실측 사례:
  · `open-sens_l_grasp_sensor` 4종은 등록되고 `_setup_fabrics` 에서 RuntimeError 로 죽는다
    (`fabric_class=None`).
  · `open-*_b_pour_fab` 12종은 warm 뱅크 파일이 없어 부팅하지 못한다.
  · 씬 USD `assets/env/usd/env.usd` 는 최근까지 git 미추적이라 새 머신에서 전부 죽었다.
이 세 부류는 **Isaac 을 띄워야만 드러났고**, 그때는 이미 GPU 를 잡은 뒤였다.

판정은 두 단계다.
  BLOCK — 부팅이 불가능하다. 하나라도 있으면 종료코드 1.
  WARN  — 부팅은 되지만 신뢰할 수 없다(예: palm 박스를 다른 로봇에서 물려받아 미실측).

사용:
    python3 scripts/tools/task_matrix.py              # 표 출력, BLOCK 있으면 exit 1
    python3 scripts/tools/task_matrix.py --write-doc  # docs/TASK_MATRIX.md 갱신
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HDGP_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = HDGP_ROOT / "assets"
AGNOSTIC_DIR = HDGP_ROOT / "source/openarm/openarm/agnostic"
GRIPPER_LEFT_DIR = HDGP_ROOT / "source/openarm/openarm/gripper/left/grasp_sensor"
FABRICS_ROOT = HDGP_ROOT / "source/FABRICS/src/fabrics_sim"
FABRIC_URDF_DIR = FABRICS_ROOT / "models/robots/urdf"
FABRIC_WORLD_DIR = FABRICS_ROOT / "worlds"
DOC_PATH = HDGP_ROOT / "docs/TASK_MATRIX.md"

if str(HDGP_ROOT / "source/openarm") not in sys.path:
    sys.path.insert(0, str(HDGP_ROOT / "source/openarm"))

_TRACKED_CACHE = None

BLOCK = "BLOCK"
WARN = "WARN"
OK = "OK"

# gym id 접미사 — 세 agnostic 태스크가 공유하는 규약(train/play × mlp/lstm).
AGNOSTIC_SUFFIXES: tuple[str, ...] = ("", "-play", "-lstm", "-play-lstm")


# =============================================================================
# 자료구조
# =============================================================================
@dataclass(frozen=True)
class Gate:
    """부팅 전제조건 하나에 대한 판정."""

    name: str
    ok: bool
    severity: str          # BLOCK | WARN
    detail: str


@dataclass(frozen=True)
class TaskRow:
    task: str
    variant: str                    # 프로필명 / 쌍 이름 / 제어기 변형
    gym_ids: tuple[str, ...]
    gates: tuple[Gate, ...]
    canonical: bool = True          # 같은 태스크의 여러 변형 중 정본인가
    contract: str = ""              # 알려진 경우의 act/obs/state (참고용)

    @property
    def blockers(self) -> tuple[Gate, ...]:
        return tuple(g for g in self.gates if not g.ok and g.severity == BLOCK)

    @property
    def warns(self) -> tuple[Gate, ...]:
        return tuple(g for g in self.gates if not g.ok and g.severity == WARN)

    @property
    def verdict(self) -> str:
        if self.blockers:
            return BLOCK
        return WARN if self.warns else OK


# =============================================================================
# 게이트 헬퍼 — 존재하지 않는 것을 "없음"으로 조용히 넘기지 않는다
# =============================================================================
def gate_path(name: str, path: Path, severity: str = BLOCK,
              collect: list | None = None) -> Gate:
    if collect is not None:
        collect.append(path)
    exists = path.exists()
    try:
        shown = path.relative_to(HDGP_ROOT)
    except ValueError:
        shown = path
    return Gate(name, exists, severity, f"{'있음' if exists else '없음'}: {shown}")


def _tracked_paths() -> frozenset:
    """git 이 추적 중인 파일 집합 (저장소 루트 기준 상대경로)."""
    global _TRACKED_CACHE
    if _TRACKED_CACHE is None:
        out = subprocess.run(
            ["git", "-C", str(HDGP_ROOT), "ls-files", "-z", "assets"],
            capture_output=True, text=True, check=True,
        ).stdout
        _TRACKED_CACHE = frozenset(x for x in out.split("\0") if x)
    return _TRACKED_CACHE


def gate_assets_tracked(paths: list) -> Gate:
    """자산이 git 에 들어 있는가.

    ★파일이 **존재하는데 미추적**이면 이 머신에서만 돈다. 실측: `assets/env/usd/env.usd` 와
      `assets/visdex_objects/USD/cup_middle/` 이 정확히 그 상태로 세 머신에 수동 복사돼
      있었고, 새 머신은 clone 만으로는 부팅하지 못했다. 존재 검사만으로는 절대 안 잡힌다.
    """
    tracked = _tracked_paths()
    missing = []
    for path in paths:
        try:
            rel = str(path.relative_to(HDGP_ROOT))
        except ValueError:
            continue
        if rel not in tracked:
            missing.append(rel)
    return Gate(
        "assets_tracked", not missing, WARN,
        "" if not missing else "git 미추적(이 머신에서만 동작): " + ", ".join(sorted(missing)),
    )


def gate_true(name: str, value: bool, severity: str, detail_false: str) -> Gate:
    return Gate(name, bool(value), severity, "" if value else detail_false)


def fabric_gates(fabric_class: str | None, robot_dir: str | None) -> list[Gate]:
    """Fabrics 로만 도는 태스크의 팔 제어기 전제조건."""
    have = fabric_class is not None and robot_dir is not None
    gates = [gate_true(
        "fabric_class", have, BLOCK,
        "fabric_class/fabric_robot_dir 가 None — env 가 RuntimeError 로 멈춘다",
    )]
    if have:
        base = FABRIC_URDF_DIR / robot_dir
        gates.append(gate_path("fabric_urdf", base / f"{robot_dir}.urdf"))
        # ★manifest 는 **런타임 의존이 아니다** — fabrics_sim 도 태스크도 읽지 않는다(grep 0건).
        #   f5baea0 이 재생성한 4종에만 있고 레거시 3종(openarm_tesollo, _left, _sensor)에는
        #   의도적으로 없다(재생성 금지 대상). 부팅을 막으면 거짓 BLOCK 이 된다 → WARN.
        gates.append(gate_path("fabric_manifest",
                               base / f"{robot_dir}_manifest.yaml", severity=WARN))
    return gates


def agent_yaml_gates(config_dir: Path, names: tuple[str, ...]) -> list[Gate]:
    return [gate_path(f"agent:{n}", config_dir / "agents" / n) for n in names]


def _agnostic_ids(short: str, side: str, task_slug: str) -> tuple[str, ...]:
    return tuple(f"open-{short}_{side}_{task_slug}{s}" for s in AGNOSTIC_SUFFIXES)


# =============================================================================
# 태스크별 행 구성
# =============================================================================
def build_grasp_sensor_rows() -> list[TaskRow]:
    """agnostic/tasks/grasp_sensor — 자체 robot_profiles.py (프로필 2)."""
    from openarm.agnostic.tasks.grasp_sensor import robot_profiles as rp

    cfg_dir = AGNOSTIC_DIR / "tasks/grasp_sensor/config"
    # config/__init__.py 의 _CFGS 키 = gym id 의 로봇+side 슬롯.
    tags = {"tesollo_right": "sens_r", "gripper_left": "sens_l"}
    contracts = {"tesollo_right": "23 / 114 / 121", "gripper_left": "7 / 48 / 55"}

    rows: list[TaskRow] = []
    for name, profile in sorted(rp.PROFILES.items()):
        short, side = tags[name].split("_")
        assets: list = []
        gates = [gate_path("robot_usd", ASSETS_DIR / profile.usd_relpath, collect=assets)]
        gates += fabric_gates(profile.fabric_class, profile.fabric_robot_dir)
        gates.append(gate_path("scene_usd", ASSETS_DIR / "env/usd/env.usd", collect=assets))
        gates.append(gate_path("object_usd", ASSETS_DIR / "cup/cup_big_rl.usd", collect=assets))
        gates += agent_yaml_gates(cfg_dir, ("rl_games_ppo_cfg.yaml",
                                            "rl_games_ppo_lstm_cfg.yaml"))
        gates.append(gate_true(
            "palm_box_verified", profile.palm_box_verified, WARN,
            "palm 박스를 probe 로 실측하지 않았다 — 다른 로봇 값을 물려받았을 수 있다",
        ))
        gates.append(gate_assets_tracked(assets))
        rows.append(TaskRow(
            task="agnostic/grasp_sensor", variant=name,
            gym_ids=_agnostic_ids(short, side, "grasp_sensor"),
            gates=tuple(gates), contract=contracts[name],
        ))
    return rows


def build_grasp_lift_fabric_rows() -> list[TaskRow]:
    """agnostic/tasks/grasp_lift_fabric — 공용 modules/robots.py (프로필 8)."""
    from openarm.agnostic.modules import object_bank as ob
    from openarm.agnostic.modules import robots as rb

    cfg_dir = AGNOSTIC_DIR / "tasks/grasp_lift_fabric/config"
    bank = ob.get("single_cup")            # env_cfg 기본값

    rows: list[TaskRow] = []
    for name, profile in sorted(rb.PROFILES.items()):
        assets: list = [Path(spec.usd_path) for spec in bank.specs]
        gates = [gate_path("robot_usd", ASSETS_DIR / profile.asset.usd_relpath,
                           collect=assets)]
        gates += fabric_gates(profile.fabric_class, profile.fabric_robot_dir)
        gates.append(gate_path("scene_usd", ASSETS_DIR / "env/usd/env.usd", collect=assets))
        for missing in bank.missing_files():
            gates.append(Gate("object_usd", False, BLOCK, f"없음: {missing}"))
        if not bank.missing_files():
            gates.append(Gate("object_usd", True, BLOCK,
                              f"있음: 뱅크 {bank.name} ({len(bank)}종)"))
        gates += agent_yaml_gates(cfg_dir, ("rl_games_ppo_cfg.yaml",
                                            "rl_games_ppo_lstm_cfg.yaml"))
        gates.append(gate_true(
            "palm_box_verified", profile.palm_box_verified, WARN,
            "palm 박스 미실측 — bis_right 값을 물려받으면 62% 가 도달 불가였다(실측)",
        ))
        gates.append(gate_true(
            "probe_verified", profile.probe_verified, WARN,
            "물리/IK probe 미통과 — 선언만 된 프로필",
        ))
        gates.append(gate_assets_tracked(assets))
        rows.append(TaskRow(
            task="agnostic/grasp_lift_fabric", variant=name,
            gym_ids=_agnostic_ids(profile.asset.short, profile.side, "grasp_lift_fab"),
            gates=tuple(gates),
            contract="19 / 121 / 127" if name == "bis_right" else "",
        ))
    return rows


def build_pour_fabric_rows() -> list[TaskRow]:
    """agnostic/tasks/pour_fabric — 양팔 쌍 3종. warm 뱅크가 필수 전제다."""
    from openarm.agnostic.tasks.pour_fabric import bimanual as bm

    cfg_dir = AGNOSTIC_DIR / "tasks/pour_fabric/config"
    data_dir = HDGP_ROOT / "data"

    rows: list[TaskRow] = []
    for name, pair in sorted(bm.PAIRS.items()):
        assets: list = []
        gates = [gate_path("robot_usd", ASSETS_DIR / pair.asset.usd_relpath, collect=assets)]
        gates += fabric_gates(pair.source.fabric_class, pair.source.fabric_robot_dir)
        gates += fabric_gates(pair.receiver.fabric_class, pair.receiver.fabric_robot_dir)
        gates.append(gate_path("object_usd", ASSETS_DIR / "cup/cup_big_sdf.usd",
                               collect=assets))
        # ★require_warm_bank=True — 두 파일 중 하나만 없어도 부팅하지 못한다.
        src = data_dir / f"pour_fab_warm_{pair.name}_src.hdf5"
        rcv = data_dir / f"pour_fab_warm_{pair.name}_rcv.hdf5"
        have = src.exists() and rcv.exists()
        gates.append(Gate(
            "warm_bank", have, BLOCK,
            "있음" if have else f"없음: {src.name} / {rcv.name} (수집 필요)",
        ))
        gates += agent_yaml_gates(cfg_dir, ("rl_games_ppo_cfg.yaml",
                                            "rl_games_ppo_lstm_cfg.yaml"))
        gates.append(gate_assets_tracked(assets))
        rows.append(TaskRow(
            task="agnostic/pour_fabric", variant=name,
            gym_ids=_agnostic_ids(pair.name, "b", "pour_fab"),
            gates=tuple(gates),
            contract="9 / 210 / 229" if name == "bis" else "",
        ))
    return rows


def build_gripper_left_rows() -> list[TaskRow]:
    """gripper/left/grasp_sensor — ManagerBased, 제어기 3변형. 정본은 _fab."""
    from openarm.gripper.left.grasp_sensor import grasp_left_preset as preset

    cfg_dir = GRIPPER_LEFT_DIR / "config"
    robot_usd = ASSETS_DIR / "robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd"
    variants = (
        ("joint", "", "rl_games_ppo_cfg.yaml", "8 / 36 / —", False),
        ("ik", "_ik", "rl_games_ppo_cfg.yaml", "7 / 35 / —", False),
        ("fab", "_fab", "rl_games_ppo_fab_cfg.yaml", "7 / 35 / —", True),
    )

    rows: list[TaskRow] = []
    for variant, id_suffix, agent_yaml, contract, canonical in variants:
        assets: list = []
        gates = [
            gate_path("robot_usd", robot_usd, collect=assets),
            gate_path("scene_usd", ASSETS_DIR / preset.ENV_USD_REL, collect=assets),
            gate_path("object_usd", ASSETS_DIR / "cup" / preset.CUP_USD_NAME,
                      collect=assets),
        ]
        gates += agent_yaml_gates(cfg_dir, (agent_yaml,))
        if variant == "fab":
            gates += fabric_gates("OpenArmGripperLeftPoseFabric", preset.FABRIC_ROBOT_DIR)
            gates.append(gate_path(
                "fabric_world",
                FABRIC_WORLD_DIR / f"{preset.FABRIC_WORLD_FILENAME}.yaml",
            ))
        gates.append(gate_assets_tracked(assets))
        rows.append(TaskRow(
            task="gripper/left/grasp_sensor", variant=variant,
            gym_ids=(f"open-grip_l_grasp_sensor{id_suffix}",
                     f"open-grip_l_grasp_sensor{id_suffix}-play"),
            gates=tuple(gates), canonical=canonical, contract=contract,
        ))
    return rows


def build_rows() -> tuple[TaskRow, ...]:
    return tuple(
        build_grasp_sensor_rows()
        + build_grasp_lift_fabric_rows()
        + build_pour_fabric_rows()
        + build_gripper_left_rows()
    )


# =============================================================================
# 출력
# =============================================================================
_MARK = {OK: "OK   ", WARN: "WARN ", BLOCK: "BLOCK"}


def _reasons(row: TaskRow) -> str:
    parts = [f"{g.name}: {g.detail}" for g in row.blockers]
    parts += [f"({g.name})" for g in row.warns]
    return " · ".join(parts)


def render_table(rows: tuple[TaskRow, ...]) -> str:
    out = [f"{'판정':6}{'태스크':30}{'구성':16}{'ID':>4}  사유"]
    out.append("-" * 110)
    for row in rows:
        out.append(
            f"{_MARK[row.verdict]:6}{row.task:30}{row.variant:16}"
            f"{len(row.gym_ids):>4}  {_reasons(row)}"
        )
    n_block = sum(1 for r in rows if r.verdict == BLOCK)
    n_warn = sum(1 for r in rows if r.verdict == WARN)
    ids_ok = sum(len(r.gym_ids) for r in rows if r.verdict != BLOCK)
    ids_block = sum(len(r.gym_ids) for r in rows if r.verdict == BLOCK)
    out.append("-" * 110)
    out.append(
        f"구성 {len(rows)}개 — OK {len(rows) - n_block - n_warn} / WARN {n_warn} / "
        f"BLOCK {n_block}   |   gym id 부팅가능 {ids_ok} / 불가 {ids_block}"
    )
    return "\n".join(out)


def render_markdown(rows: tuple[TaskRow, ...]) -> str:
    lines = [
        "# 태스크 부팅 가능성 매트릭스",
        "",
        "> `scripts/tools/task_matrix.py` 가 생성한다. **직접 편집하지 말 것.**",
        "> 정적 판정이라 Isaac Sim·GPU 없이 돈다 — 학습이 GPU 를 쓰는 중에도 갱신 가능하다.",
        "",
        "`BLOCK` = 부팅 불가(전제조건 결손). `WARN` = 부팅은 되나 검증 미비.",
        "",
        "| 판정 | 태스크 | 구성 | gym id | act/obs/state | 사유 |",
        "|---|---|---|---|---:|---|",
    ]
    for row in rows:
        canon = "" if row.canonical else " *(비정본)*"
        lines.append(
            f"| **{row.verdict}** | `{row.task}` | `{row.variant}`{canon} | "
            f"{len(row.gym_ids)} | {row.contract or '—'} | {_reasons(row) or '—'} |"
        )
    lines += [
        "",
        "## 게이트 전문",
        "",
    ]
    for row in rows:
        lines.append(f"### `{row.task}` / `{row.variant}`")
        lines.append("")
        lines.append("gym id: " + ", ".join(f"`{i}`" for i in row.gym_ids))
        lines.append("")
        for gate in row.gates:
            mark = "✅" if gate.ok else ("⛔" if gate.severity == BLOCK else "⚠️")
            lines.append(f"- {mark} `{gate.name}` — {gate.detail or '통과'}")
        lines.append("")
    lines += [
        "## 범위 밖 (이 매트릭스가 판정하지 않는 것)",
        "",
        "- **런타임 거동** — 부팅 후 물리·보상·수렴은 정적으로 알 수 없다.",
        "- **perception_plus_plus 연동** — 저장소가 이 머신에 없다(vision-3090 별도 repo).",
        "  sim 평가는 물체 pose 를 env GT 로 직독하고, `/cup_pose` ROS 경로는 실기 전용이다.",
        "  `sim2real/config/global_camera_extrinsics.yaml` 은 아직 PLACEHOLDER 다.",
        "- **체크포인트 계약** — 학습된 정책의 실제 차원은 런의 `params/env.yaml` 이 진실원천이다",
        "  (`scripts/tools/policy_contract.py`). 위 표의 act/obs/state 는 참고값이다.",
        "",
    ]
    return "\n".join(lines)


def exit_code(rows) -> int:
    return 1 if any(r.blockers for r in rows) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-doc", action="store_true",
                    help=f"{DOC_PATH.relative_to(HDGP_ROOT)} 갱신")
    args = ap.parse_args()

    rows = build_rows()
    print(render_table(rows))
    if args.write_doc:
        DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
        DOC_PATH.write_text(render_markdown(rows), encoding="utf-8")
        print(f"\n→ {DOC_PATH.relative_to(HDGP_ROOT)}")
    return exit_code(rows)


if __name__ == "__main__":
    raise SystemExit(main())
