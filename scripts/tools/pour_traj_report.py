#!/usr/bin/env python3
"""pour 궤적 HDF5 → 사람이 읽는 Markdown 리포트. **Isaac 무의존.**

왜 필요한가. 궤적의 진실원천은 HDF5 지만, 실제로 판단할 때 매번 h5py 를 여는 것은
현실적이지 않다. 그렇다고 손으로 옮겨 적으면 그 순간 두 벌이 되고 두 벌은 반드시 갈린다.
그래서 **파일에서 뽑아 렌더링**한다 — 리포트가 낡으면 다시 돌리면 된다.

담는 것은 s2r 에서 실제로 쓰이는 것만:
  · 출처(체크포인트·커밋·재현 명령) — 이 궤적이 무엇인지
  · pour 초기 세팅 관절값 **전량** — 실기가 이송해야 할 목표
  · 관절 속도 peak/p95 — 실기 프로필 한계와 대볼 값 (넘으면 rate-limit 이 아니라 시간을 늘린다)
  · 지령↔측정 추종오차 — 관절 재생이 성립하는지
  · 컵–손 상대자세 — grasp 인계가 맞춰야 할 공차

사용: python pour_traj_report.py <traj.hdf5> [-o out.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pour_traj_io as PT  # noqa: E402

#: 속도 요약에 쓰는 백분위. peak 만 보면 순간 스파이크에 속고, p95 만 보면 넘침을 놓친다.
SPEED_PERCENTILE = 95
#: 표에 관절을 몇 개까지 늘어놓을지 (그 이상은 그룹 요약만).
MAX_JOINT_ROWS = 40


# ---------------------------------------------------------------------------
# 계산 (순수)
# ---------------------------------------------------------------------------

def tilt_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """포즈 시계열의 up 축이 +z 에서 벗어난 각도. 붓기 깊이를 이 값으로 읽는다."""
    x, y = quat_wxyz[:, 1], quat_wxyz[:, 2]
    return np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def episode_rows(meta: PT.TrajMeta, episodes) -> list[dict]:
    rows = []
    for i, ep in enumerate(episodes):
        cup = ep.arrays.get("cup_in_hand_pose")
        dist = np.linalg.norm(cup[:, :3], axis=1) if cup is not None else None
        src = ep.arrays.get("source_cup_pose")
        rows.append({
            "ep": i,
            "steps": ep.n_steps,
            "seconds": ep.n_steps * meta.dt,
            "bead": ep.bead_frac,
            "spill": ep.bead_spill,
            "cup_in_hand_t0_mm": None if dist is None else dist[0] * 1e3,
            "cup_in_hand_drift_mm": None if dist is None else (dist.max() - dist.min()) * 1e3,
            "max_tilt_deg": None if src is None else float(tilt_deg(src[:, 3:]).max()),
        })
    return rows


def joint_speed_stats(meta: PT.TrajMeta, episodes) -> list[dict]:
    """관절별 |속도| peak · p95 (rad/s). 실기 프로필 한계와 대보는 값이다."""
    if "joint_vel" in meta.missing_channels:
        return []
    speeds = np.abs(np.concatenate([e.arrays["joint_vel"] for e in episodes], axis=0))
    peak = speeds.max(axis=0)
    p95 = np.percentile(speeds, SPEED_PERCENTILE, axis=0)
    return [
        {"joint": name, "peak": float(peak[i]), "p95": float(p95[i])}
        for i, name in enumerate(meta.joint_names)
    ]


def tracking_stats(meta: PT.TrajMeta, episodes) -> dict:
    """지령↔측정 오차. 관절 재생의 진실원천이 지령이므로 이 격차를 안다."""
    out: dict = {}
    if "joint_pos_target" not in meta.missing_channels:
        err = np.concatenate(
            [np.abs(e.arrays["joint_pos_target"] - e.arrays["joint_pos"]) for e in episodes],
            axis=0,
        )
        out["joint_mean_rad"] = float(err.mean())
        out["joint_max_rad"] = float(err.max())
    if "right_palm_cmd_pose" not in meta.missing_channels:
        err = np.concatenate([
            np.linalg.norm(e.arrays["right_palm_cmd_pose"][:, :3]
                           - e.arrays["right_palm_pose"][:, :3], axis=1)
            for e in episodes
        ])
        out["palm_mean_m"] = float(err.mean())
        out["palm_max_m"] = float(err.max())
    return out


def group_ranges(meta: PT.TrajMeta, episodes) -> list[dict]:
    """관절 그룹별 이동폭. 정지 그룹(=왼팔 고정)을 한눈에 드러낸다."""
    groups = {
        "우팔": meta.right_arm_joint_names,
        "우손": meta.right_hand_joint_names,
        "좌팔": meta.left_arm_joint_names,
        "좌그리퍼": meta.left_gripper_joint_names,
    }
    q = np.concatenate([e.arrays["joint_pos"] for e in episodes], axis=0)
    rows = []
    for label, names in groups.items():
        cols = list(PT.column_index(meta.joint_names, names))
        span = q[:, cols].max(axis=0) - q[:, cols].min(axis=0)
        rows.append({"group": label, "n": len(names),
                     "max_span_rad": float(span.max()), "mean_span_rad": float(span.mean())})
    return rows


# ---------------------------------------------------------------------------
# 렌더링 (순수)
# ---------------------------------------------------------------------------

def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _vec(values: np.ndarray, digits: int = 4) -> str:
    return "[" + ", ".join(f"{v:.{digits}f}" for v in np.asarray(values).reshape(-1)) + "]"


def render_markdown(meta: PT.TrajMeta, episodes, source: Path) -> str:
    rows = episode_rows(meta, episodes)
    best = max(range(len(episodes)), key=lambda i: (episodes[i].bead_frac, -episodes[i].n_steps))
    init = PT.init_state(meta, episodes[best])

    out: list[str] = []
    add = out.append

    add(f"# pour 궤적 — `{source.name}`\n")
    add("> 이 문서는 `scripts/tools/pour_traj_report.py` 가 HDF5 에서 **생성**한다. "
        "손으로 고치지 말 것 — 값이 두 벌이 된다.\n")

    add("## 출처\n")
    add(_table(["항목", "값"], [
        ["task", f"`{meta.task_id}`"],
        ["checkpoint", f"`{meta.checkpoint}`"],
        ["checkpoint sha256", f"`{meta.checkpoint_sha256[:16]}…`" if meta.checkpoint_sha256 else "—"],
        ["코드 커밋", f"`{meta.git_commit}`" if meta.git_commit else "—"],
        ["robot USD", f"`{Path(meta.robot_usd).name}`" if meta.robot_usd else "—"],
        ["기록 시각", meta.recorded_at],
        ["dt / decimation", f"{meta.dt:.6f} s ({1/meta.dt:.0f} Hz) / {meta.decimation}"],
        ["bead 수", str(meta.num_beads)],
        ["관절 수", f"{len(meta.joint_names)} "
                  f"(우팔 {len(meta.right_arm_joint_names)} · 우손 {len(meta.right_hand_joint_names)} · "
                  f"좌팔 {len(meta.left_arm_joint_names)} · 좌그리퍼 {len(meta.left_gripper_joint_names)})"],
        ["액션 차원", str(episodes[0].arrays["action"].shape[1]) if "action" in episodes[0].arrays else "—"],
        ["결손 채널", ", ".join(f"`{c}`" for c in meta.missing_channels) or "없음"],
    ]))
    add("")

    add("## 에피소드\n")
    add(_table(["ep", "steps", "초", "bead", "spill", "컵–손 t0 (mm)", "드리프트 (mm)", "최대 기울기 (°)"],
               [[str(r["ep"]), str(r["steps"]), f"{r['seconds']:.2f}",
                 f"{r['bead']:.2f}", f"{r['spill']:.2f}",
                 "—" if r["cup_in_hand_t0_mm"] is None else f"{r['cup_in_hand_t0_mm']:.1f}",
                 "—" if r["cup_in_hand_drift_mm"] is None else f"{r['cup_in_hand_drift_mm']:.1f}",
                 "—" if r["max_tilt_deg"] is None else f"{r['max_tilt_deg']:.1f}"] for r in rows]))
    beads = [r["bead"] for r in rows]
    add(f"\nbead 평균 **{np.mean(beads):.3f}** · 범위 [{min(beads):.2f}, {max(beads):.2f}] · "
        f"길이 평균 {np.mean([r['steps'] for r in rows]):.0f} step\n")

    add(f"## pour 초기 세팅 — ep_{best:03d} (bead {episodes[best].bead_frac:.2f})\n")
    add("실기는 재생 시작 **전에** 아래 자세에 도달해 있어야 한다. "
        "`*_q` 는 측정(파킹 목표), `*_q_cmd` 는 재생 첫 지령이며 **둘은 같지 않다**.\n")
    for label, key, names in (
        ("우팔", "right_arm_q", meta.right_arm_joint_names),
        ("좌팔", "left_arm_q", meta.left_arm_joint_names),
        ("좌그리퍼", "left_gripper_q", meta.left_gripper_joint_names),
        ("우손", "right_hand_q", meta.right_hand_joint_names),
    ):
        if key not in init:
            continue
        add(f"**{label}** (`{names[0]}` … `{names[-1]}`, rad)\n")
        add("```")
        add(f"{key:<16} {_vec(init[key])}")
        if f"{key}_cmd" in init:
            add(f"{key + '_cmd':<16} {_vec(init[f'{key}_cmd'])}")
        add("```\n")

    add("**포즈** (pos xyz [m] + quat wxyz, env 원점 = robot base 기준)\n")
    pose_rows = []
    for key, label in (("right_palm_pose", "우 palm_ee"), ("left_ee_pose", "좌 EE"),
                       ("source_cup_pose", "소스컵"), ("target_cup_pose", "타겟컵"),
                       ("cup_in_hand_pose", "컵–손 상대")):
        if key in init:
            pose_rows.append([label, f"`{key}`", _vec(init[key])])
    add(_table(["", "채널", "값"], pose_rows))
    add("")

    add("## 관절 그룹별 이동폭\n")
    add(_table(["그룹", "관절 수", "최대 이동폭 (rad)", "평균 이동폭 (rad)"],
               [[r["group"], str(r["n"]), f"{r['max_span_rad']:.5f}", f"{r['mean_span_rad']:.5f}"]
                for r in group_ranges(meta, episodes)]))
    add("")

    speeds = joint_speed_stats(meta, episodes)
    if speeds:
        add("## 관절 속도 (전 에피소드 합산, |rad/s|)\n")
        add("실기 프로필 한계와 대볼 값이다. ★넘으면 rate-limit 클램프가 아니라 "
            "**시간을 늘려서**(`rate_scale`) 맞춘다 — 클램프는 궤적 모양을 뭉갠다.\n")
        moving = [s for s in speeds if s["peak"] > 1e-6]
        add(f"움직인 관절 {len(moving)} / {len(speeds)}개. "
            f"전체 peak **{max(s['peak'] for s in speeds):.3f}** · "
            f"p{SPEED_PERCENTILE} 최대 **{max(s['p95'] for s in speeds):.3f}**\n")
        shown = sorted(moving, key=lambda s: -s["peak"])[:MAX_JOINT_ROWS]
        add(_table(["관절", "peak", f"p{SPEED_PERCENTILE}"],
                   [[f"`{s['joint']}`", f"{s['peak']:.3f}", f"{s['p95']:.3f}"] for s in shown]))
        add("")

    track = tracking_stats(meta, episodes)
    if track:
        add("## 지령 ↔ 측정 추종오차\n")
        trows = []
        if "joint_mean_rad" in track:
            trows.append(["관절 `joint_pos_target` − `joint_pos`",
                          f"{track['joint_mean_rad']:.4f} rad", f"{track['joint_max_rad']:.4f} rad"])
        if "palm_mean_m" in track:
            trows.append(["palm_ee 지령 − 실제",
                          f"{track['palm_mean_m']:.4f} m", f"{track['palm_max_m']:.4f} m"])
        add(_table(["항목", "평균", "최대"], trows))
        add("")

    add("## 채널 명세 (에피소드당)\n")
    widths = {k: v.shape[1] for k, v in episodes[0].arrays.items()}
    add(_table(["채널", "shape", "설명"], [
        [f"`{k}`", f"[T, {widths[k]}]", d] for k, d in [
            ("joint_pos", "측정 관절각"),
            ("joint_vel", "측정 관절속도"),
            ("joint_pos_target", "**지령** 관절각 — JTC 로 나가는 것"),
            ("action", "정책 액션"),
            ("right_palm_pose", "우 palm_ee 실제 포즈"),
            ("right_palm_cmd_pose", "우 palm_ee Fabrics 지령 포즈"),
            ("left_ee_pose", "좌 EE(`l_hl_gripper_base`) 실제 포즈"),
            ("left_ee_cmd_pose", "좌 EE 지령 포즈 (both 트랙 전용)"),
            ("source_cup_pose", "소스컵 포즈"),
            ("target_cup_pose", "타겟컵 포즈"),
            ("cup_in_hand_pose", "소스컵을 palm_ee 프레임에서 본 상대자세"),
        ] if k in widths
    ]))
    add("\n포즈는 전부 `[x, y, z, qw, qx, qy, qz]`, env 원점 기준. "
        "`ep_XXX/init` 그룹에 t=0 단면이 중복 저장돼 있다.\n")

    add("## 관절 순서\n")
    add("```")
    for i in range(0, len(meta.joint_names), 6):
        add("  " + " ".join(f"{j:>3}:{n:<18}" for j, n in
                            enumerate(meta.joint_names[i:i + 6], start=i)))
    add("```")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="pour 궤적 HDF5 → Markdown 리포트")
    parser.add_argument("traj", type=Path, help="pour_traj_io v2 형식 HDF5")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="출력 경로 (기본: 입력과 같은 폴더의 .md)")
    args = parser.parse_args()

    meta, episodes = PT.read_traj(args.traj)
    problems = PT.validate(meta, episodes)
    if problems:
        raise SystemExit("궤적이 계약을 어긴다:\n  " + "\n  ".join(problems))

    out = args.out or args.traj.with_suffix(".md")
    out.write_text(render_markdown(meta, episodes, args.traj), encoding="utf-8")
    print(f"[리포트] {out}  ({len(episodes)} 에피소드)")


if __name__ == "__main__":
    main()
