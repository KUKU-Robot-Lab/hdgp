"""델타 맵 — STATE−CAMERA 비교 리포트 (순수 함수, Isaac 무관).

설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp2-camera-design.md §4.4
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 서버 headless 대비
import matplotlib.pyplot as plt


def build_delta(cells_state: list[dict], cells_camera: list[dict]) -> list[dict]:
    """STATE−CAMERA 셀 대조 후 델타 계산.

    Args:
        cells_state: report.aggregate 출력 (cell_idx, x, y, success_rate, lifted_rate, perception_fail_rate, ...)
        cells_camera: 동일 스키마 (perception_fail_rate는 사용됨)

    Returns:
        각 행: cell_idx, x, y, delta_success (state−camera), delta_lifted, perception_fail_rate (from camera),
               n_state (state 쪽 non-None), n_camera (camera 쪽 non-None)

    Raises:
        ValueError: 셀 개수 불일치 또는 (x, y) 불일치
    """
    if len(cells_state) != len(cells_camera):
        raise ValueError(
            f"셀 개수 불일치: state {len(cells_state)} ≠ camera {len(cells_camera)}"
        )

    # 카메라 셀을 cell_idx로 색인
    camera_by_idx = {c["cell_idx"]: c for c in cells_camera}

    rows = []
    for state_cell in cells_state:
        state_idx = state_cell["cell_idx"]
        camera_cell = camera_by_idx.get(state_idx)

        if camera_cell is None:
            raise ValueError(f"camera에 cell_idx {state_idx}가 없음")

        # (x, y) 대조 (부동소수점 비교는 approx 사용)
        state_x, state_y = state_cell["x"], state_cell["y"]
        camera_x, camera_y = camera_cell["x"], camera_cell["y"]
        x_close = abs(state_x - camera_x) < 1e-6
        y_close = abs(state_y - camera_y) < 1e-6
        if not (x_close and y_close):
            raise ValueError(
                f"cell {state_idx}: (x, y) 불일치 state({state_x}, {state_y}) ≠ camera({camera_x}, {camera_y})"
            )

        # 델타 계산
        state_success = state_cell.get("success_rate")
        camera_success = camera_cell.get("success_rate")
        delta_success = None
        if state_success is not None and camera_success is not None:
            delta_success = state_success - camera_success

        state_lifted = state_cell.get("lifted_rate")
        camera_lifted = camera_cell.get("lifted_rate")
        delta_lifted = None
        if state_lifted is not None and camera_lifted is not None:
            delta_lifted = state_lifted - camera_lifted

        # perception_fail_rate는 camera 쪽에서만
        perception_fail_rate = camera_cell.get("perception_fail_rate")

        # n_state, n_camera: 각 쪽 non-None 메트릭 개수 (success_rate 기준)
        n_state = 1 if state_success is not None else 0
        n_camera = 1 if camera_success is not None else 0

        rows.append({
            "cell_idx": state_idx,
            "x": state_x,
            "y": state_y,
            "delta_success": delta_success,
            "delta_lifted": delta_lifted,
            "perception_fail_rate": perception_fail_rate,
            "n_state": n_state,
            "n_camera": n_camera,
        })

    return rows


def write_delta_csv(rows: list[dict], path: str) -> None:
    """델타 행 목록을 CSV로 저장.

    Args:
        rows: build_delta 출력
        path: 저장 경로
    """
    fields = ["cell_idx", "x", "y", "delta_success", "delta_lifted",
              "perception_fail_rate", "n_state", "n_camera"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def write_delta_heatmap(rows: list[dict], nx: int, ny: int, metric: str, path: str) -> None:
    """델타 히트맵 히트맵 생성.

    Args:
        rows: build_delta 출력
        nx: 격자 x 축 셀 수
        ny: 격자 y 축 셀 수
        metric: "delta_success" 또는 "delta_lifted"
        path: 저장 경로

    Raises:
        ValueError: 알 수 없는 메트릭
    """
    allowed_metrics = ("delta_success", "delta_lifted")
    if metric not in allowed_metrics:
        raise ValueError(f"unknown metric {metric!r} (choose from {allowed_metrics})")

    # 격자 구성 (cell_idx = xi * ny + yi, x-major)
    grid = [[None] * nx for _ in range(ny)]
    perception_grid = [[False] * nx for _ in range(ny)]
    for r in rows:
        xi, yi = r["cell_idx"] // ny, r["cell_idx"] % ny
        val = r[metric]
        grid[yi][xi] = val if val is not None else float("nan")
        # perception_fail_rate > 0인 셀에 'P' 마크
        if r["perception_fail_rate"] is not None and r["perception_fail_rate"] > 0:
            perception_grid[yi][xi] = True

    # 히트맵 렌더링
    data = grid
    fig, ax = plt.subplots(figsize=(1.2 * nx + 2, 1.0 * ny + 2))
    im = ax.imshow(data, origin="lower", cmap="RdBu", vmin=-1.0, vmax=1.0)

    # 축 레이블
    xs = sorted({r["x"] for r in rows})
    ys = sorted({r["y"] for r in rows})
    ax.set_xticks(range(nx), [f"{v:.3f}" for v in xs])
    ax.set_yticks(range(ny), [f"{v:.3f}" for v in ys])
    ax.set_xlabel("object x [m]")
    ax.set_ylabel("object y [m]")
    ax.set_title(f"{metric} (STATE−CAMERA)")

    # 셀 주석 및 P 마커
    for yi in range(ny):
        for xi in range(nx):
            v = grid[yi][xi]
            if not (v != v):  # not NaN (NaN != NaN는 True)
                text = f"{v:.2f}"
                if perception_grid[yi][xi]:
                    text += "P"
                ax.text(xi, yi, text, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    """CLI 엔트리포인트.

    python3 scripts/eval_s2r/delta_report.py --state <state.json> --camera <camera.json> --out <dir>
    """
    parser = argparse.ArgumentParser(
        description="STATE−CAMERA 델타 맵 리포트"
    )
    parser.add_argument("--state", required=True, help="state summary.json 경로")
    parser.add_argument("--camera", required=True, help="camera summary.json 경로")
    parser.add_argument("--out", required=True, help="출력 디렉토리")

    args = parser.parse_args()

    # 로드
    with open(args.state) as f:
        state_data = json.load(f)
    with open(args.camera) as f:
        camera_data = json.load(f)

    cells_state = state_data["cells"]
    cells_camera = camera_data["cells"]

    # nx, ny 추출
    try:
        meta_args = state_data["meta"]["args"]
        nx = meta_args["grid_nx"]
        ny = meta_args["grid_ny"]
    except (KeyError, TypeError):
        # 폴백: 고유한 x, y 개수로 추정
        xs = sorted({c["x"] for c in cells_state})
        ys = sorted({c["y"] for c in cells_state})
        nx = len(xs)
        ny = len(ys)

    # 검증
    if nx * ny != len(cells_state):
        raise ValueError(
            f"grid mismatch: nx={nx}, ny={ny} → {nx*ny} 셀, but got {len(cells_state)}"
        )

    # 델타 계산
    rows = build_delta(cells_state, cells_camera)

    # 출력
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "delta_success.csv"
    write_delta_csv(rows, str(csv_path))
    print(f"Written {csv_path}")

    hm_success_path = outdir / "heatmap_delta_success.png"
    write_delta_heatmap(rows, nx=nx, ny=ny, metric="delta_success", path=str(hm_success_path))
    print(f"Written {hm_success_path}")

    hm_lifted_path = outdir / "heatmap_delta_lifted.png"
    write_delta_heatmap(rows, nx=nx, ny=ny, metric="delta_lifted", path=str(hm_lifted_path))
    print(f"Written {hm_lifted_path}")

    # 요약 출력
    valid = [r for r in rows if r["delta_success"] is not None]
    if valid:
        mean_delta_success = sum(r["delta_success"] for r in valid) / len(valid)
        mean_delta_lifted = sum(r["delta_lifted"] for r in valid) / len(valid)
        print(f"delta_success mean: {mean_delta_success:.3f} ({len(valid)}/{len(rows)} 유효)")
        print(f"delta_lifted mean: {mean_delta_lifted:.3f}")
    else:
        print("No valid deltas computed")


if __name__ == "__main__":
    main()
