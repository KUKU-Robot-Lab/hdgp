"""grasp_v1 sim2real 평가 SP2 — Pass 2 FoundationPose 배치(standalone).

**실행 환경 주의**: 이 스크립트는 vision-3090 머신의 py3.8 FoundationPose 도커
컨테이너 내부에서 실행된다(perception repo 런북 참조 —
docs/FOUNDATIONPOSE_PLUSPLUS.md, SETUP_GUIDE 및 fpplusplus_smoke 계열 진입점과
동일 환경). hdgp 저장소는 이 파일을 **버전관리만** 한다 — hdgp 쪽에서 직접
실행하지 않는다(FP/trimesh가 이 repo 환경엔 설치돼 있지 않다).

입력: Task 3 render_pass.py가 만든 frames/<robot>/env_<i>.npz + object_map.json.
출력: poses/<robot>.json(Task 2 CameraFileProvider가 소비하는 스키마 그대로) +
      poses/<robot>_stats.json(성공률·사유별 실패 수·GT 위치오차 mean/median/p95[cm]·
      물체별 mesh bbox extents 로그) + stderr 진행 로그.

py3.8 호환: 상단 import는 numpy/json/argparse/pathlib뿐이다. FP(estimater 등)와
trimesh는 run_fp() 안에서만 지연 import한다 — 그래야 이 모듈의 순수 함수
(load_frame/mesh_path_for/pose_entry/error_cm/collect_stats)를 FP 미설치
환경(메인 hdgp repo, scripts/eval_s2r/tests/test_fp_batch_prep.py)에서도
import·테스트할 수 있다. match 문·`X | Y` 타입 유니언은 쓰지 않는다
(`from __future__ import annotations`로 애노테이션은 지연 평가되므로
`list[...]`/`X | None` 표기 자체는 안전 — transforms.py/providers.py와 동일 관례).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# render_pass.py _capture_env()가 저장하는 키 전체(리뷰 F2 추가 키 mask_pixels 포함).
REQUIRED_NPZ_KEYS = (
    "rgb", "depth", "mask", "K", "T_local_cam",
    "gt_obj_pos_local", "gt_obj_pos_init", "gt_obj_rot_wxyz",
    "obj_idx", "cell_idx", "cell_x", "cell_y", "mask_pixels",
)


def load_frame(npz_path: str) -> dict:
    """env_<i>.npz 하나를 로드하고 필수 키 존재를 검증한다.

    누락 키가 있으면 어떤 키가 빠졌는지 이름을 밝힌 ValueError를 낸다(경계 입력 검증).
    """
    with np.load(npz_path) as data:
        present = set(data.files)
        missing = [k for k in REQUIRED_NPZ_KEYS if k not in present]
        if missing:
            raise ValueError(
                f"load_frame({npz_path}): 필수 키 누락 {missing} "
                f"(요구 스키마: {list(REQUIRED_NPZ_KEYS)})"
            )
        return {k: data[k] for k in REQUIRED_NPZ_KEYS}


def mesh_path_for(obj_idx: int, object_map: dict, mesh_dir: str | None) -> tuple:
    """obj_idx의 mesh .obj 경로를 결정한다.

    탐색 순서:
    1) object_map[obj_idx]["usd_path"]와 같은 디렉토리의 sibling `<usd stem>.obj`
    2) `<mesh_dir>/<id>.obj` (mesh_dir이 주어진 경우만, Task 6 export 산출물)
    둘 다 없으면 (None, scale) — 호출자가 해당 물체 전 env를 reason="mesh_missing"으로
    FAIL 처리한다.

    반환: (path_str_or_None, scale) — scale은 object_map 값을 그대로(list) 반환한다.
    """
    key = str(obj_idx)
    if key not in object_map:
        raise ValueError(f"mesh_path_for: obj_idx {obj_idx!r} not in object_map (keys={list(object_map.keys())})")
    spec = object_map[key]
    usd_path = Path(spec["usd_path"])
    scale = list(spec["scale"])

    sibling = usd_path.parent / f"{usd_path.stem}.obj"
    if sibling.exists():
        return str(sibling), scale

    if mesh_dir is not None:
        candidate = Path(mesh_dir) / f"{spec['id']}.obj"
        if candidate.exists():
            return str(candidate), scale

    return None, scale


def pose_entry(ok: bool, T=None, reason: str | None = None) -> dict:
    """poses.json의 env별 엔트리 스키마(Task 2 CameraFileProvider 계약과 리터럴 일치).

    ok=True  → {"ok": True, "T_cam_obj": [[4x4]]}  (T 필수, shape (4,4))
    ok=False → {"ok": False, "reason": "..."}       (reason 필수)
    """
    if ok:
        if T is None:
            raise ValueError("pose_entry: ok=True requires T")
        T_arr = np.asarray(T, dtype=float)
        if T_arr.shape != (4, 4):
            raise ValueError(f"pose_entry: T must be shape (4,4), got {T_arr.shape}")
        return {"ok": True, "T_cam_obj": T_arr.tolist()}
    if reason is None:
        raise ValueError("pose_entry: ok=False requires reason")
    return {"ok": False, "reason": str(reason)}


def error_cm(T_cam_obj, T_local_cam, gt_pos_local) -> float:
    """FP 추정 pose의 GT 대비 위치 오차[cm].

    # transforms.py 와 동일 수학 (compose_local_pose): env-local 물체 위치 =
    # (T_local_cam @ T_cam_obj) 의 평행이동부. 단일 파일 자립을 위해 여기 numpy로 복제한다.
    """
    T_cam_obj = np.asarray(T_cam_obj, dtype=float)
    T_local_cam = np.asarray(T_local_cam, dtype=float)
    gt = np.asarray(gt_pos_local, dtype=float)
    if T_cam_obj.shape != (4, 4):
        raise ValueError(f"error_cm: T_cam_obj must be shape (4,4), got {T_cam_obj.shape}")
    if T_local_cam.shape != (4, 4):
        raise ValueError(f"error_cm: T_local_cam must be shape (4,4), got {T_local_cam.shape}")
    if gt.shape != (3,):
        raise ValueError(f"error_cm: gt_pos_local must be shape (3,), got {gt.shape}")
    pos_local = (T_local_cam @ T_cam_obj)[:3, 3]
    return float(np.linalg.norm(pos_local - gt) * 100.0)


def collect_stats(entries: dict, errors: dict) -> dict:
    """poses.json 엔트리 + error_cm 결과로부터 stats json 페이로드를 구성한다.

    entries가 비어 있어도(empty-safe) 예외 없이 None/0 값으로 채운 dict를 반환한다.
    error 통계(mean/median/p95)는 ok=True인 env 중 errors에 값이 있는 것만 대상으로 한다
    (GT 위치오차는 성공 env에만 정의됨).
    """
    total = len(entries)
    ok_ids = [k for k, v in entries.items() if v.get("ok")]
    fail_ids = [k for k, v in entries.items() if not v.get("ok")]

    reason_counts: dict = {}
    for k in fail_ids:
        reason = entries[k].get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    err_vals = list(errors.values())
    if err_vals:
        arr = np.asarray(err_vals, dtype=float)
        error_stats = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "n": int(arr.size),
        }
    else:
        error_stats = {"mean": None, "median": None, "p95": None, "n": 0}

    return {
        "num_envs": total,
        "num_ok": len(ok_ids),
        "num_failed": len(fail_ids),
        "success_rate": (len(ok_ids) / total) if total > 0 else None,
        "failure_reasons": reason_counts,
        "error_cm": error_stats,
    }


def _discover_frames(frames_dir: Path) -> list:
    """frames_dir/env_<i>.npz 전부를 env_id 오름차순으로 나열한다."""
    files = sorted(
        frames_dir.glob("env_*.npz"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return [(int(p.stem.split("_")[1]), p) for p in files]


def _build_estimator(mesh, scorer, refiner, glctx):
    """물체별 FoundationPose estimator 1회 생성.

    호출 시그니처는 이 저장소의 perception FoundationPose++ vendor 코드
    (repo/FoundationPose-plus-plus/FoundationPose/{run_demo,run_ycb_video}.py)의
    `FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
    mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx)` 패턴을 표본으로 삼았다.
    **정확한 인자명은 vision-3090 컨테이너 쪽 estimater.py가 진실** — 컨테이너 게이트에서
    맞지 않으면 이 함수만 조정하면 된다(register() 호출부는 안 건드림).
    """
    from estimater import FoundationPose  # noqa: E402  (지연 import, py3.8 컨테이너 전용)
    return FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
    )


def run_fp(frames_dir: str, mesh_dir: str | None, out_path: str, stats_path: str,
           robot: str, iteration: int) -> None:
    """FP 배치 본체(지연 import) — estimater/trimesh 없는 환경에서는 절대 호출되면 안 된다."""
    import trimesh  # noqa: E402
    import nvdiffrast.torch as dr  # noqa: E402
    from estimater import PoseRefinePredictor, ScorePredictor  # noqa: E402

    frames_path = Path(frames_dir)
    with open(frames_path / "object_map.json") as f:
        object_map = json.load(f)

    frame_list = _discover_frames(frames_path)
    if not frame_list:
        raise RuntimeError(f"run_fp: {frames_dir}에 env_*.npz가 없음")
    num_envs = len(frame_list)
    print(f"[INFO] frames={frames_dir} num_envs={num_envs} iteration={iteration}", file=sys.stderr)

    frames_by_env: dict = {}
    group_by_obj: dict = {}
    for env_id, path in frame_list:
        frame = load_frame(str(path))
        frames_by_env[env_id] = frame
        obj_idx = int(frame["obj_idx"])
        group_by_obj.setdefault(obj_idx, []).append(env_id)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    entries: dict = {}
    errors: dict = {}
    mesh_bbox_log: dict = {}

    t0 = time.time()
    done = 0
    for obj_idx in sorted(group_by_obj.keys()):
        env_ids = group_by_obj[obj_idx]
        path, scale = mesh_path_for(obj_idx, object_map, mesh_dir)
        if path is None:
            print(f"[WARN] obj_idx={obj_idx}: mesh 못 찾음 (env {len(env_ids)}개 mesh_missing 처리)",
                  file=sys.stderr)
            for env_id in env_ids:
                entries[env_id] = pose_entry(False, reason="mesh_missing")
            done += len(env_ids)
            continue

        mesh = trimesh.load(path)
        # 비균일 스케일 포함 — 정점에 직접 곱해 적용(단위 오류를 mesh bbox로 즉시 탐지, 스펙 위험 1).
        mesh.vertices = np.asarray(mesh.vertices, dtype=float) * np.asarray(scale, dtype=float)
        extents = (mesh.bounds[1] - mesh.bounds[0]).tolist() if len(mesh.vertices) else [0.0, 0.0, 0.0]
        mesh_bbox_log[str(obj_idx)] = {
            "id": object_map[str(obj_idx)]["id"],
            "path": path,
            "scale": scale,
            "bbox_extents_m": extents,
        }

        est = _build_estimator(mesh, scorer, refiner, glctx)

        for env_id in env_ids:
            frame = frames_by_env[env_id]
            if int(frame["mask_pixels"]) == 0:
                entries[env_id] = pose_entry(False, reason="empty_mask")
            else:
                try:
                    pose = est.register(
                        K=frame["K"], rgb=frame["rgb"], depth=frame["depth"],
                        ob_mask=frame["mask"], iteration=iteration,
                    )
                    T_cam_obj = np.asarray(pose, dtype=float).reshape(4, 4)
                    if not np.isfinite(T_cam_obj).all():
                        raise ValueError("register() returned non-finite pose")
                    entries[env_id] = pose_entry(True, T=T_cam_obj)
                    errors[env_id] = error_cm(T_cam_obj, frame["T_local_cam"], frame["gt_obj_pos_local"])
                except Exception as e:  # noqa: BLE001 — env 하나 실패로 배치 전체를 막지 않는다
                    entries[env_id] = pose_entry(False, reason=type(e).__name__)

            done += 1
            if done % 10 == 0 or done == num_envs:
                elapsed = time.time() - t0
                print(f"[INFO] {done}/{num_envs} 처리 ({elapsed:.1f}s 경과)", file=sys.stderr)

    poses_payload = {
        "robot": robot,
        "num_envs": num_envs,
        "poses": {str(k): v for k, v in entries.items()},
    }
    with open(out_path, "w") as f:
        json.dump(poses_payload, f, indent=2, ensure_ascii=False)

    stats = collect_stats(entries, errors)
    stats["mesh_bbox_extents"] = mesh_bbox_log
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[INFO] poses={out_path} stats={stats_path} "
          f"success_rate={stats['success_rate']}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="grasp_v1 s2r 평가 SP2 — Pass 2 FoundationPose 배치(vision-3090 컨테이너 전용)"
    )
    parser.add_argument("--robot", required=True, choices=["left", "right"])
    parser.add_argument("--frames", required=True, help="frames/<robot> 디렉토리(render_pass.py 출력)")
    parser.add_argument("--mesh_dir", default=None, help="mesh 대체 탐색 디렉토리(Task 6 export 산출물)")
    parser.add_argument("--out", default=None, help="poses json 출력 경로(기본 poses/<robot>.json)")
    parser.add_argument("--iteration", type=int, default=5, help="FoundationPose register() refine iteration 수")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path("poses") / f"{args.robot}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = out_path.with_name(f"{out_path.stem}_stats.json")

    run_fp(
        frames_dir=args.frames,
        mesh_dir=args.mesh_dir,
        out_path=str(out_path),
        stats_path=str(stats_path),
        robot=args.robot,
        iteration=args.iteration,
    )


if __name__ == "__main__":
    main()
