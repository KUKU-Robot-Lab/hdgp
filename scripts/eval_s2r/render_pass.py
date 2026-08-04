"""grasp_v1 sim2real 평가 SP2 — Pass 1 렌더 (Isaac 엔트리).

설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp2-camera-design.md §4.1
사용 예는 위 설계 §4.1·§6(게이트). GPU 실행은 사용자 게이트(§6 프리뷰 게이트가 1번).

그리드 스폰(SP1 훅 재사용) → head pan/tilt 명령 → D435i 헤드캠 안정화 후 1프레임/env 캡처
→ frames/<robot>/env_<i>.npz + meta.json + object_map.json. 정책 실행 없음(체크포인트 불필요).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

# hdgp 루트를 sys.path에 추가 (scripts.eval_s2r 패키지 import용) — eval_sim2real.py:20-23 패턴 복제
_HDGP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _HDGP_ROOT not in sys.path:
    sys.path.insert(0, _HDGP_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="grasp_v1 s2r 평가 SP2 — Pass 1 렌더(D435i 헤드캠 NPZ 캡처)")
parser.add_argument("--robot", required=True, choices=["left", "right"])
parser.add_argument("--grid_x", nargs=2, type=float, metavar=("MIN", "MAX"), required=True)
parser.add_argument("--grid_y", nargs=2, type=float, metavar=("MIN", "MAX"), required=True)
parser.add_argument("--grid_nx", type=int, required=True)
parser.add_argument("--grid_ny", type=int, required=True)
parser.add_argument("--grid_repeats", type=int, default=8)
parser.add_argument("--head_tilt", type=float, default=0.0, help="head_j_tilt 목표각 [rad]")
parser.add_argument("--head_pan", type=float, default=0.0, help="head_j_pan 목표각 [rad]")
parser.add_argument("--camera_info", type=str, default=None,
                    help="실기 camera_info YAML(K + width/height) — 지정 시 D435i 공칭값을 대체")
parser.add_argument("--out", type=str, required=True, help="frames/<robot> 출력 디렉토리")
parser.add_argument("--preview", action="store_true",
                    help="중심 셀 1 env만 렌더 → preview_env0.png 저장 + 정보 출력 후 종료(NPZ/meta 미생성)")
# --headless/--enable_cameras/--device 는 AppLauncher가 이미 제공(중복 정의 금지).
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if hydra_args:
    parser.error(f"알 수 없는 인자: {hydra_args}")

# ---- Isaac 기동 전 fail-fast (eval_sim2real.py:95-104 패턴 복제): GridSpec 검증만 순수 파이썬으로 미리 ----
from scripts.eval_s2r.grid import GridSpec  # noqa: E402  (torch 미의존 — 기동 전 import 안전)

try:
    _GRID_SPEC = GridSpec(
        args_cli.grid_x[0], args_cli.grid_x[1], args_cli.grid_nx,
        args_cli.grid_y[0], args_cli.grid_y[1], args_cli.grid_ny, args_cli.grid_repeats,
    )
except ValueError as e:
    parser.error(str(e))
if _GRID_SPEC.repeats % 8 != 0:
    print(f"[WARN] repeats={_GRID_SPEC.repeats} 가 8의 배수가 아님 — 셀마다 물체(8종 % 배정) 구성이 달라짐")

args_cli.enable_cameras = True  # TiledCamera/rtx 센서 필수 — scripts/distillation/place_camera.py 패턴 복제
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Isaac 기동 후 import (eval_sim2real.py:113-137 패턴 복제) ----
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml as _yaml  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import openarm.tasks  # noqa: F401,E402

from scripts.eval_s2r.grid import build_cells, build_spawn_tensor, single_spawn_tensor  # noqa: E402
from scripts.eval_s2r.transforms import k_from_pinhole, pinhole_from_k  # noqa: E402

TASK_BY_ROBOT = {
    "left": "open-tesol_l_grasp_v1-play-lstm",
    "right": "open-tesol_r_grasp_v1-play-lstm",
}
# object_map.json 소스 — _ACTIVE_OBJECT_SPECS(8종, env_id % 8 결정론적 배정)를 그대로 덤프.
ENV_CFG_MODULE_BY_ROBOT = {
    "left": "openarm.tesollo.left.grasp_v1.grasp_left_env_cfg",
    "right": "openarm.tesollo.right.grasp_v1.grasp_right_env_cfg",
}
# head 카메라 부착 링크 — 좌우 URDF(openarm_tesollo_sensor_rl/bi_rl) 둘 다
# head_base -> head_mid(head_j_pan) -> head_camera(head_j_tilt) -> head_cam_view(고정) 체인이
# 동일 이름으로 존재함을 확인(assets/robot/{openarm_tesollo_sensor_rl,openarm_tesollo_bi_rl}/*.urdf
# 918행대/1393행대). 로봇 prim_path도 둘 다 "/World/envs/env_.*/Robot" 이라 prim_path 하나로 공용.
HEAD_CAM_PRIM_PATH = "/World/envs/env_.*/Robot/head_cam_view/Camera"
GRASPOBJ_SEMANTIC_CLASS = "graspobj"

# D435i 공칭 intrinsics — grasp_right_preset.py:352-359 상수를 값만 복제(env import 회피, 소스 주석).
D435I_WIDTH = 320
D435I_HEIGHT = 180
D435I_FOCAL_LENGTH = 20.0
D435I_HORIZONTAL_APERTURE = 37.9586
D435I_CLIPPING_RANGE = (0.3, 3.0)

N_SETTLE = 30  # 카메라 노출 안정화 물리 스텝 수


def _resolve_camera_intrinsics(camera_info_path: str | None) -> tuple[int, int, float, float, str]:
    """--camera_info YAML(K+width/height)이 있으면 pinhole_from_k로 환산, 없으면 D435i 공칭값.

    반환: (width, height, focal_length, horizontal_aperture, k_source)
    """
    if camera_info_path is None:
        return D435I_WIDTH, D435I_HEIGHT, D435I_FOCAL_LENGTH, D435I_HORIZONTAL_APERTURE, "nominal"
    with open(camera_info_path) as f:
        info = _yaml.safe_load(f)
    width = int(info["width"])
    height = int(info["height"])
    k_raw = np.asarray(info["K"], dtype=float)
    k_mat = k_raw.reshape(3, 3) if k_raw.size == 9 else k_raw
    focal, aperture = pinhole_from_k(k_mat, width, height)
    return width, height, focal, aperture, camera_info_path


def _inject_semantic_tags(env_cfg) -> None:
    """cup MultiAsset 각 서브 cfg에 semantic_tags 주입 — instance seg 대신 semantic seg로 마스크 추출.

    파일 무수정: env_cfg 객체(런타임 인스턴스)에만 프로그램적으로 설정한다.
    """
    for asset_cfg in env_cfg.cup_cfg.spawn.assets_cfg:
        asset_cfg.semantic_tags = [("class", GRASPOBJ_SEMANTIC_CLASS)]


def _build_camera_cfg(width: int, height: int, focal: float, aperture: float) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path=HEAD_CAM_PRIM_PATH,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal,
            focus_distance=400.0,
            horizontal_aperture=aperture,
            clipping_range=D435I_CLIPPING_RANGE,
        ),
        width=width,
        height=height,
    )


def _create_head_camera(ge, camera_cfg: TiledCameraCfg) -> TiledCamera:
    """post-hoc TiledCamera 부착 — grasp_right_env.py:566-571(TiledCamera(cfg) 생성 +
    scene.sensors[...] 등록) 패턴을 그대로 복제하되, grasp_v1 cfg엔 카메라 필드가 없어
    env 생성 완료 후(post-hoc)에 붙인다.

    IsaacLab 센서는 timeline PLAY 콜백에서만 초기화되는데(SensorBase._register_callbacks
    → _initialize_callback), DirectRLEnv.__init__ 안의 1회뿐인 sim.reset()/play(direct_rl_env.py:167)는
    gym.make() 시점에 이미 지나갔다 — 그래서 여기서 새로 만든 카메라는 그 콜백을 영영 못 받는다.
    IsaacLab 공식 튜토리얼들은 전부 "센서/에셋 생성 → 단일 sim.reset()" 순서(스크립트 최초 1회)를
    따르는데, 우리는 카메라를 늦게 붙이므로 그 reset을 등록 직후 한 번 더 호출해 재현한다.
    이 시점(그리드 스폰 이전, eval_fixed_spawn_local·env.reset() 호출 전)엔 아직 물체가 최종
    위치에 놓이기 전이라, 이 reset이 유발하는 물리 상태 초기화는 이후의 정상 env.reset()이
    전부 덮어써 안전하다.
    """
    camera = TiledCamera(camera_cfg)
    ge.scene.sensors["head_camera"] = camera
    ge.sim.reset()
    return camera


def _command_head_pose(ge, tilt: float, pan: float) -> None:
    ids, names = ge.robot.find_joints(["head_j_pan", "head_j_tilt"], preserve_order=True)
    if len(ids) != 2:
        raise RuntimeError(f"head_j_pan/head_j_tilt 조인트를 찾지 못함 (matched={names})")
    values = {"head_j_pan": pan, "head_j_tilt": tilt}
    target = torch.zeros(ge.num_envs, len(ids), device=ge.device)
    for col, name in enumerate(names):
        target[:, col] = values[name]
    ge.robot.set_joint_position_target(target, joint_ids=ids)
    ge.scene.write_data_to_sim()


def _settle(ge, n_steps: int) -> None:
    """카메라 노출 안정화 — 정책 없이 raw 물리 스텝만 반복 (eval_sim2real.py _replay_reverse 패턴 복제:
    set_joint_position_target → write_data_to_sim → sim.step → scene.update)."""
    phys_dt = ge.sim.get_physics_dt()
    for _ in range(n_steps):
        ge.scene.write_data_to_sim()
        ge.sim.step()
        ge.scene.update(phys_dt)


def _quat_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    """(w,x,y,z) 쿼터니언 → 3x3 회전행렬 (numpy only, Isaac 무관)."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _assert_camera_ready(camera: TiledCamera, clip_range: tuple[float, float]) -> None:
    """카메라가 실제로 초기화·렌더됐는지 하드 검증 — 조용히 200장의 오염된 NPZ를 쓰지 않기 위한 가드.

    리뷰 F1: post-hoc TiledCamera 생성 + sim.reset() 워크어라운드(_create_head_camera 참조)가
    실패해도 이전엔 아무 에러 없이 검은/빈 프레임이 그대로 저장됐다. 이제 3중 체크로 즉시 중단한다.
    (a) is_initialized(속성 있으면 하드 체크, 없으면 (b)(c) 데이터 존재 체크로 대체)
    (b) env0 rgb가 전부 0이 아님 — 렌더 타깃이 비어있으면(빈 화면) 감지.
    (c) env0 depth가 clipping range 안에 유한값을 하나 이상 포함 — depth 파이프라인이 죽어있으면 감지.
    """
    if hasattr(camera, "is_initialized") and not camera.is_initialized:
        raise RuntimeError(
            "TiledCamera가 초기화되지 않음(is_initialized=False) — post-hoc 생성 후 "
            "sim.reset() 워크어라운드가 PLAY 콜백을 트리거하지 못한 것으로 보인다. "
            "--enable_cameras 플래그·head_cam_view prim 경로를 확인하라."
        )
    rgb0 = camera.data.output["rgb"][0].detach().cpu().numpy()
    if not rgb0.any():
        raise RuntimeError(
            "env0 rgb 버퍼가 전부 0 — 카메라가 렌더링되지 않았다(빈 화면). "
            "settle 스텝 수·prim_path·조명 설정을 확인하라."
        )
    depth0 = camera.data.output["distance_to_image_plane"][0, ..., 0].detach().cpu().numpy()
    finite = np.isfinite(depth0)
    in_range = finite & (depth0 >= clip_range[0]) & (depth0 <= clip_range[1])
    if not in_range.any():
        raise RuntimeError(
            f"env0 depth 버퍼에 clipping range {clip_range} 안의 유한값이 하나도 없음 — "
            "depth 파이프라인이 죽어있거나 카메라가 아무 것도 보고 있지 않다."
        )


def _resolve_semantic_id(info: dict, class_name: str) -> int | None:
    """semantic_segmentation info의 idToLabels에서 class_name과 일치하는 정수 ID를 찾는다.

    값 포맷이 Kit/Replicator 버전에 따라 dict({"class": name}) 또는 str(name) 둘 다 관측되므로
    양쪽을 방어적으로 처리한다. 매칭 실패 시 None(마스크는 전부 False, 상위에서 경고 출력).
    """
    id_to_labels = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    for id_str, label in id_to_labels.items():
        name = label.get("class") if isinstance(label, dict) else label
        if name == class_name:
            try:
                return int(id_str)
            except (TypeError, ValueError):
                continue
    return None


def _capture_env(ge, camera: TiledCamera, env_idx: int, graspobj_id: int | None,
                 K: np.ndarray, cells: list[tuple[float, float]], repeats: int) -> dict:
    """env_idx 하나의 npz payload 구성. 키는 스펙 §4.1 표 + `mask_pixels`(리뷰 F2, 추가 키)."""
    rgb = camera.data.output["rgb"][env_idx].detach().cpu().numpy().astype(np.uint8)
    depth = camera.data.output["distance_to_image_plane"][env_idx, ..., 0].detach().cpu().numpy().astype(np.float32)
    seg = camera.data.output["semantic_segmentation"][env_idx, ..., 0].detach().cpu().numpy()
    mask = (seg == graspobj_id) if graspobj_id is not None else np.zeros(seg.shape, dtype=bool)

    env_origin = ge.scene.env_origins[env_idx].detach().cpu().numpy()
    cam_pos_w = camera.data.pos_w[env_idx].detach().cpu().numpy()
    cam_quat_ros = camera.data.quat_w_ros[env_idx].detach().cpu().numpy()  # (w,x,y,z)
    T_local_cam = np.eye(4, dtype=np.float64)
    T_local_cam[:3, :3] = _quat_to_rot(cam_quat_ros)
    T_local_cam[:3, 3] = cam_pos_w - env_origin

    # object_pos는 _get_dones()(→_compute_intermediate_values) 안에서만 갱신되는데(env.step() 경로),
    # 이 스크립트는 env.step()을 한 번도 호출하지 않는다 — 그래서 stale일 수 있는 ge.object_pos 대신
    # cup RigidObject의 root pose를 직접 읽어 정착 후 값을 구한다(StateFrozenProvider와 같은 이유로
    # object_init_pos는 스폰 시점 GT라 그대로 유효 — 둘 다 저장해 정착 전/후 차이를 남긴다).
    gt_obj_pos_init = ge.object_init_pos[env_idx].detach().cpu().numpy().astype(np.float64)
    gt_obj_pos_settled = (ge.cup.data.root_pos_w[env_idx] - ge.scene.env_origins[env_idx])
    gt_obj_pos_settled = gt_obj_pos_settled.detach().cpu().numpy().astype(np.float64)
    gt_obj_rot_wxyz = ge.cup.data.root_quat_w[env_idx].detach().cpu().numpy().astype(np.float64)

    cell_idx = env_idx // repeats
    cell_x, cell_y = cells[cell_idx]

    return {
        "rgb": rgb,
        "depth": depth,
        "mask": mask,
        "K": K.astype(np.float64),
        "T_local_cam": T_local_cam,
        "gt_obj_pos_local": gt_obj_pos_settled,
        "gt_obj_pos_init": gt_obj_pos_init,
        "gt_obj_rot_wxyz": gt_obj_rot_wxyz,
        "obj_idx": env_idx % 8,
        "cell_idx": cell_idx,
        "cell_x": cell_x,
        "cell_y": cell_y,
        # 리뷰 F2 추가 키(additive) — Pass 2가 mask 없는(FOV 밖 등) env를 사전 필터링할 수 있게.
        "mask_pixels": int(mask.sum()),
    }


def _save_preview(frame: dict, path: Path) -> None:
    """rgb + mask 외곽선(순수 numpy edge) 오버레이 PNG 저장."""
    rgb = frame["rgb"].copy()
    mask = frame["mask"]
    edge = np.zeros_like(mask)
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    edge[:-1, :] |= mask[:-1, :] != mask[1:, :]
    rgb[edge] = np.array([255, 0, 0], dtype=np.uint8)
    Image.fromarray(rgb).save(path)


def _dump_object_map(robot: str, out_dir: Path) -> None:
    """object_map.json — env_cfg 모듈의 _ACTIVE_OBJECT_SPECS를 그대로 덤프(obj_idx = 리스트 index)."""
    module = importlib.import_module(ENV_CFG_MODULE_BY_ROBOT[robot])
    specs = module._ACTIVE_OBJECT_SPECS
    payload = {
        str(i): {"id": s["id"], "usd_path": s["usd_path"], "scale": list(s["scale"])}
        for i, s in enumerate(specs)
    }
    with open(out_dir / "object_map.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    task = TASK_BY_ROBOT[args_cli.robot]
    cells = build_cells(_GRID_SPEC)

    if args_cli.preview:
        # 중심 셀 (x-major 순, build_cells와 동일 규약 — grid.py:44-48)
        center_idx = (_GRID_SPEC.nx // 2) * _GRID_SPEC.ny + (_GRID_SPEC.ny // 2)
        local_cells = [cells[center_idx]]
        num_envs = 1
        repeats = 1
    else:
        local_cells = cells
        num_envs = len(cells) * _GRID_SPEC.repeats
        repeats = _GRID_SPEC.repeats

    print(f"[INFO] robot={args_cli.robot} task={task} num_envs={num_envs} preview={args_cli.preview}")

    width, height, focal, aperture, k_source = _resolve_camera_intrinsics(args_cli.camera_info)

    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=num_envs)
    env_cfg.scene.num_envs = num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # ---- ADR 항상 off + demo/warm 스폰 분기 차단 (eval_sim2real.py:502-520 패턴 복제) ----
    _disabled_adr = []
    for adr_attr in (
        "enable_adr", "enable_noise_adr", "enable_bead_count_adr",
        "enable_success_adr", "enable_spill_adr",
    ):
        if hasattr(env_cfg, adr_attr):
            setattr(env_cfg, adr_attr, False)
            _disabled_adr.append(adr_attr)
    if _disabled_adr:
        print(f"[INFO] ADR disabled for render: {', '.join(_disabled_adr)}")
    if hasattr(env_cfg, "enable_demo_grasp_reset"):
        env_cfg.enable_demo_grasp_reset = False
    if hasattr(env_cfg, "enable_warm_state_export"):
        env_cfg.enable_warm_state_export = False

    _inject_semantic_tags(env_cfg)

    env = gym.make(task, cfg=env_cfg, render_mode=None)
    ge = env.unwrapped

    camera_cfg = _build_camera_cfg(width, height, focal, aperture)
    camera = _create_head_camera(ge, camera_cfg)

    # eval_fixed_spawn_local: env_ids로 인덱싱한 뒤 .to(self.device) 하므로 미리 .to(ge.device)
    # (eval_sim2real.py:555-564 패턴 복제).
    if args_cli.preview:
        spawn_tensor = single_spawn_tensor(local_cells[0][0], local_cells[0][1], float("nan"), 1)
    else:
        spawn_tensor = build_spawn_tensor(local_cells, repeats, z=float("nan"))
    ge.eval_fixed_spawn_local = spawn_tensor.to(ge.device)

    env.reset()

    _command_head_pose(ge, tilt=args_cli.head_tilt, pan=args_cli.head_pan)
    _settle(ge, N_SETTLE)
    camera.update(ge.sim.get_physics_dt())
    _assert_camera_ready(camera, D435I_CLIPPING_RANGE)  # 리뷰 F1: fail-loud 가드

    seg_info = camera.data.info.get("semantic_segmentation", {})
    graspobj_id = _resolve_semantic_id(seg_info, GRASPOBJ_SEMANTIC_CLASS)
    if graspobj_id is None:
        id_to_labels = seg_info.get("idToLabels", seg_info) if isinstance(seg_info, dict) else seg_info
        if args_cli.preview:
            # 프리뷰는 바로 이 상황(마스크 클래스 부재)을 진단하려고 존재 — 경고만 내고 계속.
            print(f"[WARN] semantic id for class={GRASPOBJ_SEMANTIC_CLASS!r} 못 찾음 "
                  f"— mask 전부 False로 저장됨. idToLabels={id_to_labels}")
        else:
            # 리뷰 F2: mask 없이는 배치 200장 전체가 Pass 2(FoundationPose) 입력으로 무의미하다 —
            # 조용히 mask=False로 채운 NPZ를 쓰는 대신 즉시 중단한다.
            raise RuntimeError(
                f"semantic id for class={GRASPOBJ_SEMANTIC_CLASS!r} 못 찾음 — mask 없이는 "
                f"배치 전체가 무의미. semantic_tags 주입(_inject_semantic_tags)이나 카메라 "
                f"data_types 설정을 확인하라. idToLabels={id_to_labels}"
            )

    K = k_from_pinhole(focal, aperture, width, height)

    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args_cli.preview:
        frame = _capture_env(ge, camera, 0, graspobj_id, K, local_cells, repeats)
        _save_preview(frame, out_dir / "preview_env0.png")
        print(f"[PREVIEW] K=\n{K}")
        print(f"[PREVIEW] head_tilt={args_cli.head_tilt} head_pan={args_cli.head_pan} "
              f"mask_px={frame['mask_pixels']}/{frame['mask'].size}")
        print(f"[PREVIEW] 저장: {out_dir / 'preview_env0.png'}")
        env.close()
        simulation_app.close()
        return

    T_local_cam_all: dict[str, list] = {}
    zero_mask_envs: list[int] = []
    for i in range(ge.num_envs):
        frame = _capture_env(ge, camera, i, graspobj_id, K, local_cells, repeats)
        if frame["mask_pixels"] == 0:
            zero_mask_envs.append(i)
        np.savez(out_dir / f"env_{i}.npz", **frame)
        T_local_cam_all[str(i)] = frame["T_local_cam"].tolist()

    if zero_mask_envs:
        # raise 하지 않음 — FOV 밖 그리드 극단 셀에서는 정상적으로 일어날 수 있는 경우(설계 §7 위험 5).
        # Pass 2가 이 env들을 mask_pixels==0으로 사전 필터링할 수 있게 여기서도 요약만 남긴다.
        print(f"[WARN] mask 픽셀 0인 env {len(zero_mask_envs)}/{ge.num_envs}개 "
              f"(FP 등록 실패 예상 — FOV 밖 등이 원인일 수 있음): {zero_mask_envs}")

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HDGP_ROOT).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        git_sha = None
        print(f"[WARN] git SHA 조회 실패: {e}")

    meta = {
        "robot": args_cli.robot,
        "num_envs": ge.num_envs,
        "grid": {
            "x_min": _GRID_SPEC.x_min, "x_max": _GRID_SPEC.x_max, "nx": _GRID_SPEC.nx,
            "y_min": _GRID_SPEC.y_min, "y_max": _GRID_SPEC.y_max, "ny": _GRID_SPEC.ny,
            "repeats": _GRID_SPEC.repeats,
        },
        "head_tilt": args_cli.head_tilt,
        "head_pan": args_cli.head_pan,
        "T_local_cam": T_local_cam_all,
        "git_sha": git_sha,
        "k_source": k_source,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    _dump_object_map(args_cli.robot, out_dir)

    print(f"[INFO] {ge.num_envs}개 프레임 저장 완료: {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
