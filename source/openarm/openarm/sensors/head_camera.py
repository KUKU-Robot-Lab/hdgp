"""실기 head 카메라를 sim 에 그대로 붙인다 — distillation 용 SIM = REAL.

**왜 필요한가.** 자산의 `head_cam_view` 프레임(head_v1 CAD 의 RGB 렌즈, 2026-09-05 교체)과
hand-eye 실측 `T_neck_cam` 병진이 **69 mm** 다르다(옛 head 프레임과는 59.5 mm, 2026-09-01
재투영 1.214 px → 50.268 px). hand-eye 병진은 목 회전폭 24.6° 로 약하게 구속된 값이라
어느 쪽이 참인지 미확정이며, 실기 head 마운트 높이(B4)도 미검증이다. 그 프레임에 카메라를
붙이면 sim 에서 본 그림이 실기와 다르고, 그 위에서 학습한 학생망은 실기에서 어긋난다.
자산을 고치려면 URDF·USD 재빌드가 필요하므로 **건드리지 않고**, 실측 `T_neck_cam` 으로
tilt 링크(`head_camera`)에 카메라를 **새로** 붙인다.

캘리브값 출처: `sim2real/config/head_camera_sim.json`
(생성: `sim2real/scripts/sim_head_camera.py`, 검증: `probe_sim_head_camera.py`)

전체 고리 실측: 실기 카메라 → 캘리브 → base 좌표 → sim 소환 → Isaac 렌더 → 화소
비교에서 **평균 1.51 px**(0.64 m 에서 1.6 mm).

    from openarm.sensors.head_camera import attach_head_camera
    camera = attach_head_camera(env)          # env 를 만든 **뒤에** 부른다
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: 기본 캘리브 경로. 저장소 배치가 바뀌면 여기만 고치면 된다.
DEFAULT_SPEC_JSON = (Path(__file__).resolve().parents[5]
                     / "sim2real" / "config" / "head_camera_sim.json")
#: 카메라를 붙일 링크. tilt 가 움직이는 링크라 목을 돌리면 카메라도 따라간다.
NECK_LINK = "head_camera"
DEFAULT_PRIM_NAME = "head_cam_real"


@dataclass(frozen=True)
class HeadCameraSpec:
    link: str
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    width: int
    height: int
    intrinsic_matrix: tuple[float, ...]
    clipping_range: tuple[float, float]


def load_spec(path: Path | str = DEFAULT_SPEC_JSON) -> HeadCameraSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in ("link", "pos", "quat_wxyz", "width", "height",
                           "intrinsic_matrix", "clipping_range") if k not in raw]
    if missing:
        raise ValueError(f"캘리브 JSON 에 빠진 항목: {', '.join(missing)}")
    return HeadCameraSpec(
        link=str(raw["link"]), pos=tuple(raw["pos"]), quat_wxyz=tuple(raw["quat_wxyz"]),
        width=int(raw["width"]), height=int(raw["height"]),
        intrinsic_matrix=tuple(raw["intrinsic_matrix"]),
        clipping_range=tuple(raw["clipping_range"]),
    )


def head_camera_cfg(spec: HeadCameraSpec | None = None,
                    data_types: tuple[str, ...] = ("rgb",),
                    prim_name: str = DEFAULT_PRIM_NAME):
    """실측 extrinsics·intrinsics 를 담은 `CameraCfg`.

    ★`convention="ros"` 가 핵심이다. `T_neck_cam` 의 목적지가 카메라 optical 프레임이고,
    그것이 곧 ROS 규약(+z 전방 · +x 우 · +y 하)이다. 빠뜨리면 엉뚱한 곳을 본다.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    spec = spec or load_spec()
    return CameraCfg(
        prim_path=f"/World/envs/env_.*/Robot/{spec.link}/{prim_name}",
        update_period=0.0, height=spec.height, width=spec.width,
        data_types=list(data_types),
        offset=CameraCfg.OffsetCfg(pos=spec.pos, rot=spec.quat_wxyz, convention="ros"),
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=list(spec.intrinsic_matrix),
            width=spec.width, height=spec.height,
            clipping_range=spec.clipping_range),
    )


def attach_head_camera(env, spec: HeadCameraSpec | None = None, **kwargs):
    """**env 를 만든 뒤** 카메라를 붙이고 초기화까지 마친다.

    ★씬 cfg 에 넣으면 안 된다. DirectRLEnv 는 로봇을 `_setup_scene()` 에서 추가하는데
    씬 cfg 의 센서는 그**보다 먼저** 만들어져 `head_camera` prim 이 아직 없다
    (`RuntimeError: Unable to find source prim path`).

    ★센서 초기화는 sim **play 이벤트**에 걸려 있다. env 가 이미 play 중이면 그 콜백이
    다시 뜨지 않아 버퍼가 0 으로 남는다 — 여기서 직접 태운다.
    """
    from isaaclab.sensors import Camera

    camera = Camera(head_camera_cfg(spec, **kwargs))
    if not camera.is_initialized:
        camera._initialize_impl()
        camera._is_initialized = True
    return camera


def urdf_head_angles(pan_encoder_deg: float, tilt_encoder_deg: float
                     ) -> tuple[float, float]:
    """실기 인코더 각(deg) → URDF 관절 각(deg). **pan 만 부호가 반대다.**

    URDF pan 축이 `(0,0,-1)` 이라 인코더의 양의 방향과 반대로 돈다. 2026-09-01 hand-eye
    로 판정했다 — 뒤집어야 보드가 테이블 높이(z=+0.23 m)에 놓이고, 안 뒤집으면
    z=+1.48 m 로 카메라(z=0.82)보다 66 cm 위에 나온다.

    ★sim 에서 목 자세를 **유지**하려면 매 물리 스텝 다시 명령해야 한다. head 의
    `ImplicitActuator` 가 약해서, 상태를 한 번만 써 넣으면 스텝당 1.6°씩 0 으로 끌려간다
    (−20° 지령이 12스텝 뒤 −7.4°).
    """
    return (-float(pan_encoder_deg), float(tilt_encoder_deg))
