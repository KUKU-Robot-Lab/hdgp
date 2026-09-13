# IKER 앞단(스냅샷·키포인트·프롬프트·게이트) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 구성 하나(config_00)에 대해 머리 카메라 스냅샷, 번호가 찍힌 키포인트 영상, 논문 프롬프트를 만든다. 그리고 사람 기준선과 VLM 형식 응답을 게이트에 통과시키는 데까지를, 학습 없이 hdgp 안에 구현한다.

**Architecture:** 역할은 세 층으로 나뉜다.
- `modules/iker`(numpy·torch·PIL): 기하, 프롬프트, 응답 실행, 게이트, 보상.
- `tasks/iker_shoe`: 장면 상수와 Isaac 스폰 설정.
- `scripts/iker`: Isaac 실행(자산 변환·스냅샷)과 CLI(메타 생성·사람 기준선·응답 수집).

산출물은 `assets/iker_shoe/`(자산)와 `iker_runs/shoe_place/config_XX/`(구성별 파일)에 남는다.

**Tech Stack:**
- Python 3.10(시스템, 순수 테스트), Python 3.11(Isaac Sim 5.1 / Isaac Lab 0.50.5)
- numpy, torch, Pillow, scipy(메타 생성만), pytest, git LFS

**Spec:** `docs/superpowers/specs/2026-09-13-iker-vlm-keypoint-reward-design.md`

**실행 위치:** 워크트리 `~/rl_ws/hdgp-iker`(브랜치 `iker-front-end`, main 83f8974a 에서 분기). 학습과 다른 세션이 쓰는 공유 체크아웃 `~/rl_ws/hdgp` 는 건드리지 않는다(사용자 요청 2026-09-13).

**계획 분할(3개 중 1번):**
- 2번: 파지 뱅크, 학습 환경(`iker_shoe_env*`, gym 등록, PPO), 사람 기준선 구성 1개 학습·평가(스펙 §7·§8, 마일스톤 1·4).
- 3번: Claude Code 생성 루프, Qwen 배선, 구성 10개 평가, 보고서(스펙 §6 백엔드·§9, 마일스톤 5~7).

**검증 상태(2026-09-13):** 이 계획의 코드 블록은 스크래치패드의 hdgp 복제 구조에서 실제로 돌린 파일을 스크립트로 옮겨 넣은 것이다.
- 순수 테스트 103개 통과.
- Isaac 자산 변환 5 s, 스냅샷 10 s 에 통과.
- 스냅샷(구성 0): 회전 −22.7°, 테두리 여유 +49.1 px, 번호 최소 간격 25.3 px, 머리 각 오차 0.14°, 안착 움직임 0.00 mm, 받침 깊이 오차 0.13 mm.
- 게이트: 사람 기준선 통과, 상대 좌표 응답 통과, 공개 데이터 간격 응답은 G4 정규화 0.118 로 실패.

프로토타입에서 계획에 반영한 발견은 세 가지다.
- 보상: 레거시 성공·실패 카운터는 연속이 아니라 **누적**이다(스펙 §7 수정 완료).
- 사람 기준선: 안착한 신발이 4~5° 기울어 있어 기울기를 유지한다(스펙 §6 수정 완료).
- 응답 검사: 최상위 문 검사보다 금지 노드 검사를 먼저 해야 오류 문구가 맞는다.

## Global Constraints

- t2r(`modules/t2r`, `scripts/reward_gen`)과 grasp_s2r 을 import 하지 않는다. IKER 단독 성능 측정이 목적이다.
- `modules/iker` 는 표준 라이브러리·numpy·torch·PIL 만 import 한다. scipy 는 `scripts/iker/build_shoe_meta.py` 에서만 쓴다.
- 로봇 값은 `modules/robot_profiles.py`·`modules/vendor_gains.py` 에서만 읽는다(숫자 복사 금지).
- GPU 는 로컬만 쓴다(서버 미사용).
- 순수 테스트: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest <경로> -q -p no:cacheprovider` (시스템 python3 3.10; scipy·requests 경고 줄은 무시).
- Isaac 실행: `cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh <스크립트> --headless`.
  - `isaaclab.sh` 는 쓰지 않는다. 그 런처가 `~/rl_ws/hdgp/source/openarm`(공유 체크아웃)을 PYTHONPATH 맨 앞에 넣어 워크트리 코드를 가린다(`isaaclab.sh:32-36`). 공유 런처는 고치지 않는다.
  - 스크립트 docstring 의 `cd ~/rl_ws/hdgp && ... isaaclab.sh -p` 사용법은 병합 후 기준이라 코드는 그대로 둔다.
- Isaac 스크립트의 종료 규칙:
  - `sys.excepthook` 에서 `os._exit(1)`, 끝에서 `os._exit(...)` 로 나간다(종료 시 멈춤 방지).
  - 성공 판정은 종료 코드가 아니라 출력 표식(`CONVERT DONE`, `SNAPSHOT ... passed True`)과 산출 파일로 한다.
- 스크립트는 `scripts/` 바로 아래 한 단계(`scripts/iker/`)에만 둔다.
- `tasks/iker_shoe/__init__.py` 는 빈 파일로 둔다. 이 계획은 gym 등록을 하지 않는다(계획 2).
- `.usd` 는 기존 `.gitattributes`(`*.usd filter=lfs`)로 LFS 에 들어간다. `.gitattributes` 는 고치지 않는다.
- 워크트리는 깨끗한 상태에서 시작한다. `git add -A`·`git add .` 금지, 각 커밋 단계에 적힌 경로만 add 하고 push·병합하지 않는다.
- 공유 체크아웃 `~/rl_ws/hdgp` 에서는 어떤 명령도 실행하지 않는다.
- 커밋 메시지 끝에는 다음 두 줄을 붙인다(각 커밋 단계에 이미 들어 있다).
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4`
- 비교 가드는 양의 형태로 쓴다(`if not value <= limit:`). 그래야 NaN 이 통과하지 못한다.
- 설정값은 스펙에서 그대로 가져온다.

| 항목 | 값 |
|---|---|
| 지시문 | `"Place the shoe on the rack next to the other shoe."` |
| 머리 인코더 각 | pan −20°, tilt −20° |
| 스냅샷 | 겹침 제거 15 px, 테두리 여유 ≥ 15 px |
| 게이트 | G4 평가 5 cm·정규화 0.10, G5 관통 1 cm, G7 들어올림 2 cm |
| 받침 | x 0.11~0.43, y −0.33~0.03, 상면 0.325 |
| 테이블·신발 | 테이블 상면 0.205, 다른 신발 y −0.15 |
| 구성 시드 | 20260913 |

## 파일 구조

`M = source/openarm/openarm/agnostic/modules/iker`, `T = source/openarm/openarm/agnostic/tasks/iker_shoe`

| 파일 | 책임 | 태스크 |
|---|---|---|
| `M/rotations.py` | 쿼터니언(wxyz)·행렬·xyz 외부축 오일러·유한값 검사 | 1 |
| `M/keypoints.py` | 정지 자세 수평 두 축 끝점, 받침 격자, 번호, 겹침 제거 | 1 |
| `M/projection.py` | 링크 자세 ⊗ 광학 오프셋 → 카메라, 투영, +x 위 회전각, 회전 후 좌표 | 2 |
| `M/annotate.py` | 영상 회전, 번호 원, 좌표축 범례 | 2 |
| `M/scene_image.py` | 한 프레임 → 투영·가림·번호·겹침·그림(`AnnotatedSnapshot`) | 2 |
| `M/prompts.py`, `M/prompt_text/*.txt` | 논문 부록 C 원문과 채우기 | 3 |
| `M/interaction.py` | 응답 코드 블록 추출·AST 검사·제한 실행·반환값 파싱 | 4 |
| `M/gate.py` | G1~G7, Kabsch, `gate_response` | 5 |
| `M/reward.py` | 레거시 `compute_xarm_reward` 이식(학습은 계획 2) | 6 |
| `M/baseline.py` | 공개 목표 관계 → 사람 기준선 목표 | 7 |
| `M/run_files.py` | `shoe_meta.json`·`keypoints.json`·`interaction_<source>.json` | 8 |
| `T/layout.py` | 장면 상수, 구성 샘플링, 스폰 자세, 게이트 설정(순수) | 9 |
| `scripts/iker/build_shoe_meta.py` → `assets/iker_shoe/shoe_meta.json` | 메시 키포인트·정지 자세·헐·레거시 관계 | 9 |
| `T/robot.py`, `T/scene_cfg.py` | 로봇 ArticulationCfg(프로필·벤더 게인), 장면 스폰 설정 | 10 |
| `scripts/iker/convert_shoe_assets.py` → `assets/iker_shoe/shoe_*/` | URDF → USD | 10 |
| `scripts/iker/snapshot.py` → `iker_runs/shoe_place/config_00/` | 스냅샷·검증 | 11 |
| `scripts/iker/human_baseline.py`, `scripts/iker/ingest.py` | 사람 기준선, VLM 응답 게이트 | 12 |

---

### Task 1: 설계·계획 문서 커밋, 회전·키포인트 모듈

**Files:**
- Commit: `docs/superpowers/specs/2026-09-13-iker-vlm-keypoint-reward-design.md`, `docs/superpowers/plans/2026-09-13-iker-front-end.md`
- Create: `M/__init__.py`, `M/rotations.py`, `M/keypoints.py`, `M/tests/__init__.py`, `M/tests/test_rotations.py`, `M/tests/test_keypoints.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `rotations`
    - `finite_array(name, value, shape) -> np.ndarray`
    - `quat_wxyz_to_matrix(quat) -> (3,3)`
    - `matrix_to_quat_wxyz(matrix) -> (4,)`(w ≥ 0)
    - `rpy_xyz_to_matrix(rpy) -> (3,3)`
    - `yaw_matrix(yaw_rad) -> (3,3)`
  - `keypoints`
    - `snap_rotation(matrix) -> (3,3)`
    - `horizontal_axes(rest_rotation, vertices) -> (int, int)`
    - `extremity_offsets(vertices, axes) -> (4,3)`(순서 +a0, −a0, +a1, −a1)
    - `surface_grid(x_range, y_range, z, inset, nx, ny) -> (nx·ny, 3)`(x 우선)
    - `assign_labels(object_names, is_static, u) -> int ndarray`
    - `overlap_keep_mask(uv, labels, is_static, min_px) -> bool ndarray`

- [ ] **Step 1: 워크트리 브랜치를 확인하고 설계·계획 문서를 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker
test "$(git branch --show-current)" = iker-front-end
git add docs/superpowers/specs/2026-09-13-iker-vlm-keypoint-reward-design.md docs/superpowers/plans/2026-09-13-iker-front-end.md
git commit -F - <<'MSG'
docs(iker): IKER VLM 키포인트 보상 프레임워크 설계와 앞단 구현 계획

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

- [ ] **Step 2: 패키지와 실패하는 테스트를 만든다**

```bash
cd ~/rl_ws/hdgp-iker && mkdir -p source/openarm/openarm/agnostic/modules/iker/tests
: > source/openarm/openarm/agnostic/modules/iker/tests/__init__.py
```

Create `source/openarm/openarm/agnostic/modules/iker/__init__.py`:

````python
"""IKER (VLM-generated iterative keypoint rewards) building blocks, independent of the simulator.

Design: `docs/superpowers/specs/2026-09-13-iker-vlm-keypoint-reward-design.md`.
Only numpy, torch and PIL are imported here; the t2r and grasp_s2r tracks are never imported.
"""
````

Create `source/openarm/openarm/agnostic/modules/iker/tests/test_rotations.py`:

````python
"""rotations — quaternion/matrix/Euler helpers (no Isaac)."""

import numpy as np
import pytest

from openarm.agnostic.modules.iker import rotations as rot

SHOE_REST = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def test_quat_matrix_round_trip_keeps_w_non_negative():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        back = rot.matrix_to_quat_wxyz(rot.quat_wxyz_to_matrix(q))
        assert back[0] >= 0.0
        assert np.allclose(back, q if q[0] >= 0.0 else -q, atol=1e-9)


def test_shoe_rest_rotation_has_the_expected_quaternion():
    assert np.allclose(rot.matrix_to_quat_wxyz(SHOE_REST), [0.5, -0.5, 0.5, -0.5])


def test_rpy_matches_scipy_extrinsic_xyz():
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    rpy = (-1.5636626780948757, 0.028418572882054294, -1.6710602751696029)
    assert np.allclose(rot.rpy_xyz_to_matrix(rpy), rotation.from_euler("xyz", rpy).as_matrix(), atol=1e-12)


def test_non_finite_and_improper_inputs_raise():
    with pytest.raises(ValueError, match="non-finite"):
        rot.quat_wxyz_to_matrix([np.nan, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="proper rotation"):
        rot.matrix_to_quat_wxyz(np.diag([1.0, 1.0, -1.0]))
````

Create `source/openarm/openarm/agnostic/modules/iker/tests/test_keypoints.py`:

````python
"""keypoints — mesh extremities, rack grid, labels, overlap filter (no Isaac)."""

import numpy as np
import pytest

from openarm.agnostic.modules.iker import keypoints as kp
from openarm.agnostic.modules.iker.rotations import rpy_xyz_to_matrix

SHOE_REST = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# A box with the shoe's extents: width x 0.096, height y 0.104, length z 0.250.
SHOE_BOX = np.array([[sx * 0.048, sy * 0.052, sz * 0.125] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def test_snap_rotation_removes_a_small_scan_tilt():
    tilted = SHOE_REST @ rpy_xyz_to_matrix((0.05, -0.03, 0.04))
    assert np.array_equal(kp.snap_rotation(tilted), SHOE_REST)


def test_snap_rotation_rejects_a_reflection():
    with pytest.raises(ValueError, match="axis-aligned"):
        kp.snap_rotation(np.diag([1.0, 1.0, -1.0]))


def test_horizontal_axes_are_length_then_width_not_height():
    # Sorting all three axes by size would return (2, 1): the height beats the width.
    assert kp.horizontal_axes(SHOE_REST, SHOE_BOX) == (2, 0)


def test_extremity_offsets_follow_axis_order_and_keep_asymmetry():
    offsets = kp.extremity_offsets(SHOE_BOX + [0.0, 0.0, 0.01], (2, 0))
    assert np.allclose(offsets, [[0, 0, 0.135], [0, 0, -0.115], [0.048, 0, 0], [-0.048, 0, 0]])


def test_surface_grid_is_inset_and_x_major():
    grid = kp.surface_grid((0.11, 0.43), (-0.33, 0.03), 0.325, 0.05, 3, 4)
    assert grid.shape == (12, 3)
    assert np.allclose(grid[0], [0.16, -0.28, 0.325])
    assert np.allclose(grid[3], [0.16, -0.02, 0.325])
    assert np.allclose(grid[11], [0.38, -0.02, 0.325])


def test_labels_number_movable_objects_left_to_right_then_static():
    names = ["shoe_move"] * 4 + ["shoe_other"] * 4 + ["rack"] * 2
    static = [False] * 8 + [True] * 2
    u = [300, 310, 320, 330, 100, 110, 120, 130, 50, 60]
    assert kp.assign_labels(names, static, u).tolist() == [5, 6, 7, 8, 1, 2, 3, 4, 9, 10]


def test_labels_reject_an_object_with_mixed_kinds():
    with pytest.raises(ValueError, match="mixes"):
        kp.assign_labels(["a", "a"], [False, True], [0.0, 1.0])


def test_overlap_filter_drops_static_first_then_the_higher_label():
    uv = np.array([[100.0, 100.0], [105.0, 100.0], [300.0, 300.0], [306.0, 300.0]])
    keep = kp.overlap_keep_mask(uv, [10, 3, 1, 2], [True, False, False, False], 15.0)
    assert keep.tolist() == [False, True, True, False]
````

- [ ] **Step 3: 테스트가 실패하는지 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_rotations.py source/openarm/openarm/agnostic/modules/iker/tests/test_keypoints.py -q -p no:cacheprovider`
Expected: FAIL. 수집 단계에서 `ImportError`(`rotations`·`keypoints` 가 아직 없음).

- [ ] **Step 4: 구현한다**

Create `source/openarm/openarm/agnostic/modules/iker/rotations.py`:

````python
"""Rotation helpers for the IKER modules (numpy only).

Quaternions are ``wxyz`` (Isaac Lab order). Matrices rotate column vectors.
"""

from __future__ import annotations

import numpy as np

_ORTHONORMAL_TOL = 1e-6


def finite_array(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    """``value`` as a float array of ``shape``; raises unless every entry is finite."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def quat_wxyz_to_matrix(quat) -> np.ndarray:
    q = finite_array("quat", quat, (4,))
    norm = float(np.linalg.norm(q))
    if not norm > 1e-9:
        raise ValueError("quat has zero norm")
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(matrix) -> np.ndarray:
    """Unit quaternion with ``w >= 0`` (Shepperd's method)."""
    m = finite_array("matrix", matrix, (3, 3))
    if not (np.allclose(m.T @ m, np.eye(3), atol=_ORTHONORMAL_TOL) and np.linalg.det(m) > 0.0):
        raise ValueError("matrix is not a proper rotation")
    trace = float(np.trace(m))
    k = int(np.argmax([trace, m[0, 0], m[1, 1], m[2, 2]]))
    if k == 0:
        w = np.sqrt(1.0 + trace) / 2.0
        q = [w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)]
    elif k == 1:
        x = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) / 2.0
        q = [(m[2, 1] - m[1, 2]) / (4 * x), x, (m[0, 1] + m[1, 0]) / (4 * x), (m[0, 2] + m[2, 0]) / (4 * x)]
    elif k == 2:
        y = np.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2]) / 2.0
        q = [(m[0, 2] - m[2, 0]) / (4 * y), (m[0, 1] + m[1, 0]) / (4 * y), y, (m[1, 2] + m[2, 1]) / (4 * y)]
    else:
        z = np.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2]) / 2.0
        q = [(m[1, 0] - m[0, 1]) / (4 * z), (m[0, 2] + m[2, 0]) / (4 * z), (m[1, 2] + m[2, 1]) / (4 * z), z]
    quat = np.asarray(q, dtype=float)
    return quat if quat[0] >= 0.0 else -quat


def rpy_xyz_to_matrix(rpy) -> np.ndarray:
    """Extrinsic x-y-z Euler angles (scipy ``from_euler("xyz")``), the convention of the IKER assets."""
    roll, pitch, yaw = finite_array("rpy", rpy, (3,))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def yaw_matrix(yaw_rad: float) -> np.ndarray:
    return rpy_xyz_to_matrix((0.0, 0.0, yaw_rad))
````

Create `source/openarm/openarm/agnostic/modules/iker/keypoints.py`:

````python
"""IKER candidate keypoints (paper §III-A, design spec §4).

Movable objects get the mesh extremities on the two local axes that are horizontal at rest; static
objects get a uniform grid on their top surface. Labels are grouped by object, movable objects
first from left to right in the derotated image. The overlap filter follows the paper: static
points near movable ones are removed first, and among movable points the lower label survives.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

HORIZONTAL_TOL = 1e-6


def snap_rotation(matrix) -> np.ndarray:
    """Nearest rotation whose columns are signed world axes (removes a scan's residual tilt)."""
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError("matrix must be a finite 3x3 array")
    snapped = np.zeros((3, 3))
    for col in range(3):
        row = int(np.argmax(np.abs(m[:, col])))
        snapped[row, col] = np.sign(m[row, col])
    is_permutation = np.allclose(np.abs(snapped).sum(axis=1), 1.0)
    if not (is_permutation and np.linalg.det(snapped) > 0.0):
        raise ValueError("matrix does not snap to an axis-aligned proper rotation")
    return snapped


def horizontal_axes(rest_rotation, vertices) -> tuple[int, int]:
    """The two local axes that are horizontal at rest, longest mesh extent first.

    Sorting all three axes by size would pick the shoe's height (0.104 m) over its width (0.096 m).
    """
    rotation = np.asarray(rest_rotation, dtype=float)
    points = _points(vertices, "vertices")
    horizontal = [axis for axis in range(3) if abs(rotation[2, axis]) < HORIZONTAL_TOL]
    if len(horizontal) != 2:
        raise ValueError(f"rest rotation must keep exactly two local axes horizontal, got {horizontal}")
    extent = points.max(axis=0) - points.min(axis=0)
    first, second = sorted(horizontal, key=lambda axis: -extent[axis])
    return first, second


def extremity_offsets(vertices, axes: tuple[int, int]) -> np.ndarray:
    """Mesh extremities ``(+a0, -a0, +a1, -a1)`` on the given local axes, shape (4, 3)."""
    points = _points(vertices, "vertices")
    if len(set(axes)) != 2 or not all(axis in (0, 1, 2) for axis in axes):
        raise ValueError(f"axes must be two distinct local axes, got {axes}")
    offsets = np.zeros((4, 3))
    for i, axis in enumerate(axes):
        offsets[2 * i, axis] = points[:, axis].max()
        offsets[2 * i + 1, axis] = points[:, axis].min()
    return offsets


def surface_grid(x_range, y_range, z: float, inset: float, nx: int, ny: int) -> np.ndarray:
    """``nx x ny`` grid on a horizontal rectangle, ``inset`` from its edges, x-major order."""
    (x0, x1), (y0, y1) = x_range, y_range
    if not (x1 - x0 > 2 * inset and y1 - y0 > 2 * inset and nx >= 1 and ny >= 1):
        raise ValueError("rectangle is too small for the inset or the grid is empty")
    xs = np.linspace(x0 + inset, x1 - inset, nx)
    ys = np.linspace(y0 + inset, y1 - inset, ny)
    return np.array([[x, y, z] for x in xs for y in ys], dtype=float)


def assign_labels(object_names: Sequence[str], is_static: Sequence[bool], u: Sequence[float]) -> np.ndarray:
    """1-based labels grouped by object: movable objects by mean image ``u`` (left first), then static ones.

    Within an object the labels follow the input order, so they run in the object's keypoint order.
    """
    names = list(object_names)
    static = [bool(s) for s in is_static]
    columns = np.asarray(u, dtype=float)
    if not (len(names) == len(static) == len(columns)):
        raise ValueError("object_names, is_static and u must have the same length")
    if not np.all(np.isfinite(columns)):
        raise ValueError("u contains non-finite values")
    kind: dict[str, bool] = {}
    for name, flag in zip(names, static):
        if kind.setdefault(name, flag) != flag:
            raise ValueError(f"object {name!r} mixes static and movable keypoints")
    members = {name: [i for i, n in enumerate(names) if n == name] for name in kind}
    movable = sorted((n for n in kind if not kind[n]), key=lambda n: float(columns[members[n]].mean()))
    labels = np.zeros(len(names), dtype=int)
    next_label = 1
    for name in movable + [n for n in kind if kind[n]]:
        for i in members[name]:
            labels[i] = next_label
            next_label += 1
    return labels


def overlap_keep_mask(uv, labels, is_static, min_px: float) -> np.ndarray:
    """Greedy paper rule: movable before static, lower label first; a point is kept only if it is
    at least ``min_px`` from every point already kept."""
    points = np.asarray(uv, dtype=float)
    label_arr = np.asarray(labels, dtype=int)
    static = [bool(s) for s in is_static]
    if points.ndim != 2 or points.shape[1] != 2 or not (len(points) == len(label_arr) == len(static)):
        raise ValueError("uv must be (N, 2) and match labels and is_static")
    if not np.all(np.isfinite(points)):
        raise ValueError("uv contains non-finite values")
    keep = np.zeros(len(points), dtype=bool)
    kept: list[int] = []
    for i in sorted(range(len(points)), key=lambda j: (static[j], int(label_arr[j]))):
        if all(np.hypot(*(points[i] - points[j])) >= min_px for j in kept):
            keep[i] = True
            kept.append(i)
    return keep


def _points(value, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"{name} must be (N, 3) with N >= 2")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains non-finite values")
    return points
````

- [ ] **Step 5: 통과를 확인한다**

Run: Step 3 과 같은 명령
Expected: `12 passed`

- [ ] **Step 6: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker && M=source/openarm/openarm/agnostic/modules/iker
git add $M/__init__.py $M/rotations.py $M/keypoints.py $M/tests/__init__.py $M/tests/test_rotations.py $M/tests/test_keypoints.py
git commit -F - <<'MSG'
feat(iker): 회전 헬퍼와 메시 키포인트·번호·겹침 제거

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 2: 카메라 투영·영상 회전·키포인트 영상

**Files:**
- Create: `M/projection.py`, `M/annotate.py`, `M/scene_image.py`, `M/tests/test_projection.py`, `M/tests/test_scene_image.py`

**Interfaces:**
- Consumes: `rotations.finite_array`, `rotations.quat_wxyz_to_matrix`, `keypoints.assign_labels`, `keypoints.overlap_keep_mask`
- Produces:
  - `projection`
    - `CameraPose(position (3,), rotation (3,3); 열 = 광학 x·y·z)`
    - `camera_pose_from_link(link_pos, link_quat_wxyz, offset_pos, offset_quat_wxyz) -> CameraPose`
    - `project_points(points_w, camera, intrinsic) -> (uv (N,2), depth (N,))`
    - `derotation_deg(anchor_w, camera, intrinsic) -> float`(PIL 반시계)
    - `rotate_image_points(uv, theta_deg, size) -> (N,2)`
    - `image_margin(uv, size) -> (N,)`
  - `annotate`
    - `Marker(label, uv, color, visible)`
    - `rotate_image(rgb, theta_deg) -> Image`
    - `draw_markers(image, markers, axes) -> Image`
    - 상수 `MOVABLE_COLORS`, `STATIC_COLOR`
  - `scene_image`
    - `PointGroup(object_name, is_static, points_w)`
    - `KeypointRecord(label, object_name, is_static, world, uv_raw, uv, depth, render_depth, visible, kept)`
    - `AnnotatedSnapshot(image, derotation_deg, records, axes, min_margin_px, min_gap_px)`
    - `annotate_snapshot(rgb, depth, camera, intrinsic, groups, anchor_w, overlap_min_px, occlusion_tol_m=0.01) -> AnnotatedSnapshot`

- [ ] **Step 1: 실패하는 테스트를 작성한다**

Create `source/openarm/openarm/agnostic/modules/iker/tests/test_projection.py`:

````python
"""projection — camera pose, pinhole projection, derotation (no Isaac)."""

import numpy as np
import pytest
from PIL import Image

from openarm.agnostic.modules.iker import projection
from openarm.agnostic.modules.iker.projection import CameraPose

# Looking straight down from 1 m: optical x = world +x, optical y = world -y, optical z = world -z.
DOWN = CameraPose(position=np.array([0.0, 0.0, 1.0]), rotation=np.diag([1.0, -1.0, -1.0]))
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def test_camera_pose_composes_link_pose_and_optical_offset():
    link_quat = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]  # 90 deg about z
    pose = projection.camera_pose_from_link([1.0, 2.0, 3.0], link_quat, [0.1, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(pose.position, [1.0, 2.1, 3.0])
    assert np.allclose(pose.rotation[:, 0], [0.0, 1.0, 0.0])


def test_projection_reproduces_pinhole_pixels_and_depth():
    uv, depth = projection.project_points([[0.1, 0.0, 0.0], [0.0, 0.1, 0.5]], DOWN, K)
    assert np.allclose(uv, [[380.0, 240.0], [320.0, 120.0]])
    assert np.allclose(depth, [1.0, 0.5])


def test_points_behind_the_camera_raise():
    with pytest.raises(ValueError, match="behind"):
        projection.project_points([[0.0, 0.0, 2.0]], DOWN, K)


def test_non_finite_camera_raises_instead_of_projecting_nan():
    bad = CameraPose(position=np.array([0.0, np.nan, 1.0]), rotation=np.eye(3))
    with pytest.raises(ValueError, match="non-finite"):
        projection.project_points([[0.0, 0.0, 0.0]], bad, K)


def test_derotation_turns_world_x_up_and_world_y_left():
    theta = projection.derotation_deg([0.0, 0.0, 0.0], DOWN, K)
    assert theta == pytest.approx(90.0)
    uv, _ = projection.project_points([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], DOWN, K)
    rotated = projection.rotate_image_points(uv, theta, (640, 480))
    x_dir, y_dir = rotated[1] - rotated[0], rotated[2] - rotated[0]
    assert x_dir[1] < 0.0 and abs(x_dir[0]) < 1e-6
    assert y_dir[0] < 0.0 and abs(y_dir[1]) < 1e-6


def test_rotate_image_points_matches_pil_rotation():
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    pixels[95:106, 495:506] = 255
    theta = 30.0
    rotated = np.asarray(Image.fromarray(pixels).rotate(theta, resample=Image.Resampling.BICUBIC, expand=False))
    rows, cols = np.nonzero(rotated[:, :, 0] > 127)
    expected = projection.rotate_image_points([[500.5, 100.5]], theta, (640, 480))[0]
    assert abs(cols.mean() + 0.5 - expected[0]) < 1.0
    assert abs(rows.mean() + 0.5 - expected[1]) < 1.0


def test_image_margin_is_negative_outside_the_image():
    assert np.allclose(projection.image_margin([[10.0, 240.0], [-5.0, 100.0]], (640, 480)), [10.0, -5.0])
````

Create `source/openarm/openarm/agnostic/modules/iker/tests/test_scene_image.py`:

````python
"""scene_image + annotate — labelled keypoint image from a synthetic camera (no Isaac)."""

import numpy as np
import pytest
from PIL import Image

from openarm.agnostic.modules.iker import annotate
from openarm.agnostic.modules.iker.projection import CameraPose
from openarm.agnostic.modules.iker.scene_image import PointGroup, annotate_snapshot

DOWN = CameraPose(position=np.array([0.0, 0.0, 1.0]), rotation=np.diag([1.0, -1.0, -1.0]))
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
RGB = np.full((480, 640, 3), 40, dtype=np.uint8)
DEPTH = np.full((480, 640), 1.0)  # the plane z = 0 is 1 m below the camera


def _shoe(y: float) -> np.ndarray:
    return np.array([[0.12, y, 0.0], [-0.12, y, 0.0], [0.0, y - 0.05, 0.0], [0.0, y + 0.05, 0.0]])


def _groups(rack_points) -> list[PointGroup]:
    return [
        PointGroup("shoe_move", False, _shoe(0.15)),
        PointGroup("shoe_other", False, _shoe(-0.15)),
        PointGroup("rack", True, np.asarray(rack_points, dtype=float)),
    ]


def test_left_object_gets_the_first_labels_and_static_points_come_last():
    snap = annotate_snapshot(RGB, DEPTH, DOWN, K, _groups([[0.1, -0.35, 0.0], [-0.1, -0.35, 0.0]]), [0, 0, 0], 15.0)
    by_object: dict[str, list[int]] = {}
    for record in snap.records:
        by_object.setdefault(record.object_name, []).append(record.label)
    assert by_object == {"shoe_move": [1, 2, 3, 4], "shoe_other": [5, 6, 7, 8], "rack": [9, 10]}
    assert snap.derotation_deg == pytest.approx(90.0)


def test_static_point_on_a_shoe_keypoint_is_dropped_and_occlusion_is_recorded():
    depth = DEPTH.copy()
    depth[240 - 90, 320 + 72] = 0.5  # raw pixel of shoe_move's first keypoint (x 0.12, y 0.15)
    snap = annotate_snapshot(RGB, depth, DOWN, K, _groups([[0.12, -0.15, 0.0], [-0.1, -0.35, 0.0]]), [0, 0, 0], 15.0)
    assert [r.kept for r in snap.records if r.object_name == "rack"] == [False, True]
    first = next(r for r in snap.records if r.label == 1)
    assert first.visible is False and first.render_depth == pytest.approx(0.5)
    assert all(r.visible for r in snap.records if r.label != 1)


def test_overlapping_movable_keypoints_abort_the_snapshot():
    groups = [PointGroup("shoe_move", False, _shoe(0.0)), PointGroup("shoe_other", False, _shoe(0.0) + [0.0, 0.004, 0.0])]
    with pytest.raises(ValueError, match="removed movable"):
        annotate_snapshot(RGB, DEPTH, DOWN, K, groups, [0, 0, 0], 15.0)


def test_markers_and_legend_are_drawn_on_the_derotated_image():
    snap = annotate_snapshot(RGB, DEPTH, DOWN, K, _groups([[-0.1, -0.35, 0.0]]), [0, 0, 0], 15.0)
    assert snap.image.size == (640, 480)
    for label, color in ((1, annotate.MOVABLE_COLORS[0]), (5, annotate.MOVABLE_COLORS[1]), (9, annotate.STATIC_COLOR)):
        u, v = next(r for r in snap.records if r.label == label).uv
        assert snap.image.getpixel((round(u - 8), round(v))) == color
    assert snap.axes["+x"][1] < 0.0 and abs(snap.axes["+x"][0]) < 1e-6
    assert snap.axes["+y"][0] < 0.0 and abs(snap.axes["+y"][1]) < 1e-6
    assert snap.min_margin_px > 15.0 and snap.min_gap_px >= 15.0


def test_draw_markers_refuses_non_finite_coordinates():
    marker = annotate.Marker(1, (float("nan"), 3.0), (255, 0, 0), True)
    with pytest.raises(ValueError, match="non-finite"):
        annotate.draw_markers(Image.new("RGB", (64, 48)), [marker], {})
````

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_projection.py source/openarm/openarm/agnostic/modules/iker/tests/test_scene_image.py -q -p no:cacheprovider`
Expected: FAIL. 수집 단계에서 `ImportError`(`projection`·`annotate`·`scene_image` 없음).

- [ ] **Step 3: 구현한다**

Create `source/openarm/openarm/agnostic/modules/iker/projection.py`:

````python
"""Head-camera geometry (design spec §5).

The camera pose is the ``head_camera`` link pose composed with the calibrated optical offset
(``sim2real/config/head_camera_sim.json``, ROS optical frame: +x right, +y down, +z forward).
Isaac Lab's ``Camera.data.pos_w`` / ``quat_w_ros`` stay zero/NaN for a camera attached after the
scene is built, so the pose is always computed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rotations import finite_array, quat_wxyz_to_matrix

MIN_DEPTH_M = 1e-6


@dataclass(frozen=True)
class CameraPose:
    position: np.ndarray  # (3,) world
    rotation: np.ndarray  # (3, 3) columns: optical x, y, z axes in world


def camera_pose_from_link(link_pos, link_quat_wxyz, offset_pos, offset_quat_wxyz) -> CameraPose:
    link_rotation = quat_wxyz_to_matrix(link_quat_wxyz)
    offset = finite_array("offset_pos", offset_pos, (3,))
    position = finite_array("link_pos", link_pos, (3,)) + link_rotation @ offset
    return CameraPose(position=position, rotation=link_rotation @ quat_wxyz_to_matrix(offset_quat_wxyz))


def project_points(points_w, camera: CameraPose, intrinsic) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (N, 2) and optical depth (N,) of world points."""
    points = np.asarray(points_w, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_w must be a finite (N, 3) array")
    k = finite_array("intrinsic", intrinsic, (3, 3))
    position = finite_array("camera.position", camera.position, (3,))
    rotation = finite_array("camera.rotation", camera.rotation, (3, 3))
    cam = (points - position) @ rotation
    depth = cam[:, 2]
    if not np.all(depth > MIN_DEPTH_M):
        raise ValueError("a keypoint lies behind the camera")
    u = k[0, 0] * cam[:, 0] / depth + k[0, 2]
    v = k[1, 1] * cam[:, 1] / depth + k[1, 2]
    return np.stack([u, v], axis=1), depth


def derotation_deg(anchor_w, camera: CameraPose, intrinsic, step_m: float = 0.1) -> float:
    """Counter-clockwise image rotation (PIL ``Image.rotate``) that makes projected world +x point up."""
    anchor = finite_array("anchor_w", anchor_w, (3,))
    uv, _ = project_points(np.stack([anchor, anchor + [step_m, 0.0, 0.0]]), camera, intrinsic)
    du, dv = uv[1] - uv[0]
    if not math.hypot(du, dv) > 1e-6:
        raise ValueError("world +x projects to a single pixel at the anchor")
    return 90.0 - math.degrees(math.atan2(-dv, du))


def rotate_image_points(uv, theta_deg: float, size: tuple[int, int]) -> np.ndarray:
    """Where pixels land after ``Image.rotate(theta_deg, expand=False)`` about the image centre."""
    points = np.asarray(uv, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("uv must be a finite (N, 2) array")
    if not math.isfinite(theta_deg):
        raise ValueError("theta_deg must be finite")
    width, height = size
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    x = points[:, 0] - width / 2.0
    y = -(points[:, 1] - height / 2.0)
    return np.stack([width / 2.0 + (c * x - s * y), height / 2.0 - (s * x + c * y)], axis=1)


def image_margin(uv, size: tuple[int, int]) -> np.ndarray:
    """Distance (px) from each point to the nearest image border; negative outside the image."""
    points = np.asarray(uv, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("uv must be a finite (N, 2) array")
    width, height = size
    sides = np.stack([points[:, 0], width - points[:, 0], points[:, 1], height - points[:, 1]], axis=1)
    return sides.min(axis=1)
````

Create `source/openarm/openarm/agnostic/modules/iker/annotate.py`:

````python
"""Draw IKER keypoint markers and the axis legend (design spec §5-6)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MARKER_RADIUS_PX = 11
LABEL_FONT_PX = 14
LEGEND_ARROW_PX = 38
LEGEND_HEAD_PX = 8
LEGEND_INSET_PX = 50
LABEL_COLOR = (255, 255, 255)
STATIC_COLOR = (70, 130, 255)
OCCLUDED_COLOR = (150, 150, 150)
LEGEND_COLOR = (255, 255, 0)
MOVABLE_COLORS = ((230, 60, 60), (60, 200, 90), (240, 170, 40), (190, 80, 220))


@dataclass(frozen=True)
class Marker:
    label: int
    uv: tuple[float, float]
    color: tuple[int, int, int]
    visible: bool


def movable_color(order_index: int) -> tuple[int, int, int]:
    return MOVABLE_COLORS[order_index % len(MOVABLE_COLORS)]


def rotate_image(rgb, theta_deg: float) -> Image.Image:
    """Rotate counter-clockwise about the centre, keeping the input size (black fill)."""
    pixels = np.asarray(rgb)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("rgb must be an (H, W, 3) uint8 array")
    if not math.isfinite(theta_deg):
        raise ValueError("theta_deg must be finite")
    return Image.fromarray(pixels).rotate(
        theta_deg, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0)
    )


def draw_markers(image: Image.Image, markers: Sequence[Marker], axes: Mapping[str, tuple[float, float]]) -> Image.Image:
    """A copy of ``image`` with numbered markers and a bottom-right axis legend."""
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=LABEL_FONT_PX)
    for marker in markers:
        u, v = marker.uv
        if not (math.isfinite(u) and math.isfinite(v)):
            raise ValueError(f"marker {marker.label} has non-finite coordinates")
        box = [u - MARKER_RADIUS_PX, v - MARKER_RADIUS_PX, u + MARKER_RADIUS_PX, v + MARKER_RADIUS_PX]
        if marker.visible:
            draw.ellipse(box, fill=marker.color, outline=LABEL_COLOR)
        else:
            draw.ellipse(box, outline=OCCLUDED_COLOR, width=2)
        draw.text((u, v), str(marker.label), fill=LABEL_COLOR, font=font, anchor="mm")
    _draw_legend(draw, out.size, axes, font)
    return out


def _draw_legend(draw: ImageDraw.ImageDraw, size: tuple[int, int], axes, font) -> None:
    width, height = size
    ox, oy = width - LEGEND_INSET_PX, height - LEGEND_INSET_PX
    for name, (du, dv) in axes.items():
        norm = math.hypot(du, dv)
        if not norm > 1e-9:
            raise ValueError(f"legend axis {name} has zero length")
        ux, uy = du / norm, dv / norm
        tip = (ox + LEGEND_ARROW_PX * ux, oy + LEGEND_ARROW_PX * uy)
        draw.line([(ox, oy), tip], fill=LEGEND_COLOR, width=3)
        back = (tip[0] - LEGEND_HEAD_PX * ux, tip[1] - LEGEND_HEAD_PX * uy)
        side = (-uy * LEGEND_HEAD_PX / 2.0, ux * LEGEND_HEAD_PX / 2.0)
        draw.polygon(
            [tip, (back[0] + side[0], back[1] + side[1]), (back[0] - side[0], back[1] - side[1])], fill=LEGEND_COLOR
        )
        label_at = (ox + (LEGEND_ARROW_PX + 12) * ux, oy + (LEGEND_ARROW_PX + 12) * uy)
        draw.text(label_at, name, fill=LEGEND_COLOR, font=font, anchor="mm")
````

Create `source/openarm/openarm/agnostic/modules/iker/scene_image.py`:

````python
"""Head-camera frame -> numbered IKER keypoint image (design spec §4-5), independent of the simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from . import annotate, keypoints, projection
from .projection import CameraPose

AXIS_STEP_M = 0.1


@dataclass(frozen=True)
class PointGroup:
    object_name: str
    is_static: bool
    points_w: np.ndarray  # (K, 3) world, in the object's keypoint order


@dataclass(frozen=True)
class KeypointRecord:
    label: int
    object_name: str
    is_static: bool
    world: tuple[float, float, float]
    uv_raw: tuple[float, float]
    uv: tuple[float, float]
    depth: float
    render_depth: float | None
    visible: bool
    kept: bool


@dataclass(frozen=True)
class AnnotatedSnapshot:
    image: Image.Image
    derotation_deg: float
    records: tuple[KeypointRecord, ...]  # sorted by label
    axes: Mapping[str, tuple[float, float]]
    min_margin_px: float
    min_gap_px: float | None


def annotate_snapshot(
    rgb,
    depth,
    camera: CameraPose,
    intrinsic,
    groups: Sequence[PointGroup],
    anchor_w,
    overlap_min_px: float,
    occlusion_tol_m: float = 0.01,
) -> AnnotatedSnapshot:
    """Project, label, filter and draw the keypoints of one head-camera frame.

    ``depth`` is the renderer's distance to the image plane. A keypoint is occluded when the rendered
    surface is more than ``occlusion_tol_m`` in front of it. Raises if the overlap filter would remove
    a movable keypoint, because the reward tracks every one of them.
    """
    pixels = np.asarray(rgb)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("rgb must be (H, W, 3)")
    height, width = pixels.shape[:2]
    depth_image = np.asarray(depth, dtype=float)
    if depth_image.shape != (height, width):
        raise ValueError(f"depth must have shape {(height, width)}, got {depth_image.shape}")
    names, static, points_w = _flatten(groups)
    uv_raw, cam_depth = projection.project_points(points_w, camera, intrinsic)
    render_depth = _sample_depth(depth_image, uv_raw)
    visible = [d is not None and d + occlusion_tol_m >= z for d, z in zip(render_depth, cam_depth)]
    theta = projection.derotation_deg(anchor_w, camera, intrinsic)
    uv = projection.rotate_image_points(uv_raw, theta, (width, height))
    labels = keypoints.assign_labels(names, static, uv[:, 0])
    kept = keypoints.overlap_keep_mask(uv, labels, static, overlap_min_px)
    dropped = sorted(int(labels[i]) for i in range(len(labels)) if not static[i] and not kept[i])
    if dropped:
        raise ValueError(f"overlap filter removed movable keypoints {dropped}; every movable keypoint is needed")
    axes = _axes(anchor_w, camera, intrinsic, theta, (width, height))
    markers = _markers(names, static, labels, uv, visible, kept)
    image = annotate.draw_markers(annotate.rotate_image(pixels, theta), markers, axes)
    records = tuple(
        sorted(
            (
                KeypointRecord(
                    label=int(labels[i]),
                    object_name=names[i],
                    is_static=static[i],
                    world=tuple(float(c) for c in points_w[i]),
                    uv_raw=(float(uv_raw[i, 0]), float(uv_raw[i, 1])),
                    uv=(float(uv[i, 0]), float(uv[i, 1])),
                    depth=float(cam_depth[i]),
                    render_depth=render_depth[i],
                    visible=bool(visible[i]),
                    kept=bool(kept[i]),
                )
                for i in range(len(names))
            ),
            key=lambda record: record.label,
        )
    )
    kept_uv = uv[kept]
    gaps = [float(np.hypot(*(a - b))) for i, a in enumerate(kept_uv) for b in kept_uv[i + 1 :]]
    margin = float(projection.image_margin(kept_uv, (width, height)).min())
    return AnnotatedSnapshot(image, theta, records, axes, margin, min(gaps) if gaps else None)


def _flatten(groups: Sequence[PointGroup]) -> tuple[list[str], list[bool], np.ndarray]:
    if not groups:
        raise ValueError("no keypoint groups")
    names: list[str] = []
    static: list[bool] = []
    blocks = []
    for group in groups:
        points = np.asarray(group.points_w, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(f"{group.object_name}: points_w must be (K, 3) with K >= 1")
        names += [group.object_name] * len(points)
        static += [bool(group.is_static)] * len(points)
        blocks.append(points)
    return names, static, np.concatenate(blocks)


def _sample_depth(depth_image: np.ndarray, uv: np.ndarray) -> list[float | None]:
    height, width = depth_image.shape
    samples: list[float | None] = []
    for u, v in uv:
        col, row = math.floor(u), math.floor(v)
        inside = 0 <= col < width and 0 <= row < height
        value = float(depth_image[row, col]) if inside else math.nan
        samples.append(value if math.isfinite(value) else None)
    return samples


def _axes(anchor_w, camera: CameraPose, intrinsic, theta: float, size: tuple[int, int]) -> dict[str, tuple[float, float]]:
    anchor = np.asarray(anchor_w, dtype=float)
    probe = np.stack([anchor, anchor + [AXIS_STEP_M, 0.0, 0.0], anchor + [0.0, AXIS_STEP_M, 0.0]])
    uv, _ = projection.project_points(probe, camera, intrinsic)
    rotated = projection.rotate_image_points(uv, theta, size)
    dx, dy = rotated[1] - rotated[0], rotated[2] - rotated[0]
    return {"+x": (float(dx[0]), float(dx[1])), "+y": (float(dy[0]), float(dy[1]))}


def _markers(names, static, labels, uv, visible, kept) -> list[annotate.Marker]:
    movable_order: list[str] = []
    for i in np.argsort(labels):
        if not static[i] and names[i] not in movable_order:
            movable_order.append(names[i])
    markers = []
    for i in range(len(names)):
        if not kept[i]:
            continue
        color = annotate.STATIC_COLOR if static[i] else annotate.movable_color(movable_order.index(names[i]))
        markers.append(annotate.Marker(int(labels[i]), (float(uv[i, 0]), float(uv[i, 1])), color, bool(visible[i])))
    return markers
````

- [ ] **Step 4: 통과를 확인한다**

Run: Step 2 와 같은 명령
Expected: `12 passed`

- [ ] **Step 5: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker && M=source/openarm/openarm/agnostic/modules/iker
git add $M/projection.py $M/annotate.py $M/scene_image.py $M/tests/test_projection.py $M/tests/test_scene_image.py
git commit -F - <<'MSG'
feat(iker): 머리 카메라 투영·+x 위 회전·번호 키포인트 영상

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 3: 논문 부록 C 프롬프트

**Files:**
- Create: `M/prompt_text/single_step.txt`, `M/prompt_text/multi_step.txt`, `M/prompts.py`, `M/tests/test_prompts.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `prompts`
    - `PROMPT_DIR`, `SINGLE_STEP`, `MULTI_STEP`, `TASK_MARKER="[TASK]"`, `IMAGE_MARKER="[IMAGE_WITH_KEYPOINTS]"`, `QUERY_HEADER="## Query"`
    - `StageRecord(image_marker, code)`
    - `load_template(kind) -> str`
    - `fill_single_step(task) -> str`
    - `fill_multi_step(task, history) -> str`

프롬프트 파일은 PDF 에서 옮긴 사본이다(두 단 조판 줄바꿈 결합, 인용부호 ’ → ', 문장 묶음마다 줄바꿈).
2026-09-13 에 모든 줄을 `pdftotext -raw` 추출본과 대조했다. 아래 heredoc 은 파일을 바이트 단위로 재현하며, Step 2 의 해시가 그것을 확인한다.

- [ ] **Step 1: 실패하는 테스트를 작성한다**

Create `source/openarm/openarm/agnostic/modules/iker/tests/test_prompts.py`:

````python
"""prompts — paper Appendix C text and filling (no Isaac)."""

import hashlib

import pytest

from openarm.agnostic.modules.iker import prompts

TASK = "Place the shoe on the rack next to the other shoe."
# Pinned transcriptions; a change here must be a deliberate prompt edit.
PROMPT_SHA256 = {
    prompts.SINGLE_STEP: "ddcb3beeb3db0bc54edbe2036eb78a486a976fd8d404e7b33151d69a5aae597b",
    prompts.MULTI_STEP: "f6335691c1f28c0a0ddfb99320d91111f44506f490e333748c098386e53597e0",
}


def test_prompt_files_are_the_pinned_appendix_c_transcription():
    for kind, digest in PROMPT_SHA256.items():
        assert hashlib.sha256((prompts.PROMPT_DIR / f"{kind}.txt").read_bytes()).hexdigest() == digest


def test_single_step_fill_inserts_the_task_and_keeps_the_image_marker():
    text = prompts.fill_single_step(TASK)
    assert text.startswith("## Instructions\nYour job is to help with moving rigid objects in real-world by writing code in python.\n")
    assert f"Query Task: '{TASK}'" in text
    assert prompts.TASK_MARKER not in text
    assert text.count(prompts.IMAGE_MARKER) == 1


def test_multi_step_prompt_differs_where_the_paper_differs():
    single = prompts.load_template(prompts.SINGLE_STEP)
    multi = prompts.load_template(prompts.MULTI_STEP)
    assert "Generally, big objects can only be pushed" in multi and "Generally, big objects" not in single
    assert "done = ?" in multi and "done = ?" not in single


def test_multi_step_history_is_placed_before_the_query():
    text = prompts.fill_multi_step(TASK, [prompts.StageRecord("[IMAGE_STAGE_1]", "def f():\n    return\n")])
    assert text.index("## History") < text.index("## Query")
    assert "Stage 1 image: [IMAGE_STAGE_1]\nStage 1 code:\n```python\ndef f():\n    return\n```" in text


@pytest.mark.parametrize("task", ["", "   ", "two\nlines"])
def test_task_must_be_a_single_non_empty_line(task):
    with pytest.raises(ValueError, match="single line"):
        prompts.fill_single_step(task)
````

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_prompts.py -q -p no:cacheprovider`
Expected: FAIL. `ImportError`(`prompts` 없음).

- [ ] **Step 2: 프롬프트 파일을 만들고 해시를 확인한다**

```bash
cd ~/rl_ws/hdgp-iker && mkdir -p source/openarm/openarm/agnostic/modules/iker/prompt_text
```

````bash
cat > source/openarm/openarm/agnostic/modules/iker/prompt_text/single_step.txt <<'IKER_PROMPT_EOF'
## Instructions
Your job is to help with moving rigid objects in real-world by writing code in python.
The task is given as an image of the environment, overlayed with keypoints marked with their indices, along with a text instruction.
These keypoints are in 3D space, and are projected onto the 2D image. They are attached with the objects, and move along with them.
So to determine where a specific point should go, you should specify where its corresponding keypoint should go.
The coordinate system is marked at the bottom right in the image, with a vertical arrow pointing forward in the positive x direction and the horizontal arrow pointing to the left in the positive y direction.
The code should predict the final keypoint locations relative to their matching keypoints. Use all matching keypoints to determine the final position of the moving object, not just a single reference point.
Note:
- You should determine if you need to grasp or push the object. You should output a boolean grasp_mode for that.
- Some objects should not be moved. Hence, the final location of key points on them should be the same as the initial locations.
- If you need to interact with an object, the final location of only the keypoints marked on it should change. Hence, you should first try to understand which object should move.
- You should try to understand where the moving object should go relative to other stationary objects and use all the matching keypoints for alignment. Then you can give the final locations of keypoints of moving objects relative to keypoints on stationary objects.
- Positive x direction points towards up and positive y direction points towards left.
- The input to the function is a dictionary of keypoint coordinates. So keys will be strings like "1", "2", ... and their values will be numpy arrays ([x, y, z]) representing the 3D location of the keypoint corresponding to that index.
- To represent coordinates relative to other keypoints, you can make predictions like:
keypoint_coordinates['1'] = keypoint_coordinates['2'] + np.array([delta_x, delta_y, delta_z]).
For instance, if keypoint 1 needs to be placed to the left of keypoint 2, then delta_x = 0, delta_y = 0.1, and delta_z = 0.
So can predict keypoint_coordinates['1'] = keypoint_coordinates['2'] + np.array([0, 0.1, 0]).
The units here are in meters and left direction corresponds to + y-axis.
- Make use of semantics. Some objects should be placed in a certain way, like a left shoe should be placed on the left of the right shoe.
Structure your output in a single python code block as follows:
def get_interaction_data(keypoint_coordinates):
    """ Put your explanation here. """
    object_to_interact = ?
    keypoint_indices_to_interact = ?
    grasp_mode = ?
    ## final keypoint calculation for each keypoint in keypoint_indices_to_interact
    keypoint_coordinates['keypoint_indices_to_interact[0]'] = ? # Write calculation here. You may use multiple lines
    # Repeat for other keypoints
    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
## Query
Query Task: '[TASK]'
Query Image: [IMAGE_WITH_KEYPOINTS]
IKER_PROMPT_EOF
````

````bash
cat > source/openarm/openarm/agnostic/modules/iker/prompt_text/multi_step.txt <<'IKER_PROMPT_EOF'
## Instructions
Your job is to help with moving rigid objects in real-world by writing code in python.
The task is given as an image of the environment, overlayed with keypoints marked with their indices, along with a text instruction.
These keypoints are in 3D space, and are projected onto the 2D image. They are attached with the objects, and move along with them.
So to determine where a specific point should go, you should specify where its corresponding keypoint should go.
The coordinate system is marked at the bottom right in the image, with a vertical arrow pointing forward in the positive x direction and the horizontal arrow pointing to the left in the positive y direction.
The code should predict the final keypoint locations relative to their matching keypoints. Use all matching keypoints to determine the final position of the moving object, not just a single reference point.
Note:
- You should determine if you need to grasp or push the object. You should output a boolean grasp_mode for that. Generally, big objects can only be pushed as they are too big to be grasped.
- Some objects should not be moved. Hence, the final location of key points on them should be the same as the initial locations.
- If you need to interact with an object, the final location of only the keypoints marked on it should change. Hence, you should first try to understand which object should move.
- You should try to understand where the moving object should go relative to other stationary objects and use all the matching keypoints for alignment. Then you can give the final locations of keypoints of moving objects relative to keypoints on stationary objects.
- Positive x direction points towards up and positive y direction points towards left.
- The input to the function is a dictionary of keypoint coordinates. So keys will be strings like "1", "2", ... and their values will be numpy arrays ([x, y, z]) representing the 3D location of the keypoint corresponding to that index.
- To represent coordinates relative to other keypoints, you can make predictions like: keypoint_coordinates['1'] = keypoint_coordinates['2'] + np.array([delta_x, delta_y, delta_z]).
For instance, if the keypoint 1 needs to be placed to the left of keypoint 2, then delta_x = 0, delta_y = 0.1, and delta_z = 0.
So can predict keypoint_coordinates['1'] = keypoint_coordinates['2'] + np.array([0, 0.1, 0]).
The units here are in meters and left direction corresponds to + y-axis.
- Make use of semantics. Some objects should be placed in a certain way, like a left shoe should be placed on the left of the right shoe.
- Some tasks involve multiple stages, so you will also predict the overall plan. Then you will write the description and code for the current stage. You will interact with only one object in a stage. Placing or pushing a single object will be considered a single stage. Grasping is not considered as a separate stage.
- You are free to make minor changes to the plan, or change the plan altogether if you think is necessary.
- We will keep adding the previous states of the environment as images and the corresponding code to the description. This will show how the task progressed. At the start, you will only see the task description.
- You should predict done=True when the task is complete, otherwise False. Only predict done=True when you see that the task is completed.
Structure your output in a single python code block as follows:
def get_interaction_data(keypoint_coordinates):
    """
    Previous Plan Description
    Current Plan Description
    Current stage description
    """
    done = ?
    if done:
        return
    object_to_interact = ?
    keypoint_indices_to_interact = ?
    grasp_mode = ?
    ## final keypoint calculation for each keypoint in keypoint_indices_to_interact
    keypoint_coordinates['keypoint_indices_to_interact[0]'] = ? # Write calculation here. You may use multiple lines
    # Repeat for other keypoints
    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
## Query
Query Task: '[TASK]'
Query Image: [IMAGE_WITH_KEYPOINTS]
IKER_PROMPT_EOF
````

Run: `cd ~/rl_ws/hdgp-iker && sha256sum source/openarm/openarm/agnostic/modules/iker/prompt_text/*.txt`
Expected:
```text
f6335691c1f28c0a0ddfb99320d91111f44506f490e333748c098386e53597e0  source/openarm/openarm/agnostic/modules/iker/prompt_text/multi_step.txt
ddcb3beeb3db0bc54edbe2036eb78a486a976fd8d404e7b33151d69a5aae597b  source/openarm/openarm/agnostic/modules/iker/prompt_text/single_step.txt
```

- [ ] **Step 3: 구현한다**

Create `source/openarm/openarm/agnostic/modules/iker/prompts.py`:

````python
"""IKER VLM prompts (paper Appendix C), stored as text files in ``prompt_text/``.

The files transcribe the PDF: two-column line wraps are joined, typographic quotes become ASCII
apostrophes, and each sentence group starts on its own line as the paper sets it. Every line was
checked against the text extracted from the PDF (2026-09-13). Tests pin the file hashes, so any
edit to a prompt is deliberate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROMPT_DIR = Path(__file__).resolve().parent / "prompt_text"
SINGLE_STEP = "single_step"
MULTI_STEP = "multi_step"
TASK_MARKER = "[TASK]"
IMAGE_MARKER = "[IMAGE_WITH_KEYPOINTS]"
QUERY_HEADER = "## Query"


@dataclass(frozen=True)
class StageRecord:
    """One earlier stage of a multi-step task: where its image goes and the code the VLM wrote."""

    image_marker: str
    code: str


def load_template(kind: str) -> str:
    if kind not in (SINGLE_STEP, MULTI_STEP):
        raise ValueError(f"unknown prompt kind {kind!r}")
    text = (PROMPT_DIR / f"{kind}.txt").read_text(encoding="utf-8")
    for marker in (TASK_MARKER, IMAGE_MARKER, QUERY_HEADER):
        if text.count(marker) != 1:
            raise ValueError(f"{kind} prompt must contain {marker!r} exactly once")
    return text


def fill_single_step(task: str) -> str:
    _check_task(task)
    return load_template(SINGLE_STEP).replace(TASK_MARKER, task)


def fill_multi_step(task: str, history: Sequence[StageRecord]) -> str:
    """Multi-step prompt; earlier stages are listed in a ``## History`` section before the query."""
    _check_task(task)
    text = load_template(MULTI_STEP).replace(TASK_MARKER, task)
    if not history:
        return text
    lines = ["## History"]
    for number, stage in enumerate(history, start=1):
        lines.append(f"Stage {number} image: {stage.image_marker}")
        lines.append(f"Stage {number} code:\n```python\n{stage.code.strip()}\n```")
    return text.replace(QUERY_HEADER, "\n".join(lines) + "\n" + QUERY_HEADER)


def _check_task(task: str) -> None:
    if not isinstance(task, str) or not task.strip() or "\n" in task:
        raise ValueError("task must be a non-empty single line")
````

- [ ] **Step 4: 통과를 확인한다**

Run: Step 1 과 같은 pytest 명령
Expected: `7 passed`

- [ ] **Step 5: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker && M=source/openarm/openarm/agnostic/modules/iker
git add $M/prompt_text/single_step.txt $M/prompt_text/multi_step.txt $M/prompts.py $M/tests/test_prompts.py
git commit -F - <<'MSG'
feat(iker): 논문 부록 C 프롬프트 원문(단일·다단계)과 채우기

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 4: VLM 응답 수집(코드 추출·검사·제한 실행)

**Files:**
- Create: `M/interaction.py`, `M/tests/test_interaction.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `interaction`
    - `InteractionError(ValueError)`
    - `Interaction(object_name, keypoint_ids: tuple[int,...], grasp_mode: bool, coordinates: Mapping[int, (x,y,z)], done: bool)`
    - `extract_code_block(response) -> str`
    - `check_code(code) -> ast.Module`
    - `run_interaction(code, keypoints: Mapping[int, xyz], timeout_s=2.0) -> Interaction`
    - 상수 `DEFAULT_TIMEOUT_S`

- [ ] **Step 1: 실패하는 테스트를 작성한다**

Create `source/openarm/openarm/agnostic/modules/iker/tests/test_interaction.py`:

````python
"""interaction — code block extraction, restricted execution, result parsing (no Isaac)."""

import numpy as np
import pytest

from openarm.agnostic.modules.iker.interaction import InteractionError, check_code, extract_code_block, run_interaction

KEYPOINTS = {1: (0.39, 0.14, 0.25), 2: (0.15, 0.14, 0.25), 3: (0.27, 0.09, 0.25), 4: (0.27, 0.19, 0.25), 5: (0.40, -0.15, 0.38)}
GOOD_RESPONSE = '''The shoe goes on the rack.
```python
import numpy as np

def get_interaction_data(keypoint_coordinates):
    """Move the left shoe next to the other shoe."""
    object_to_interact = "left shoe"
    keypoint_indices_to_interact = [1, 2, 3, 4]
    grasp_mode = True
    for k in ["1", "2", "3", "4"]:
        keypoint_coordinates[k] = keypoint_coordinates[k] + np.array([0.0, -0.2, 0.12])
    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
```
'''


def _function(body: str) -> str:
    return "def get_interaction_data(k):\n" + "".join(f"    {line}\n" for line in body.splitlines())


def test_good_response_runs_and_parses():
    result = run_interaction(extract_code_block(GOOD_RESPONSE), KEYPOINTS)
    assert (result.object_name, result.keypoint_ids, result.grasp_mode, result.done) == ("left shoe", (1, 2, 3, 4), True, False)
    assert np.allclose(result.coordinates[1], (0.39, -0.06, 0.37))
    assert np.allclose(result.coordinates[5], KEYPOINTS[5])


def test_caller_keypoints_are_not_mutated():
    arrays = {k: np.array(v) for k, v in KEYPOINTS.items()}
    run_interaction(extract_code_block(GOOD_RESPONSE), arrays)
    assert all(np.array_equal(arrays[k], np.array(v)) for k, v in KEYPOINTS.items())


def test_string_ids_are_accepted():
    code = _function("return 'shoe', ['1', '2'], False, k")
    assert run_interaction(code, KEYPOINTS).keypoint_ids == (1, 2)


@pytest.mark.parametrize("response", ["no code at all", "```python\nx = 1\n```\n```python\ny = 2\n```"])
def test_exactly_one_python_block_is_required(response):
    with pytest.raises(InteractionError, match="exactly one"):
        extract_code_block(response)


@pytest.mark.parametrize(
    "code, fragment",
    [
        ("import os", "only numpy"),
        ("from numpy import array", "from-imports"),
        (_function("return open('x')"), "open"),
        (_function("return k.__class__"), "attribute"),
        (_function("return eval('1')"), "eval"),
        (_function("return __builtins__"), "__builtins__"),
        ("x = 1\n" + _function("return None"), "top-level"),
        (_function("try:\n    return None\nexcept Exception:\n    return None"), "Try"),
        ("def other(k):\n    return None\n", "no top-level function"),
        ("def get_interaction_data(k:\n", "syntax error"),
    ],
)
def test_forbidden_code_is_rejected_before_running(code, fragment):
    with pytest.raises(InteractionError, match=fragment):
        check_code(code)


def test_runtime_errors_and_timeouts_become_interaction_errors():
    with pytest.raises(InteractionError, match="ZeroDivisionError"):
        run_interaction(_function("return 1 / 0"), KEYPOINTS)
    with pytest.raises(InteractionError, match="exceeded"):
        run_interaction(_function("while True:\n    pass"), KEYPOINTS, timeout_s=0.2)


def test_multi_step_done_is_reported():
    result = run_interaction(_function("done = True\nif done:\n    return"), KEYPOINTS)
    assert result.done is True and result.keypoint_ids == ()


@pytest.mark.parametrize(
    "body, fragment",
    [
        ("return 1, 2", "must return"),
        ("return '', [1], True, k", "object_to_interact"),
        ("return 'shoe', [], True, k", "keypoint_indices"),
        ("return 'shoe', [1, 1], True, k", "duplicates"),
        ("return 'shoe', [1], 'yes', k", "grasp_mode"),
        ("return 'shoe', [True], True, k", "not an integer"),
        ("k['1'] = [1.0, 2.0]\nreturn 'shoe', [1], True, k", "shape"),
    ],
)
def test_malformed_return_values_are_rejected(body, fragment):
    with pytest.raises(InteractionError, match=fragment):
        run_interaction(_function(body), KEYPOINTS)
````

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_interaction.py -q -p no:cacheprovider`
Expected: FAIL. `ModuleNotFoundError: No module named 'openarm.agnostic.modules.iker.interaction'`

- [ ] **Step 3: 구현한다**

금지 노드 검사(`ast.walk`)를 최상위 문 검사보다 **먼저** 한다. 순서가 반대면 `from numpy import array` 가 "top-level statement" 로 먼저 잡혀 문구 테스트가 깨진다(프로토타입에서 확인).

Create `source/openarm/openarm/agnostic/modules/iker/interaction.py`:

````python
"""Run a VLM ``get_interaction_data`` response in a restricted namespace (design spec §6).

The response must contain exactly one fenced Python block that defines ``get_interaction_data``.
Only ``import numpy`` is allowed. Dunder names and attributes, ``try``, classes and the builtins that
reach the interpreter (``open``, ``exec``, ``eval``, ``getattr`` ...) are rejected before anything
runs, and the call itself runs under a wall-clock limit.
"""

from __future__ import annotations

import ast
import builtins
import re
import signal
import threading
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

FUNCTION_NAME = "get_interaction_data"
DEFAULT_TIMEOUT_S = 2.0
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_BANNED_NAMES = frozenset(
    {
        "open", "exec", "eval", "compile", "globals", "locals", "vars", "getattr", "setattr", "delattr",
        "input", "breakpoint", "help", "exit", "quit", "memoryview", "type", "object", "super",
    }
)
_BANNED_NODES = (
    ast.ClassDef, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal, ast.Try, getattr(ast, "TryStar", ast.Try),
    ast.Yield, ast.YieldFrom, ast.Await,
)
_ALLOWED_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len", "list",
        "max", "min", "range", "reversed", "round", "sorted", "str", "sum", "tuple", "zip",
    )
}


class InteractionError(ValueError):
    """The response cannot be turned into an interaction."""


@dataclass(frozen=True)
class Interaction:
    object_name: str
    keypoint_ids: tuple[int, ...]
    grasp_mode: bool
    coordinates: Mapping[int, tuple[float, float, float]]
    done: bool


class _Timeout(BaseException):
    """Raised by the alarm handler; a BaseException so the checked code cannot catch it."""


def extract_code_block(response: str) -> str:
    blocks = _CODE_BLOCK.findall(response)
    if len(blocks) != 1:
        raise InteractionError(f"expected exactly one python code block, found {len(blocks)}")
    return blocks[0]


def check_code(code: str) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise InteractionError(f"syntax error: {exc.msg} (line {exc.lineno})") from None
    for node in ast.walk(tree):
        _check_node(node)
    for node in tree.body:
        is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not (is_docstring or isinstance(node, (ast.Import, ast.FunctionDef))):
            raise InteractionError(f"top-level statement not allowed: {type(node).__name__} (line {node.lineno})")
    if not any(isinstance(node, ast.FunctionDef) and node.name == FUNCTION_NAME for node in tree.body):
        raise InteractionError(f"no top-level function named {FUNCTION_NAME}")
    return tree


def _check_node(node: ast.AST) -> None:
    line = getattr(node, "lineno", "?")
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name != "numpy":
                raise InteractionError(f"only numpy may be imported, got {alias.name!r} (line {line})")
    elif isinstance(node, ast.ImportFrom):
        raise InteractionError(f"from-imports are not allowed (line {line})")
    elif isinstance(node, _BANNED_NODES):
        raise InteractionError(f"{type(node).__name__} is not allowed (line {line})")
    elif isinstance(node, ast.Name) and (node.id in _BANNED_NAMES or node.id.startswith("__")):
        raise InteractionError(f"name {node.id!r} is not allowed (line {line})")
    elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
        raise InteractionError(f"attribute {node.attr!r} is not allowed (line {line})")


def run_interaction(
    code: str, keypoints: Mapping[int, Sequence[float]], timeout_s: float = DEFAULT_TIMEOUT_S
) -> Interaction:
    """Execute checked code on a copy of ``{"1": np.array([x, y, z]), ...}`` and parse its return value."""
    tree = check_code(code)
    inputs = {}
    for key, xyz in keypoints.items():
        point = np.asarray(xyz, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"keypoint {key} must be a finite xyz triple")
        inputs[str(int(key))] = point.copy()
    namespace = {"__builtins__": dict(_ALLOWED_BUILTINS, __import__=_import_numpy_only), "np": np, "numpy": np}
    try:
        exec(compile(tree, "<interaction>", "exec"), namespace)
    except Exception as exc:  # noqa: BLE001 - any failure of untrusted code is reported, not raised
        raise InteractionError(f"module execution failed: {type(exc).__name__}: {exc}") from None
    try:
        result = _call_with_timeout(namespace[FUNCTION_NAME], inputs, timeout_s)
    except InteractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InteractionError(f"{FUNCTION_NAME} raised {type(exc).__name__}: {exc}") from None
    return _parse_result(result)


def _import_numpy_only(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002 - builtin signature
    if name != "numpy" or level != 0:
        raise ImportError(f"import of {name!r} is not allowed")
    return np


def _call_with_timeout(fn, argument, timeout_s: float):
    if threading.current_thread() is not threading.main_thread():
        raise InteractionError("interaction code must run on the main thread (the time limit uses SIGALRM)")

    def _on_alarm(signum, frame):
        raise _Timeout()

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return fn(argument)
    except _Timeout:
        raise InteractionError(f"{FUNCTION_NAME} exceeded {timeout_s} s") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _parse_result(result) -> Interaction:
    if result is None:
        return Interaction(object_name="", keypoint_ids=(), grasp_mode=False, coordinates={}, done=True)
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise InteractionError(
            f"{FUNCTION_NAME} must return (object_to_interact, keypoint_indices_to_interact, grasp_mode, "
            "keypoint_coordinates)"
        )
    name, ids, grasp_mode, coordinates = result
    if not isinstance(name, str) or not name.strip():
        raise InteractionError("object_to_interact must be a non-empty string")
    if isinstance(ids, (str, bytes)) or not isinstance(ids, (list, tuple, np.ndarray)) or len(ids) == 0:
        raise InteractionError("keypoint_indices_to_interact must be a non-empty list")
    keypoint_ids = tuple(_as_keypoint_id(value) for value in ids)
    if len(set(keypoint_ids)) != len(keypoint_ids):
        raise InteractionError(f"keypoint_indices_to_interact has duplicates: {list(keypoint_ids)}")
    if not isinstance(grasp_mode, (bool, np.bool_)):
        raise InteractionError("grasp_mode must be a boolean")
    if not isinstance(coordinates, dict):
        raise InteractionError("keypoint_coordinates must be a dict")
    parsed = {}
    for key, value in coordinates.items():
        try:
            point = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise InteractionError(f"coordinate {key!r} is not numeric") from None
        if point.shape != (3,):
            raise InteractionError(f"coordinate {key!r} must have shape (3,), got {point.shape}")
        parsed[_as_keypoint_id(key)] = (float(point[0]), float(point[1]), float(point[2]))
    return Interaction(name.strip(), keypoint_ids, bool(grasp_mode), parsed, done=False)


def _as_keypoint_id(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise InteractionError(f"keypoint id {value!r} is not an integer")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise InteractionError(f"keypoint id {value!r} is not an integer")
````

- [ ] **Step 4: 통과를 확인한다**

Run: Step 2 와 같은 명령
Expected: `24 passed`

- [ ] **Step 5: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker && M=source/openarm/openarm/agnostic/modules/iker
git add $M/interaction.py $M/tests/test_interaction.py
git commit -F - <<'MSG'
feat(iker): VLM 응답 코드 추출·AST 검사·시간 제한 실행

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 5: 게이트 G1~G7

**Files:**
- Create: `M/gate.py`, `M/tests/test_gate.py`

**Interfaces:**
- Consumes: `interaction.Interaction`, `InteractionError`, `extract_code_block`, `run_interaction`, `DEFAULT_TIMEOUT_S`
- Produces:
  - `gate`
    - `Support(name, top_z, x_range=(-inf,inf), y_range=(-inf,inf)).contains_xy(x, y)`
    - `MovableObject(name, keypoint_ids, local_offsets (K,3), hull_local (M,3), init_keypoints (K,3), start_support_z)`
    - `GateConfig(supports, palm_box_min, palm_box_max, eval_success_m=0.05, train_success_norm=0.10, penetration_tol_m=0.01, push_lift_tol_m=0.02, min_travel_m=1e-3)`
    - `GateReport(passed, failures: tuple[str], metrics: Mapping, object_name, target_keypoints)`
    - `kabsch(source, target) -> (R, t)`
    - `evaluate_gate(interaction, movables, static_ids, cfg) -> GateReport`
    - `gate_response(response, keypoints, movables, static_ids, cfg, timeout_s) -> (Interaction | None, GateReport)`
  - 실패 문구는 `"G1: "` … `"G7: "` 로 시작한다.
  - `metrics` 키: `fitted_center`, `fitted_rotation`, `residual_m`, `mean_distance_m`, `travel_m`, `normalized_error`(이동 0 이면 None), `hull_min_z`, `support`, `support_top_z`, `in_palm_box`, `grasp_mode`

- [ ] **Step 1: 실패하는 테스트를 작성한다**

`PUBLIC_SPACING_TARGET` 은 공개 데이터의 목표 관계를 다른 신발 옆으로 옮기기만 한 값이다(스펙 §10 "간격 불일치를 G4 실패 사례로 고정").
평균 거리 0.0258 m 는 5 cm 안이고, 정규화 0.1147 이 0.10 을 넘어 실패한다.

Create `source/openarm/openarm/agnostic/modules/iker/tests/test_gate.py`:

````python
"""gate — G1-G7 on the nominal shoe scene (no Isaac)."""

import numpy as np
import pytest

from openarm.agnostic.modules.iker import gate
from openarm.agnostic.modules.iker.interaction import Interaction

REST = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
MOVE_OFFSETS = np.array([[0.0, 0.0, 0.1248], [0.0, 0.0, -0.1248], [0.0481, 0.0, 0.0], [-0.0481, 0.0, 0.0]])
OTHER_OFFSETS = np.array([[0.0, 0.0, 0.1261], [0.0, 0.0, -0.1261], [0.0485, 0.0, 0.0], [-0.0485, 0.0, 0.0]])
MOVE_HULL = np.array([[sx * 0.0481, sy * 0.05214, sz * 0.1248] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
OTHER_HULL = np.array([[sx * 0.0485, sy * 0.05502, sz * 0.1261] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
TABLE_Z, RACK_Z = 0.205, 0.325
MOVE_START = np.array([0.27, 0.14, TABLE_Z + 0.05214])
OTHER_START = np.array([0.27, -0.15, RACK_Z + 0.05502])
SUPPORTS = (gate.Support("rack", RACK_Z, (0.11, 0.43), (-0.33, 0.03)), gate.Support("table", TABLE_Z))
CFG = gate.GateConfig(supports=SUPPORTS, palm_box_min=(0.20, -0.55, 0.20), palm_box_max=(0.55, 0.22, 0.70))
MOVABLES = (
    gate.MovableObject("shoe_move", (1, 2, 3, 4), MOVE_OFFSETS, MOVE_HULL, MOVE_START + MOVE_OFFSETS @ REST.T, TABLE_Z),
    gate.MovableObject("shoe_other", (5, 6, 7, 8), OTHER_OFFSETS, OTHER_HULL, OTHER_START + OTHER_OFFSETS @ REST.T, RACK_Z),
)
STATIC_IDS = tuple(range(9, 21))
BESIDE_OTHER = [0.2573, -0.021, RACK_Z + 0.05214]
# The public IKER target relation carried beside the other shoe without refitting (spacing 0.150 / 0.100 m).
PUBLIC_SPACING_TARGET = np.array(
    [[0.3323, -0.021, 0.4065], [0.1823, -0.021, 0.4065], [0.2573, -0.071, 0.4065], [0.2573, 0.029, 0.4065]]
)


def _rigid(center) -> np.ndarray:
    return np.asarray(center, dtype=float) + MOVE_OFFSETS @ REST.T


def _interaction(targets, ids=(1, 2, 3, 4), grasp=True) -> Interaction:
    coordinates = {k: tuple(p) for k, p in zip((1, 2, 3, 4), np.asarray(targets, dtype=float).tolist())}
    return Interaction("shoe", tuple(ids), grasp, coordinates, False)


def _codes(report: gate.GateReport) -> list[str]:
    return [failure[:2] for failure in report.failures]


def test_kabsch_recovers_a_rigid_motion():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r, t = gate.kabsch(MOVE_OFFSETS, MOVE_OFFSETS @ rotation.T + [0.1, 0.2, 0.3])
    assert np.allclose(r, rotation) and np.allclose(t, [0.1, 0.2, 0.3])


def test_rigid_target_resting_beside_the_other_shoe_passes():
    report = gate.evaluate_gate(_interaction(_rigid(BESIDE_OTHER)), MOVABLES, STATIC_IDS, CFG)
    assert report.passed, report.failures
    assert report.object_name == "shoe_move" and report.metrics["support"] == "rack"
    assert report.metrics["mean_distance_m"] == pytest.approx(0.0, abs=1e-9)
    assert report.metrics["hull_min_z"] == pytest.approx(RACK_Z, abs=1e-9)
    assert np.allclose(report.target_keypoints, _rigid(BESIDE_OTHER))


def test_done_fails_g1():
    report = gate.evaluate_gate(Interaction("", (), False, {}, True), MOVABLES, STATIC_IDS, CFG)
    assert _codes(report) == ["G1"]


@pytest.mark.parametrize("ids, fragment", [((1, 99), "unknown"), ((9, 10), "static"), ((1, 5), "span")])
def test_keypoint_ids_must_belong_to_one_movable_object(ids, fragment):
    report = gate.evaluate_gate(_interaction(_rigid(BESIDE_OTHER), ids=ids), MOVABLES, STATIC_IDS, CFG)
    assert _codes(report) == ["G2"] and fragment in report.failures[0]


def test_missing_or_non_finite_targets_fail_g3():
    targets = _rigid(BESIDE_OTHER)
    targets[2, 0] = np.nan
    assert _codes(gate.evaluate_gate(_interaction(targets), MOVABLES, STATIC_IDS, CFG)) == ["G3"]
    partial = Interaction("shoe", (1, 2, 3, 4), True, {1: (0.3, 0.0, 0.3)}, False)
    report = gate.evaluate_gate(partial, MOVABLES, STATIC_IDS, CFG)
    assert _codes(report) == ["G3"] and "[2, 3, 4]" in report.failures[0]


def test_public_iker_target_spacing_fails_only_the_normalized_bound_of_g4():
    report = gate.evaluate_gate(_interaction(PUBLIC_SPACING_TARGET), MOVABLES, STATIC_IDS, CFG)
    assert _codes(report) == ["G4"] and "normalized" in report.failures[0]
    assert report.metrics["mean_distance_m"] == pytest.approx(0.0258, abs=1e-3)
    assert report.metrics["normalized_error"] == pytest.approx(0.1147, abs=2e-3)


def test_target_sunk_into_the_rack_fails_g5():
    sunk = [BESIDE_OTHER[0], BESIDE_OTHER[1], BESIDE_OTHER[2] - 0.03]
    assert _codes(gate.evaluate_gate(_interaction(_rigid(sunk)), MOVABLES, STATIC_IDS, CFG)) == ["G5"]


def test_target_outside_the_palm_box_fails_g6():
    far = [0.60, 0.10, TABLE_Z + 0.05214]
    assert _codes(gate.evaluate_gate(_interaction(_rigid(far)), MOVABLES, STATIC_IDS, CFG)) == ["G6"]


def test_push_cannot_lift_the_shoe_onto_the_rack_g7():
    report = gate.evaluate_gate(_interaction(_rigid(BESIDE_OTHER), grasp=False), MOVABLES, STATIC_IDS, CFG)
    assert _codes(report) == ["G7"]


def test_push_along_the_table_passes():
    report = gate.evaluate_gate(_interaction(_rigid([0.27, 0.05, TABLE_Z + 0.05214]), grasp=False), MOVABLES, STATIC_IDS, CFG)
    assert report.passed, report.failures


RELATIVE_RESPONSE = '''```python
import numpy as np

def get_interaction_data(keypoint_coordinates):
    """Place the shoe to the left of the other shoe, on the rack."""
    object_to_interact = "right shoe"
    keypoint_indices_to_interact = [1, 2, 3, 4]
    grasp_mode = True
    for moving, reference in (("1", "5"), ("2", "6"), ("3", "7"), ("4", "8")):
        keypoint_coordinates[moving] = keypoint_coordinates[reference] + np.array([0.0, 0.129, -0.0029])
    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
```'''


def _all_keypoints() -> dict[int, tuple[float, float, float]]:
    points = list(MOVABLES[0].init_keypoints) + list(MOVABLES[1].init_keypoints)
    points += [(x, y, RACK_Z) for x in (0.16, 0.27, 0.38) for y in (-0.28, -0.1933, -0.1067, -0.02)]
    return {label: tuple(float(c) for c in p) for label, p in enumerate(points, start=1)}


def test_gate_response_runs_and_gates_a_relative_response():
    interaction, report = gate.gate_response(RELATIVE_RESPONSE, _all_keypoints(), MOVABLES, STATIC_IDS, CFG)
    assert interaction is not None and interaction.object_name == "right shoe"
    assert report.passed, report.failures


def test_gate_response_turns_unusable_text_into_g1():
    interaction, report = gate.gate_response("I would move the shoe.", _all_keypoints(), MOVABLES, STATIC_IDS, CFG)
    assert interaction is None and _codes(report) == ["G1"] and "code block" in report.failures[0]
````

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_gate.py -q -p no:cacheprovider`
Expected: FAIL. `ImportError`(`gate` 없음).

- [ ] **Step 3: 구현한다**

Create `source/openarm/openarm/agnostic/modules/iker/gate.py`:

````python
"""Mechanical gate for a VLM interaction (design spec §6, G1-G7).

The gate never edits the prompt. It checks that the targets describe a pose the policy can reach:
G1 the code ran and is not ``done``; G2 the ids belong to one movable object; G3 its targets are
finite; G4 the best rigid fit of the object's keypoints meets the evaluation (5 cm) and training
(normalized 0.10) success bounds; G5 that pose does not sink into its support; G6 its centre lies in
the arm's palm box; G7 a push does not raise the object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .interaction import DEFAULT_TIMEOUT_S, Interaction, InteractionError, extract_code_block, run_interaction


@dataclass(frozen=True)
class Support:
    name: str
    top_z: float
    x_range: tuple[float, float] = (-math.inf, math.inf)
    y_range: tuple[float, float] = (-math.inf, math.inf)

    def contains_xy(self, x: float, y: float) -> bool:
        return self.x_range[0] <= x <= self.x_range[1] and self.y_range[0] <= y <= self.y_range[1]


@dataclass(frozen=True)
class MovableObject:
    name: str
    keypoint_ids: tuple[int, ...]
    local_offsets: np.ndarray  # (K, 3); row i belongs to keypoint_ids[i]
    hull_local: np.ndarray  # (M, 3) convex-hull vertices in the object frame
    init_keypoints: np.ndarray  # (K, 3) world, at the snapshot
    start_support_z: float


@dataclass(frozen=True)
class GateConfig:
    supports: tuple[Support, ...]  # the first support whose rectangle holds the fitted centre is used
    palm_box_min: tuple[float, float, float]
    palm_box_max: tuple[float, float, float]
    eval_success_m: float = 0.05
    train_success_norm: float = 0.10
    penetration_tol_m: float = 0.01
    push_lift_tol_m: float = 0.02
    min_travel_m: float = 1e-3


@dataclass(frozen=True)
class GateReport:
    passed: bool
    failures: tuple[str, ...]
    metrics: Mapping[str, object]
    object_name: str | None = None
    target_keypoints: tuple[tuple[float, float, float], ...] = ()


def kabsch(source, target) -> tuple[np.ndarray, np.ndarray]:
    """Proper rotation ``R`` and translation ``t`` minimising ``sum |R source_i + t - target_i|^2``."""
    p = np.asarray(source, dtype=float)
    q = np.asarray(target, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
        raise ValueError("source and target must both be (N, 3) with N >= 3")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
        raise ValueError("kabsch inputs contain non-finite values")
    p_center, q_center = p.mean(axis=0), q.mean(axis=0)
    u, _, vt = np.linalg.svd((p - p_center).T @ (q - q_center))
    d = 1.0 if np.linalg.det(vt.T @ u.T) >= 0.0 else -1.0
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, q_center - rotation @ p_center


def gate_response(
    response: str,
    keypoints: Mapping[int, Sequence[float]],
    movables: Sequence[MovableObject],
    static_ids: Sequence[int],
    cfg: GateConfig,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[Interaction | None, GateReport]:
    """Extract, run and gate one VLM response; a response that cannot run fails G1."""
    try:
        interaction = run_interaction(extract_code_block(response), keypoints, timeout_s)
    except InteractionError as exc:
        return None, GateReport(passed=False, failures=(f"G1: {exc}",), metrics={})
    return interaction, evaluate_gate(interaction, movables, static_ids, cfg)


def evaluate_gate(
    interaction: Interaction, movables: Sequence[MovableObject], static_ids: Sequence[int], cfg: GateConfig
) -> GateReport:
    if interaction.done:
        return _fail("G1: done=True, but a single-step task needs an interaction")
    owner = {kid: obj for obj in movables for kid in obj.keypoint_ids}
    ids = list(interaction.keypoint_ids)
    unknown = [kid for kid in ids if kid not in owner and kid not in set(static_ids)]
    if unknown:
        return _fail(f"G2: unknown keypoint ids {unknown}")
    on_static = [kid for kid in ids if kid not in owner]
    if on_static:
        return _fail(f"G2: keypoint ids {on_static} belong to a static object")
    names = sorted({owner[kid].name for kid in ids})
    if len(names) != 1:
        return _fail(f"G2: keypoint ids span objects {names}")
    obj = owner[ids[0]]
    missing = [kid for kid in obj.keypoint_ids if kid not in interaction.coordinates]
    if missing:
        return _fail(f"G3: no target coordinates for keypoints {missing} of {obj.name}", obj.name)
    targets = np.array([interaction.coordinates[kid] for kid in obj.keypoint_ids], dtype=float)
    if not np.all(np.isfinite(targets)):
        return _fail(f"G3: non-finite target coordinates for {obj.name}", obj.name)
    return _check_fit(interaction, obj, targets, cfg)


def _check_fit(interaction: Interaction, obj: MovableObject, targets: np.ndarray, cfg: GateConfig) -> GateReport:
    rotation, center = kabsch(obj.local_offsets, targets)
    residual = np.linalg.norm(obj.local_offsets @ rotation.T + center - targets, axis=1)
    travel = np.linalg.norm(targets - obj.init_keypoints, axis=1)
    hull_min_z = float((obj.hull_local @ rotation.T + center)[:, 2].min())
    support = next((s for s in cfg.supports if s.contains_xy(center[0], center[1])), None)
    in_box = bool(np.all(center >= np.asarray(cfg.palm_box_min)) and np.all(center <= np.asarray(cfg.palm_box_max)))
    mean_distance = float(residual.mean())
    moves = bool(np.all(travel > cfg.min_travel_m))
    normalized = float((residual / travel).mean()) if moves else None

    failures = []
    if not mean_distance <= cfg.eval_success_m:
        failures.append(f"G4: mean keypoint distance at the best rigid fit {mean_distance:.4f} m > {cfg.eval_success_m} m")
    if normalized is None:
        failures.append(f"G4: target equals the start for some keypoints (travel {np.round(travel, 4).tolist()} m)")
    elif not normalized <= cfg.train_success_norm:
        failures.append(f"G4: normalized error at the best rigid fit {normalized:.4f} > {cfg.train_success_norm}")
    if support is None:
        failures.append(f"G5: fitted centre {np.round(center, 4).tolist()} is above no support")
    elif not hull_min_z >= support.top_z - cfg.penetration_tol_m:
        failures.append(f"G5: fitted pose sinks {support.top_z - hull_min_z:.4f} m into {support.name}")
    if not in_box:
        failures.append(f"G6: fitted centre {np.round(center, 4).tolist()} is outside the palm box")
    if not interaction.grasp_mode and not hull_min_z <= obj.start_support_z + cfg.push_lift_tol_m:
        failures.append(f"G7: push mode cannot raise {obj.name} to {hull_min_z:.4f} m (start support {obj.start_support_z} m)")

    metrics = {
        "fitted_center": center.tolist(),
        "fitted_rotation": rotation.tolist(),
        "residual_m": residual.tolist(),
        "mean_distance_m": mean_distance,
        "travel_m": travel.tolist(),
        "normalized_error": normalized,
        "hull_min_z": hull_min_z,
        "support": support.name if support else None,
        "support_top_z": support.top_z if support else None,
        "in_palm_box": in_box,
        "grasp_mode": interaction.grasp_mode,
    }
    return GateReport(
        passed=not failures,
        failures=tuple(failures),
        metrics=metrics,
        object_name=obj.name,
        target_keypoints=tuple(tuple(float(c) for c in row) for row in targets),
    )


def _fail(message: str, object_name: str | None = None) -> GateReport:
    return GateReport(passed=False, failures=(message,), metrics={}, object_name=object_name)
````

- [ ] **Step 4: 통과를 확인한다**

Run: Step 2 와 같은 명령
Expected: `14 passed`

- [ ] **Step 5: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker && M=source/openarm/openarm/agnostic/modules/iker
git add $M/gate.py $M/tests/test_gate.py
git commit -F - <<'MSG'
feat(iker): 도달성 게이트 G1-G7 과 응답 게이트 진입점

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 6: 고정 IKER 보상(레거시 이식)

**Files:**
- Create: `M/reward.py`, `M/tests/test_reward.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `reward`
    - `IkerRewardCfg`(frozen 아님; 기본 높이는 레거시 절대값 1.0/0.9)
    - `RewardOutput(reward, terminated, success_count, failure_count)`
    - `reward_cfg_for_start_support(start_support_z, **overrides) -> IkerRewardCfg`(0.205 → under 0.268, fall 0.168)
    - `transform_keypoints(position (N,3), quaternion_wxyz (N,4), offsets (K,3)) -> (N,K,3)`
    - `compute_reward_and_termination(actions, eef_pos, object_pos, current_keypoints, init_keypoints, target_keypoints, progress, success_count, failure_count, force_penalty=None, cfg=None) -> RewardOutput`

동등성은 식 하나가 아니라 인터페이스 단위로 확인한다(관찰 0156).
- 출처: `repo/skill_gen/IKER/isaacgymenvs/tasks/shoe_place.py:722-778`(2026-09-13 원문 읽음).
- 테스트는 그 함수를 줄 단위로 옮긴 사본과 무작위 상태 256개에서 보상·종료·카운터를 비교한다.
- 식 밖의 차이(끝단 몸체, 높이 기준, 힘 벌점, 방향 항, 카운터 리셋 주체, 시간 초과 처리)는 모듈 docstring 표에 적혀 있다.

- [ ] **Step 1: 실패하는 테스트를 작성한다**

Create `source/openarm/openarm/agnostic/modules/iker/tests/test_reward.py`:

````python
"""reward — legacy IKER reward port (no Isaac)."""

import pytest
import torch

from openarm.agnostic.modules.iker.reward import (
    IkerRewardCfg,
    RewardOutput,
    compute_reward_and_termination,
    reward_cfg_for_start_support,
    transform_keypoints,
)

OFFSETS = torch.tensor([[0.0, 0.0, 0.12], [0.0, 0.0, -0.12], [0.0, 0.075, 0.0], [0.0, -0.075, 0.0]])
IDENTITY = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


def _keypoints(n: int, base: torch.Tensor) -> torch.Tensor:
    return base.reshape(1, 4, 3).repeat(n, 1, 1)


def _inputs(n: int = 2, **overrides) -> dict:
    target = _keypoints(n, OFFSETS + torch.tensor([0.6, 0.1, 1.2]))
    init = _keypoints(n, OFFSETS + torch.tensor([0.3, 0.0, 1.2]))
    inputs = dict(
        actions=torch.zeros(n, 6),
        eef_pos=torch.tensor([[0.3, 0.0, 1.3]]).repeat(n, 1),
        object_pos=torch.tensor([[0.3, 0.0, 1.2]]).repeat(n, 1),
        current_keypoints=init.clone(),
        init_keypoints=init,
        target_keypoints=target,
        progress=torch.zeros(n, dtype=torch.long),
        success_count=torch.zeros(n),
        failure_count=torch.zeros(n),
        force_penalty=torch.zeros(n),
    )
    inputs.update(overrides)
    return inputs


def _legacy_compute_xarm_reward(reset_buf, progress_buf, success_buf, fail_buf, actions, states, max_episode_length):
    """Transcription of shoe_place.py:722-778 without TorchScript; `force` and `eef_acc` are passed as zeros."""
    d = torch.norm(states["obj_pos"] - states["eef_pos"], dim=-1)
    d = torch.clamp(d, min=0.05, max=0.5)
    dist_reward = 1 - torch.tanh(5.0 * d)
    action_penalty = torch.sum(actions**2, dim=-1)
    under_penalty = (states["eef_pos"][:, 2] < 1.0) * 1
    target_dir_vec = states["target"] - states["init_pos"]
    dist_dir_vec = states["track_pos"] - states["init_pos"]
    cos_theta = torch.sum(target_dir_vec * dist_dir_vec, dim=-1) / (
        torch.norm(target_dir_vec, dim=-1) * torch.norm(dist_dir_vec, dim=-1) + 0.1
    )
    dir_reward = torch.clamp(cos_theta, min=0, max=1.0)
    dir_reward = torch.sum(dir_reward, dim=-1) / states["track_pos"].shape[1]
    target_loc = states["target"]
    d_target = torch.norm(states["track_pos"] - target_loc, dim=-1) / torch.norm(target_dir_vec, dim=-1)
    align_reward = torch.clamp(0.5 * (1 - d_target**2) + 0.5 * (1 - torch.tanh(2.0 * d_target)), min=0.0, max=1.0)
    align_reward = torch.sum(align_reward, dim=-1) / states["track_pos"].shape[1]
    success_buf = success_buf + (torch.sum(d_target, dim=-1) / states["track_pos"].shape[1] <= 0.10) * 1.0
    fall_penalty = (states["obj_pos"][:, 2] < 0.9) * 1
    fail_buf = fail_buf + fall_penalty
    force_penalty = 0.5 * torch.clamp(torch.norm(states["force"][:, 8], dim=-1) * 0.1, min=0.0, max=1.0)
    force_penalty += 0.5 * torch.clamp(10 * torch.norm(states["eef_acc"][:, :3], dim=-1), min=0, max=1.0)
    success = success_buf > 20
    fail = fail_buf > 20
    bonus = torch.max(10 * (max_episode_length - progress_buf), 300 * torch.ones_like(progress_buf))
    rewards = (
        0.2 * dist_reward + 4 * align_reward + bonus * success + 2.0 * dir_reward - fall_penalty
        - 0.1 * under_penalty - 750 * fail - 0.20 * action_penalty - 3.0 * force_penalty
    )
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.logical_or(reset_buf, success)
    reset_buf = torch.logical_or(reset_buf, fail)
    return rewards, reset_buf, success_buf, fail_buf


def test_matches_the_legacy_function_on_random_states():
    torch.manual_seed(0)
    n = 256
    init = torch.rand(n, 4, 3) + torch.tensor([0.0, 0.0, 1.0])
    target = init + 0.3 * torch.randn(n, 4, 3)
    near = (torch.arange(n) % 2 == 0)[:, None, None]
    current = torch.where(near, target + 0.01 * torch.randn(n, 4, 3), init + 0.2 * torch.randn(n, 4, 3))
    eef = torch.rand(n, 3) + torch.tensor([0.0, 0.0, 0.5])
    obj = torch.rand(n, 3) + torch.tensor([0.0, 0.0, 0.5])
    actions = torch.randn(n, 6)
    progress = torch.randint(0, 200, (n,))
    success = torch.randint(0, 25, (n,)).float()
    fail = torch.randint(0, 25, (n,)).float()
    states = {
        "obj_pos": obj, "eef_pos": eef, "target": target, "init_pos": init, "track_pos": current,
        "force": torch.zeros(n, 9, 3), "eef_acc": torch.zeros(n, 6),
    }
    legacy = _legacy_compute_xarm_reward(torch.zeros(n, dtype=torch.long), progress, success, fail, actions, states, 200.0)
    ours = compute_reward_and_termination(
        actions=actions, eef_pos=eef, object_pos=obj, current_keypoints=current, init_keypoints=init,
        target_keypoints=target, progress=progress, success_count=success, failure_count=fail,
    )
    assert torch.allclose(ours.reward, legacy[0].float(), atol=1e-4)
    assert torch.equal(ours.terminated, legacy[1].bool())
    assert torch.equal(ours.success_count, legacy[2].float())
    assert torch.equal(ours.failure_count, legacy[3].float())
    assert 0 < int(ours.success_count.gt(success).sum()) < n  # both branches of the success counter ran


def test_start_support_moves_the_legacy_height_offsets():
    cfg = reward_cfg_for_start_support(0.205)
    assert cfg.under_height == pytest.approx(0.268)
    assert cfg.fall_height == pytest.approx(0.168)
    assert IkerRewardCfg().under_height == 1.0 and IkerRewardCfg().fall_height == 0.9


def test_transform_keypoints_rotates_offsets():
    half = 0.5**0.5
    out = transform_keypoints(torch.zeros(1, 3), torch.tensor([[half, 0.0, 0.0, half]]), torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(out[0, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    with pytest.raises(ValueError, match="offsets"):
        transform_keypoints(torch.zeros(1, 3), IDENTITY, torch.zeros(4, 2))


def test_output_shapes_and_types():
    out = compute_reward_and_termination(**_inputs(3))
    assert isinstance(out, RewardOutput)
    assert out.reward.shape == (3,) and out.terminated.dtype == torch.bool


def test_zero_action_avoids_the_action_penalty():
    r0 = compute_reward_and_termination(**_inputs(1)).reward
    r1 = compute_reward_and_termination(**_inputs(1, actions=torch.ones(1, 6))).reward
    assert torch.isclose(r0 - r1, torch.tensor([0.20 * 6.0]))


def test_sustained_success_terminates_and_pays_the_bonus():
    inputs = _inputs(1, success_count=torch.tensor([20.0]))
    inputs["current_keypoints"] = inputs["target_keypoints"].clone()
    out = compute_reward_and_termination(**inputs)
    assert out.success_count.item() == 21.0 and out.terminated.item() is True
    assert out.reward.item() > 1000.0


def test_success_counter_is_cumulative_not_consecutive():
    inputs = _inputs(1, success_count=torch.tensor([7.0]))  # the object is away from the target this step
    assert compute_reward_and_termination(**inputs).success_count.item() == 7.0


def test_sustained_failure_terminates_with_the_penalty():
    inputs = _inputs(1, object_pos=torch.tensor([[0.3, 0.0, 0.5]]), failure_count=torch.tensor([20.0]))
    out = compute_reward_and_termination(**inputs)
    assert out.terminated.item() is True and out.reward.item() < -700.0


def test_timeout_terminates_only_on_the_last_step():
    assert compute_reward_and_termination(**_inputs(1, progress=torch.tensor([198]))).terminated.item() is False
    assert compute_reward_and_termination(**_inputs(1, progress=torch.tensor([199]))).terminated.item() is True


def test_bonus_floor_is_300_late_in_the_episode():
    late = _inputs(1, progress=torch.tensor([190]), success_count=torch.tensor([20.0]))
    late["current_keypoints"] = late["target_keypoints"].clone()
    base = _inputs(1, progress=torch.tensor([190]))
    base["current_keypoints"] = base["target_keypoints"].clone()
    diff = compute_reward_and_termination(**late).reward - compute_reward_and_termination(**base).reward
    assert torch.isclose(diff, torch.tensor([300.0]))


def test_force_penalty_is_off_by_default_and_scalable():
    off = compute_reward_and_termination(**_inputs(1, force_penalty=torch.ones(1)))
    on = compute_reward_and_termination(**_inputs(1, force_penalty=torch.ones(1)), cfg=IkerRewardCfg(force_penalty_scale=3.0))
    assert torch.isclose(off.reward - on.reward, torch.tensor([3.0]))


def test_non_finite_keypoints_raise():
    inputs = _inputs(1)
    inputs["current_keypoints"][0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="current_keypoints"):
        compute_reward_and_termination(**inputs)


def test_cfg_fields_are_assignable_for_the_hydra_round_trip():
    cfg = IkerRewardCfg()
    cfg.max_episode_length = 300
    assert cfg.max_episode_length == 300
````

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_reward.py -q -p no:cacheprovider`
Expected: FAIL. `ModuleNotFoundError: No module named 'openarm.agnostic.modules.iker.reward'`

- [ ] **Step 3: 구현한다**

Create `source/openarm/openarm/agnostic/modules/iker/reward.py`:

````python
"""Fixed IKER reward and termination: the legacy ``compute_xarm_reward`` in pure PyTorch (design spec §7).

Source, read 2026-09-13: ``repo/skill_gen/IKER/isaacgymenvs/tasks/shoe_place.py:722-778``.
The equations and constants are unchanged; the interface around them differs as follows.

| | legacy | this module |
|---|---|---|
| end effector | xArm ``eef_pos`` | the caller's palm body position |
| heights | absolute 1.0 (under) and 0.9 (fall) in the lifted legacy world | ``under_height`` / ``fall_height`` in the caller's frame; ``reward_cfg_for_start_support`` keeps the legacy offsets from the start support (legacy slab 0.937) |
| force penalty | ``3.0 * (contact force + eef acceleration)`` | ``force_penalty_scale`` 0.0 until a signal exists |
| orientation term | computed, never added to the reward | not computed |
| counters | cumulative within an episode, zeroed on reset (``shoe_place.py:632-633``) | returned; the environment zeroes them on reset |
| termination | timeout or success (count > 20) or fail (count > 20) | same; the environment reports the timeout as truncation |

Quaternions entering this module are ``wxyz``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

NUM_KEYPOINTS = 4
LEGACY_START_SUPPORT_Z = 0.937
LEGACY_UNDER_HEIGHT = 1.0
LEGACY_FALL_HEIGHT = 0.9


@dataclass
class IkerRewardCfg:
    """Legacy constants with the implicit numbers named.

    Not frozen: Isaac Lab's hydra round trip assigns every nested config field with ``setattr``
    (``isaaclab/utils/dict.py``) and a frozen dataclass raises ``FrozenInstanceError`` there.
    """

    max_episode_length: int = 200
    under_height: float = LEGACY_UNDER_HEIGHT
    fall_height: float = LEGACY_FALL_HEIGHT
    success_tolerance: float = 0.10
    sustain_steps: int = 20
    dist_scale: float = 0.2
    align_scale: float = 4.0
    dir_scale: float = 2.0
    under_scale: float = 0.1
    fail_scale: float = 750.0
    action_scale: float = 0.20
    force_penalty_scale: float = 0.0
    bonus_per_remaining_step: float = 10.0
    bonus_floor: float = 300.0


@dataclass(frozen=True)
class RewardOutput:
    reward: torch.Tensor
    terminated: torch.Tensor
    success_count: torch.Tensor
    failure_count: torch.Tensor


def reward_cfg_for_start_support(start_support_z: float, **overrides) -> IkerRewardCfg:
    """Legacy height thresholds moved to a scene whose moving object starts on ``start_support_z``."""
    return IkerRewardCfg(
        under_height=start_support_z + (LEGACY_UNDER_HEIGHT - LEGACY_START_SUPPORT_Z),
        fall_height=start_support_z - (LEGACY_START_SUPPORT_Z - LEGACY_FALL_HEIGHT),
        **overrides,
    )


def transform_keypoints(position: torch.Tensor, quaternion_wxyz: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Map body-frame keypoint ``offsets`` (K, 3) to the world frame -> (N, K, 3)."""
    if offsets.ndim != 2 or offsets.shape[-1] != 3:
        raise ValueError(f"offsets must have shape (K, 3), got {tuple(offsets.shape)}")
    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError(f"position must have shape (N, 3), got {tuple(position.shape)}")
    if quaternion_wxyz.shape != (position.shape[0], 4):
        raise ValueError(f"quaternion_wxyz must have shape ({position.shape[0]}, 4), got {tuple(quaternion_wxyz.shape)}")
    n, k = position.shape[0], offsets.shape[0]
    quat = quaternion_wxyz[:, None, :].expand(n, k, 4)
    vec = offsets.to(position)[None].expand(n, k, 3)
    w, xyz = quat[..., :1], quat[..., 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return position[:, None, :] + vec + w * t + torch.cross(xyz, t, dim=-1)


def compute_reward_and_termination(
    actions: torch.Tensor,
    eef_pos: torch.Tensor,
    object_pos: torch.Tensor,
    current_keypoints: torch.Tensor,
    init_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    progress: torch.Tensor,
    success_count: torch.Tensor,
    failure_count: torch.Tensor,
    force_penalty: torch.Tensor | None = None,
    cfg: IkerRewardCfg | None = None,
) -> RewardOutput:
    """Legacy ``compute_xarm_reward`` with the same equations. All positions share one frame."""
    cfg = cfg if cfg is not None else IkerRewardCfg()
    n = actions.shape[0]
    for name, value in (
        ("eef_pos", eef_pos),
        ("object_pos", object_pos),
        ("current_keypoints", current_keypoints),
        ("init_keypoints", init_keypoints),
        ("target_keypoints", target_keypoints),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    for name, value in (
        ("current_keypoints", current_keypoints),
        ("init_keypoints", init_keypoints),
        ("target_keypoints", target_keypoints),
    ):
        if tuple(value.shape) != (n, NUM_KEYPOINTS, 3):
            raise ValueError(f"{name} must have shape {(n, NUM_KEYPOINTS, 3)}, got {tuple(value.shape)}")

    d = torch.norm(object_pos - eef_pos, dim=-1).clamp(min=0.05, max=0.5)
    dist_reward = 1.0 - torch.tanh(5.0 * d)
    action_penalty = torch.sum(actions**2, dim=-1)
    under_penalty = (eef_pos[:, 2] < cfg.under_height).float()

    target_dir = target_keypoints - init_keypoints
    moved_dir = current_keypoints - init_keypoints
    target_norm = torch.norm(target_dir, dim=-1)
    cos_theta = torch.sum(target_dir * moved_dir, dim=-1) / (target_norm * torch.norm(moved_dir, dim=-1) + 0.1)
    dir_reward = cos_theta.clamp(0.0, 1.0).mean(dim=-1)

    d_target = torch.norm(current_keypoints - target_keypoints, dim=-1) / target_norm
    align_reward = (0.5 * (1.0 - d_target**2) + 0.5 * (1.0 - torch.tanh(2.0 * d_target))).clamp(0.0, 1.0).mean(dim=-1)

    success_count = success_count + (d_target.mean(dim=-1) <= cfg.success_tolerance).float()
    fall_penalty = (object_pos[:, 2] < cfg.fall_height).float()
    failure_count = failure_count + fall_penalty
    if force_penalty is None:
        force_penalty = torch.zeros_like(dist_reward)

    success = success_count > cfg.sustain_steps
    fail = failure_count > cfg.sustain_steps
    remaining = (cfg.max_episode_length - progress).to(dist_reward)
    bonus = torch.maximum(cfg.bonus_per_remaining_step * remaining, torch.full_like(remaining, cfg.bonus_floor))

    reward = (
        cfg.dist_scale * dist_reward
        + cfg.align_scale * align_reward
        + bonus * success.float()
        + cfg.dir_scale * dir_reward
        - fall_penalty
        - cfg.under_scale * under_penalty
        - cfg.fail_scale * fail.float()
        - cfg.action_scale * action_penalty
        - cfg.force_penalty_scale * force_penalty
    )
    terminated = (progress >= cfg.max_episode_length - 1) | success | fail
    return RewardOutput(reward=reward, terminated=terminated, success_count=success_count, failure_count=failure_count)
````

- [ ] **Step 4: 통과를 확인한다**

Run: Step 2 와 같은 명령
Expected: `13 passed`

- [ ] **Step 5: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker && M=source/openarm/openarm/agnostic/modules/iker
git add $M/reward.py $M/tests/test_reward.py
git commit -F - <<'MSG'
feat(iker): 레거시 compute_xarm_reward 순수 torch 이식과 원문 대조 테스트

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 7: 사람 기준선 목표

**Files:**
- Create: `M/baseline.py`, `M/tests/test_baseline.py`

**Interfaces:**
- Consumes: `gate.Support`, `gate.kabsch`, `rotations.finite_array`, `rotations.rpy_xyz_to_matrix`, `rotations.yaw_matrix`
- Produces:
  - `baseline`
    - `HumanTarget(keypoints (4,3), rotation, center, relation_residual_m (4,), support_name, width_pair_swapped)`
    - `legacy_relation_targets(legacy_other_pos, legacy_other_rpy, legacy_targets, other_pos, other_rotation) -> (4,3)`
    - `heading(rotation, long_axis) -> float`
    - `human_target(relation_targets, local_offsets, hull_local, start_rotation, long_axis, supports) -> HumanTarget`

기울기는 스냅샷에서 안착한 자세를 유지하고 방향과 위치만 맞춘다(스펙 §6). 기대값의 출처는 다음과 같다.
- `EXPECTED_TARGET`: 공개 데이터를 우리 장면(다른 신발 (0.27, −0.15))에 옮겨 계산한 값.
- 관계 잔차(49.8·49.8·1.9·1.9 mm): 공개 데이터 간격 불일치의 크기.

- [ ] **Step 1: 실패하는 테스트를 작성한다**

Create `source/openarm/openarm/agnostic/modules/iker/tests/test_baseline.py`:

````python
"""baseline — public IKER relation carried to our scene and refitted (no Isaac)."""

import math

import numpy as np
import pytest

from openarm.agnostic.modules.iker import baseline
from openarm.agnostic.modules.iker.gate import Support
from openarm.agnostic.modules.iker.rotations import rpy_xyz_to_matrix, yaw_matrix

REST = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
LONG_AXIS = 2
MOVE_OFFSETS = np.array([[0.0, 0.0, 0.1248], [0.0, 0.0, -0.1248], [0.0481, 0.0, 0.0], [-0.0481, 0.0, 0.0]])
MOVE_HULL = np.array([[sx * 0.0481, sy * 0.05214, sz * 0.1248] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
SUPPORTS = (Support("rack", 0.325, (0.11, 0.43), (-0.33, 0.03)), Support("table", 0.205))
OTHER_POS = np.array([0.27, -0.15, 0.325 + 0.05502])
# IKER envs/shoe_place: object_pose.json object_1 (lifted +1.0 m as the legacy task spawns it) and target.json.
LEGACY_OTHER_POS = np.array([0.6329157289252264, -0.02441016504, 0.17804389875793722 + 1.0])
LEGACY_OTHER_RPY = (-1.5636626780948757, 0.028418572882054294, -1.6710602751696029)
LEGACY_TARGETS = np.array(
    [
        [0.7075356401032895, 0.09688803021587822, 1.2085786984839113],
        [0.5582958177471633, 0.11193229987877429, 1.207509099031963],
        [0.6279129482614393, 0.05468136477793894, 1.2066231614852063],
        [0.6379185095890135, 0.15413896531671356, 1.2094646360306682],
    ]
)
EXPECTED_TARGET = np.array(
    [[0.3821, -0.021, 0.3771], [0.1325, -0.021, 0.3771], [0.2573, -0.0691, 0.3771], [0.2573, 0.0271, 0.3771]]
)


def _relation() -> np.ndarray:
    return baseline.legacy_relation_targets(LEGACY_OTHER_POS, LEGACY_OTHER_RPY, LEGACY_TARGETS, OTHER_POS, REST)


def test_relation_puts_the_target_beside_the_other_shoe_and_keeps_the_public_spacing():
    relation = _relation()
    assert np.allclose(relation.mean(axis=0) - OTHER_POS, [-0.0127, 0.129, 0.0264], atol=5e-4)
    assert np.linalg.norm(relation[0] - relation[1]) == pytest.approx(0.150, abs=1e-3)
    assert np.linalg.norm(relation[2] - relation[3]) == pytest.approx(0.100, abs=1e-3)


def test_human_target_is_rigid_upright_and_resting_on_the_rack():
    target = baseline.human_target(_relation(), MOVE_OFFSETS, MOVE_HULL, REST, LONG_AXIS, SUPPORTS)
    assert np.allclose(target.keypoints, EXPECTED_TARGET, atol=5e-4)
    assert target.support_name == "rack" and not target.width_pair_swapped
    assert np.linalg.norm(target.keypoints[0] - target.keypoints[1]) == pytest.approx(2 * 0.1248)
    assert np.allclose(target.relation_residual_m, [0.0498, 0.0498, 0.0019, 0.0019], atol=5e-4)
    assert np.allclose(target.rotation, REST, atol=1e-6)


def test_mirrored_width_order_is_detected_and_gives_the_same_target():
    target = baseline.human_target(_relation()[[0, 1, 3, 2]], MOVE_OFFSETS, MOVE_HULL, REST, LONG_AXIS, SUPPORTS)
    assert target.width_pair_swapped
    assert np.allclose(target.keypoints, EXPECTED_TARGET, atol=5e-4)


def test_settled_attitude_is_kept_and_only_the_heading_changes():
    # The shoe settled on the table 8 deg off its nominal heading and rolled 4 deg on its uneven sole.
    start = yaw_matrix(math.radians(8.0)) @ REST @ rpy_xyz_to_matrix((0.0, 0.0, math.radians(4.0)))
    target = baseline.human_target(_relation(), MOVE_OFFSETS, MOVE_HULL, start, LONG_AXIS, SUPPORTS)
    assert baseline.heading(target.rotation, LONG_AXIS) == pytest.approx(baseline.heading(REST, LONG_AXIS), abs=1e-3)
    assert np.allclose(target.rotation[2], start[2], atol=1e-9)  # same tilt as at the start
    assert float((MOVE_HULL @ target.rotation.T + target.center)[:, 2].min()) == pytest.approx(0.325)
    assert np.allclose(target.keypoints[:, :2].mean(axis=0), EXPECTED_TARGET[:, :2].mean(axis=0), atol=5e-4)
````

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_baseline.py -q -p no:cacheprovider`
Expected: FAIL. `ImportError`(`baseline` 없음).

- [ ] **Step 3: 구현한다**

Create `source/openarm/openarm/agnostic/modules/iker/baseline.py`:

````python
"""Human baseline target (design spec §6): the public IKER target relation, refitted to our keypoints.

The public ``target.json`` places the moving shoe's four keypoints relative to the other shoe, but
its spacing (0.150 / 0.100 m) matches no rigid shoe pose, so copying it would make the baseline
unreachable. The relation is carried into our scene through the other shoe's frame, and the moving
shoe is placed where it best matches that relation while keeping the attitude it settled into at
the snapshot (a scanned sole is not flat; the settled attitude is the one it will rest in again):
only its heading and position change, and it is set down on the support under it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .gate import Support, kabsch
from .rotations import finite_array, rpy_xyz_to_matrix, yaw_matrix

# Our offsets are ordered (+long, -long, +width, -width). The legacy targets use the same long pair,
# but their width pair can be in either order; the pairing that keeps the object upright is used.
WIDTH_PAIRINGS = ((0, 1, 2, 3), (0, 1, 3, 2))


@dataclass(frozen=True)
class HumanTarget:
    keypoints: np.ndarray  # (4, 3) world, in the object's keypoint order
    rotation: np.ndarray  # (3, 3)
    center: np.ndarray  # (3,)
    relation_residual_m: np.ndarray  # (4,) free rigid-fit residual against the carried relation
    support_name: str
    width_pair_swapped: bool


def legacy_relation_targets(legacy_other_pos, legacy_other_rpy, legacy_targets, other_pos, other_rotation) -> np.ndarray:
    """Legacy target keypoints expressed in the legacy other shoe's frame, placed on our other shoe."""
    legacy_rotation = rpy_xyz_to_matrix(legacy_other_rpy)
    legacy = finite_array("legacy_targets", legacy_targets, (4, 3))
    in_other_frame = (legacy - finite_array("legacy_other_pos", legacy_other_pos, (3,))) @ legacy_rotation
    rotation = finite_array("other_rotation", other_rotation, (3, 3))
    return finite_array("other_pos", other_pos, (3,)) + in_other_frame @ rotation.T


def heading(rotation, long_axis: int) -> float:
    """World yaw (rad) of the object's long local axis."""
    direction = finite_array("rotation", rotation, (3, 3))[:, long_axis]
    if not math.hypot(direction[0], direction[1]) > 1e-6:
        raise ValueError("the long axis is vertical; heading is undefined")
    return math.atan2(direction[1], direction[0])


def human_target(
    relation_targets, local_offsets, hull_local, start_rotation, long_axis: int, supports: Sequence[Support]
) -> HumanTarget:
    relation = finite_array("relation_targets", relation_targets, (4, 3))
    offsets = finite_array("local_offsets", local_offsets, (4, 3))
    hull = np.asarray(hull_local, dtype=float)
    start = finite_array("start_rotation", start_rotation, (3, 3))
    vertical_axis = int(np.argmax(np.abs(start[2])))
    for pairing in WIDTH_PAIRINGS:
        ordered = relation[list(pairing)]
        fit_rotation, fit_center = kabsch(offsets, ordered)
        if np.sign(fit_rotation[2, vertical_axis]) == np.sign(start[2, vertical_axis]):
            break
    else:
        raise ValueError("no width pairing keeps the object upright")
    residual = np.linalg.norm(offsets @ fit_rotation.T + fit_center - ordered, axis=1)
    rotation = yaw_matrix(heading(fit_rotation, long_axis) - heading(start, long_axis)) @ start
    center = ordered.mean(axis=0) - (offsets @ rotation.T).mean(axis=0)  # least squares for a fixed rotation
    support = next((s for s in supports if s.contains_xy(center[0], center[1])), None)
    if support is None:
        raise ValueError(f"target centre {np.round(center, 4).tolist()} is above no support")
    center[2] += support.top_z - float((hull @ rotation.T + center)[:, 2].min())
    return HumanTarget(
        keypoints=offsets @ rotation.T + center,
        rotation=rotation,
        center=center,
        relation_residual_m=residual,
        support_name=support.name,
        width_pair_swapped=pairing != WIDTH_PAIRINGS[0],
    )
````

- [ ] **Step 4: 통과를 확인한다**

Run: Step 2 와 같은 명령
Expected: `4 passed`

- [ ] **Step 5: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker && M=source/openarm/openarm/agnostic/modules/iker
git add $M/baseline.py $M/tests/test_baseline.py
git commit -F - <<'MSG'
feat(iker): 공개 목표 관계를 안착 자세로 정합한 사람 기준선

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 8: 실행 산출물 파일

**Files:**
- Create: `M/run_files.py`, `M/tests/test_run_files.py`

**Interfaces:**
- Consumes: `gate.GateReport`, `gate.MovableObject`, `gate.Support`, `interaction.Interaction`, `projection.CameraPose`, `scene_image.AnnotatedSnapshot`
- Produces:
  - `run_files`
    - `SCHEMA_VERSION=1`
    - `read_json(path) -> dict`(schema 검사)
    - `write_json(path, doc) -> Path`(원자적, NaN 거부)
    - `load_shoe_meta(path) -> dict`
    - `keypoints_document(*, scene_config, camera, intrinsic, snapshot, image_size, object_poses, head, checks) -> dict`
    - `vlm_keypoints(doc) -> {label: (x,y,z)}`
    - `static_keypoint_ids(doc) -> tuple`
    - `movable_objects(doc, meta, supports) -> tuple[MovableObject]`
    - `gate_document(report) -> dict`
    - `interaction_document(source, interaction, report, snapshot_name="snapshot.png", extra=None) -> dict`
  - `keypoints.json` 키: `schema`, `scene_config`, `image_size`, `camera{position,rotation,intrinsic,derotation_deg}`, `axes`, `head`, `objects{name:{position,quat_wxyz}}`, `keypoints[KeypointRecord 필드]`, `checks`
  - `interaction_<source>.json` 키: `schema`, `source`, `snapshot`, `object`, `interaction{object_to_interact,keypoint_ids,grasp_mode}`, `target_keypoints`(물체 키포인트 순서), `gate{passed,failures,object,metrics}`, `extra`(선택)

- [ ] **Step 1: 실패하는 테스트를 작성한다**

Create `source/openarm/openarm/agnostic/modules/iker/tests/test_run_files.py`:

````python
"""run_files — keypoints.json / shoe_meta.json / interaction documents (no Isaac)."""

import copy
import json

import numpy as np
import pytest

from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.modules.iker.gate import GateReport, Support
from openarm.agnostic.modules.iker.interaction import Interaction
from openarm.agnostic.modules.iker.projection import CameraPose
from openarm.agnostic.modules.iker.scene_image import PointGroup, annotate_snapshot

REST = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
OFFSETS = [[0.0, 0.0, 0.1248], [0.0, 0.0, -0.1248], [0.0481, 0.0, 0.0], [-0.0481, 0.0, 0.0]]
HULL = [[sx * 0.0481, sy * 0.05214, sz * 0.1248] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
SUPPORTS = (Support("rack", 0.325, (0.11, 0.43), (-0.33, 0.03)), Support("table", 0.205))
MOVE_POS, OTHER_POS = (0.27, 0.14, 0.257), (0.27, -0.15, 0.377)
CAMERA = CameraPose(position=np.array([0.27, 0.0, 1.2]), rotation=np.diag([1.0, -1.0, -1.0]))
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
META_OBJECT = {
    "source": "object_0", "rest_rotation": REST.tolist(), "rest_quat_wxyz": [0.5, -0.5, 0.5, -0.5],
    "horizontal_axes": [2, 0], "keypoint_offsets": OFFSETS, "rest_height": 0.05214, "hull_local": HULL,
}
META = {"schema": 1, "legacy": {}, "objects": {"shoe_move": META_OBJECT, "shoe_other": META_OBJECT}}


def _doc() -> dict:
    groups = [
        PointGroup("shoe_move", False, np.add(MOVE_POS, np.asarray(OFFSETS) @ REST.T)),
        PointGroup("shoe_other", False, np.add(OTHER_POS, np.asarray(OFFSETS) @ REST.T)),
        PointGroup("rack", True, np.array([[0.16, -0.28, 0.325], [0.38, -0.02, 0.325]])),
    ]
    snap = annotate_snapshot(
        np.zeros((480, 640, 3), np.uint8), np.full((480, 640), 5.0), CAMERA, K, groups, [0.27, 0.0, 0.2], 15.0
    )
    return run_files.keypoints_document(
        scene_config={"index": 0},
        camera=CAMERA,
        intrinsic=K,
        snapshot=snap,
        image_size=(640, 480),
        object_poses={"shoe_move": (MOVE_POS, (0.5, -0.5, 0.5, -0.5)), "shoe_other": (OTHER_POS, (0.5, -0.5, 0.5, -0.5))},
        head={"pan_cmd_deg": -20.0},
        checks={"settle_disp_m": 0.001},
    )


def test_keypoints_document_round_trips_through_json(tmp_path):
    doc = _doc()
    path = run_files.write_json(tmp_path / "run" / "keypoints.json", doc)
    assert run_files.read_json(path) == json.loads(json.dumps(doc))
    assert not (tmp_path / "run" / "keypoints.json.tmp").exists()


def test_write_rejects_nan_and_read_checks_the_schema(tmp_path):
    with pytest.raises(ValueError):
        run_files.write_json(tmp_path / "bad.json", {"schema": 1, "x": float("nan")})
    (tmp_path / "old.json").write_text('{"schema": 0}')
    with pytest.raises(ValueError, match="schema"):
        run_files.read_json(tmp_path / "old.json")
    with pytest.raises(FileNotFoundError):
        run_files.read_json(tmp_path / "missing.json")


def test_movable_objects_follow_labels_and_start_supports():
    doc = _doc()
    move, other = run_files.movable_objects(doc, META, SUPPORTS)
    assert move.name == "shoe_move" and move.keypoint_ids == (1, 2, 3, 4) and move.start_support_z == 0.205
    assert other.keypoint_ids == (5, 6, 7, 8) and other.start_support_z == 0.325
    assert np.allclose(move.init_keypoints, np.add(MOVE_POS, np.asarray(OFFSETS) @ REST.T))
    assert set(run_files.vlm_keypoints(doc)) == {r["label"] for r in doc["keypoints"] if r["kept"]}
    assert run_files.static_keypoint_ids(doc) == (9, 10)


def test_movable_objects_refuse_a_removed_keypoint():
    doc = copy.deepcopy(_doc())
    next(r for r in doc["keypoints"] if r["label"] == 2)["kept"] = False
    with pytest.raises(ValueError, match="removed"):
        run_files.movable_objects(doc, META, SUPPORTS)


def test_load_shoe_meta_names_the_missing_field(tmp_path):
    broken = copy.deepcopy(META)
    del broken["objects"]["shoe_move"]["hull_local"]
    run_files.write_json(tmp_path / "meta.json", broken)
    with pytest.raises(ValueError, match="hull_local"):
        run_files.load_shoe_meta(tmp_path / "meta.json")


def test_interaction_document_carries_targets_and_gate():
    interaction = Interaction("shoe", (1, 2, 3, 4), True, {}, False)
    report = GateReport(True, (), {"mean_distance_m": 0.0}, "shoe_move", ((0.1, 0.2, 0.3),) * 4)
    doc = run_files.interaction_document("human", interaction, report, extra={"note": "x"})
    assert doc["object"] == "shoe_move" and doc["target_keypoints"][0] == [0.1, 0.2, 0.3]
    assert doc["gate"]["passed"] is True and doc["extra"] == {"note": "x"}
    with pytest.raises(ValueError, match="source"):
        run_files.interaction_document("gpt", interaction, report)
````

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_run_files.py -q -p no:cacheprovider`
Expected: FAIL. `ImportError`(`run_files` 없음).

- [ ] **Step 3: 구현한다**

Create `source/openarm/openarm/agnostic/modules/iker/run_files.py`:

````python
"""IKER run artifacts (design spec §5-6).

``assets/iker_shoe/shoe_meta.json``    keypoint offsets, rest pose and hull of each shoe, legacy relation
``<run>/keypoints.json``               one head-camera snapshot: camera, object poses, labelled keypoints
``<run>/interaction_<source>.json``    gated target keypoints from the human baseline or a VLM response
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .gate import GateReport, MovableObject, Support
from .interaction import Interaction
from .projection import CameraPose
from .scene_image import AnnotatedSnapshot

SCHEMA_VERSION = 1
INTERACTION_SOURCES = ("human", "vlm")
META_OBJECT_KEYS = (
    "source", "rest_rotation", "rest_quat_wxyz", "horizontal_axes", "keypoint_offsets", "rest_height", "hull_local",
)


def read_json(path) -> dict:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"missing IKER artifact: {file}")
    doc = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"{file}: expected a JSON object with schema {SCHEMA_VERSION}")
    return doc


def write_json(path, doc: Mapping) -> Path:
    """Write atomically. NaN and infinity are rejected so a broken value never reaches a reader."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, allow_nan=False) + "\n"
    tmp = file.with_name(file.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(file)
    return file


def load_shoe_meta(path) -> dict:
    meta = read_json(path)
    objects = meta.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise ValueError(f"{path}: 'objects' is missing")
    for name, obj in objects.items():
        missing = [key for key in META_OBJECT_KEYS if key not in obj]
        if missing:
            raise ValueError(f"{path}: object {name!r} lacks {missing}")
    if "legacy" not in meta:
        raise ValueError(f"{path}: 'legacy' is missing")
    return meta


def keypoints_document(
    *,
    scene_config: Mapping,
    camera: CameraPose,
    intrinsic,
    snapshot: AnnotatedSnapshot,
    image_size: Sequence[int],
    object_poses: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    head: Mapping,
    checks: Mapping,
) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "scene_config": dict(scene_config),
        "image_size": [int(v) for v in image_size],
        "camera": {
            "position": np.asarray(camera.position, dtype=float).tolist(),
            "rotation": np.asarray(camera.rotation, dtype=float).tolist(),
            "intrinsic": np.asarray(intrinsic, dtype=float).tolist(),
            "derotation_deg": float(snapshot.derotation_deg),
        },
        "axes": {name: [float(v) for v in direction] for name, direction in snapshot.axes.items()},
        "head": dict(head),
        "objects": {
            name: {"position": [float(v) for v in pos], "quat_wxyz": [float(v) for v in quat]}
            for name, (pos, quat) in object_poses.items()
        },
        "keypoints": [asdict(record) for record in snapshot.records],
        "checks": {**dict(checks), "min_margin_px": snapshot.min_margin_px, "min_gap_px": snapshot.min_gap_px},
    }


def vlm_keypoints(doc: Mapping) -> dict[int, tuple[float, float, float]]:
    """The ``keypoint_coordinates`` input: every keypoint drawn on the image, by label."""
    return {int(r["label"]): tuple(float(v) for v in r["world"]) for r in doc["keypoints"] if r["kept"]}


def static_keypoint_ids(doc: Mapping) -> tuple[int, ...]:
    return tuple(int(r["label"]) for r in doc["keypoints"] if r["is_static"] and r["kept"])


def movable_objects(doc: Mapping, meta: Mapping, supports: Sequence[Support]) -> tuple[MovableObject, ...]:
    movables = []
    for name, obj in meta["objects"].items():
        records = sorted((r for r in doc["keypoints"] if r["object_name"] == name), key=lambda r: r["label"])
        offsets = np.asarray(obj["keypoint_offsets"], dtype=float)
        if len(records) != len(offsets):
            raise ValueError(f"{name}: snapshot has {len(records)} keypoints, shoe meta has {len(offsets)}")
        if not all(r["kept"] for r in records):
            raise ValueError(f"{name}: a keypoint was removed by the overlap filter")
        position = np.asarray(doc["objects"][name]["position"], dtype=float)
        support = next((s for s in supports if s.contains_xy(position[0], position[1])), None)
        if support is None:
            raise ValueError(f"{name}: start position {position.tolist()} is above no support")
        movables.append(
            MovableObject(
                name=name,
                keypoint_ids=tuple(int(r["label"]) for r in records),
                local_offsets=offsets,
                hull_local=np.asarray(obj["hull_local"], dtype=float),
                init_keypoints=np.asarray([r["world"] for r in records], dtype=float),
                start_support_z=support.top_z,
            )
        )
    return tuple(movables)


def gate_document(report: GateReport) -> dict:
    return {
        "passed": report.passed,
        "failures": list(report.failures),
        "object": report.object_name,
        "metrics": dict(report.metrics),
    }


def interaction_document(
    source: str,
    interaction: Interaction,
    report: GateReport,
    snapshot_name: str = "snapshot.png",
    extra: Mapping | None = None,
) -> dict:
    if source not in INTERACTION_SOURCES:
        raise ValueError(f"source must be one of {INTERACTION_SOURCES}, got {source!r}")
    doc = {
        "schema": SCHEMA_VERSION,
        "source": source,
        "snapshot": snapshot_name,
        "object": report.object_name,
        "interaction": {
            "object_to_interact": interaction.object_name,
            "keypoint_ids": list(interaction.keypoint_ids),
            "grasp_mode": interaction.grasp_mode,
        },
        "target_keypoints": [list(point) for point in report.target_keypoints],
        "gate": gate_document(report),
    }
    if extra:
        doc["extra"] = dict(extra)
    return doc
````

- [ ] **Step 4: 통과를 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests -q -p no:cacheprovider`
Expected: `92 passed`(modules/iker 전체)

- [ ] **Step 5: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker && M=source/openarm/openarm/agnostic/modules/iker
git add $M/run_files.py $M/tests/test_run_files.py
git commit -F - <<'MSG'
feat(iker): keypoints·shoe_meta·interaction 산출물 읽기·쓰기

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 9: 장면 배치와 신발 메타 파일

**Files:**
- Create: `T/__init__.py`(빈 파일), `T/layout.py`, `T/tests/__init__.py`, `T/tests/test_layout.py`, `T/tests/test_shoe_meta.py`, `scripts/iker/build_shoe_meta.py`, `assets/iker_shoe/shoe_meta.json`(스크립트 산출)

**Interfaces:**
- Consumes:
  - `gate.GateConfig`, `gate.Support`, `keypoints.surface_grid`, `keypoints.snap_rotation`, `keypoints.horizontal_axes`, `keypoints.extremity_offsets`
  - `rotations.matrix_to_quat_wxyz`, `rotations.rpy_xyz_to_matrix`, `rotations.yaw_matrix`
  - `run_files.write_json`, `run_files.load_shoe_meta`, `robot_profiles.PROFILES`
- Produces:
  - 경로·이름 상수
    - 경로: `layout.HDGP_ROOT`, `ASSETS_DIR`, `SHOE_ASSET_DIR`, `SHOE_META_PATH`, `TABLE_USD_PATH`, `RUNS_DIR`
    - 이름: `PROFILE_NAME`, `TASK_INSTRUCTION`, `MOVING_SHOE="shoe_move"`, `OTHER_SHOE="shoe_other"`, `RACK="rack"`
  - 수치 상수
    - 받침·테이블: `TABLE_TOP_Z`, `RACK_X_RANGE`, `RACK_Y_RANGE`, `RACK_TOP_Z`
    - 다른 신발·스폰: `OTHER_SHOE_Y`, `OTHER_SHOE_YAW_DEG`, `SPAWN_CLEARANCE_M`
    - 구성 샘플링: `CONFIG_SEED`, `MOVE_X_RANGE`, `MOVE_Y_RANGE`, `MOVE_YAW_DEG_RANGE`, `OTHER_X_RANGE`
    - 스냅샷: `HEAD_PAN_ENCODER_DEG`, `HEAD_TILT_ENCODER_DEG`, `OVERLAP_MIN_PX`, `MIN_BORDER_MARGIN_PX`
  - 함수
    - `SceneConfig(index, move_x, move_y, move_yaw_deg, other_x)`
    - `config_stream(seed)`, `sample_configs(count, seed) -> tuple`
    - `shoe_start_poses(config, meta) -> {name: (pos (3,), quat_wxyz (4,))}`
    - `supports() -> (rack, table)`, `rack_keypoints() -> (12,3)`, `gate_config() -> GateConfig`
  - `shoe_meta.json` 키:
    - 최상위: `schema`, `source{iker_dir, sha256}`, `legacy{z_lift, object_pose, target_position}`
    - `objects{shoe_move|shoe_other: {source, rest_rotation, rest_quat_wxyz, horizontal_axes, keypoint_offsets, rest_height, hull_local}}`

- [ ] **Step 1: 실패하는 배치 테스트를 작성한다**

```bash
cd ~/rl_ws/hdgp-iker && mkdir -p source/openarm/openarm/agnostic/tasks/iker_shoe/tests scripts/iker assets/iker_shoe
: > source/openarm/openarm/agnostic/tasks/iker_shoe/__init__.py
: > source/openarm/openarm/agnostic/tasks/iker_shoe/tests/__init__.py
```

첫 구성 값 `(0.241444, 0.134132, -4.040029, 0.27183)` 은 `np.random.default_rng(20260913)` 의 PCG64 스트림에서 나온다. numpy 버전과 무관하게 같다.

Create `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_layout.py`:

````python
"""layout — scene constants, configuration sampling, gate config (no Isaac)."""

import numpy as np

from openarm.agnostic.modules import robot_profiles
from openarm.agnostic.modules.iker import rotations
from openarm.agnostic.tasks.iker_shoe import layout


def test_first_configuration_is_pinned_to_the_seed():
    first, second = layout.sample_configs(2)
    values = (first.index, round(first.move_x, 6), round(first.move_y, 6), round(first.move_yaw_deg, 6), round(first.other_x, 6))
    assert values == (0, 0.241444, 0.134132, -4.040029, 0.27183)
    assert second.index == 1


def test_sampled_configurations_stay_inside_the_spec_ranges():
    for config in layout.sample_configs(200):
        assert layout.MOVE_X_RANGE[0] <= config.move_x <= layout.MOVE_X_RANGE[1]
        assert layout.MOVE_Y_RANGE[0] <= config.move_y <= layout.MOVE_Y_RANGE[1]
        assert layout.MOVE_YAW_DEG_RANGE[0] <= config.move_yaw_deg <= layout.MOVE_YAW_DEG_RANGE[1]
        assert layout.OTHER_X_RANGE[0] <= config.other_x <= layout.OTHER_X_RANGE[1]


def test_rack_is_checked_before_the_table():
    rack, table = layout.supports()
    assert rack.name == "rack" and rack.contains_xy(0.27, layout.OTHER_SHOE_Y)
    assert not rack.contains_xy(0.27, 0.14) and table.contains_xy(0.27, 0.14)


def test_rack_keypoints_form_the_inset_grid():
    grid = layout.rack_keypoints()
    assert grid.shape == (12, 3)
    assert np.allclose(grid[0], [0.16, -0.28, 0.325]) and np.allclose(grid[-1], [0.38, -0.02, 0.325])


def test_shoe_start_poses_rest_on_their_supports_with_the_config_yaw():
    rest = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    meta = {"objects": {name: {"rest_rotation": rest, "rest_height": 0.05} for name in (layout.MOVING_SHOE, layout.OTHER_SHOE)}}
    config = layout.SceneConfig(index=0, move_x=0.25, move_y=0.14, move_yaw_deg=90.0, other_x=0.27)
    poses = layout.shoe_start_poses(config, meta)
    move_pos, move_quat = poses[layout.MOVING_SHOE]
    other_pos, other_quat = poses[layout.OTHER_SHOE]
    assert np.allclose(move_pos, [0.25, 0.14, layout.TABLE_TOP_Z + 0.05 + layout.SPAWN_CLEARANCE_M])
    assert np.allclose(other_pos, [0.27, layout.OTHER_SHOE_Y, layout.RACK_TOP_Z + 0.05 + layout.SPAWN_CLEARANCE_M])
    assert np.allclose(other_quat, [0.5, -0.5, 0.5, -0.5])
    long_axis = rotations.quat_wxyz_to_matrix(move_quat)[:, 2]  # local z is the shoe length
    assert np.allclose(long_axis, [0.0, 1.0, 0.0], atol=1e-9)  # rest points it along +x; +90 deg yaw turns it to +y


def test_gate_config_takes_the_palm_box_from_the_robot_profile():
    profile = robot_profiles.PROFILES[layout.PROFILE_NAME]
    cfg = layout.gate_config()
    assert cfg.palm_box_min == tuple(profile.palm_box_min) and cfg.palm_box_max == tuple(profile.palm_box_max)


def test_paths_resolve_inside_the_hdgp_checkout():
    assert (layout.HDGP_ROOT / "source" / "openarm").is_dir()
    assert layout.TABLE_USD_PATH.is_file()
````

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_layout.py -q -p no:cacheprovider`
Expected: FAIL. `ImportError`(`layout` 없음).

- [ ] **Step 2: 배치 모듈을 구현한다**

Create `source/openarm/openarm/agnostic/tasks/iker_shoe/layout.py`:

````python
"""IKER shoe-placement scene, snapshot settings and evaluation configurations (design spec §4, §5, §9).

Pure Python: importable without Isaac Sim. Positions are in the robot frame (origin on the mount
plate top), metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from openarm.agnostic.modules import robot_profiles
from openarm.agnostic.modules.iker.gate import GateConfig, Support
from openarm.agnostic.modules.iker.keypoints import surface_grid
from openarm.agnostic.modules.iker.rotations import matrix_to_quat_wxyz, yaw_matrix

HDGP_ROOT = Path(__file__).resolve().parents[6]
ASSETS_DIR = HDGP_ROOT / "assets"
SHOE_ASSET_DIR = ASSETS_DIR / "iker_shoe"
SHOE_META_PATH = SHOE_ASSET_DIR / "shoe_meta.json"
TABLE_USD_PATH = ASSETS_DIR / "simulation_setting" / "env_v1" / "usd" / "env_v1.usda"
RUNS_DIR = HDGP_ROOT / "iker_runs" / "shoe_place"

PROFILE_NAME = "tesollo_right"
TASK_INSTRUCTION = "Place the shoe on the rack next to the other shoe."

TABLE_TOP_Z = 0.205
RACK_X_RANGE = (0.11, 0.43)
RACK_Y_RANGE = (-0.33, 0.03)
RACK_TOP_Z = 0.325
RACK_KEYPOINT_INSET = 0.05
RACK_GRID_SHAPE = (3, 4)

MOVING_SHOE = "shoe_move"
OTHER_SHOE = "shoe_other"
RACK = "rack"
OTHER_SHOE_Y = -0.15
OTHER_SHOE_YAW_DEG = 0.0
SPAWN_CLEARANCE_M = 0.002  # drop the shoes from just above their support; the snapshot uses the settled pose

CONFIG_SEED = 20260913
MOVE_X_RANGE = (0.22, 0.32)
MOVE_Y_RANGE = (0.12, 0.16)
MOVE_YAW_DEG_RANGE = (-10.0, 10.0)
OTHER_X_RANGE = (0.25, 0.29)

HEAD_PAN_ENCODER_DEG = -20.0
HEAD_TILT_ENCODER_DEG = -20.0
OVERLAP_MIN_PX = 15.0
MIN_BORDER_MARGIN_PX = 15.0


@dataclass(frozen=True)
class SceneConfig:
    index: int
    move_x: float
    move_y: float
    move_yaw_deg: float
    other_x: float


def config_stream(seed: int = CONFIG_SEED) -> Iterator[SceneConfig]:
    """Configurations drawn in a fixed order; a rejected one is replaced by the next draw."""
    rng = np.random.default_rng(seed)
    index = 0
    while True:
        yield SceneConfig(
            index=index,
            move_x=float(rng.uniform(*MOVE_X_RANGE)),
            move_y=float(rng.uniform(*MOVE_Y_RANGE)),
            move_yaw_deg=float(rng.uniform(*MOVE_YAW_DEG_RANGE)),
            other_x=float(rng.uniform(*OTHER_X_RANGE)),
        )
        index += 1


def sample_configs(count: int, seed: int = CONFIG_SEED) -> tuple[SceneConfig, ...]:
    if count < 0:
        raise ValueError("count must be non-negative")
    stream = config_stream(seed)
    return tuple(next(stream) for _ in range(count))


def shoe_start_poses(config: SceneConfig, meta: Mapping) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Spawn position and ``wxyz`` quaternion of both shoes for one configuration (before settling)."""
    placements = {
        MOVING_SHOE: ((config.move_x, config.move_y), config.move_yaw_deg, TABLE_TOP_Z),
        OTHER_SHOE: ((config.other_x, OTHER_SHOE_Y), OTHER_SHOE_YAW_DEG, RACK_TOP_Z),
    }
    poses = {}
    for name, ((x, y), yaw_deg, support_z) in placements.items():
        obj = meta["objects"][name]
        rotation = yaw_matrix(math.radians(yaw_deg)) @ np.asarray(obj["rest_rotation"], dtype=float)
        position = np.array([x, y, support_z + float(obj["rest_height"]) + SPAWN_CLEARANCE_M])
        poses[name] = (position, matrix_to_quat_wxyz(rotation))
    return poses


def supports() -> tuple[Support, ...]:
    return (Support(RACK, RACK_TOP_Z, RACK_X_RANGE, RACK_Y_RANGE), Support("table", TABLE_TOP_Z))


def rack_keypoints() -> np.ndarray:
    return surface_grid(RACK_X_RANGE, RACK_Y_RANGE, RACK_TOP_Z, RACK_KEYPOINT_INSET, *RACK_GRID_SHAPE)


def gate_config() -> GateConfig:
    profile = robot_profiles.PROFILES[PROFILE_NAME]
    return GateConfig(
        supports=supports(),
        palm_box_min=tuple(float(v) for v in profile.palm_box_min),
        palm_box_max=tuple(float(v) for v in profile.palm_box_max),
    )
````

Run: Step 1 의 pytest 명령
Expected: `7 passed`

- [ ] **Step 3: 메타 파일 테스트를 작성하고 실패를 확인한다**

Create `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_shoe_meta.py`:

````python
"""The committed shoe meta file matches the measured IKER mesh geometry (no Isaac)."""

import numpy as np
import pytest

from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.tasks.iker_shoe import layout


@pytest.fixture(scope="module")
def meta() -> dict:
    return run_files.load_shoe_meta(layout.SHOE_META_PATH)


def test_shoes_map_to_the_iker_objects(meta):
    assert meta["objects"][layout.MOVING_SHOE]["source"] == "object_0"
    assert meta["objects"][layout.OTHER_SHOE]["source"] == "object_1"
    assert {"object_0_mesh.obj", "object_1_mesh.obj", "object_pose.json", "target.json"} <= set(meta["source"]["sha256"])


@pytest.mark.parametrize("name, half_length, half_width, rest_height", [
    (layout.MOVING_SHOE, 0.1248, 0.0481, 0.05214),
    (layout.OTHER_SHOE, 0.1261, 0.0485, 0.05502),
])
def test_keypoints_are_length_and_width_extremities_at_rest(meta, name, half_length, half_width, rest_height):
    obj = meta["objects"][name]
    assert obj["horizontal_axes"] == [2, 0]
    expected = [[0, 0, half_length], [0, 0, -half_length], [half_width, 0, 0], [-half_width, 0, 0]]
    assert np.allclose(obj["keypoint_offsets"], expected, atol=5e-4)
    assert obj["rest_height"] == pytest.approx(rest_height, abs=5e-4)
    assert np.allclose(obj["rest_quat_wxyz"], [0.5, -0.5, 0.5, -0.5])
    assert len(obj["hull_local"]) > 100


def test_legacy_relation_is_recorded(meta):
    assert meta["legacy"]["z_lift"] == 1.0
    assert len(meta["legacy"]["target_position"]) == 4
    assert set(meta["legacy"]["object_pose"]) == {"object_0", "object_1"}
````

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_shoe_meta.py -q -p no:cacheprovider`
Expected: FAIL. `FileNotFoundError: missing IKER artifact: .../assets/iker_shoe/shoe_meta.json`

- [ ] **Step 4: 메타 생성 스크립트를 만들고 실행한다**

Create `scripts/iker/build_shoe_meta.py`:

````python
#!/usr/bin/env python3
"""Build assets/iker_shoe/shoe_meta.json from the IKER shoe meshes (design spec §4). No Isaac needed.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm python3 scripts/iker/build_shoe_meta.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from openarm.agnostic.modules.iker import keypoints, run_files
from openarm.agnostic.modules.iker.rotations import matrix_to_quat_wxyz, rpy_xyz_to_matrix
from openarm.agnostic.tasks.iker_shoe import layout

DEFAULT_IKER_DIR = Path("/home/user/rl_ws/repo/skill_gen/IKER/envs/shoe_place")
SOURCES = {layout.MOVING_SHOE: "object_0", layout.OTHER_SHOE: "object_1"}
# The legacy task spawns object_pose.json 1.0 m higher (shoe_place.py:650 `sampled_obj_state[:, 2] += 1`);
# target.json is already in that lifted frame.
LEGACY_Z_LIFT = 1.0
IDENTITY_ORIGIN = '<origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>'
DECIMALS = 6


def parse_obj_vertices(path: Path) -> np.ndarray:
    rows = [line.split()[1:4] for line in path.read_text().splitlines() if line.startswith("v ")]
    if len(rows) < 4:
        raise ValueError(f"{path}: fewer than 4 vertices")
    return np.asarray(rows, dtype=float)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_urdf_frame(urdf: Path, mesh_name: str) -> None:
    """Keypoint offsets are measured in the mesh frame, so the mesh must sit at the link origin."""
    text = urdf.read_text()
    if f'filename="{mesh_name}"' not in text or text.count(IDENTITY_ORIGIN) != 2:
        raise ValueError(f"{urdf}: expected visual and collision {mesh_name} at the link origin")


def object_meta(iker_dir: Path, source: str, legacy_rpy) -> dict:
    mesh = iker_dir / f"{source}_mesh.obj"
    check_urdf_frame(iker_dir / f"{source}_object.urdf", mesh.name)
    vertices = parse_obj_vertices(mesh)
    hull = vertices[ConvexHull(vertices).vertices]
    rest = keypoints.snap_rotation(rpy_xyz_to_matrix(legacy_rpy))
    axes = keypoints.horizontal_axes(rest, vertices)
    return {
        "source": source,
        "rest_rotation": rest.tolist(),
        "rest_quat_wxyz": matrix_to_quat_wxyz(rest).round(DECIMALS).tolist(),
        "horizontal_axes": list(axes),
        "keypoint_offsets": keypoints.extremity_offsets(vertices, axes).round(DECIMALS).tolist(),
        "rest_height": round(-float((hull @ rest.T)[:, 2].min()), DECIMALS),
        "hull_local": hull.round(DECIMALS).tolist(),
    }


def build(iker_dir: Path) -> dict:
    poses = json.loads((iker_dir / "object_pose.json").read_text())
    target = json.loads((iker_dir / "target.json").read_text())["target_position"]
    inputs = ["object_pose.json", "target.json"]
    inputs += [f"{source}_{kind}" for source in SOURCES.values() for kind in ("mesh.obj", "object.urdf")]
    return {
        "schema": run_files.SCHEMA_VERSION,
        "source": {"iker_dir": str(iker_dir), "sha256": {name: sha256(iker_dir / name) for name in inputs}},
        "legacy": {"z_lift": LEGACY_Z_LIFT, "object_pose": poses, "target_position": target},
        "objects": {name: object_meta(iker_dir, source, poses[source]["rpy"]) for name, source in SOURCES.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the IKER shoe keypoint meta file.")
    parser.add_argument("--iker-dir", type=Path, default=DEFAULT_IKER_DIR)
    parser.add_argument("--out", type=Path, default=layout.SHOE_META_PATH)
    args = parser.parse_args(argv)
    required = [args.iker_dir / name for name in ("object_pose.json", "target.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(f"missing IKER inputs: {missing}", file=sys.stderr)
        return 2
    meta = build(args.iker_dir)
    run_files.write_json(args.out, meta)
    for name, obj in meta["objects"].items():
        print(f"{name}: offsets {obj['keypoint_offsets']} rest_height {obj['rest_height']} hull {len(obj['hull_local'])}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 scripts/iker/build_shoe_meta.py`
Expected(경고 줄 제외):
```text
shoe_move: offsets [[0.0, 0.0, 0.124778], [0.0, 0.0, -0.124778], [0.048112, 0.0, 0.0], [-0.048112, 0.0, 0.0]] rest_height 0.052145 hull 1776
shoe_other: offsets [[0.0, 0.0, 0.126099], [0.0, 0.0, -0.126099], [0.048512, 0.0, 0.0], [-0.048512, 0.0, 0.0]] rest_height 0.05502 hull 1643
wrote /home/user/rl_ws/hdgp-iker/assets/iker_shoe/shoe_meta.json
```
파일 크기는 약 200 KB 다.

- [ ] **Step 5: 통과를 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests -q -p no:cacheprovider`
Expected: `11 passed`

- [ ] **Step 6: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker && T=source/openarm/openarm/agnostic/tasks/iker_shoe
git add $T/__init__.py $T/layout.py $T/tests/__init__.py $T/tests/test_layout.py $T/tests/test_shoe_meta.py scripts/iker/build_shoe_meta.py assets/iker_shoe/shoe_meta.json
git commit -F - <<'MSG'
feat(iker): 신발 장면 배치·구성 샘플링과 메시 키포인트 메타 파일

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 10: 로봇·장면 설정과 신발 USD 변환(Isaac)

**Files:**
- Create: `T/robot.py`, `T/scene_cfg.py`, `scripts/iker/convert_shoe_assets.py`
- Create(스크립트 산출): `assets/iker_shoe/shoe_move/`, `assets/iker_shoe/shoe_other/` 아래 파일
  - `<name>.usd`, `configuration/<name>_{base,physics,robot,sensor}.usd`
  - `configuration/materials/textures/object_N_material.png`, `config.yaml`, `.asset_hash`

**Interfaces:**
- Consumes: `layout.*` 경로·상수, `run_files.load_shoe_meta`, `robot_profiles.PROFILES`, `vendor_gains.load_joint_inertia`, `vendor_gains.hand_gains_by_joint`
- Produces:
  - `robot`
    - 상수 `ROBOT_PRIM_PATH`, `GRAVITY_COMPENSATION_JOINTS="[rl]_aj_[1-7]"`, `HEAD_PAN_JOINT`, `HEAD_TILT_JOINT`
    - `profile()`, `robot_cfg() -> ArticulationCfg`
  - `scene_cfg`
    - `shoe_usd_path(name) -> Path`
    - `table_cfg()`, `rack_cfg() -> AssetBaseCfg`
    - `shoe_cfg(name, position, quat_wxyz) -> RigidObjectCfg`
    - `light_cfg() -> AssetBaseCfg`

이 파일들은 isaaclab 을 import 하므로 시스템 python 유닛 테스트가 없다. 검증 경로는 두 가지다.
- 이 태스크의 변환 실행.
- Task 11 스냅샷의 안착·깊이 검사.

- [ ] **Step 1: 설정 모듈을 만든다**

Create `source/openarm/openarm/agnostic/tasks/iker_shoe/robot.py`:

````python
"""Robot articulation for the IKER shoe task, assembled from the hdgp robot profile (design spec §3, §7).

Every robot value comes from ``modules/robot_profiles.py`` and ``modules/vendor_gains.py``; grasp_s2r
is not imported. Conventions kept from the validated hdgp tracks: gravity on and compensated by the
caller on both arms (``GRAVITY_COMPENSATION_JOINTS``), robot solver iterations 32/1 as in the vendor
DG-5F Isaac USD, and DG-5F hand gains proportional to joint inertia (the vendor Isaac rule).
"""

from __future__ import annotations

import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from openarm.agnostic.modules import robot_profiles, vendor_gains

from . import layout

ROBOT_PRIM_PATH = "/World/envs/env_.*/Robot"
SOLVER_POSITION_ITERATIONS = 32
SOLVER_VELOCITY_ITERATIONS = 1
GRAVITY_COMPENSATION_JOINTS = "[rl]_aj_[1-7]"
HEAD_PAN_JOINT = "head_j_pan"
HEAD_TILT_JOINT = "head_j_tilt"


def profile() -> robot_profiles.RobotProfile:
    return robot_profiles.PROFILES[layout.PROFILE_NAME]


def _actuators(prof: robot_profiles.RobotProfile, asset_dir: str) -> dict[str, ImplicitActuatorCfg]:
    inertia = vendor_gains.load_joint_inertia(asset_dir)
    actuators = {}
    for name, spec in prof.actuator_specs.items():
        kwargs = dict(spec)
        if name.endswith("hand"):
            patterns = [re.compile(pattern) for pattern in kwargs["joint_names_expr"]]
            joints = tuple(joint for joint in sorted(inertia) if any(p.fullmatch(joint) for p in patterns))
            if not joints:
                raise RuntimeError(f"{asset_dir}: no joint in the inertia table matches actuator {name!r}")
            kwargs["stiffness"], kwargs["damping"] = vendor_gains.hand_gains_by_joint(asset_dir, joints)
        actuators[name] = ImplicitActuatorCfg(**kwargs)
    return actuators


def robot_cfg() -> ArticulationCfg:
    prof = profile()
    usd = layout.ASSETS_DIR / prof.usd_relpath
    if not usd.is_file():
        raise FileNotFoundError(f"robot USD for profile {prof.name!r} not found: {usd}")
    return ArticulationCfg(
        prim_path=ROBOT_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1000.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=dict(prof.init_joint_pos)
        ),
        actuators=_actuators(prof, str(usd.parent)),
        soft_joint_pos_limit_factor=1.0,
    )
````

Create `source/openarm/openarm/agnostic/tasks/iker_shoe/scene_cfg.py`:

````python
"""Spawn configurations for the IKER shoe scene (design spec §4): env_v1 table, kinematic rack, two shoes."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from . import layout

RACK_COLOR = (0.62, 0.45, 0.28)
RACK_MASS_KG = 1.0  # kinematic; the value only satisfies the mass API
LIGHT_INTENSITY = 2500.0
SHOE_MAX_DEPENETRATION_VELOCITY = 1.0


def shoe_usd_path(name: str) -> Path:
    return layout.SHOE_ASSET_DIR / name / f"{name}.usd"


def table_cfg() -> AssetBaseCfg:
    if not layout.TABLE_USD_PATH.is_file():
        raise FileNotFoundError(f"env_v1 table USD not found: {layout.TABLE_USD_PATH}")
    return AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table", spawn=sim_utils.UsdFileCfg(usd_path=str(layout.TABLE_USD_PATH))
    )


def rack_cfg() -> AssetBaseCfg:
    (x0, x1), (y0, y1) = layout.RACK_X_RANGE, layout.RACK_Y_RANGE
    height = layout.RACK_TOP_Z - layout.TABLE_TOP_Z
    return AssetBaseCfg(
        prim_path="/World/envs/env_.*/Rack",
        spawn=sim_utils.CuboidCfg(
            size=(x1 - x0, y1 - y0, height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=RACK_MASS_KG),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=RACK_COLOR),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=((x0 + x1) / 2, (y0 + y1) / 2, layout.TABLE_TOP_Z + height / 2)),
    )


def shoe_cfg(name: str, position: Sequence[float], quat_wxyz: Sequence[float]) -> RigidObjectCfg:
    usd = shoe_usd_path(name)
    if not usd.is_file():
        raise FileNotFoundError(f"shoe USD not found: {usd} (run scripts/iker/convert_shoe_assets.py)")
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=SHOE_MAX_DEPENETRATION_VELOCITY),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(float(v) for v in position), rot=tuple(float(v) for v in quat_wxyz)
        ),
    )


def light_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=LIGHT_INTENSITY))
````

- [ ] **Step 2: 변환 스크립트를 만든다**

Create `scripts/iker/convert_shoe_assets.py`:

````python
"""Convert the IKER shoe URDFs to USD under assets/iker_shoe/<shoe>/ (design spec §4).

The URDF and mesh hashes must match assets/iker_shoe/shoe_meta.json, so the USD and the keypoint
offsets always come from the same mesh.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/convert_shoe_assets.py --headless
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

DEFAULT_IKER_DIR = Path("/home/user/rl_ws/repo/skill_gen/IKER/envs/shoe_place")

parser = argparse.ArgumentParser(description="Convert the IKER shoe URDFs to USD.")
parser.add_argument("--iker-dir", type=Path, default=DEFAULT_IKER_DIR)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    # Isaac Sim can hang on shutdown after an exception; report and leave immediately.
    traceback.print_exception(exc_type, exc, tb)
    print("CONVERT FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout, scene_cfg  # noqa: E402

SHOE_DENSITY = 100.0  # IKER URDF <density value="100.0"/>


def _check_hash(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{path} changed since shoe_meta.json was built ({actual} != {expected})")


meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
for name, obj in meta["objects"].items():
    urdf = args.iker_dir / f"{obj['source']}_object.urdf"
    mesh = args.iker_dir / f"{obj['source']}_mesh.obj"
    for path in (urdf, mesh):
        _check_hash(path, meta["source"]["sha256"][path.name])
    target = scene_cfg.shoe_usd_path(name)
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(urdf),
            usd_dir=str(target.parent),
            usd_file_name=target.name,
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=False,
            joint_drive=None,
            link_density=SHOE_DENSITY,
            collider_type="convex_hull",
        )
    )
    print(f"CONVERT {name}: {urdf.name} -> {converter.usd_path}", flush=True)
print("CONVERT DONE", flush=True)
os._exit(0)
````

- [ ] **Step 3: 변환을 실행한다**

Run: `cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh scripts/iker/convert_shoe_assets.py --headless 2>&1 | grep -E "CONVERT|Traceback|Error"`
Expected(약 5 s):
```text
CONVERT shoe_move: object_0_object.urdf -> /home/user/rl_ws/hdgp-iker/assets/iker_shoe/shoe_move/shoe_move.usd
CONVERT shoe_other: object_1_object.urdf -> /home/user/rl_ws/hdgp-iker/assets/iker_shoe/shoe_other/shoe_other.usd
CONVERT DONE
```

Run: `cd ~/rl_ws/hdgp-iker && find assets/iker_shoe -type f | sort`
Expected: `shoe_meta.json` 외에 신발마다 파일 8개. 대략 크기는 다음과 같다.
- `shoe_move.usd` 1.4 KB, `configuration/shoe_move_base.usd` 5.9 MB, 텍스처 png 0.4 MB
- `shoe_other_base.usd` 4.3 MB, 텍스처 png 0.5 MB

- [ ] **Step 4: LFS 로 들어가는지 확인하고 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker && T=source/openarm/openarm/agnostic/tasks/iker_shoe
git add $T/robot.py $T/scene_cfg.py scripts/iker/convert_shoe_assets.py assets/iker_shoe/shoe_move assets/iker_shoe/shoe_other
git lfs status | grep -c "iker_shoe/.*\.usd (LFS"
```
Expected: `10`(신발마다 USD 5개가 LFS 로 stage 됨; png·yaml·`.asset_hash` 는 일반 git)

```bash
cd ~/rl_ws/hdgp-iker && git commit -F - <<'MSG'
feat(iker): 프로필 기반 로봇 설정·신발 장면 스폰 설정과 IKER 신발 USD 변환

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 11: 머리 카메라 스냅샷(Isaac)

**Files:**
- Create: `scripts/iker/snapshot.py`
- Create(스크립트 산출): `iker_runs/shoe_place/config_00/snapshot_raw.png`, `snapshot.png`, `keypoints.json`

**Interfaces:**
- Consumes:
  - `layout.sample_configs`, `layout.shoe_start_poses`, `layout.rack_keypoints`, `layout.supports`, `layout` 스냅샷 상수
  - `robot.robot_cfg`, `robot.GRAVITY_COMPENSATION_JOINTS`, `robot.HEAD_*`, `scene_cfg.*`
  - `projection.camera_pose_from_link`, `scene_image.annotate_snapshot`, `run_files.keypoints_document`, `run_files.movable_objects`
  - `openarm.sensors.head_camera.head_camera_cfg`, `load_spec`, `urdf_head_angles`
- Produces: `config_XX/keypoints.json`(Task 8 스키마, `checks.passed` 포함)
  - 계획 3 의 생성·평가가 이 파일을 읽는다.

스크립트가 따르는 규칙은 다음과 같다.
- 머리 각은 매 물리 스텝 다시 지령한다(한 번만 쓰면 스텝당 1.6° 씩 0 으로 끌려간다).
- 팔 중력보상을 매 스텝 건다.
- 카메라 자세는 센서 값이 아니라 `head_camera` 링크 ⊗ 보정 오프셋이다.
- 120 스텝 안착 중 마지막 10 스텝만 렌더한다.

- [ ] **Step 1: 스냅샷 스크립트를 만든다**

Create `scripts/iker/snapshot.py`:

````python
"""Head-camera snapshot of one IKER configuration (design spec §5).

Spawns the scene, holds the head at the snapshot angles while the shoes settle, renders RGB and
depth, and writes iker_runs/shoe_place/config_XX/{snapshot_raw.png, snapshot.png, keypoints.json}.
The files are written even when a check fails (``checks.passed`` is false) and the exit code is 1.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/snapshot.py \
        --config-index 0 --headless
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Head-camera snapshot of one IKER configuration.")
parser.add_argument("--config-index", type=int, required=True)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--render-steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    # Isaac Sim can hang on shutdown after an exception; report and leave immediately.
    traceback.print_exception(exc_type, exc, tb)
    print("SNAPSHOT FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import isaacsim.core.utils.prims as prim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from isaaclab.assets import Articulation, RigidObject  # noqa: E402
from isaaclab.sensors import Camera  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.modules.iker.projection import camera_pose_from_link  # noqa: E402
from openarm.agnostic.modules.iker.rotations import quat_wxyz_to_matrix  # noqa: E402
from openarm.agnostic.modules.iker.scene_image import PointGroup, annotate_snapshot  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout, robot, scene_cfg  # noqa: E402
from openarm.sensors.head_camera import head_camera_cfg, load_spec, urdf_head_angles  # noqa: E402

PHYSICS_DT = 1.0 / 120.0
ENV_PRIM = "/World/envs/env_0"
SETTLE_WINDOW_STEPS = 30
MAX_SETTLE_MOTION_M = 0.005
MAX_HEAD_ERROR_DEG = 0.5
MAX_RACK_DEPTH_ERROR_M = 0.005


def build_scene(poses):
    sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT, render_interval=1, device=args.device))
    prim_utils.create_prim(ENV_PRIM, "Xform")
    for cfg, prim in ((scene_cfg.table_cfg(), f"{ENV_PRIM}/Table"), (scene_cfg.rack_cfg(), f"{ENV_PRIM}/Rack")):
        cfg.spawn.func(prim, cfg.spawn, translation=cfg.init_state.pos)
    light = scene_cfg.light_cfg()
    light.spawn.func(light.prim_path, light.spawn)
    arm = Articulation(robot.robot_cfg().replace(prim_path=f"{ENV_PRIM}/Robot"))
    shoes = {
        name: RigidObject(scene_cfg.shoe_cfg(name, pos, quat).replace(prim_path=f"{ENV_PRIM}/{name}"))
        for name, (pos, quat) in poses.items()
    }
    spec = load_spec()
    camera = Camera(
        head_camera_cfg(spec, data_types=("rgb", "distance_to_image_plane")).replace(
            prim_path=f"{ENV_PRIM}/Robot/{spec.link}/head_cam_real"
        )
    )
    sim.reset()
    if not camera.is_initialized:  # the camera is created after the robot; see openarm.sensors.head_camera
        camera._initialize_impl()
        camera._is_initialized = True
    return sim, arm, shoes, camera, spec


def settle(sim, arm, shoes, camera, target, gravity_ids):
    """Hold the head (re-commanded every step) with arm gravity compensation; return shoe position traces."""
    traces = {name: [] for name in shoes}
    for step in range(args.settle_steps):
        arm.set_joint_position_target(target)
        tau = arm.root_physx_view.get_gravity_compensation_forces()
        arm.set_joint_effort_target(tau[:, gravity_ids], joint_ids=gravity_ids)
        arm.write_data_to_sim()
        render = step >= args.settle_steps - args.render_steps
        sim.step(render=render)
        arm.update(PHYSICS_DT)
        for name, shoe in shoes.items():
            shoe.update(PHYSICS_DT)
            traces[name].append(shoe.data.root_pos_w[0].cpu().numpy().copy())
        if render:
            camera.update(PHYSICS_DT)
    return {name: np.asarray(trace) for name, trace in traces.items()}


def main() -> int:
    if not args.settle_steps > max(SETTLE_WINDOW_STEPS, args.render_steps):
        raise ValueError("--settle-steps must exceed both the settle window and --render-steps")
    config = layout.sample_configs(args.config_index + 1)[args.config_index]
    meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
    sim, arm, shoes, camera, spec = build_scene(layout.shoe_start_poses(config, meta))

    home = arm.data.default_joint_pos.clone()
    arm.write_joint_state_to_sim(home, torch.zeros_like(home))
    pan_id = arm.find_joints(robot.HEAD_PAN_JOINT)[0][0]
    tilt_id = arm.find_joints(robot.HEAD_TILT_JOINT)[0][0]
    pan_deg, tilt_deg = urdf_head_angles(layout.HEAD_PAN_ENCODER_DEG, layout.HEAD_TILT_ENCODER_DEG)
    target = home.clone()
    target[:, pan_id], target[:, tilt_id] = math.radians(pan_deg), math.radians(tilt_deg)
    gravity_ids, _ = arm.find_joints(robot.GRAVITY_COMPENSATION_JOINTS)
    traces = settle(sim, arm, shoes, camera, target, gravity_ids)

    link = arm.find_bodies(spec.link)[0][0]
    camera_pose = camera_pose_from_link(
        arm.data.body_pos_w[0, link].cpu().numpy(), arm.data.body_quat_w[0, link].cpu().numpy(), spec.pos, spec.quat_wxyz
    )
    intrinsic = camera.data.intrinsic_matrices[0].cpu().numpy()
    rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
    depth = camera.data.output["distance_to_image_plane"][0, :, :, 0].cpu().numpy()
    poses = {name: (s.data.root_pos_w[0].cpu().numpy(), s.data.root_quat_w[0].cpu().numpy()) for name, s in shoes.items()}

    groups = [
        PointGroup(name, False, pos + np.asarray(meta["objects"][name]["keypoint_offsets"]) @ quat_wxyz_to_matrix(quat).T)
        for name, (pos, quat) in poses.items()
    ]
    groups.append(PointGroup(layout.RACK, True, layout.rack_keypoints()))
    anchor = np.concatenate([group.points_w for group in groups]).mean(axis=0)
    snapshot = annotate_snapshot(rgb, depth, camera_pose, intrinsic, groups, anchor, layout.OVERLAP_MIN_PX)

    head_error = max(
        abs(math.degrees(arm.data.joint_pos[0, pan_id].item()) - pan_deg),
        abs(math.degrees(arm.data.joint_pos[0, tilt_id].item()) - tilt_deg),
    )
    settle_motion = max(
        float(np.linalg.norm(trace[-SETTLE_WINDOW_STEPS:] - trace[-1], axis=1).max()) for trace in traces.values()
    )
    rack_errors = [abs(r.render_depth - r.depth) for r in snapshot.records if r.is_static and r.visible and r.kept]
    rack_depth_error = max(rack_errors) if rack_errors else None
    failures = []
    if not snapshot.min_margin_px >= layout.MIN_BORDER_MARGIN_PX:
        failures.append(f"keypoint border margin {snapshot.min_margin_px:.1f} px < {layout.MIN_BORDER_MARGIN_PX}")
    if not head_error <= MAX_HEAD_ERROR_DEG:
        failures.append(f"head angle error {head_error:.2f} deg > {MAX_HEAD_ERROR_DEG}")
    if not settle_motion <= MAX_SETTLE_MOTION_M:
        failures.append(f"shoes still moving {settle_motion * 1000:.1f} mm over the last {SETTLE_WINDOW_STEPS} steps")
    if rack_depth_error is None or not rack_depth_error <= MAX_RACK_DEPTH_ERROR_M:
        failures.append(f"rack keypoint depth vs render depth {rack_depth_error} m > {MAX_RACK_DEPTH_ERROR_M}")
    try:
        run_files.movable_objects(
            {"keypoints": [vars(r) for r in snapshot.records], "objects": {n: {"position": p.tolist()} for n, (p, _) in poses.items()}},
            meta,
            layout.supports(),
        )
    except ValueError as exc:
        failures.append(str(exc))

    run_dir = layout.RUNS_DIR / f"config_{config.index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(run_dir / "snapshot_raw.png")
    snapshot.image.save(run_dir / "snapshot.png")
    checks = {
        "passed": not failures,
        "failures": failures,
        "head_error_deg": head_error,
        "settle_motion_m": settle_motion,
        "rack_depth_error_max_m": rack_depth_error,
    }
    head = {"pan_encoder_deg": layout.HEAD_PAN_ENCODER_DEG, "tilt_encoder_deg": layout.HEAD_TILT_ENCODER_DEG}
    doc = run_files.keypoints_document(
        scene_config=vars(config), camera=camera_pose, intrinsic=intrinsic, snapshot=snapshot,
        image_size=(rgb.shape[1], rgb.shape[0]), object_poses=poses, head=head, checks=checks,
    )
    run_files.write_json(run_dir / "keypoints.json", doc)
    print(
        f"SNAPSHOT config {config.index:02d} derotation {snapshot.derotation_deg:+.1f} deg margin {snapshot.min_margin_px:+.1f} px "
        f"gap {snapshot.min_gap_px} px head_err {head_error:.2f} deg settle {settle_motion * 1000:.2f} mm "
        f"rack_depth_err {rack_depth_error} m passed {not failures} -> {run_dir}",
        flush=True,
    )
    for failure in failures:
        print(f"SNAPSHOT CHECK FAILED: {failure}", flush=True)
    return 0 if not failures else 1


os._exit(main())
````

- [ ] **Step 2: 구성 0 을 찍는다**

Run: `cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh scripts/iker/snapshot.py --config-index 0 --headless 2>&1 | grep -E "SNAPSHOT|Traceback|Error"`
Expected(약 10 s): 한 줄의 `SNAPSHOT config 00 ... passed True`. `SNAPSHOT CHECK FAILED` 줄은 없어야 한다.

| 지표 | 프로토타입 값 | 허용 |
|---|---|---|
| derotation | −22.7° | 참고 |
| margin | +49.1 px | ≥ 15 |
| gap | 25.3 px | ≥ 15 |
| head_err | 0.14° | ≤ 0.5 |
| settle | 0.00 mm | ≤ 5 |
| rack_depth_err | 0.00013 m | ≤ 0.005 |

- [ ] **Step 3: 영상과 번호를 확인한다**

`iker_runs/shoe_place/config_00/snapshot.png` 를 열어 다음을 확인한다.
- 옮길 신발(테이블 위, 왼쪽): 빨간 원 1~4.
- 받침 위 다른 신발: 초록 원 5~8.
- 받침 격자: 파란 원 9~20.
- 가려진 점: 회색 테두리.
- 오른쪽 아래 노란 범례: +x 위, +y 왼쪽.

Run:
```bash
cd ~/rl_ws/hdgp-iker && python3 -c "
import json; d = json.load(open('iker_runs/shoe_place/config_00/keypoints.json'))
print({n: [r['label'] for r in d['keypoints'] if r['object_name'] == n] for n in ('shoe_move', 'shoe_other')}, d['checks']['passed'])"
```
Expected: `{'shoe_move': [1, 2, 3, 4], 'shoe_other': [5, 6, 7, 8]} True`

- [ ] **Step 4: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker
git add scripts/iker/snapshot.py iker_runs/shoe_place/config_00/snapshot_raw.png iker_runs/shoe_place/config_00/snapshot.png iker_runs/shoe_place/config_00/keypoints.json
git commit -F - <<'MSG'
feat(iker): 머리 카메라 스냅샷 스크립트와 구성 0 스냅샷

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 12: 사람 기준선과 응답 수집 CLI

**Files:**
- Create: `scripts/iker/human_baseline.py`, `scripts/iker/ingest.py`
- Create(스크립트 산출): `iker_runs/shoe_place/config_00/interaction_human.json`

**Interfaces:**
- Consumes: `run_files.*`, `baseline.legacy_relation_targets`, `baseline.human_target`, `gate.evaluate_gate`, `gate.gate_response`, `rotations.quat_wxyz_to_matrix`, `layout.*`
- Produces:
  - `human_baseline.py --run-dir <dir> [--meta]`: `interaction_human.json` 을 쓴다. 게이트 통과 시 종료 코드 0.
  - `ingest.py --run-dir <dir> --attempt N [--meta]`
    - `attempt_NN/response.md` 를 읽어 `attempt_NN/gate.json` 을 쓴다.
    - 통과하면 `interaction_vlm.json` 도 쓰고 종료 코드 0.
  - 계획 3 의 생성 루프가 `ingest.py` 를 그대로 부른다.

검증용 응답 두 개는 저장소가 아니라 임시 복사본에서만 쓴다. `attempt_NN` 은 계획 3 의 실제 VLM 시도 자리다.

- [ ] **Step 1: 스크립트를 만든다**

Create `scripts/iker/human_baseline.py`:

````python
#!/usr/bin/env python3
"""Human baseline interaction for one IKER configuration (design spec §6). No Isaac needed.

Reads <run-dir>/keypoints.json and assets/iker_shoe/shoe_meta.json, carries the public IKER target
relation onto the snapshot's other shoe, refits it to the moving shoe, gates it like a VLM result
and writes <run-dir>/interaction_human.json. Exit code 0 only when the gate passes.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm python3 scripts/iker/human_baseline.py \
        --run-dir iker_runs/shoe_place/config_00
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openarm.agnostic.modules.iker import baseline, gate, run_files
from openarm.agnostic.modules.iker.interaction import Interaction
from openarm.agnostic.modules.iker.rotations import quat_wxyz_to_matrix
from openarm.agnostic.tasks.iker_shoe import layout

LEGACY_OTHER_SHOE = "object_1"


def human_interaction(doc: dict, meta: dict) -> tuple[Interaction, gate.GateReport, baseline.HumanTarget]:
    supports = layout.supports()
    movables = {obj.name: obj for obj in run_files.movable_objects(doc, meta, supports)}
    legacy = meta["legacy"]
    legacy_other = legacy["object_pose"][LEGACY_OTHER_SHOE]
    other = doc["objects"][layout.OTHER_SHOE]
    relation = baseline.legacy_relation_targets(
        legacy_other_pos=np.add(legacy_other["xyz"], [0.0, 0.0, legacy["z_lift"]]),
        legacy_other_rpy=legacy_other["rpy"],
        legacy_targets=legacy["target_position"],
        other_pos=other["position"],
        other_rotation=quat_wxyz_to_matrix(other["quat_wxyz"]),
    )
    moving = movables[layout.MOVING_SHOE]
    target = baseline.human_target(
        relation,
        moving.local_offsets,
        moving.hull_local,
        start_rotation=quat_wxyz_to_matrix(doc["objects"][layout.MOVING_SHOE]["quat_wxyz"]),
        long_axis=int(meta["objects"][layout.MOVING_SHOE]["horizontal_axes"][0]),
        supports=supports,
    )
    interaction = Interaction(
        object_name=layout.MOVING_SHOE,
        keypoint_ids=moving.keypoint_ids,
        grasp_mode=True,
        coordinates={kid: tuple(point) for kid, point in zip(moving.keypoint_ids, target.keypoints.tolist())},
        done=False,
    )
    report = gate.evaluate_gate(interaction, tuple(movables.values()), run_files.static_keypoint_ids(doc), layout.gate_config())
    return interaction, report, target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the human baseline interaction for one configuration.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--meta", type=Path, default=layout.SHOE_META_PATH)
    args = parser.parse_args(argv)
    run_dir = args.run_dir if args.run_dir.is_absolute() else layout.HDGP_ROOT / args.run_dir
    doc = run_files.read_json(run_dir / "keypoints.json")
    meta = run_files.load_shoe_meta(args.meta)
    interaction, report, target = human_interaction(doc, meta)
    extra = {
        "relation_residual_m": target.relation_residual_m.tolist(),
        "width_pair_swapped": target.width_pair_swapped,
        "support": target.support_name,
    }
    out = run_files.write_json(
        run_dir / "interaction_human.json", run_files.interaction_document("human", interaction, report, extra=extra)
    )
    print(f"gate passed={report.passed} failures={list(report.failures)}")
    print(f"targets {np.round(report.target_keypoints, 4).tolist()} -> {out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
````

Create `scripts/iker/ingest.py`:

````python
#!/usr/bin/env python3
"""Gate one VLM response for an IKER configuration (design spec §6). No Isaac needed.

Reads <run-dir>/attempt_NN/response.md, writes attempt_NN/gate.json, and on a pass writes
<run-dir>/interaction_vlm.json. Exit code 0 only when the gate passes.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm python3 scripts/iker/ingest.py \
        --run-dir iker_runs/shoe_place/config_00 --attempt 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openarm.agnostic.modules.iker import gate, run_files
from openarm.agnostic.tasks.iker_shoe import layout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate one VLM response.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--meta", type=Path, default=layout.SHOE_META_PATH)
    args = parser.parse_args(argv)
    run_dir = args.run_dir if args.run_dir.is_absolute() else layout.HDGP_ROOT / args.run_dir
    attempt_dir = run_dir / f"attempt_{args.attempt:02d}"
    response_path = attempt_dir / "response.md"
    if not response_path.is_file():
        raise FileNotFoundError(f"no VLM response at {response_path}")
    doc = run_files.read_json(run_dir / "keypoints.json")
    meta = run_files.load_shoe_meta(args.meta)
    movables = run_files.movable_objects(doc, meta, layout.supports())
    interaction, report = gate.gate_response(
        response_path.read_text(encoding="utf-8"),
        run_files.vlm_keypoints(doc),
        movables,
        run_files.static_keypoint_ids(doc),
        layout.gate_config(),
    )
    run_files.write_json(attempt_dir / "gate.json", {"schema": run_files.SCHEMA_VERSION, **run_files.gate_document(report)})
    print(f"attempt {args.attempt:02d}: gate passed={report.passed} failures={list(report.failures)}")
    if report.passed and interaction is not None:
        out = run_files.write_json(
            run_dir / "interaction_vlm.json",
            run_files.interaction_document("vlm", interaction, report, extra={"attempt": args.attempt}),
        )
        print(f"wrote {out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
````

- [ ] **Step 2: 사람 기준선을 만든다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 scripts/iker/human_baseline.py --run-dir iker_runs/shoe_place/config_00 2>&1 | grep -v -i warn`
Expected:
- 출력 `gate passed=True failures=[]`. 목표 중심은 받침 위 약 (0.261, −0.015, 0.379)(프로토타입 값).
- 대상 줄 `targets [[0.3858, -0.0149, 0.38], [0.1363, -0.0144, 0.3779], [0.261, -0.0627, 0.3814], [0.2612, 0.0334, 0.3765]]`(안착 기울기에 따라 mm 단위로 달라질 수 있음).
- 종료 코드 0.

- [ ] **Step 3: VLM 형식 응답 두 개를 임시 복사본에서 게이트에 넣는다**

첫 응답은 다른 신발 키포인트를 옆으로 옮긴 상대 좌표라 통과해야 한다. 둘째는 공개 데이터 간격을 그대로 쓴 목표라 G4 에서 실패해야 한다.
Task 11 Step 3 에서 번호가 `shoe_move` 1~4, `shoe_other` 5~8 이 아니었다면 응답의 번호를 그 번호로 바꾼다.

````bash
cd ~/rl_ws/hdgp-iker && export PYTHONPATH=source/openarm
RUN=$(mktemp -d)/config_00 && cp -r iker_runs/shoe_place/config_00 "$RUN" && mkdir "$RUN/attempt_00" "$RUN/attempt_01"
cat > "$RUN/attempt_00/response.md" <<'EOF'
The left shoe (keypoints 1-4) goes on the rack, to the left of the other shoe (keypoints 5-8).

```python
import numpy as np

def get_interaction_data(keypoint_coordinates):
    """Copy the other shoe's keypoints 0.129 m to the left, lowered to the moving shoe's resting height."""
    object_to_interact = "left shoe"
    keypoint_indices_to_interact = [1, 2, 3, 4]
    grasp_mode = True
    for moving, reference in (("1", "5"), ("2", "6"), ("3", "7"), ("4", "8")):
        keypoint_coordinates[moving] = keypoint_coordinates[reference] + np.array([0.0, 0.129, -0.0029])
    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
```
EOF
cat > "$RUN/attempt_01/response.md" <<'EOF'
```python
import numpy as np

def get_interaction_data(keypoint_coordinates):
    """The public IKER target relation (spacing 0.150 / 0.100 m) around the other shoe's centre."""
    object_to_interact = "left shoe"
    keypoint_indices_to_interact = [1, 2, 3, 4]
    grasp_mode = True
    centre = sum(keypoint_coordinates[k] for k in ("5", "6", "7", "8")) / 4
    keypoint_coordinates["1"] = centre + np.array([0.0623, 0.129, 0.0265])
    keypoint_coordinates["2"] = centre + np.array([-0.0877, 0.129, 0.0265])
    keypoint_coordinates["3"] = centre + np.array([-0.0127, 0.079, 0.0265])
    keypoint_coordinates["4"] = centre + np.array([-0.0127, 0.179, 0.0265])
    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
```
EOF
python3 scripts/iker/ingest.py --run-dir "$RUN" --attempt 0; echo "exit=$?"
python3 scripts/iker/ingest.py --run-dir "$RUN" --attempt 1; echo "exit=$?"
ls "$RUN"
````
Expected(경고 줄 제외):
```text
attempt 00: gate passed=True failures=[]
wrote <RUN>/interaction_vlm.json
exit=0
attempt 01: gate passed=False failures=['G4: normalized error at the best rigid fit 0.118 > 0.1']
exit=1
```
- attempt 01 의 수치는 0.11~0.12.
- `ls` 에는 `attempt_00 attempt_01 interaction_human.json interaction_vlm.json keypoints.json snapshot.png snapshot_raw.png` 가 나온다.
- 저장소의 `iker_runs/shoe_place/config_00` 에는 `attempt_*`·`interaction_vlm.json` 이 **없어야** 한다.

- [ ] **Step 4: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker
git add scripts/iker/human_baseline.py scripts/iker/ingest.py iker_runs/shoe_place/config_00/interaction_human.json
git commit -F - <<'MSG'
feat(iker): 사람 기준선·VLM 응답 게이트 CLI 와 구성 0 사람 기준선

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 13: 종료 게이트(커밋 단위·작업 트리·인계)

"모든 단계 완료"가 곧 "쓸 수 있는 상태"가 되도록, 커밋·트리·산출물을 직접 확인하고 인계한다(관찰 0155).

- [ ] **Step 1: 전체 순수 테스트**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests source/openarm/openarm/agnostic/tasks/iker_shoe/tests -q -p no:cacheprovider`
Expected: `103 passed`

- [ ] **Step 2: 기존 모듈 테스트가 깨지지 않았는지 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/tests -q -p no:cacheprovider`
Expected: `234 passed`(2026-09-13 착수 전 기준). 이 계획은 그 폴더를 건드리지 않으므로, 달라졌으면 멈추고 보고한다.

- [ ] **Step 3: 커밋 단위와 작업 트리를 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && git log --oneline main..HEAD && git status --short`
Expected:
- `git log`: 커밋 13개(문서 1 + 태스크 1~12 의 기능 커밋 12).
- `git status --short`: 출력 없음(`.superpowers/` 는 자체 .gitignore 로 무시된다).

Run: `cd ~/rl_ws/hdgp-iker && git lfs ls-files | grep -c iker_shoe`
Expected: `10`

- [ ] **Step 4: 인계**

사용자에게 다음을 보고한다.
1. 커밋 13개 목록과 테스트 수(iker 103, 기존 모듈 234).
2. `iker_runs/shoe_place/config_00/snapshot.png`(영상 첨부)와 스냅샷 수치.
3. 사람 기준선 게이트 결과와 두 검증 응답 결과.
4. 브랜치 `iker-front-end` 는 push·병합하지 않았다. 병합 여부는 사용자가 정한다.
5. 다음 계획(2번: 파지 뱅크·학습 환경·사람 기준선 학습) 착수 조건: `config_00/keypoints.json` 의 `checks.passed=true`, `interaction_human.json` 의 `gate.passed=true`.
