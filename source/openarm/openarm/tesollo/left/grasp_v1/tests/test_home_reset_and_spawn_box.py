"""고정 홈 리셋 + 스폰 박스 계약 (2026-08-18).

perception_plus_plus 연결 요건을 코드로 고정한다:
  1) 컵 스폰 박스가 x 0.25~0.35, y 10.10~10.30 이다
  2) 리셋 홈 palm 이 그 박스 **밖**이다 (side 접근 전제)
  3) 홈에서 액션만으로 모든 컵에 도달한다 (palm_delta_xyz 의 y 성분)
  4) 유휴 팔이 중립 접힘 자세다
하나라도 깨지면 학습을 돌려도 s2r 전제가 무너진다.

cfg 는 isaaclab→pxr 를 끌어와 Isaac 앱 없이는 import 가 안 된다. 그래서
dataclass 필드 기본값을 ast 로 직접 읽는다(테스트가 무거워지지 않도록).
"""
import ast
from pathlib import Path

from openarm.tesollo.left.grasp_v1.grasp_left_preset import RIGHT_ARM_REST_JOINT_POS

SIGN = 1
_CFG_SRC = Path(__file__).resolve().parents[1] / "grasp_left_env_cfg.py"


def _cfg_literals():
    """GraspLeftEnvCfg 의 필드 기본값을 리터럴로 추출."""
    tree = ast.parse(_CFG_SRC.read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "GraspLeftEnvCfg")
    out = {}
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign) or node.value is None:
            continue
        name = node.target.id
        try:
            out[name] = ast.literal_eval(node.value)
        except ValueError:
            # field(default_factory=lambda: {...}) → 람다 본문의 dict 를 집는다.
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Lambda):
                    out[name] = ast.literal_eval(sub.body)
                    break
    return out


def _spawn_box(c):
    xr = c["adr_custom_cfg"]["spawn"]["x_range"][1]
    yr = c["adr_custom_cfg"]["spawn"]["y_range"][1]
    return ((c["object_spawn_x_center"] - xr, c["object_spawn_x_center"] + xr),
            (c["object_spawn_y_center"] - yr, c["object_spawn_y_center"] + yr))


def test_spawn_box_matches_requested_range():
    (x0, x1), (y0, y1) = _spawn_box(_cfg_literals())
    assert (round(x0, 3), round(x1, 3)) == (0.25, 0.35)
    assert sorted(round(v, 3) for v in (y0, y1)) == sorted((SIGN * 0.10, SIGN * 0.30))


def test_home_palm_is_outside_spawn_box():
    c = _cfg_literals()
    assert c["reset_from_fixed_home"] is True
    hx, hy = c["reset_home_palm_pose"][0], c["reset_home_palm_pose"][1]
    (x0, x1), (y0, y1) = _spawn_box(c)
    inside = (x0 <= hx <= x1) and (min(y0, y1) <= hy <= max(y0, y1))
    assert not inside, f"홈 palm ({hx}, {hy}) 이 스폰 박스 안이다"
    # side 접근: 홈은 컵보다 바깥쪽(|y| 큰 쪽)이어야 한다.
    assert abs(hy) > max(abs(y0), abs(y1)) - 1e-9


def test_action_range_reaches_every_cup():
    c = _cfg_literals()
    hy = c["reset_home_palm_pose"][1]
    d = c["palm_delta_xyz"]
    dy = d if isinstance(d, (int, float)) else d[1]
    (_, _), (y0, y1) = _spawn_box(c)
    far = max(abs(y0 - hy), abs(y1 - hy))
    assert far <= dy + 1e-9, f"먼 쪽 컵까지 {far:.3f}m 인데 액션 y 범위는 {dy:.3f}m"


def test_idle_arm_rest_is_neutral_fold():
    """유휴 팔은 중립 접힘 [0,0,0,1.4,0,0,0] — 좌우 동일 값(_ARM_SIGN[3]=+1)."""
    vals = [RIGHT_ARM_REST_JOINT_POS[f"r_aj_{i}"] for i in range(1, 8)]
    assert vals == [0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0], vals
