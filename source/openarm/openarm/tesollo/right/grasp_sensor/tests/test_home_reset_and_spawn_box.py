"""고정 홈 리셋 + 스폰 박스 계약 (2026-08-18).

perception_plus_plus 연결 요건을 코드로 고정한다:
  1) 컵 스폰 박스가 x 0.25~0.35, y -10.10~-10.30 이다
  2) 리셋 홈 palm 이 그 박스 **밖**이다 (side 접근 전제)
  3) 홈에서 액션만으로 모든 컵에 도달한다 (palm_delta_xyz 의 y 성분)
  4) 유휴 팔이 중립 접힘 자세다
하나라도 깨지면 학습을 돌려도 s2r 전제가 무너진다.

cfg 는 isaaclab→pxr 를 끌어와 Isaac 앱 없이는 import 가 안 된다. 그래서
dataclass 필드 기본값을 ast 로 직접 읽는다(테스트가 무거워지지 않도록).
"""
import ast
from pathlib import Path

from openarm.tesollo.right.grasp_sensor.grasp_right_preset import LEFT_ARM_REST_JOINT_POS

REST = LEFT_ARM_REST_JOINT_POS
OTHER = "l"
SIGN = -1
_CFG_SRC = Path(__file__).resolve().parents[1] / "grasp_right_env_cfg.py"


def _cfg_literals():
    """GraspRightEnvCfg 의 필드 기본값을 리터럴로 추출."""
    tree = ast.parse(_CFG_SRC.read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "GraspRightEnvCfg")
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


def test_action_reference_is_cup_anchored():
    """★2026-08-19(A0) 고정 홈 모드의 액션 기준점 = 컵-정준 pregrasp.

    구 계약("홈에서 액션 y 범위만으로 모든 컵 도달" → palm_delta y=0.35)은 폐기.
    이제 물리 리셋만 홈이고 기준점은 컵 pregrasp 이므로, 도달성은 기준점이 보장하고
    delta 는 미세조정 용도다. 회귀 = 고정 홈 분기가 기준점을 홈으로 되덮는 것.
    """
    env_src = (Path(__file__).resolve().parents[1] / "grasp_right_env.py").read_text()
    assert "pregrasp_palm_pose = self.home_palm_pose" not in env_src, (
        "고정 홈 분기가 액션 기준점을 홈으로 되덮고 있다 (A0 회귀)"
    )


def test_action_range_covers_pregrasp_offset():
    """기준점(컵 pregrasp)에서 delta 액션으로 컵 파지 중심까지 도달 가능해야 한다.

    |pregrasp_offset| + 기준점 노이즈 ≤ palm_delta (축별). 이게 깨지면 action=+1 로도
    컵에 못 닿는 축이 생긴다.
    """
    c = _cfg_literals()
    d = c["palm_delta_xyz"]
    d = (d, d, d) if isinstance(d, (int, float)) else d
    off = (c["pregrasp_offset_x"], c["pregrasp_offset_y"], c["pregrasp_offset_z"])
    noise = (c["pregrasp_noise_x"], c["pregrasp_noise_y"], c["pregrasp_noise_z"])
    for ax, (o, nz, dx) in enumerate(zip(off, noise, d)):
        assert abs(o) + nz <= dx + 1e-9, (
            f"축 {ax}: |pregrasp_offset|({abs(o)})+noise({nz}) > palm_delta({dx})"
        )


def test_idle_arm_rest_mirrors_grasp_home():
    """유휴 팔 rest = 파지 팔 홈의 부호 미러.

    좌우 완전 대칭 + 양팔 pour 초기 자세 정합을 위한 계약.
    (env._build_home_pose 가 Fabrics 로 푼 실제 q_home 과의 일치를 런타임에도 검사한다.)
    """
    ARM_SIGN = [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
    # ★08.18 grasp_sensor(a1/DG-5F) 실측 q_home — env._build_home_pose 첫 부팅 출력값.
    #   bi_s(DG-5FS) 값([0.3082, 0.5785, ...])과 최대 0.265rad 다르다(palm 오프셋 5.5cm).
    HOME_R = [0.0431, 0.6706, 0.0961, 0.7342, 0.3750, 0.5678, 0.6709]
    want = HOME_R if OTHER == "r" else [s * v for s, v in zip(ARM_SIGN, HOME_R)]
    have = [REST[f"{OTHER}_aj_{i}"] for i in range(1, 8)]
    assert all(abs(a - b) < 1e-6 for a, b in zip(have, want)), (have, want)


def test_idle_arm_is_not_neutral_zero():
    """전 관절 0 이면 팔이 앞으로 뻗어 파지 팔·카메라와 겹친다 — 회귀 방지."""
    have = [REST[f"{OTHER}_aj_{i}"] for i in range(1, 8)]
    assert any(abs(v) > 0.05 for v in have)
