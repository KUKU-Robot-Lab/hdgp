"""수직 리프트 + 컵 밀기 억제 계약 (2026-08-19).

구 리프트(j7 ±0.31rad)는 palm 자세를 17.76° 회전시켜 쥔 컵을 같이 기울였다.
실측 cup/tilt_deg 9.07° 은 그 절반 = 손 안에서 미끄러진다는 뜻이고,
stabilize_upright_max_deg=5.0 을 못 넘어 stabilize 항이 구조적으로 죽어 있었다.

여기서 고정하는 것:
  1) 리프트 높이가 성공 임계보다 충분히 크다
  2) 보상 정규화 기준(lift_height_ref)이 리프트 높이와 일치 — 안 맞으면 미끄러짐을 못 잰다
  3) 컵 밀림 soft 감쇠가 켜져 있다
  4) j7 리프트 잔재가 코드에서 사라졌다

cfg 는 isaaclab→pxr 를 끌어와 Isaac 앱 없이 import 가 안 되므로 ast 로 리터럴을 읽는다.
"""
import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "grasp_left_env_cfg.py"
_ENV = Path(__file__).resolve().parents[1] / "grasp_left_env.py"


def _cfg():
    tree = ast.parse(_SRC.read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "GraspLeftEnvCfg")
    out = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            try:
                out[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def test_lift_height_exceeds_success_threshold():
    c = _cfg()
    assert c["lift_height_delta"] >= 2.0 * c["lift_success_height"], (
        f"리프트 {c['lift_height_delta']} 가 성공 임계 {c['lift_success_height']} 대비 여유 부족")


def test_reward_normalizer_matches_lift_height():
    """정규화 기준이 리프트 높이와 다르면 미끄러짐을 구분하지 못한다.

    ref < delta 면 컵이 ref 만큼만 따라와도 만점 → 나머지 미끄러짐이 공짜.
    """
    c = _cfg()
    assert c["lift_height_ref"] == c["lift_height_delta"]


def test_cup_push_decay_enabled():
    c = _cfg()
    assert c["cup_xy_disp_limit"] > 0.0
    # 하드 게이트가 아니라 감쇠여야 한다 — 성공 임계(grasp_xy_threshold)보다 커야
    # 임계 근처에서 보상이 절벽처럼 끊기지 않는다.
    assert c["cup_xy_disp_limit"] > c["grasp_xy_threshold"]


def test_push_penalty_is_meaningful_but_not_dominant():
    """페널티가 너무 작으면 무시되고, 너무 크면 '컵에 안 다가감' 국소최적이 된다."""
    c = _cfg()
    w = c["approach_xy_penalty_weight"]
    assert 10.0 <= w <= 60.0, f"approach_xy_penalty_weight={w} 가 권장 범위 밖"


def test_j7_lift_removed_from_env():
    """구 관절공간 리프트 잔재가 남아 있으면 두 경로가 공존해 조용히 어긋난다."""
    src = _ENV.read_text()
    for dead in ("prelift_arm_pos_buf", "lift_arm_start_buf",
                 "compute_joint7_lift_wait_target"):
        assert dead not in src, f"구 리프트 잔재: {dead}"
    assert "lift_palm_pose_buf" in src
