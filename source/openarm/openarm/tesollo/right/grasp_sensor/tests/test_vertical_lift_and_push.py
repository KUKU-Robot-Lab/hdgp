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

_SRC = Path(__file__).resolve().parents[1] / "grasp_right_env_cfg.py"
_ENV = Path(__file__).resolve().parents[1] / "grasp_right_env.py"


def _cfg():
    tree = ast.parse(_SRC.read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "GraspRightEnvCfg")
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
    """2026-08-20: 컵 밀기 억제가 **곱셈 감쇠 → 덧셈 페널티**로 바뀌었다.

    구 disp_factor(=1-clamp(disp/limit))는 lift/success_bonus 에 곱해져서
    실측 밀림 0.084 에서 정확히 0 이 됐다 = 하드게이트. 가중치 50짜리 두 항이
    통째로 소거되고, 감쇠 없는 grasp(12.0)만 남아 손끝 파지가 최적해가 됐다.
    신 설계는 상한 있는 덧셈 페널티라 gradient 가 절대 소실되지 않는다.
    """
    c = _cfg()
    assert c["push_penalty_weight"] > 0.0
    assert c["push_penalty_ref"] > 0.0
    core = (Path(__file__).resolve().parents[1] / "grasp_reward.py").read_text()
    # 주석은 제외하고 **코드 라인만** 검사한다(헤더가 구 설계를 설명하므로).
    code = "\n".join(
        l for l in core.split("\n") if not l.lstrip().startswith("#")
    )
    assert "* disp_factor" not in code, "컵 밀림이 다시 곱셈 감쇠가 됐다"
    assert "push_penalty = -" in code


def test_push_penalty_is_meaningful_but_not_dominant():
    """페널티가 너무 작으면 무시되고, 너무 크면 '컵에 안 다가감' 국소최적이 된다.

    신 설계 기준: 페널티 상한(push_penalty_weight)이 lift+hold 합의 5~20% 구간.
    """
    c = _cfg()
    w = c["push_penalty_weight"]
    stage_max = c["lift_weight"] + c["hold_weight"]
    assert 0.05 * stage_max <= w <= 0.20 * stage_max, (
        f"push_penalty_weight={w} 가 lift+hold({stage_max})의 5~20% 밖"
    )
def test_j7_lift_removed_from_env():
    """구 관절공간 리프트 잔재가 남아 있으면 두 경로가 공존해 조용히 어긋난다."""
    src = _ENV.read_text()
    for dead in ("prelift_arm_pos_buf", "lift_arm_start_buf",
                 "compute_joint7_lift_wait_target"):
        assert dead not in src, f"구 리프트 잔재: {dead}"
    assert "lift_palm_pose_buf" in src


def test_fabric_freeze_removed_during_lift():
    """리프트 중 Fabrics 동결이 남아 있으면 palm 램프가 물리적으로 죽는다.

    동결 = fabric_q 팔 상태를 매 스텝 실측으로 되돌리고 속도를 0으로 초기화
    → integrator 가 목표로 전진하지 못해 z 상승 ≈ 0 (GPU 프로브 실측 -0.3mm).
    35cee1b 가 이 블록을 지우지 않은 채 커밋된 사고의 회귀 방지.
    """
    src = _ENV.read_text()
    assert "freeze_mask" not in src, "리프트 중 Fabrics 동결 블록 잔재"
