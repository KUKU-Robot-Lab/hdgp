"""task_matrix 정적 판정 테스트 — Isaac·GPU 불필요.

이 테스트가 지키는 것은 두 가지다.
  ① 부팅 불가 구성이 **BLOCK 으로 드러나는가** (조용히 통과하면 매트릭스가 무의미하다)
  ② gym id 조립 규칙이 실제 config 소스와 어긋나지 않는가 (드리프트 차단)
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import task_matrix as TM  # noqa: E402


# ---------------------------------------------------------------- 구조
def test_rows_cover_all_four_upgraded_task_families():
    tasks = {r.task for r in TM.build_rows()}
    assert tasks == {
        "agnostic/grasp_sensor",
        "agnostic/grasp_lift_fabric",
        "agnostic/pour_fabric",
        "gripper/left/grasp_sensor",
    }


def test_every_row_carries_at_least_one_gate_and_a_gym_id():
    for row in TM.build_rows():
        assert row.gates, f"{row.task}/{row.variant} 에 게이트가 없다"
        if row.registered:
            assert row.gym_ids, f"{row.task}/{row.variant} 에 gym id 가 없다"
        else:
            assert not row.gym_ids, "미등록인데 id 를 주장한다"


# ---------------------------------------------------------------- 판정
def _row(task: str, variant: str) -> TM.TaskRow:
    for r in TM.build_rows():
        if r.task == task and r.variant == variant:
            return r
    raise AssertionError(f"행 없음: {task}/{variant}")


def test_gripper_left_profile_of_grasp_sensor_is_blocked_by_missing_fabric_class():
    """sens_l 은 등록되지만 _setup_fabrics 에서 RuntimeError 로 죽는다."""
    row = _row("agnostic/grasp_sensor", "gripper_left")
    names = {g.name for g in row.blockers}
    assert "fabric_class" in names, f"BLOCK 목록: {names}"
    assert row.verdict == TM.BLOCK


def test_tesollo_right_profile_of_grasp_sensor_is_bootable():
    row = _row("agnostic/grasp_sensor", "tesollo_right")
    assert row.blockers == (), f"예상치 못한 BLOCK: {[g.name for g in row.blockers]}"


def test_pour_fabric_rows_are_blocked_by_missing_warm_bank():
    rows = [r for r in TM.build_rows() if r.task == "agnostic/pour_fabric"]
    assert rows, "pour_fabric 행이 없다"
    for row in rows:
        assert "warm_bank" in {g.name for g in row.blockers}, row.variant


def test_unverified_palm_box_is_a_warning_not_a_blocker():
    """bis_right 만 실측 박스다 — 나머지는 경고로 드러나되 부팅은 막지 않는다."""
    verified = _row("agnostic/grasp_lift_fabric", "bis_right")
    inherited = _row("agnostic/grasp_lift_fabric", "bi_right")

    assert "palm_box_verified" not in {g.name for g in verified.warns}
    assert "palm_box_verified" in {g.name for g in inherited.warns}
    assert "palm_box_verified" not in {g.name for g in inherited.blockers}


def test_gripper_left_fabric_variant_is_the_canonical_one():
    """사용자 결정: _fab 이 정본. 매트릭스가 그걸 표기해야 한다."""
    row = _row("gripper/left/grasp_sensor", "fab")
    assert row.canonical is True
    assert _row("gripper/left/grasp_sensor", "joint").canonical is False


# ---------------------------------------------------------------- 드리프트 차단
def test_gym_id_rule_matches_grasp_lift_fabric_config_source():
    src = (TM.AGNOSTIC_DIR / "tasks/grasp_lift_fabric/config/__init__.py").read_text()
    assert 'f"open-{_p.asset.short}_{_side}_grasp_lift_fab"' in src, (
        "config 의 id 조립식이 바뀌었다 — task_matrix 의 규칙도 함께 고쳐야 한다"
    )
    ids = _row("agnostic/grasp_lift_fabric", "bis_right").gym_ids
    assert "open-bis_r_grasp_lift_fab" in ids


def test_gym_id_rule_matches_grasp_sensor_config_source():
    src = (TM.AGNOSTIC_DIR / "tasks/grasp_sensor/config/__init__.py").read_text()
    assert 'f"open-{_tag}_grasp_sensor{_suffix}"' in src
    ids = _row("agnostic/grasp_sensor", "tesollo_right").gym_ids
    assert "open-sens_r_grasp_sensor" in ids
    assert "open-sens_r_grasp_sensor-lstm" in ids


def test_gripper_left_ids_match_its_config_source():
    src = (TM.HDGP_ROOT
           / "source/openarm/openarm/gripper/left/grasp_sensor/config/__init__.py").read_text()
    for row in [r for r in TM.build_rows() if r.task == "gripper/left/grasp_sensor"]:
        for gid in row.gym_ids:
            assert f'id="{gid}"' in src, f"{gid} 가 config 에 없다"


# ---------------------------------------------------------------- 출력
def test_markdown_lists_every_row_and_states_the_verdict():
    rows = TM.build_rows()
    md = TM.render_markdown(rows)
    for row in rows:
        assert row.variant in md
    assert "BLOCK" in md


def test_exit_code_is_nonzero_while_any_blocker_exists():
    rows = TM.build_rows()
    assert TM.exit_code(rows) == (1 if any(r.blockers for r in rows) else 0)


def test_exit_code_is_zero_for_a_clean_row_set():
    clean = (TM.TaskRow(task="t", variant="v", gym_ids=("open-x",),
                        gates=(TM.Gate("g", True, TM.BLOCK, "ok"),)),)
    assert TM.exit_code(clean) == 0


def test_fabric_manifest_is_documentation_not_a_boot_dependency():
    """레거시 fabric 디렉터리 3종은 manifest 가 없는 게 정상이다.

    fabrics_sim 도 agnostic 태스크도 manifest 를 읽지 않는다(런타임 의존 0).
    BLOCK 으로 두면 멀쩡히 도는 구성을 부팅 불가로 오보한다.
    """
    src = (TM.HDGP_ROOT / "source/FABRICS/src/fabrics_sim").rglob("*.py")
    assert not [f for f in src if "manifest" in f.read_text()], (
        "fabrics_sim 이 manifest 를 읽기 시작했다면 게이트 등급을 재검토해야 한다"
    )
    for row in TM.build_rows():
        for gate in row.gates:
            if gate.name == "fabric_manifest":
                assert gate.severity == TM.WARN


# ---------------------------------------------------------------- 이식성
def test_asset_tracking_gate_flags_a_file_that_exists_but_is_untracked():
    """존재 검사만으로는 못 잡는 부류를 잡는지 확인한다."""
    untracked = TM.HDGP_ROOT / "docs/eval/fig1_task_sequence.png"
    assert untracked.exists(), "표본 파일이 사라졌다 — 다른 미추적 파일로 교체할 것"

    gate = TM.gate_assets_tracked([untracked])
    assert gate.ok is False
    assert "미추적" in gate.detail


def test_asset_tracking_gate_passes_for_a_tracked_file():
    tracked = TM.ASSETS_DIR / "cup/cup_big_rl.usd"
    assert TM.gate_assets_tracked([tracked]).ok is True


def test_every_referenced_asset_is_committed():
    """새 머신이 clone 만으로 부팅할 수 있는가 — 회귀 잠금."""
    offenders = {
        f"{r.task}/{r.variant}": g.detail
        for r in TM.build_rows()
        for g in r.gates
        if g.name == "assets_tracked" and not g.ok
    }
    assert not offenders, f"git 미추적 자산: {offenders}"


# ---------------------------------------------------------------- perception 이음매
def test_perception_seam_gate_recognizes_the_existing_override_hook():
    legacy = (TM.HDGP_ROOT
              / "source/openarm/openarm/tesollo/right/grasp_v1/grasp_right_env.py")
    assert TM.gate_perception_seam(legacy).ok is True


def test_upgraded_tasks_are_flagged_as_lacking_the_perception_seam():
    """`eval_cup_pos_override` 는 구 env 3종에만 있다 — 신규 4태스크엔 전무(실측).

    이 이음매가 없으면 perception 이 낸 물체 pose 를 정책 관측에 넣을 방법이 없다.
    부팅은 되므로 WARN 이지만, 평가 계획의 전제라 보이게 둬야 한다.
    """
    flagged = {
        r.task for r in TM.build_rows()
        for g in r.gates if g.name == "perception_seam" and not g.ok
    }
    assert flagged == {
        "agnostic/grasp_sensor",
        "agnostic/grasp_lift_fabric",
        "agnostic/pour_fabric",
        "gripper/left/grasp_sensor",
    }


def test_perception_seam_is_a_warning_not_a_blocker():
    for row in TM.build_rows():
        for gate in row.gates:
            if gate.name == "perception_seam":
                assert gate.severity == TM.WARN


def test_profiles_without_fabrics_are_reported_as_unregistered():
    """config 가 SKIPPED 로 건너뛰므로 그 id 들은 **존재하지 않는다**.

    "4개 id 가 있는데 BLOCK" 과 "id 가 아예 없다" 는 다른 사실이다.
    """
    assert _row("agnostic/grasp_sensor", "gripper_left").registered is False
    assert _row("agnostic/grasp_lift_fabric", "rh56_left").registered is False
    assert _row("agnostic/grasp_sensor", "tesollo_right").registered is True
