from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm import call_openai_chat, call_ollama_chat

_ANALYZER_DIR = Path(__file__).resolve().parent
_GUIDE_MAX_CHARS = 12000


def _guide_candidates(file_name: str) -> list[Path]:
    return [
        _ANALYZER_DIR / file_name,
        (_ANALYZER_DIR / ".." / "hdgp" / "atuo" / file_name).resolve(),
        (_ANALYZER_DIR / ".." / "SkillBLender_Manipulation" / "atuo" / file_name).resolve(),
    ]


@dataclass
class AnalysisResult:
    issues: list[str]
    observations: list[str]
    llm_summary: str | None
    llm_overrides: list[str]
    applied_overrides: list[str]
    new_rules: dict[str, list[str]] | None = None  # LLM이 제안한 새로운 규칙


def _mean_pair(a: float | None, b: float | None) -> float:
    a = a if a is not None else 0.0
    b = b if b is not None else 0.0
    return (a + b) / 2.0


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _get_scalar(scalars: dict, key: str, field: str = "mean_last_100") -> float | None:
    """Safely get a scalar metric value."""
    entry = scalars.get(key, {})
    if isinstance(entry, dict):
        return _safe_float(entry.get(field))
    return _safe_float(entry)


def load_learned_rules(config_dir: Path) -> dict:
    """learned_rules.json 로드. 없으면 빈 dict 반환."""
    learned_path = config_dir / "learned_rules.json"
    if learned_path.is_file():
        try:
            return json.loads(learned_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"issue_to_overrides": {}}


def save_learned_rules(config_dir: Path, new_rules: dict[str, list[str]]) -> None:
    """새로운 규칙을 learned_rules.json에 추가."""
    learned_path = config_dir / "learned_rules.json"
    existing = load_learned_rules(config_dir)

    for issue_name, overrides in new_rules.items():
        if issue_name and overrides:
            # 기존 규칙이 있으면 병합, 없으면 새로 추가
            if issue_name in existing["issue_to_overrides"]:
                # 중복 제거하며 병합
                existing_set = set(existing["issue_to_overrides"][issue_name])
                for ov in overrides:
                    existing_set.add(ov)
                existing["issue_to_overrides"][issue_name] = list(existing_set)
            else:
                existing["issue_to_overrides"][issue_name] = overrides

    learned_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"[analyzer] Saved new rules to {learned_path}")


def merge_rules(base_rules: dict, learned_rules: dict) -> dict:
    """failure_rules.json과 learned_rules.json을 병합."""
    merged = {"issue_to_overrides": dict(base_rules.get("issue_to_overrides", {}))}

    for issue_name, overrides in learned_rules.get("issue_to_overrides", {}).items():
        if issue_name in merged["issue_to_overrides"]:
            # learned rules가 우선 (더 최신)
            merged["issue_to_overrides"][issue_name] = overrides
        else:
            merged["issue_to_overrides"][issue_name] = overrides

    return merged


def rule_based_issues(payload: dict, thresholds: dict) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    observations: list[str] = []

    task_name = str(payload.get("task", "")).lower()
    train = payload.get("train", {}).get("scalars", {})
    eval_metrics = payload.get("eval", {})

    # ── 기존 aggregate metrics ──
    mean_reward_last = _get_scalar(train, "mean_reward")
    mean_reward_max = _get_scalar(train, "mean_reward", "max")
    entropy_last = _get_scalar(train, "entropy", "last")

    lift_success = _mean_pair(
        _safe_float(eval_metrics.get("lift_success_left")),
        _safe_float(eval_metrics.get("lift_success_right")),
    )
    goal_track_success = _mean_pair(
        _safe_float(eval_metrics.get("goal_track_success_left")),
        _safe_float(eval_metrics.get("goal_track_success_right")),
    )
    goal_dist_mean = _mean_pair(
        _safe_float(eval_metrics.get("goal_dist_min_left_mean")),
        _safe_float(eval_metrics.get("goal_dist_min_right_mean")),
    )

    if mean_reward_last is None:
        issues.append("no_train_metrics")
        observations.append("train.mean_reward.mean_last_100 missing")

    if mean_reward_last is not None:
        min_reward = float(thresholds.get("min_train_reward", 0.0))
        if mean_reward_last < min_reward:
            issues.append("no_learning")
            observations.append(f"mean_reward={mean_reward_last:.3f} < {min_reward}")

    if mean_reward_last is not None and mean_reward_max is not None:
        collapse_ratio = float(thresholds.get("collapse_ratio", 0.7))
        if mean_reward_max > 0.0 and mean_reward_last < mean_reward_max * collapse_ratio:
            issues.append("training_collapse")
            observations.append(
                f"mean_reward={mean_reward_last:.3f} < {collapse_ratio} * max({mean_reward_max:.3f})"
            )

    if entropy_last is not None:
        entropy_min = float(thresholds.get("entropy_min", 0.0))
        if entropy_last < entropy_min:
            issues.append("entropy_collapse")
            observations.append(f"entropy={entropy_last:.4f} < {entropy_min}")

    lift_min = float(thresholds.get("lift_success_min", 0.0))
    if lift_success < lift_min:
        issues.append("low_lift_success")
        observations.append(f"lift_success={lift_success:.3f} < {lift_min}")

    goal_track_min = float(thresholds.get("goal_track_success_min", 0.0))
    if goal_track_success < goal_track_min:
        issues.append("tracking_fail")
        observations.append(f"goal_track_success={goal_track_success:.3f} < {goal_track_min}")

    goal_dist_max = float(thresholds.get("goal_dist_mean_max", 1e9))
    if goal_dist_mean > goal_dist_max:
        issues.append("tracking_dist_high")
        observations.append(f"goal_dist_mean={goal_dist_mean:.4f} > {goal_dist_max}")

    # ── 개별 reward term 기반 진단 (새로 추가) ──
    _diagnose_per_reward(train, issues, observations, thresholds)
    if "5g_pour_right" in task_name:
        _diagnose_pour_right_v2(train, issues, observations, thresholds)

    return issues, observations


def _diagnose_per_reward(scalars: dict, issues: list[str], observations: list[str], thresholds: dict) -> None:
    """개별 reward term 기반 고급 진단."""

    # 1. Hand inactivity: 그리퍼가 전혀 움직이지 않음
    left_hand_closure = _get_scalar(scalars, "reward_left_hand_closure_diag")
    right_hand_closure = _get_scalar(scalars, "reward_right_hand_closure_diag")
    left_hand_norm = _get_scalar(scalars, "reward_left_hand_action_norm_diag")
    right_hand_norm = _get_scalar(scalars, "reward_right_hand_action_norm_diag")

    hand_inactive_threshold = float(thresholds.get("hand_closure_min", 0.05))
    hand_inactive = False
    if left_hand_closure is not None and left_hand_closure < hand_inactive_threshold:
        hand_inactive = True
        observations.append(f"left_hand_closure={left_hand_closure:.4f} < {hand_inactive_threshold} (gripper not closing)")
    if right_hand_closure is not None and right_hand_closure < hand_inactive_threshold:
        hand_inactive = True
        observations.append(f"right_hand_closure={right_hand_closure:.4f} < {hand_inactive_threshold} (gripper not closing)")
    if left_hand_norm is not None and left_hand_norm < 0.01:
        hand_inactive = True
        observations.append(f"left_hand_action_norm={left_hand_norm:.4f} (hand not actuated)")
    if right_hand_norm is not None and right_hand_norm < 0.01:
        hand_inactive = True
        observations.append(f"right_hand_action_norm={right_hand_norm:.4f} (hand not actuated)")
    if hand_inactive:
        issues.append("hand_inactive")

    # 1.5 Gripper premature close: Phase 0에서 그리퍼가 너무 많이 닫혀있음
    left_phase = _get_scalar(scalars, "reward_left_grasp2g_phase")
    right_phase = _get_scalar(scalars, "reward_right_grasp2g_phase")
    hand_premature_threshold = float(thresholds.get("hand_closure_premature_max", 0.5))
    phase_0_threshold = float(thresholds.get("phase_0_stuck_threshold", 0.1))

    gripper_premature = False
    if left_phase is not None and left_phase < phase_0_threshold:
        if left_hand_closure is not None and left_hand_closure > hand_premature_threshold:
            gripper_premature = True
            observations.append(
                f"left_hand_closure={left_hand_closure:.4f} > {hand_premature_threshold} while phase={left_phase:.3f} "
                "(gripper closing prematurely before reaching object)"
            )
    if right_phase is not None and right_phase < phase_0_threshold:
        if right_hand_closure is not None and right_hand_closure > hand_premature_threshold:
            gripper_premature = True
            observations.append(
                f"right_hand_closure={right_hand_closure:.4f} > {hand_premature_threshold} while phase={right_phase:.3f} "
                "(gripper closing prematurely before reaching object)"
            )
    if gripper_premature:
        issues.append("gripper_premature_close")

    # 2. Reaching plateau: EEF 거리가 높은 상태로 고착
    left_dist = _get_scalar(scalars, "reward_left_eef_dist_diag")
    right_dist = _get_scalar(scalars, "reward_right_eef_dist_diag")
    reaching_stuck_threshold = float(thresholds.get("reaching_stuck_dist", 0.12))

    reaching_stuck = False
    if left_dist is not None and left_dist > reaching_stuck_threshold:
        reaching_stuck = True
        observations.append(f"left_eef_dist={left_dist:.4f} > {reaching_stuck_threshold} (not reaching close enough)")
    if right_dist is not None and right_dist > reaching_stuck_threshold:
        reaching_stuck = True
        observations.append(f"right_eef_dist={right_dist:.4f} > {reaching_stuck_threshold} (not reaching close enough)")
    if reaching_stuck:
        issues.append("reaching_stuck")

    # 2.5 EEF stalled: 손이 움직이지 않음 (eef_dist_delta ≈ 0)
    left_delta = _get_scalar(scalars, "reward_left_eef_dist_delta_diag")
    right_delta = _get_scalar(scalars, "reward_right_eef_dist_delta_diag")
    eef_delta_min = float(thresholds.get("eef_dist_delta_min", 0.0001))

    eef_stalled = False
    if left_delta is not None and abs(left_delta) < eef_delta_min and reaching_stuck:
        eef_stalled = True
        observations.append(f"left_eef_dist_delta={left_delta:.6f} ≈ 0 (arm stalled, not approaching)")
    if right_delta is not None and abs(right_delta) < eef_delta_min and reaching_stuck:
        eef_stalled = True
        observations.append(f"right_eef_dist_delta={right_delta:.6f} ≈ 0 (arm stalled, not approaching)")
    if eef_stalled:
        if "eef_stalled" not in issues:
            issues.append("eef_stalled")

    # 3. Phase stuck: phase 값이 낮은 상태로 고착 (대부분 phase 0)
    phase_stuck_threshold = float(thresholds.get("phase_stuck_max", 0.5))

    phase_stuck = False
    if left_phase is not None and left_phase < phase_stuck_threshold:
        phase_stuck = True
        observations.append(f"left_phase={left_phase:.3f} < {phase_stuck_threshold} (stuck in early phase)")
    if right_phase is not None and right_phase < phase_stuck_threshold:
        phase_stuck = True
        observations.append(f"right_phase={right_phase:.3f} < {phase_stuck_threshold} (stuck in early phase)")
    if phase_stuck:
        issues.append("phase_stuck")

    # 3.5 Bimanual phase difference: 양손 페이즈 차이가 큼
    bimanual_diff_max = float(thresholds.get("bimanual_phase_diff_max", 0.3))
    if left_phase is not None and right_phase is not None:
        phase_diff = abs(left_phase - right_phase)
        if phase_diff > bimanual_diff_max:
            issues.append("bimanual_phase_desync")
            observations.append(
                f"phase_diff={phase_diff:.3f} > {bimanual_diff_max} "
                f"(L={left_phase:.3f}, R={right_phase:.3f}, arms out of sync)"
            )

    # 4. Reward conflict: displacement penalty가 reaching을 상쇄
    left_displace = _get_scalar(scalars, "reward_left_object_displacement_penalty")
    right_displace = _get_scalar(scalars, "reward_right_object_displacement_penalty")
    left_reach_fine = _get_scalar(scalars, "reward_left_reaching_object_fine")
    right_reach_fine = _get_scalar(scalars, "reward_right_reaching_object_fine")

    conflict_ratio = float(thresholds.get("reward_conflict_ratio", 0.5))
    reward_conflict = False

    if left_displace is not None and left_reach_fine is not None and left_reach_fine > 0:
        ratio = abs(left_displace) / left_reach_fine
        if ratio > conflict_ratio:
            reward_conflict = True
            observations.append(
                f"left displacement_penalty |{left_displace:.3f}| / reaching_fine {left_reach_fine:.3f} "
                f"= {ratio:.2f} > {conflict_ratio} (penalty fighting reaching)"
            )
    if right_displace is not None and right_reach_fine is not None and right_reach_fine > 0:
        ratio = abs(right_displace) / right_reach_fine
        if ratio > conflict_ratio:
            reward_conflict = True
            observations.append(
                f"right displacement_penalty |{right_displace:.3f}| / reaching_fine {right_reach_fine:.3f} "
                f"= {ratio:.2f} > {conflict_ratio} (penalty fighting reaching)"
            )
    if reward_conflict:
        issues.append("reward_conflict_displacement_vs_reaching")

    # 5. Grasping never triggered: grasping reward = 0
    left_grasp = _get_scalar(scalars, "reward_left_grasping_object")
    right_grasp = _get_scalar(scalars, "reward_right_grasping_object")
    if left_grasp is not None and right_grasp is not None:
        if left_grasp < 0.001 and right_grasp < 0.001:
            issues.append("grasp_never_triggered")
            observations.append(
                f"grasp rewards near zero (L={left_grasp:.4f}, R={right_grasp:.4f}), "
                "likely not reaching phase 1"
            )


def _diagnose_pour_right_v2(
    scalars: dict,
    issues: list[str],
    observations: list[str],
    thresholds: dict,
) -> None:
    """5g_pour_right_v2 전용 진단.

    이 task는 단일 오른손 파지 유지 + target cup 쪽 transport + directional tilt + bead crossing이
    핵심이므로, 기존 grasp2g phase 기반 진단만으로는 실패를 거의 놓친다.
    """

    success_rate = _get_scalar(scalars, "success_rate")
    # rl_games pour logs expose success proxies differently (no eval):
    #   - success_rate (eval)
    #   - success (custom)
    #   - bead_in_target_frame (train TB tag: bead_in_target/frame)
    #   - bead_cross_fraction (train TB tag)
    # We pick the first available metric as "success proxy" for gating loops.
    success_candidates = [
        success_rate,
        _get_scalar(scalars, "success"),
        _get_scalar(scalars, "bead_in_target_frame"),
        _get_scalar(scalars, "bead_cross_fraction"),
    ]
    success_val = next((v for v in success_candidates if v is not None), None)

    bead_cross = _get_scalar(scalars, "bead_cross_fraction")
    bead_in_target = _get_scalar(scalars, "bead_in_target_rate")
    bead_in_source = _get_scalar(scalars, "bead_in_source_rate")
    mouth_xy = _get_scalar(scalars, "mouth_xy_distance")
    mouth_z = _get_scalar(scalars, "mouth_z_clearance")
    tilt_reward = _get_scalar(scalars, "reward_tilt")
    aligned_tilt = _get_scalar(scalars, "reward_aligned_tilt")
    directional_tilt = _get_scalar(scalars, "directional_tilt_cos")
    pour_gate_xy = _get_scalar(scalars, "pour_gate_xy")
    pour_gate = _get_scalar(scalars, "pour_gate")
    transport_progress = _get_scalar(scalars, "reward_transport_progress")
    grasp_maintain = _get_scalar(scalars, "grasp_maintain")
    contact_maintain = _get_scalar(scalars, "contact_maintain")
    finger_curl = _get_scalar(scalars, "finger_curl_min")

    min_success = float(thresholds.get("pour_success_rate_min", 0.05))
    min_bead_cross = float(thresholds.get("pour_bead_cross_fraction_min", 0.02))
    max_mouth_xy = float(thresholds.get("pour_mouth_xy_max", 0.05))
    max_mouth_z = float(thresholds.get("pour_mouth_z_clearance_max", 0.12))
    min_tilt_reward = float(thresholds.get("pour_tilt_reward_min", 0.02))
    min_aligned_tilt = float(thresholds.get("pour_aligned_tilt_min", 0.01))
    min_directional_tilt = float(thresholds.get("pour_directional_tilt_cos_min", 0.70))
    min_pour_gate_xy = float(thresholds.get("pour_gate_xy_min", 0.70))
    min_transport_progress = float(thresholds.get("pour_transport_progress_min", 0.001))
    grasp_lock_success_max = float(thresholds.get("pour_grasp_lock_success_max", 0.01))
    grasp_lock_min = float(thresholds.get("pour_grasp_maintain_lock_min", 0.98))
    contact_lock_min = float(thresholds.get("pour_contact_maintain_lock_min", 0.95))
    finger_lock_min = float(thresholds.get("pour_finger_curl_lock_min", 0.85))

    if success_val is not None and success_val < min_success:
        issues.append("success_too_low")
        metric_name = "success_rate" if success_rate is not None else "success_proxy"
        observations.append(f"{metric_name}={success_val:.4f} < {min_success:.4f}")

    if bead_cross is not None and bead_cross < min_bead_cross:
        issues.append("pour_bead_transfer_fail")
        src = f", bead_in_source_rate={bead_in_source:.4f}" if bead_in_source is not None else ""
        dst = f", bead_in_target_rate={bead_in_target:.4f}" if bead_in_target is not None else ""
        observations.append(f"bead_cross_fraction={bead_cross:.4f} < {min_bead_cross:.4f}{src}{dst}")

    if mouth_xy is not None and mouth_xy > max_mouth_xy:
        issues.append("pour_transport_xy_far")
        gate_text = f", pour_gate_xy={pour_gate_xy:.4f}" if pour_gate_xy is not None else ""
        observations.append(f"mouth_xy_distance={mouth_xy:.4f} > {max_mouth_xy:.4f}{gate_text}")

    if mouth_z is not None and mouth_z > max_mouth_z:
        issues.append("pour_height_too_high")
        observations.append(f"mouth_z_clearance={mouth_z:.4f} > {max_mouth_z:.4f}")

    if tilt_reward is not None and tilt_reward < min_tilt_reward:
        issues.append("pour_tilt_reward_too_weak")
        d = f", directional_tilt_cos={directional_tilt:.4f}" if directional_tilt is not None else ""
        observations.append(f"reward_tilt={tilt_reward:.4f} < {min_tilt_reward:.4f}{d}")

    if aligned_tilt is not None and aligned_tilt < min_aligned_tilt:
        if "pour_tilt_reward_too_weak" not in issues:
            issues.append("pour_tilt_reward_too_weak")
        gate_text = f", pour_gate={pour_gate:.4f}" if pour_gate is not None else ""
        observations.append(f"reward_aligned_tilt={aligned_tilt:.4f} < {min_aligned_tilt:.4f}{gate_text}")

    if directional_tilt is not None and directional_tilt < min_directional_tilt:
        issues.append("pour_wrong_tilt_direction")
        observations.append(f"directional_tilt_cos={directional_tilt:.4f} < {min_directional_tilt:.4f}")

    if pour_gate_xy is not None and pour_gate_xy < min_pour_gate_xy:
        if "pour_transport_xy_far" not in issues:
            issues.append("pour_transport_xy_far")
        observations.append(f"pour_gate_xy={pour_gate_xy:.4f} < {min_pour_gate_xy:.4f}")

    if transport_progress is not None and transport_progress < min_transport_progress:
        issues.append("pour_transport_progress_stalled")
        observations.append(
            f"reward_transport_progress={transport_progress:.6f} < {min_transport_progress:.6f}"
        )

    if (
        (success_rate is None or success_rate <= grasp_lock_success_max)
        and grasp_maintain is not None
        and contact_maintain is not None
        and finger_curl is not None
        and grasp_maintain >= grasp_lock_min
        and contact_maintain >= contact_lock_min
        and finger_curl >= finger_lock_min
    ):
        issues.append("pour_grasp_overconstrained")
        observations.append(
            "grasp/contact/finger-curl are saturated while pour is failing: "
            f"grasp_maintain={grasp_maintain:.4f}, contact_maintain={contact_maintain:.4f}, "
            f"finger_curl_min={finger_curl:.4f}"
        )


def _format_prompt(
    payload: dict,
    issues: list[str],
    observations: list[str],
    allowed_overrides: list[str],
) -> str:
    scalars = payload.get("train", {}).get("scalars", {})
    task_name = str(payload.get("task", "unknown"))
    task_guide, guide_source = _load_task_guide(task_name)

    # ── Reward Terms Table ──
    reward_rows = []
    for key, val in sorted(scalars.items()):
        if key.startswith("reward_") and not key.endswith("_diag"):
            name = key[len("reward_"):]
            if isinstance(val, dict) and val.get("mean_last_100") is not None:
                reward_rows.append(
                    f"| {name} | {val['mean_last_100']:.4f} "
                    f"| {val.get('last', 0):.4f} "
                    f"| {val.get('max', 0):.4f} "
                    f"| {val.get('min', 0):.4f} |"
                )
    reward_table = (
        "| Term | Mean | Last | Max | Min |\n"
        "|------|------|------|-----|-----|\n"
        + "\n".join(reward_rows)
        if reward_rows else "(no reward data)"
    )

    # ── Diagnostic Metrics Table ──
    diag_keys = [
        "reward_left_hand_closure_diag", "reward_right_hand_closure_diag",
        "reward_left_eef_dist_diag", "reward_right_eef_dist_diag",
        "reward_left_eef_dist_xy_diag", "reward_right_eef_dist_xy_diag",
        "reward_left_eef_dist_z_diag", "reward_right_eef_dist_z_diag",
        "reward_left_eef_dist_delta_diag", "reward_right_eef_dist_delta_diag",
        "reward_left_object_height_diag", "reward_right_object_height_diag",
        "reward_left_object_displacement_diag", "reward_right_object_displacement_diag",
        "reward_left_hand_action_norm_diag", "reward_right_hand_action_norm_diag",
        "reward_left_arm_action_norm_diag", "reward_right_arm_action_norm_diag",
    ]
    diag_rows = []
    for key in diag_keys:
        val = scalars.get(key, {})
        if isinstance(val, dict) and val.get("mean_last_100") is not None:
            name = key[len("reward_"):]
            diag_rows.append(f"| {name} | {val['mean_last_100']:.4f} |")
    diag_table = (
        "| Metric | Value |\n|--------|-------|\n" + "\n".join(diag_rows)
        if diag_rows else "(no diagnostic data)"
    )

    # ── Phase Status Table ──
    phase_rows = []
    for side in ("left", "right"):
        phase_val = scalars.get(f"reward_{side}_grasp2g_phase", {})
        if isinstance(phase_val, dict) and phase_val.get("mean_last_100") is not None:
            phase_rows.append(
                f"| {side} | {phase_val['mean_last_100']:.3f} | {phase_val.get('max', 0):.3f} |"
            )
    phase_table = (
        "| Side | Mean | Max |\n|------|------|-----|\n" + "\n".join(phase_rows)
        if phase_rows else "(no phase data)"
    )

    # ── Aggregate Metrics ──
    mean_reward = _get_scalar(scalars, "mean_reward")
    entropy = _get_scalar(scalars, "entropy", "last")
    aggregate = (
        "| Metric | Value |\n|--------|-------|\n"
        f"| mean_reward | {mean_reward:.3f} |\n" if mean_reward is not None else ""
    )
    if entropy is not None:
        aggregate += f"| entropy | {entropy:.4f} |\n"

    observations_block = "\n".join(f"- {o}" for o in observations) if observations else "(none)"

    # ── METRIC INTERPRETATION GUIDE (새로 추가) ──
    metric_guide = """
## Metric Interpretation Guide (CRITICAL - Read Before Analysis)

### Hand/Gripper Metrics
- **hand_closure**: Scale 0.0 to 1.0
  - 0.0 = gripper fully OPEN
  - 1.0 = gripper fully CLOSED
  - PROBLEM: If hand_closure > 0.5 while phase < 0.5, gripper is closing PREMATURELY before reaching object
  - GOOD: hand_closure should be LOW (< 0.3) during phase 0 (reaching), then increase during phase 1 (grasping)

### Distance Metrics
- **eef_dist**: Distance from end-effector to object (meters)
  - Lower is better. ~0.12m means still far from object.
  - Target: < 0.05m for successful grasp approach

- **eef_dist_delta**: Change in distance per step
  - NEGATIVE = approaching object (GOOD)
  - ZERO = stalled, not moving (BAD)
  - POSITIVE = retreating from object (BAD)

- **eef_dist_xy**: Horizontal distance component
- **eef_dist_z**: Vertical distance component

### Phase Metrics
- **phase**: Current task phase (0.0 to 3.0)
  - 0.x = Phase 0 (reaching) - arm approaching object
  - 1.x = Phase 1 (grasping) - gripper closing on object
  - 2.x = Phase 2 (lifting) - lifting object
  - 3.x = Phase 3 (holding/tracking) - maintaining position
  - PROBLEM: If phase < 0.5 throughout training, stuck in reaching phase

### Object Metrics
- **object_height**: Height of object above table
  - Starts at ~0 (on table)
  - Should increase to > 0.1 after successful lift

- **object_displacement**: How much object moved from initial position
  - High during phase 0 = arm pushing object away before grasping (BAD)

### Common Failure Patterns
1. **Gripper Premature Close**: hand_closure > 0.5 while phase < 0.1
   - Cause: Policy learned to close gripper before reaching
   - Fix: Add penalty for closing during phase 0, or widen grasp band params

2. **Arm Stalled**: eef_dist_delta ≈ 0 while eef_dist > 0.1
   - Cause: Policy stuck in local minimum, penalties too strong
   - Fix: Reduce reaching penalties, increase fine reaching reward

3. **Phase Stuck**: phase never increases beyond 0.1-0.2
   - Cause: Phase transition conditions too strict, or not reaching close enough
   - Fix: Relax phase_stability_*_steps, widen reach_distance threshold
"""

    is_grasp2g = "grasp2g" in task_name.lower()
    is_pour_right = "5g_pour_right" in task_name.lower()
    is_left_only = ("left" in task_name.lower()) and ("both" not in task_name.lower()) and not is_grasp2g
    if is_pour_right:
        task_description = (
            "Single-arm right-hand pouring task.\n"
            "Success requires maintaining grasp while moving the source cup mouth near the target cup opening, "
            "then tilting toward the target direction enough to transfer beads.\n"
            "The main failure mode is over-optimizing grasp/contact closure while transport/tilt remain too weak."
        )
        reward_reference = (
            "- grasp_maintain / contact_maintain / force_balance / finger_curl: cup grasp retention and hand closure\n"
            "- reward_transport / reward_transport_progress: source mouth XY alignment to target opening\n"
            "- reward_tilt / reward_tilt_raw / reward_aligned_tilt: directional pouring tilt with proximity gate\n"
            "- pour_accuracy / bead_cross_fraction / success_rate: bead transfer objective\n"
            "- cost_spill / spill_ratio: spill penalty"
        )
        override_examples = (
            "- env.weight_tilt=16.0\n"
            "- env.weight_transport_progress=12.0\n"
            "- env.reward_tilt_distance_scale=0.12\n"
            "- env.weight_finger_curl=1.5\n"
            "- env.pour_gate_xy_far=0.13"
        )
        symmetry_instruction = "Do not add left-hand overrides; this is a right-hand pouring task with a scripted left target cup."
    elif is_left_only:
        task_description = (
            "Single-arm left-hand lift task.\n"
            "Focus on left-side reaching -> pre-grasp shaping -> wrap grasp -> lift.\n"
            "Do NOT enforce left/right symmetric overrides for this task.\n"
            "Prefer tuning left-hand reward weights and reward params (especially wrap/cylinder-related params)."
        )
        reward_reference = (
            "- reaching_object / reaching_object_fine: EE approach shaping\n"
            "- thumb_reaching_pose / pinky_reaching_pose / synergy_reaching_pose: pre-grasp open pose 유지\n"
            "- grasp_contact_persistence / grasp_contact_coverage: contact-based grasp shaping\n"
            "- grasp_strict_success: 4-finger + lift 유지 성공 지표\n"
            "- object_displacement: penalize pushing cup while approaching\n"
            "- lifting_object / object_goal_tracking: post-grasp lift objectives"
        )
        override_examples = (
            "- env.rewards.grasp_contact_coverage.weight=10.0\n"
            "- env.rewards.grasp_contact_persistence.weight=6.0\n"
            "- env.rewards.object_displacement.weight=-5.0\n"
            "- env.rewards.synergy_reaching_pose.weight=2.0\n"
            "- env.grasp_soft_gate_far=0.03"
        )
        symmetry_instruction = "Do not add right-hand overrides unless the task clearly uses both hands."
    else:
        task_description = (
            "Bimanual grasp-style task.\n"
            "Focus on reach -> grasp -> lift/hold progression and left/right coordination.\n"
            "Prefer symmetric left/right reward tuning unless metrics are clearly asymmetric."
        )
        reward_reference = (
            "- reaching_object / reaching_object_fine: approach shaping\n"
            "- object_displacement_penalty: suppress object pushing\n"
            "- grasping_object: closure band reward\n"
            "- lifting_object: lift objective\n"
            "- bimanual_* terms: synchronization and joint success"
        )
        override_examples = (
            "- env.rewards.left_reaching_object.weight=-2.0\n"
            "- env.rewards.right_reaching_object.weight=-2.0\n"
            "- env.rewards.left_grasping_object.params.close_min=0.2\n"
            "- env.rewards.right_grasping_object.params.close_min=0.2\n"
            "- env.phase_stability_reach_steps=5"
        )
        symmetry_instruction = "Apply symmetric left/right overrides when both sides are present."

    return (
        f"You are analyzing an RL training run for task `{task_name}`.\n"
        "Output JSON with keys: analysis, overrides, new_rule (optional).\n"
        "The 'analysis' field should contain your step-by-step reasoning as a string.\n"
        "The 'overrides' field should be a list of 'key=value' strings.\n"
        "The 'new_rule' field (optional) should be an object with 'issue_name' and 'overrides' if you discover a new failure pattern.\n\n"
        f"{metric_guide}\n\n"
        "## Task Description\n"
        f"{task_description}\n\n"
        + (
            f"## Task-Specific Rewards Guide (source: {guide_source})\n"
            "Use this guide as the primary reference for reward semantics, gate logic, and expected trend interpretation.\n\n"
            f"{task_guide}\n\n"
            if task_guide
            else ""
        )
        +
        (
        "## Reward Structure Reference\n"
        f"{reward_reference}\n\n"
        "## Analysis Steps (follow these in order)\n\n"
        "Step 1: Metric Interpretation\n"
        "- Check whether approach metrics improve (distance and/or distance delta)\n"
        "- Check finger closure/wrap terms vs approach terms for phase mismatch\n"
        "- Check whether lift-related terms activate at all\n\n"
        "Step 2: Reward Balance Analysis\n"
        "- List top positive and negative reward terms by magnitude\n"
        "- Identify if any penalty strongly cancels progress (>50% scale)\n"
        "- Identify if shaping terms dominate objective terms (e.g., orientation dominates lift)\n\n"
        "Step 3: Root Cause Identification\n"
        "- Pick the PRIMARY bottleneck from data (not generic advice)\n"
        "- Explain why that bottleneck blocks progression to lift/goal\n\n"
        "Step 4: Propose Overrides\n"
        "- Propose only keys that match allowed prefixes\n"
        "- Prefer reward weight and reward params tuning over unrelated config noise\n"
        f"- {symmetry_instruction}\n\n"
        "Step 5: New Rule Proposal (if applicable)\n"
        "- If this failure pattern is new, propose a `new_rule` with issue_name and overrides\n\n"
        "## Override Format Examples\n"
        f"{override_examples}\n\n"
        "---\n\n"
        "## Raw Training Data\n\n"
        f"### Reward Terms (Episode Reward, mean_last_100)\n{reward_table}\n\n"
        f"### Diagnostic Metrics\n{diag_table}\n\n"
        f"### Phase Status\n{phase_table}\n\n"
        f"### Aggregate\n{aggregate}\n\n"
        f"## Detected Issues (rule-based)\n{json.dumps(issues, indent=2)}\n\n"
        f"## Observations (rule-based)\n{observations_block}\n\n"
        f"## Allowed override keys (prefix match):\n{json.dumps(allowed_overrides)}\n\n"
        "Now analyze the raw training data step-by-step. "
        "Use only evidence from this run. "
        "Output strict JSON with keys: analysis, overrides, new_rule(optional)."
        )
    )


def _parse_llm_json(text: str) -> dict:
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try extracting JSON from markdown code block
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try finding first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {"analysis": text.strip(), "overrides": []}


def _filter_overrides(overrides: list[str], allowed_overrides: list[str]) -> list[str]:
    if not allowed_overrides:
        return overrides
    filtered = []
    for item in overrides:
        if "=" not in item:
            continue
        key = item.split("=", 1)[0].strip()
        if any(key.startswith(prefix) for prefix in allowed_overrides):
            filtered.append(item)
    return filtered


def _normalize_override_types(overrides: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in overrides:
        if "=" not in item:
            normalized.append(item)
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key.endswith(".weight"):
            # Force float literal for weight overrides to satisfy configclass typing.
            try:
                num = float(value)
            except Exception:
                normalized.append(item)
                continue
            if "." not in value and "e" not in value.lower():
                normalized.append(f"{key}={num:.1f}")
            else:
                normalized.append(f"{key}={num}")
        else:
            normalized.append(item)
    return normalized


def _load_task_guide(task_name: str) -> tuple[str, str]:
    """Load task-specific markdown guide text for prompt grounding."""
    task_lower = task_name.lower()
    guide_file = ""
    if "5g_lift_left-v2" in task_lower:
        guide_file = "lift_left_v2_REWARDS_GUIDE.md"
    elif "5g_lift_left-v1" in task_lower:
        guide_file = "lift_left_v1_REWARDS_GUIDE.md"
    elif "5g_pour_right-v3" in task_lower:
        guide_file = "5g_pour_right_v3_REWARDS_GUIDE.md"
    elif "5g_pour_right-v2" in task_lower:
        guide_file = "5g_pour_right_v2_REWARDS_GUIDE.md"
    else:
        return "", ""

    text = ""
    source = f"atuo/{guide_file}"
    for candidate in _guide_candidates(guide_file):
        try:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8").strip()
                if text:
                    source = str(candidate)
                    break
        except Exception:
            continue
    if not text:
        return "", source
    if len(text) > _GUIDE_MAX_CHARS:
        return text[:_GUIDE_MAX_CHARS] + "\n\n...(truncated)", source
    return text, source


def analyze(
    payload: dict,
    thresholds: dict,
    llm_cfg: dict,
    allowed_overrides: list[str],
    rules: dict,
    config_dir: Path | None = None,
) -> AnalysisResult:
    issues, observations = rule_based_issues(payload, thresholds)
    issue_to_overrides = rules.get("issue_to_overrides", {})

    llm_summary = None
    llm_overrides: list[str] = []
    applied_overrides: list[str] = []
    new_rules: dict[str, list[str]] | None = None

    # Collect rule-based overrides
    rule_overrides: list[str] = []
    for issue in issues:
        rule_overrides.extend(issue_to_overrides.get(issue, []))
    rule_overrides = _filter_overrides(rule_overrides, allowed_overrides)
    rule_overrides = _normalize_override_types(rule_overrides)

    if llm_cfg.get("enabled", False):
        prompt = _format_prompt(payload, issues, observations, allowed_overrides)
        provider = str(llm_cfg.get("provider", "openai"))
        if provider == "ollama":
            response = call_ollama_chat(
                prompt=prompt,
                model=str(llm_cfg.get("model", "qwen2.5:14b")),
                temperature=float(llm_cfg.get("temperature", 0.3)),
                api_base=str(llm_cfg.get("api_base", "http://localhost:11434")),
            )
        else:
            response = call_openai_chat(
                prompt=prompt,
                model=str(llm_cfg.get("model", "gpt-4o-mini")),
                temperature=float(llm_cfg.get("temperature", 0.3)),
                max_tokens=int(llm_cfg.get("max_tokens", 2048)),
                api_base=str(llm_cfg.get("api_base", "https://api.openai.com/v1")),
            )
        parsed = _parse_llm_json(response)
        llm_summary = str(parsed.get("analysis", parsed.get("summary", "")))
        raw_overrides = parsed.get("overrides", [])
        llm_overrides = [str(x) for x in raw_overrides] if isinstance(raw_overrides, list) else []
        filtered_llm = _filter_overrides(llm_overrides, allowed_overrides)
        filtered_llm = _normalize_override_types(filtered_llm)

        # LLM priority + rule-based fallback
        if len(filtered_llm) > 0:
            applied_overrides = filtered_llm
        else:
            applied_overrides = rule_overrides

        # Handle new rule proposal from LLM
        new_rule_data = parsed.get("new_rule", None)
        if new_rule_data and isinstance(new_rule_data, dict):
            issue_name = new_rule_data.get("issue_name", "")
            rule_overrides_list = new_rule_data.get("overrides", [])
            if issue_name and rule_overrides_list:
                filtered_new_rule = _filter_overrides(rule_overrides_list, allowed_overrides)
                if filtered_new_rule:
                    new_rules = {issue_name: filtered_new_rule}
                    # Save to learned_rules.json if config_dir provided
                    if config_dir is not None:
                        save_learned_rules(config_dir, new_rules)
                    print(f"[analyzer] LLM proposed new rule: {issue_name} -> {filtered_new_rule}")

        # Consistency log: overlap between LLM and rule-based
        llm_keys = {item.split("=", 1)[0].strip() for item in filtered_llm if "=" in item}
        rule_keys = {item.split("=", 1)[0].strip() for item in rule_overrides if "=" in item}
        overlap = sorted(llm_keys & rule_keys)
        payload["llm_rule_overlap"] = overlap
        payload["llm_override_count"] = len(filtered_llm)
        payload["rule_override_count"] = len(rule_overrides)
        payload["override_source"] = "llm" if len(filtered_llm) > 0 else "rule_fallback"

        payload["analysis_prompt"] = prompt
        payload["analysis_response_raw"] = response
    else:
        applied_overrides = rule_overrides

    return AnalysisResult(
        issues=issues,
        observations=observations,
        llm_summary=llm_summary,
        llm_overrides=llm_overrides,
        applied_overrides=applied_overrides,
        new_rules=new_rules,
    )
