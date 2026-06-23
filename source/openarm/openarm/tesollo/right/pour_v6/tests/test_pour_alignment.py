"""pour_point/corridor reward structure tests.

Legacy pour_alignment_score is preserved as a utility, but current reward uses
target inlet corridor context and disables direct align reward.
"""
from __future__ import annotations

from pathlib import Path

import torch

from openarm.tesollo.right.pour_v4.pour_right_utils import pour_alignment_score, pour_corridor_score


TASK_DIR = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (TASK_DIR / path).read_text()


def test_score_is_one_at_aim_point() -> None:
    # Arrange: pour_point가 target_opening + z_margin 위에 정확히 위치
    z_margin = 0.05
    target_opening = torch.tensor([[0.4, 0.1, 0.39]])
    pour_point = torch.tensor([[0.4, 0.1, 0.39 + z_margin]])

    # Act
    score = pour_alignment_score(pour_point, target_opening, z_margin, scale=8.0)

    # Assert: 정확히 조준 → score ≈ 1
    assert torch.allclose(score, torch.ones(1), atol=1e-5)


def test_xy_miss_reduces_score() -> None:
    z_margin = 0.05
    target_opening = torch.tensor([[0.4, 0.1, 0.39]])
    aimed = torch.tensor([[0.4, 0.1, 0.44]])
    xy_off = torch.tensor([[0.5, 0.1, 0.44]])  # x로 10cm 빗나감

    s_aim = pour_alignment_score(aimed, target_opening, z_margin, scale=8.0)
    s_off = pour_alignment_score(xy_off, target_opening, z_margin, scale=8.0)

    assert (s_off < s_aim).all()


def test_z_miss_reduces_score() -> None:
    z_margin = 0.05
    target_opening = torch.tensor([[0.4, 0.1, 0.39]])
    aimed = torch.tensor([[0.4, 0.1, 0.44]])         # z 정확
    z_high = torch.tensor([[0.4, 0.1, 0.60]])        # 16cm 고공

    s_aim = pour_alignment_score(aimed, target_opening, z_margin, scale=8.0)
    s_high = pour_alignment_score(z_high, target_opening, z_margin, scale=8.0)

    assert (s_high < s_aim).all()


def test_monotonic_decreasing_with_distance() -> None:
    z_margin = 0.0
    target_opening = torch.tensor([[0.0, 0.0, 0.0]])
    near = torch.tensor([[0.02, 0.0, 0.0]])
    far = torch.tensor([[0.20, 0.0, 0.0]])

    s_near = pour_alignment_score(near, target_opening, z_margin, scale=8.0)
    s_far = pour_alignment_score(far, target_opening, z_margin, scale=8.0)

    assert (s_near > s_far).all()
    assert (s_far > 0.0).all()  # 항상 양수 gradient (죽지 않음)


def test_batch_shape() -> None:
    z_margin = 0.05
    target_opening = torch.zeros(4, 3)
    pour_point = torch.randn(4, 3)
    score = pour_alignment_score(pour_point, target_opening, z_margin, scale=8.0)
    assert score.shape == (4,)


def test_corridor_score_is_flat_inside_entry_corridor() -> None:
    target_opening = torch.zeros(4, 3)
    pour_point = torch.tensor(
        [
            [0.000, 0.000, 0.000],
            [0.030, 0.000, 0.050],
            [0.000, -0.055, 0.110],
            [0.020, 0.020, -0.015],
        ]
    )

    score = pour_corridor_score(
        pour_point,
        target_opening,
        radius=0.056,
        z_min=-0.020,
        z_max=0.120,
        scale=20.0,
    )

    assert torch.allclose(score, torch.ones(4), atol=1e-5)


def test_corridor_score_does_not_prefer_center_over_inside_rim_side() -> None:
    target_opening = torch.zeros(2, 3)
    pour_point = torch.tensor(
        [
            [0.000, 0.000, 0.050],
            [0.055, 0.000, 0.050],
        ]
    )

    score = pour_corridor_score(
        pour_point,
        target_opening,
        radius=0.056,
        z_min=-0.020,
        z_max=0.120,
        scale=20.0,
    )

    assert torch.allclose(score[0], score[1], atol=1e-5)


def test_corridor_score_penalizes_only_excess_outside_corridor() -> None:
    target_opening = torch.zeros(3, 3)
    pour_point = torch.tensor(
        [
            [0.056, 0.000, 0.050],
            [0.086, 0.000, 0.050],
            [0.056, 0.000, 0.160],
        ]
    )

    score = pour_corridor_score(
        pour_point,
        target_opening,
        radius=0.056,
        z_min=-0.020,
        z_max=0.120,
        scale=20.0,
    )

    assert torch.allclose(score[0], torch.tensor(1.0), atol=1e-5)
    assert score[1] < score[0]
    assert score[2] < score[0]


def test_align_reward_is_active_additive_dexpour() -> None:
    # [Phase1] DexPour additive: r_align 활성화 + r_pour 곱셈→r_transport(w=0). v5와 동기화.
    env = _read("pour_right_env.py")
    cfg = _read("pour_right_env_cfg.py")

    stage_b = env.split("# Stage B — tilt / bead. Corridor is phase context, not direct align reward.", maxsplit=1)[1].split(
        "# ============================================================\n        # Outcome",
        maxsplit=1,
    )[0]

    assert "pour_corridor_score" in env
    assert "r_align = self.cfg.weight_align * (1.0 + self._directional_tilt_cos_c) / 2.0" in stage_b
    assert "r_stageB = r_align" in stage_b
    assert "torch.exp(-self.cfg.pour_z_scale * (self._mouth_z_clearance - self.cfg.pour_z_target).abs())" in stage_b
    assert "r_precision" not in env
    assert "r_pour_xy" not in stage_b
    assert "r_descend" not in stage_b
    assert "weight_align: float = 20.0" in cfg
    assert "weight_transport:   float = 30.0" in cfg
    assert "pour_corridor_xy_margin: float = 0.015" in cfg
    assert "pour_corridor_z_min: float = -0.02" in cfg
    assert "pour_corridor_z_max: float = 0.12" in cfg
    assert "pour_corridor_scale: float = 20.0" in cfg


def test_approach_reward_is_pre_ready_and_post_ready_corridor_miss_penalty() -> None:
    env = _read("pour_right_env.py")
    cfg = _read("pour_right_env_cfg.py")

    approach_block = env.split("[H13] Stage A — Approach", maxsplit=1)[1].split(
        "# ============================================================\n        # Stage A→B 공간 게이트",
        maxsplit=1,
    )[0]

    assert "weight_dist_to_target: float = 45.0" in cfg
    assert "weight_corridor_escape_after_ready: float = 20.0" in cfg
    assert "corridor_escape_floor_after_ready" not in cfg
    assert "approach_anti_floor" not in cfg
    assert "_approach_pt_w = (" in approach_block
    assert "(1.0 - _tilt_blend) * self._source_rim_center_w" in approach_block
    assert "_tilt_blend * self._source_pour_point_w" in approach_block
    assert "_approach_corridor_score = pour_corridor_score(" in approach_block
    assert "approach_corridor_miss = (1.0 - _approach_corridor_score).clamp(min=0.0)" in env
    assert "r_approach_pre_ready = -self.cfg.weight_dist_to_target * approach_corridor_miss" in env
    assert "corridor_escape = (1.0 - corridor_score).clamp(min=0.0)" in env
    assert "r_corridor_escape = -self.cfg.weight_corridor_escape_after_ready * corridor_escape" in env
    assert "r_approach = (" in env
    assert "(1.0 - latched_ready) * r_approach_pre_ready" in env
    assert "+ latched_ready * r_corridor_escape" in env
    assert "self.cfg.weight_corridor_escape_after_ready" in env
    assert "self._approach_xy_dist - self.cfg.rim_approach_saturate" not in approach_block


def test_tilt_reward_uses_ready_latch_not_live_corridor_context() -> None:
    env = _read("pour_right_env.py")
    cfg = _read("pour_right_env_cfg.py")

    stage_b = env.split("# Stage B — tilt / bead. Corridor is phase context, not direct align reward.", maxsplit=1)[1].split(
        "# [H10] 상시 내회전 유도",
        maxsplit=1,
    )[0]

    assert "weight_tilt: float = 35.0" in cfg
    assert "tilt_aim_floor: float = 0.35" in cfg
    assert "tilt_ready_factor = self.cfg.tilt_aim_floor + (1.0 - self.cfg.tilt_aim_floor) * latched_ready" in stage_b
    assert "r_tilt = self.cfg.weight_tilt * tilt_progress * rot_dir * tilt_ready_factor" in stage_b
    assert "* ready_context" not in stage_b
    assert "* corridor_score" not in stage_b
    assert "aim_soft" not in stage_b


def test_g_ready_gate_uses_corridor_latch_not_center_distance() -> None:
    """Stage B context is corridor based and latches after first ready entry.

    A pure center-distance sigmoid makes deep-tilt wobble erase release/tilt rewards.
    """
    env = _read("pour_right_env.py")
    cfg = _read("pour_right_env_cfg.py")

    assert "self._pour_ready_latched = torch.zeros(" in env
    assert "self._pour_ready_latched |= corridor_score >= self.cfg.ready_latch_threshold" in env
    assert "ready_context = torch.maximum(" in env
    assert "ready_latch_floor: float = 0.50" in cfg
    assert "ready_latch_threshold: float = 0.60" in cfg
    assert "g_ready = torch.sigmoid(" not in env
    assert "(self.cfg.g_ready_center - self._mouth_xy_distance)" not in env
    # r_approach uses xyz corridor miss penalty before ready; g_ready remains latch/context.
    assert "_approach_corridor_score = pour_corridor_score(" in env


def test_source_release_uses_release_context_not_align_gate() -> None:
    env = _read("pour_right_env.py")
    cfg = _read("pour_right_env_cfg.py")

    assert "release_gate_floor_after_ready: float = 0.40" in cfg
    assert "release_context = torch.maximum(" in env
    assert "r_source_release = (" in env
    assert "release_context\n            * aim_gate\n            * self.cfg.weight_source_release" in env
    assert "align_gate\n            * aim_gate\n            * self.cfg.weight_source_release" not in env


def test_bead_in_state_reward_is_disabled_for_release_delta_probe() -> None:
    """누적 bead_in_target_fraction 보상은 1-bead park를 만들 수 있어 probe에서 제거한다."""
    env = _read("pour_right_env.py")
    cfg = _read("pour_right_env_cfg.py")

    assert "weight_bead_in: float = 0.0" in cfg
    assert "weight_source_drain: float = 0.0" in cfg
    assert "r_bead_in = torch.zeros_like(self._bead_in_target_fraction)" in env
    assert "r_drain = torch.zeros_like(source_release_delta)" in env
    assert "r_bead_in = align_gate * self.cfg.weight_bead_in * self._bead_in_target_fraction" not in env


def test_source_release_delta_reward_drives_active_pouring() -> None:
    """소스 잔량 감소분만 reward로 써서 멈춰 있는 상태 보상을 제거한다."""
    env = _read("pour_right_env.py")
    cfg = _read("pour_right_env_cfg.py")

    assert "weight_source_release: float = 100.0" in cfg
    assert "source_release_delta = (-self._bead_in_source_delta).clamp(min=0.0)" in env
    assert "r_source_release = (" in env
    assert "* self.cfg.weight_source_release" in env
    assert "* source_release_delta" in env
    assert "+ r_source_release" in env
    assert '"Reward/source_release":  r_source_release.mean()' in env
    assert '"log/source_release_delta":  source_release_delta.mean()' in env


def test_target_capture_delta_reward_drives_outcome() -> None:
    env = _read("pour_right_env.py")
    cfg = _read("pour_right_env_cfg.py")

    assert "weight_target_capture_delta: float = 200.0" in cfg
    assert "weight_success: float = 0.0" in cfg
    assert "target_capture_delta = self._bead_in_target_delta.clamp(min=0.0)" in env
    assert "r_target_capture = self.cfg.weight_target_capture_delta * target_capture_delta" in env
    assert "+ r_target_capture" in env
    assert '"Reward/target_capture":  r_target_capture.mean()' in env
    assert '"log/target_capture_delta":  target_capture_delta.mean()' in env


def test_intermediate_values_are_cached_per_step_to_preserve_release_delta() -> None:
    """DirectRLEnv calls _get_dones before _get_rewards, so deltas must survive both calls."""
    env = _read("pour_right_env.py")

    compute_block = env.split("def _compute_intermediate_values(self) -> None:", maxsplit=1)[1].split(
        "def _get_observations(self) -> dict:",
        maxsplit=1,
    )[0]

    assert "self._intermediate_values_step = -1" in env
    assert "if self._intermediate_values_step == int(self.common_step_counter):" in compute_block
    assert "return" in compute_block.split("if self._intermediate_values_step == int(self.common_step_counter):", maxsplit=1)[1].split(
        "self._intermediate_values_step = int(self.common_step_counter)",
        maxsplit=1,
    )[0]
    assert "self._intermediate_values_step = int(self.common_step_counter)" in compute_block


def test_tilt_threshold_diagnostics_track_90_plus_degree_probe() -> None:
    env = _read("pour_right_env.py")

    assert '"log/tilt_frac_90"' in env
    assert '"log/tilt_frac_110"' in env
    assert '"log/tilt_frac_120"' in env
    assert '"log/tilt_frac_135"' in env


def test_corridor_diagnostics_are_logged() -> None:
    env = _read("pour_right_env.py")

    assert '"log/corridor_score":        corridor_score.mean()' in env
    assert '"log/approach_corridor_score": _approach_corridor_score.mean()' in env
    assert '"log/approach_corridor_miss": approach_corridor_miss.mean()' in env
    assert '"log/corridor_miss":         corridor_escape.mean()' in env
    assert '"log/approach_pre_ready":    r_approach_pre_ready.mean()' in env
    assert '"log/corridor_escape":       r_corridor_escape.mean()' in env
    assert '"log/ready_latched":         self._pour_ready_latched.float().mean()' in env
    assert '"log/release_context":       release_context.mean()' in env
    assert '"log/tilt_ready_factor":     tilt_ready_factor.mean()' in env
    assert '"log/tilt_latched_phase":    latched_ready.mean()' in env
    assert '"log/align_gate"' not in env
    assert '"log/pour_align_score"' not in env
    assert '"Reward_w0/align":    r_align.mean()' in env


def test_reward_and_joint_logs_are_grouped_by_dashboard_namespace() -> None:
    env = _read("pour_right_env.py")

    assert '"Reward/tilt":     r_tilt.mean()' in env
    assert '"Reward/approach": r_approach.mean()' in env
    assert '"Reward_w0/tilt_pre": torch.zeros((), device=self.device)' in env
    assert '"Reward_w0/bead_in":  r_bead_in.mean()' in env
    assert '"Reward_w0/drain":    (g_ready * r_drain).mean()' in env
    assert '"Reward_w0/success":  (self.cfg.weight_success * r_success).mean()' in env
    assert 'self.extras["log"] = reward_log' not in env
    assert '"reward/tilt"' not in env
    assert '"reward/align"' not in env

    assert '"joint_State/j1": arm_joint_pos[:, 0].mean()' in env
    assert 'f"joint_State/j{_i + 1}_sat": torch.maximum(' in env
    assert '"joint_State/palm_clamp_viol_xy": self._palm_clamp_viol_xy.mean()' in env
    assert '"joint_State/palm_clamp_viol_deep": torch.where(' in env
    assert '"log/j1"' not in env
    assert 'f"log/j{_i + 1}_sat"' not in env
    assert '"log/palm_clamp_viol_xy"' not in env


def test_action_kinematics_logs_are_grouped_separately() -> None:
    env = _read("pour_right_env.py")
    cfg = _read("pour_right_env_cfg.py")

    assert "palm_delta_rot_deg: float = 15.0" in cfg
    assert "current_palm_quat_xyzw = self.robot.data.body_quat_w[:, self.palm_body_index][:, [1, 2, 3, 0]]" in env
    assert "self.pregrasp_palm_pose_buf[:, 3:7],\n                delta_rotvec_world" not in env
    assert "self._raw_palm_action = torch.zeros(self.num_envs, 6, device=self.device)" in env
    assert "self._applied_palm_action = torch.zeros(self.num_envs, 6, device=self.device)" in env
    assert "self._action_tilt_gate = torch.ones(self.num_envs, device=self.device)" in env
    assert "self._cmd_delta_pre_gate = torch.zeros(self.num_envs, 6, device=self.device)" in env
    assert "self._cmd_delta_post_gate = torch.zeros(self.num_envs, 6, device=self.device)" in env
    assert "self._cmd_delta_rotvec_world = torch.zeros(self.num_envs, 3, device=self.device)" in env
    assert "self._palm_target_rot_error_deg = torch.zeros(self.num_envs, device=self.device)" in env
    assert "self._cup_rel_drift_deg = torch.zeros(self.num_envs, device=self.device)" in env
    assert "self._cmd_minus_actual_tilt_deg = torch.zeros(self.num_envs, device=self.device)" in env
    assert "self._prev_tilt_amount_log = torch.zeros(self.num_envs, device=self.device)" in env
    assert "self._raw_palm_action.copy_(palm_action)" in env
    assert "self._applied_palm_action.copy_(palm_action)" in env
    assert "delta_pre_gate = scale(self._ema_palm_action, self.delta_mins, self.delta_maxs)" in env
    assert "self._action_tilt_gate.copy_(tilt_gate)" in env
    assert "self._cmd_delta_pre_gate.copy_(delta_pre_gate)" in env
    assert "self._cmd_delta_post_gate.copy_(delta)" in env
    assert "self._cmd_delta_rotvec_world.copy_(delta_rotvec_world)" in env
    assert '"Action_Kinematics/gate/tilt_action": self._action_tilt_gate.mean()' in env
    assert '"Action_Kinematics/applied_action/tilt_toward": self._applied_palm_action[:, 4].mean()' in env
    assert '"Action_Kinematics/command/rot_norm": self._cmd_delta_rotvec_world.norm(dim=-1).mean()' in env
    assert '"Action_Kinematics/tracking/palm_target_rot_error_deg": self._palm_target_rot_error_deg.mean()' in env
    assert '"Action_Kinematics/tracking/cup_rel_drift_deg": self._cup_rel_drift_deg.mean()' in env
    assert '"Action_Kinematics/tracking/cmd_minus_actual_tilt_deg": self._cmd_minus_actual_tilt_deg.mean()' in env
    assert '"Action_Kinematics/kinematics/tilt_delta": tilt_amount_delta.mean()' in env
    assert '"Action_Kinematics/clamp/palm_active": (self._palm_clamp_viol_xy + self._palm_clamp_viol_z > 1e-4).float().mean()' in env
