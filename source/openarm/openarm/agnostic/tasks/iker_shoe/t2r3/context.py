"""RewardContext — the only input of a generated UNIFIED reward (one policy: pick the shoe up, place it, let go).

Contract (same convention as the t2r and t2r2 forks, no shared code):
  * every tensor is batched (N, ...) on one device; positions are env-local metres, angles rad, speeds m/s, forces N;
  * the field names and comments ARE the prompt's environment description (prompts.py renders this source), in English;
  * frozen — the reward only reads it; the environment passes copies of its buffers.
The grasp half of the field list comes from the stage-1 fork, the placement half from the stage-2 fork; both were
checked against the code on 2026-09-18 (iker_shoe_grasp_env, grasp_stage.stage1_step, place_stage.place_step, layout).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class RewardContext:
    # ---- constants (python numbers, fixed for the whole run) -----------------------------
    table_top_z: float             # height of the table top [m]
    rack_x_min: float              # the rack's footprint on the table: x from rack_x_min to rack_x_max [m]
    rack_x_max: float
    rack_y_min: float              # ... and y from rack_y_min to rack_y_max [m]
    rack_y_max: float
    rack_top_z: float              # height of the rack's top surface [m]; a placed shoe's lowest point should sit here
    episode_steps: int             # the episode ends after this many steps
    control_dt: float              # seconds per step
    # grasp thresholds (the environment's own hold predicate)
    lift_height: float             # a held step needs dz_free of at least this [m]
    hold_radius: float             # ... palm_shoe_dist of at most this [m]
    hold_slip_speed: float         # ... slip_speed below this [m/s]
    thumb_curl_min: float          # ... thumb_curl of at least this [rad]
    latch_steps: int               # latched becomes True once hold_count reaches this
    # placement thresholds
    place_tolerance: float         # keypoint_dist at or below this counts as placed [m]
    release_radius: float          # palm_shoe_dist above this counts as released (hand let go) [m]
    resting_tol: float             # shoe_bottom_z may sit this far from rack_top_z and still count as resting [m]
    still_speed: float             # shoe_lin_vel norm below this counts as still [m/s]
    stable_steps: int              # placed & released & resting & still & home must hold this many steps WITHIN the last window_steps for success
    window_steps: int              # length of that trailing window, in steps
    home_joint_tol: float          # arm_home_err at or below this counts as home (arm back in its rest posture) [rad]
    retract_steps: int             # steps the environment's scripted return of the arm to its rest posture takes
    retract_open_min: float        # grip open fraction (0 holding, 1 open) at or above which the environment takes over the arm

    # ---- hand: left Tesollo DG-5F, finger index 0 thumb, 1 index, 2 middle, 3 ring, 4 pinky ----
    palm_pos: torch.Tensor         # (N,3) palm frame origin
    palm_quat: torch.Tensor        # (N,4) palm orientation quaternion (w,x,y,z)
    palm_normal: torch.Tensor      # (N,3) unit vector pointing out of the palm's grasping side
    link_pos: torch.Tensor         # (N,5,3,3) finger link positions: [:, f, 0] link moved by joint _3, [:, f, 1] link moved by joint _4, [:, f, 2] fingertip
    link_shoe_gap: torch.Tensor    # (N,5,3) distance from each of those links to the nearest point of shoe_surface [m]
    link_shoe_force: torch.Tensor  # (N,5,3) contact force magnitude between each of those links and the shoe only [N] (0 = not touching)
    palm_shoe_force: torch.Tensor  # (N,) contact force magnitude between the palm and the shoe only [N]
    hand_q: torch.Tensor           # (N,20) finger joint angles [rad], order = the hand joint table in the robot description
    hand_qd: torch.Tensor          # (N,20) finger joint velocities [rad/s]
    hand_q_norm: torch.Tensor      # (N,20) joint angles normalised to each joint's commandable range: 0 = lower limit, 1 = upper limit
    hand_target_norm: torch.Tensor  # (N,20) filtered finger joint targets, same normalisation as hand_q_norm
    thumb_curl: torch.Tensor       # (N,) how far the thumb's _3 joint moved from its open angle towards its grip angle [rad]; negative = bent back
    hand_z_min: torch.Tensor       # (N,) height of the lowest finger link [m]

    # ---- arm: 7-DOF ------------------------------------------------------------------------
    arm_q: torch.Tensor            # (N,7) arm joint angles [rad]
    arm_qd: torch.Tensor           # (N,7) arm joint velocities [rad/s]
    home_palm_pos: torch.Tensor    # (N,3) where palm_pos sits when the arm is in its default rest posture; the same point for every env and step
    palm_home_dist: torch.Tensor   # (N,) distance from palm_pos to home_palm_pos [m]
    arm_home_err: torch.Tensor     # (N,) largest absolute difference between arm_q and the rest-posture joint angles [rad]
    retracting: torch.Tensor       # (N,) bool, the environment has taken over the arm and is returning it to the rest posture (policy actions ignored)
    open_frac: torch.Tensor        # (N,) how far the filtered finger targets have moved from the grip pose towards the open hand of a reset, averaged over the joints whose two poses differ, clamped [0,1]; EXACTLY the quantity the takeover compares with retract_open_min

    # ---- shoe: the shoe to pick up and place ------------------------------------------------
    shoe_pos: torch.Tensor         # (N,3) shoe reference point (its body origin)
    shoe_quat: torch.Tensor        # (N,4) shoe orientation quaternion (w,x,y,z)
    shoe_lin_vel: torch.Tensor     # (N,3) linear velocity of the shoe's centre of mass [m/s]
    shoe_ang_vel: torch.Tensor     # (N,3) angular velocity of the shoe [rad/s]
    shoe_start_xy: torch.Tensor    # (N,2) shoe_pos x, y at the start of the episode (where it was spawned on the table)
    shoe_surface: torch.Tensor     # (N,P,3) points on the shoe's outer surface
    shoe_bottom_z: torch.Tensor    # (N,) height of the shoe hull's lowest point [m]
    dz_free: torch.Tensor          # (N,) rise of the lowest surface point above its starting height [m]; 0 while any point is above the rack's footprint
    shoe_shift_xy: torch.Tensor    # (N,) horizontal distance of shoe_pos from shoe_start_xy [m]
    palm_gap: torch.Tensor         # (N,) distance from palm_pos to the nearest point of shoe_surface [m]
    palm_shoe_dist: torch.Tensor   # (N,) distance from palm_pos to shoe_pos [m]
    slip_speed: torch.Tensor       # (N,) speed of the shoe's centre of mass relative to the palm body frame [m/s]; 0 while it moves rigidly with the hand

    # ---- target: the placement goal ---------------------------------------------------------
    target_keypoints: torch.Tensor  # (N,4,3) the 4 target keypoints on the rack, fixed for the whole run (set once by the vision model)
    keypoints: torch.Tensor        # (N,4,3) the shoe's own 4 keypoints in its current pose
    init_keypoints: torch.Tensor   # (N,4,3) the shoe's 4 keypoints at the start of the episode
    keypoint_err: torch.Tensor     # (N,4) per-keypoint distance from keypoints to target_keypoints [m]
    keypoint_dist: torch.Tensor    # (N,) mean over the 4 keypoints of keypoint_err [m]

    # ---- task status: computed by the environment -------------------------------------------
    held: torch.Tensor             # (N,) bool, this step satisfies the grasp hold conditions (lifted clear, thumb closed, not slipping)
    hold_count: torch.Tensor       # (N,) consecutive held steps up to this one (float)
    latched: torch.Tensor          # (N,) bool, True once hold_count reached latch_steps in this episode (stays True)
    carried: torch.Tensor          # (N,) bool, True once the shoe has been lifted clear of the table in this episode (stays True)
    placed: torch.Tensor           # (N,) bool, keypoint_dist <= place_tolerance this step
    released: torch.Tensor         # (N,) bool, palm_shoe_dist > release_radius this step
    resting: torch.Tensor          # (N,) bool, |shoe_bottom_z - rack_top_z| <= resting_tol this step
    still: torch.Tensor            # (N,) bool, shoe speed below still_speed this step
    home: torch.Tensor             # (N,) bool, arm_home_err <= home_joint_tol this step
    stable_count: torch.Tensor     # (N,) how many of the last window_steps steps had placed & released & resting & still & home all true; a single bad step costs 1, it does not reset the count
    success: torch.Tensor          # (N,) bool, True on the step stable_count reaches stable_steps (the episode then ends)
    start_held: torch.Tensor       # (N,) bool, this episode began with the shoe already in the hand (a shortened start) instead of on the table
    episode_progress: torch.Tensor  # (N,) elapsed fraction [0,1] of episode_steps

    # ---- actions ---------------------------------------------------------------------------
    actions: torch.Tensor          # (N,26) policy action of this step, clipped to [-1,1]
    prev_actions: torch.Tensor     # (N,26) policy action of the previous step (zeros right after a reset)

    # ------------------------------------------------------------------
    @property
    def num_envs(self) -> int:
        return int(self.palm_pos.shape[0])

    @property
    def device(self) -> torch.device:
        return self.palm_pos.device


TENSOR_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(RewardContext) if f.type == "torch.Tensor")
SCALAR_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(RewardContext) if f.type in ("float", "int"))


def context_stub_source() -> str:
    """The class body (fields and comments) — the prompt's environment description; the helper properties are cut."""
    import inspect

    src = inspect.getsource(RewardContext)
    return src[: src.find("    # ----------")].rstrip() + "\n"
