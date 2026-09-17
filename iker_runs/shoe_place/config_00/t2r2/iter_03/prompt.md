You are an expert in robotics, reinforcement learning and code generation.

We control one 7-DOF OpenArm robot arm (the left arm) carrying a five-finger Tesollo DG-5F hand, standing at a table. The episode starts with the shoe already held in the hand (its recorded grasp pose); a second shoe stands on a raised rack beside the table and the hand must place its held shoe on the rack next to it. The rack's footprint is x in [0.11, 0.43] m and y in [-0.33, -0.02] m, and its top is at z = 0.325 m. Positions are in metres in each environment's local frame with +z up; the table top is at z = table_top_z.

The action space is a normalized `Box(-1, 1, (7,), float32)`, one action every 0.1 s, 200 steps per episode:
    actions[0:3] = change of the palm position in the robot base frame, 0.02 * a metres per step;
    actions[3:6] = change of the palm orientation in the robot base frame (axis-angle), 0.05 * a radians per step.
The palm pose change goes through damped least-squares inverse kinematics to arm joint position targets (PD, gravity compensated), clamped to the arm's joint limits.
    actions[6] = grip axis: a = -1 keeps the hand at its recorded grip pose (holding the shoe), a = +1 moves it toward the hand profile's open pose. Each step the finger joint targets move 0.4686 of the way from their previous value towards that point — an exponential moving average applied only to the grip axis, filtering its own target over consecutive steps — clamped so a finger cannot bend back past its open pose. The episode starts at a = -1 (holding). `ctx.grip_norm` reports this filtered state, normalised to [-1, 1], where -1 is the grip pose and +1 the open pose.

During training the policy's observations carry uniform noise of +-0.02 (orientations up to 0.2 rad) and its actions uniform noise of +-0.05 before they are executed.

Now I want you to help me write a reward function for reinforcement learning.

Typically, the reward function of a manipulation task consists of these parts (some are optional — include them only if really necessary):
1. the distance between the robot's hand and the target object
2. the difference between the object's current state and its goal state
3. regularization of the robot's action
4. [optional] extra constraints on the target object implied by the task
5. [optional] extra constraints on the robot implied by the task

The reward function receives a single argument `ctx`, an instance of this class (all positions env-local, metres; angles rad; speeds m/s):

```python
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
    place_tolerance: float         # keypoint_dist at or below this counts as placed [m]
    release_radius: float          # palm_shoe_dist above this counts as released (hand let go) [m]
    resting_tol: float             # shoe_bottom_z may sit this far from rack_top_z and still count as resting [m]
    still_speed: float             # shoe_lin_vel norm below this counts as still [m/s]
    stable_steps: int              # placed & released & resting & still & home must hold this many steps WITHIN the last window_steps for success
    window_steps: int              # length of that trailing window, in steps
    home_radius: float             # palm_home_dist at or below this counts as home (hand back at its rest position) [m]

    # ---- palm: left Tesollo DG-5F hand on the 7-DOF arm -------------------------------------
    palm_pos: torch.Tensor         # (N,3) palm frame origin
    palm_quat: torch.Tensor        # (N,4) palm orientation quaternion (w,x,y,z)
    palm_normal: torch.Tensor      # (N,3) unit vector pointing out of the palm's grasping side
    home_palm_pos: torch.Tensor    # (N,3) where palm_pos sits when the arm is in its default rest posture; the same point for every env and every step
    palm_home_dist: torch.Tensor   # (N,) distance from palm_pos to home_palm_pos [m]

    # ---- arm: 7-DOF ------------------------------------------------------------------------
    arm_q: torch.Tensor            # (N,7) arm joint angles [rad]
    arm_qd: torch.Tensor           # (N,7) arm joint velocities [rad/s]

    # ---- grip: the one hand action axis (finger joints are not individually observed) -------
    grip_norm: torch.Tensor        # (N,) EMA-filtered grip state normalised to [-1,1]: -1 the bank's grip pose (holding), +1 the profile's open pose

    # ---- shoe: the shoe being placed --------------------------------------------------------
    shoe_pos: torch.Tensor         # (N,3) shoe reference point (its body origin)
    shoe_quat: torch.Tensor        # (N,4) shoe orientation quaternion (w,x,y,z)
    shoe_lin_vel: torch.Tensor     # (N,3) linear velocity of the shoe's centre of mass [m/s]
    shoe_ang_vel: torch.Tensor     # (N,3) angular velocity of the shoe [rad/s]
    shoe_surface: torch.Tensor     # (N,256,3) points on the shoe's outer surface
    shoe_bottom_z: torch.Tensor    # (N,) height of the shoe hull's lowest point [m]
    palm_gap: torch.Tensor         # (N,) distance from palm_pos to the nearest point of shoe_surface [m]
    palm_shoe_dist: torch.Tensor   # (N,) distance from palm_pos to shoe_pos [m]

    # ---- target: the placement goal ---------------------------------------------------------
    target_keypoints: torch.Tensor  # (N,4,3) the 4 target keypoints on the rack, fixed for the whole run (set once by the vision model)
    keypoints: torch.Tensor        # (N,4,3) the shoe's own 4 keypoints in its current pose
    init_keypoints: torch.Tensor   # (N,4,3) the shoe's 4 keypoints at the start of the episode (already in the grasped pose)
    keypoint_err: torch.Tensor     # (N,4) per-keypoint distance from keypoints to target_keypoints [m]
    keypoint_dist: torch.Tensor    # (N,) mean over the 4 keypoints of keypoint_err [m]

    # ---- placement status: computed by the environment (see place_tolerance etc. above) -----
    placed: torch.Tensor           # (N,) bool, keypoint_dist <= place_tolerance this step
    released: torch.Tensor         # (N,) bool, palm_shoe_dist > release_radius this step
    resting: torch.Tensor          # (N,) bool, |shoe_bottom_z - rack_top_z| <= resting_tol this step
    still: torch.Tensor            # (N,) bool, shoe speed below still_speed this step
    home: torch.Tensor             # (N,) bool, palm_home_dist <= home_radius this step
    stable_count: torch.Tensor     # (N,) how many of the last window_steps steps had placed & released & resting & still & home all true; a single bad step costs 1, it does not reset the count
    success: torch.Tensor          # (N,) bool, True on the step stable_count reaches stable_steps (the episode then ends)
    episode_progress: torch.Tensor  # (N,) elapsed fraction [0,1] of episode_steps

    # ---- actions ---------------------------------------------------------------------------
    actions: torch.Tensor          # (N,7) policy action of this step, clipped to [-1,1]
    prev_actions: torch.Tensor     # (N,7) policy action of the previous step (zeros right after a reset)
```

Additional knowledge:
1. Everything is BATCHED: every tensor field has a leading dimension N (number of parallel environments). The function must return a reward tensor of shape (N,) — never a Python float — and a dict of named component tensors, each of shape (N,). Use only `torch` and `math`; do not import anything else.
2. Write staged rewards with tensor masks, not `if`/`else` on tensors, e.g. `staged = torch.where(ctx.released, r_withdraw, torch.zeros_like(r_withdraw))`.
3. Prefer bounded, smooth shaping such as `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead of raw negative distances, so that no single term dominates.
4. Goal: `ctx.target_keypoints` holds 4 fixed target points on the rack, set once for the whole run by a vision model (not part of the reward). `ctx.keypoints` are the shoe's own 4 keypoints in its current pose, `ctx.init_keypoints` the same at the start of the episode (already in the grasped pose), `ctx.keypoint_err` the per-keypoint distance to the target and `ctx.keypoint_dist` their mean.
5. Placement success is computed by the environment and cannot be redefined by the reward. A step counts toward it when all five of these hold: `ctx.keypoint_dist <= 0.03` m (`ctx.placed`), `ctx.palm_shoe_dist > 0.15` m (`ctx.released`, the hand has let go), `abs(ctx.shoe_bottom_z - ctx.rack_top_z) <= ctx.resting_tol` (`ctx.resting`, the shoe rests on the rack rather than floating or sinking), `ctx.shoe_lin_vel.norm(dim=-1) < 0.05` m/s (`ctx.still`), and `ctx.palm_home_dist <= 0.05` m (`ctx.home`, the hand is back at `ctx.home_palm_pos`, where the palm sits when the arm is in its default rest posture). `ctx.stable_count` is how many of the last 30 steps had all five true — one bad step costs one count and does NOT reset it — and `ctx.success` is True on the step `ctx.stable_count` reaches 20. The policy may therefore set the shoe down, nudge it back into place, let go, and bring the hand home, all within that window. Moving the hand anywhere other than home after letting go (for example lifting the arm high) never counts toward success. You may add a bonus on `ctx.success` or on the individual conditions such as `ctx.placed` or `ctx.released`.
6. The episode ends on a success, when the shoe falls off its support, or after `ctx.episode_steps - 1` steps. Nothing is added to the reward outside your function.
7. Do not keep any state between calls (no globals, no attributes); the function must be pure. `ctx` is read-only — never assign to one of its fields (no `ctx.x = ...`) and never call an in-place op on one (no `.clamp_()`, no `x[:, 0] = ...`).
8. Each component you return is logged separately during training (as `t2r_reward/<name>`, the sum as `t2r_reward/total`) and may be shown back to you after training, so name components meaningfully (e.g. "approach", "align", "release", "withdraw").

I want it to fulfil the following task: Place the shoe on the rack next to the other shoe: align it with the target keypoints, set it down, let go, and withdraw the hand.
1. Please think step by step and tell me what this task means and which stages the robot must go through.
2. Then write a function with exactly this signature:
```python
def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, {"component_name": component_tensor, ...}
```
   where `reward` has shape (N,) and every dict value has shape (N,).
3. Take care of tensor shapes and types; never access a field or method that is not listed in the class definition above.
4. Put the final code in ONE ```python fenced block that starts with `import torch` and `import math` and contains only the function (plus small helper functions if needed). Add short comments explaining the weights.

The previous reward function was:
```python
import torch
import math


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Place the held shoe on the rack: transport -> seat -> let go -> withdraw -> hold quiet.

    Changes vs the previous reward, driven by the measured rollouts:
      * the release/withdraw chain is learned (open_frac 0.97 where placed & resting), so its
        weights are cut hard -- they now only maintain the behaviour;
      * `align` gains a middle length scale at 2*tol and a sharper fine scale, because the
        previous shaping was nearly flat between 0.03 and 0.10 m, which is where the policy lives
        and where the newly tightened 0.03 m tolerance must now be won;
      * `settle` is re-gated and its temperature sharpened 2.5x -- the old one tracked how often
        its gate was open, not whether the shoe was still (it rose while `still` occupancy fell);
      * success terminates the episode, so the terminal bonus is paid PER REMAINING STEP at a rate
        above the dense per-step ceiling; otherwise hovering one condition short of the counter
        strictly beats finishing, which is what the previous logs show happened.
    """

    # ---- shared quantities ---------------------------------------------------------------
    dist = ctx.keypoint_dist                       # (N,) mean keypoint error to the goal pose
    zero = torch.zeros_like(dist)

    # python-side constants, guarded against degenerate values
    tol = max(float(ctx.place_tolerance), 1e-3)        # 0.03 m: the placement tolerance
    rest_tol = max(float(ctx.resting_tol), 1e-3)
    still_sp = max(float(ctx.still_speed), 1e-3)
    rel_r = max(float(ctx.release_radius), 1e-3)
    n_stable = max(int(ctx.stable_steps), 1)
    ep_steps = max(int(ctx.episode_steps), 1)

    # grip axis as an "open fraction": 0.0 = recorded grip pose (holding), 1.0 = open pose
    open_frac = torch.clamp((ctx.grip_norm + 1.0) * 0.5, 0.0, 1.0)
    ep = torch.clamp(ctx.episode_progress, 0.0, 1.0)

    lin_sp = ctx.shoe_lin_vel.norm(dim=-1)
    ang_sp = ctx.shoe_ang_vel.norm(dim=-1)

    seated = ctx.placed & ctx.resting              # shoe at the goal pose, at rack height
    handed = seated & ctx.released                 # ... and the hand is off it
    near_goal = dist <= (3.0 * tol)                # 0.09 m funnel for the height/stillness terms

    # ---- 1. align: three length scales, w=1.5 ---------------------------------------------
    # The previous version had only 0.20 m and tol. Between 3 and 10 cm the 0.20 m term moves
    # just 0.10 total and the Gaussian is numerically dead, so there was almost no gradient in
    # the band the policy actually occupies (measured median 0.046 m). The 2*tol term supplies
    # it; the 0.5*tol term supplies the final centimetre.
    coarse = 1.0 - torch.tanh(dist / 0.20)             # long-range transport pull
    mid = 1.0 - torch.tanh(dist / (2.0 * tol))         # ~0.06 m: the band that must be closed
    fine = torch.exp(-((dist / (0.5 * tol)) ** 2))     # ~0.015 m: margin inside the tolerance
    align = 1.5 * (0.30 * coarse + 0.35 * mid + 0.35 * fine)

    # ---- 2. precision: the WORST keypoint, not the mean, w=0.8 -----------------------------
    # `placed` uses the mean, which can hide one badly rotated corner. Paying the worst corner
    # buys orientation as well as position -- the difference between "next to the target" and
    # "on it".
    worst_err = ctx.keypoint_err.max(dim=-1).values
    precision = 0.8 * torch.exp(-((worst_err / (1.5 * tol)) ** 2))

    # ---- 3. seat: put the shoe's lowest point onto the rack top, w=0.5 ---------------------
    height_err = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    seat = 0.5 * torch.where(near_goal, 1.0 - torch.tanh(height_err / rest_tol), zero)

    # ---- 4. release: maintenance only, w 4.0 -> 0.8 ---------------------------------------
    # open_frac is already 0.97 in this state; this term no longer has to discover anything.
    release = 0.8 * torch.where(seated, open_frac, zero)

    # ---- 5. handsfree: paid only in proportion to the margin inside the tolerance, w=0.6 ---
    # Flat pay for `placed & resting & released` goes dead the instant the shoe is nominally
    # inside 0.03 m. With +-0.02 m observation noise the policy needs margin, so this varies
    # from 0.13 to 1.0 WITHIN the tolerance band and keeps a gradient where the old one had none.
    margin = torch.exp(-((dist / (0.7 * tol)) ** 2))
    handsfree = 0.6 * torch.where(handed, margin, zero)

    # ---- 6. withdraw: maintenance only, saturating at the release radius, w 2.0 -> 0.4 -----
    # Capped at 1.0 so there is no pay for fleeing further than `released` requires.
    withdraw = 0.4 * torch.where(
        seated, torch.clamp(ctx.palm_shoe_dist / rel_r, 0.0, 1.0), zero
    )

    # ---- 7. settle: THE FIX for `still`, w=1.2 --------------------------------------------
    # Old shape paid 0.24 at exactly the failure threshold and was gated on `released`, so its
    # logged mean tracked gate occupancy rather than stillness (it rose 0.126 -> 0.699 while
    # measured `still` fell 0.811 -> 0.614). Temperature is now 0.4*still_speed: ~0.01 at the
    # threshold, 0.24 at 40% of it, 0.76 at 10% of it -- it only pays for real margin. Gated on
    # the shoe resting near the goal, not on the hand being away.
    calm = 1.0 - torch.tanh(lin_sp / (0.4 * still_sp) + ang_sp / 3.0)
    settle = 1.2 * torch.where(ctx.resting & near_goal, calm, zero)

    # ---- 8. gates: smooth superlinear staircase over the four predicates, w=1.0 ------------
    # 2 gates -> 0.25, 3 -> 0.56, 4 -> 1.0, so the last missing condition is worth the most.
    n_gates = (
        ctx.placed.float() + ctx.resting.float() + ctx.released.float() + ctx.still.float()
    )
    gates = 1.0 * (n_gates / 4.0) ** 2

    # ---- 9. stability: the direct success proxy, now the largest dense term, max 3.0 -------
    # Previously 3.0 weight but only 0.138 realised. Linear part gives gradient from the first
    # counted step; the quadratic part pulls the last few counts toward 20/30.
    frac = torch.clamp(ctx.stable_count.float() / n_stable, 0.0, 1.0)
    stability = 2.2 * frac + 0.8 * frac ** 2

    # ---- 10. success: paid PER REMAINING STEP -------------------------------------------
    # Success ends the episode, so a flat bonus makes finishing irrational: at ~8.35/step the
    # old policy forfeited ~670 reward to collect 30. The dense ceiling above is ~9.8/step, so
    # 12.0 per remaining step makes finishing strictly better than hovering, and finishing
    # sooner strictly better than finishing later.
    steps_left = torch.clamp(1.0 - ep, 0.0, 1.0) * float(ep_steps)
    success = torch.where(ctx.success, 40.0 + 12.0 * steps_left, zero)

    # ---- 11. fall: bounded safety guard against dropping the shoe below the TABLE ----------
    # Kept although small (-0.017): it insures the ~9% drop rate, it is not a shaping term.
    fall = torch.clamp(ctx.table_top_z - ctx.shoe_bottom_z, min=0.0)
    fall_penalty = -1.5 * torch.tanh(fall / 0.05)

    # ---- 12. action regularisation: rate term raised, jitter is what breaks `still` --------
    # Still small overall, or the +-0.05 action noise would dominate and kill exploration.
    d_act = ctx.actions - ctx.prev_actions
    rate = torch.mean(d_act ** 2, dim=-1)
    mag = torch.mean(ctx.actions ** 2, dim=-1)
    arm_motion = torch.tanh(torch.mean(ctx.arm_qd ** 2, dim=-1) / 4.0)
    action_reg = -(0.08 * rate + 0.02 * mag + 0.03 * arm_motion)

    components = {
        "align": align,
        "precision": precision,
        "seat": seat,
        "release": release,
        "handsfree": handsfree,
        "withdraw": withdraw,
        "settle": settle,
        "gates": gates,
        "stability": stability,
        "success": success,
        "fall_penalty": fall_penalty,
        "action_reg": action_reg,
    }

    reward = torch.zeros_like(dist)
    for value in components.values():
        reward = reward + value

    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components (t2r_reward/*) as well as task metrics computed by the environment (the placement flags place/placed, place/released, place/resting, place/still, place/home and place/retreated, the keypoint distance iker/keypoint_distance_m, the success rate iker/success_5cm, the drop rate iker/dropped, episode length and total reward) at 10 evenly spaced points during training, plus the min / mean / max encountered:

episode_lengths/iter: [24.9, 140, 185, 163, 159, 168, 155, 167, 159, 146]  min 24.9 · mean 155 · max 199
episode_lengths/step: [28.5, 143, 175, 145, 176, 167, 154, 163, 163, 139]  min 28.5 · mean 155 · max 199
episode_lengths/time: [24.9, 140, 185, 163, 159, 168, 155, 167, 159, 146]  min 24.9 · mean 155 · max 199
iker/dropped: [0.999, 0.125, 0.105, 0.155, 0.0772, 0.102, 0.0908, 0.0919, 0.0616, 0.0959]  min 0 · mean 0.156 · max 1
iker/keypoint_distance_m: [0.664, 0.33, 0.219, 0.197, 0.156, 0.176, 0.157, 0.162, 0.133, 0.151]  min 0.13 · mean 0.331 · max 23.3
iker/success_5cm: [0.00104, 0, 0.0696, 0.163, 0.22, 0.242, 0.32, 0.326, 0.394, 0.42]  min 0 · mean 0.247 · max 0.46
place/placed: [0.000153, 4.6e-05, 0.00167, 0.00932, 0.00877, 0.0176, 0.0219, 0.0213, 0.0269, 0.035]  min 0 · mean 0.0167 · max 0.0476
place/released: [0.647, 0.995, 0.946, 0.881, 0.934, 0.913, 0.898, 0.914, 0.901, 0.893]  min 0.647 · mean 0.912 · max 1
place/resting: [0.0226, 0.0486, 0.469, 0.568, 0.627, 0.636, 0.63, 0.649, 0.645, 0.637]  min 0.00415 · mean 0.53 · max 0.663
place/retreated: [0.791, 0.00412, 0.0329, 0.058, 0.0298, 0.0432, 0.0493, 0.0451, 0.0476, 0.0532]  min 0 · mean 0.0535 · max 0.791
place/still: [0.363, 0.948, 0.854, 0.765, 0.841, 0.819, 0.808, 0.832, 0.814, 0.81]  min 0.363 · mean 0.821 · max 0.982
rewards/iter: [-4.39, 60.3, 160, 329, 426, 490, 558, 491, 574, 701]  min -4.69 · mean 409 · max 876
rewards/step: [-4.33, 56.9, 212, 425, 351, 365, 583, 554, 559, 710]  min -4.69 · mean 410 · max 876
rewards/time: [-4.39, 60.3, 160, 329, 426, 490, 558, 491, 574, 701]  min -4.69 · mean 409 · max 876
t2r_reward/action_reg: [-0.0951, -0.0622, -0.047, -0.0381, -0.0325, -0.0286, -0.0256, -0.0233, -0.0211, -0.02]  min -0.0951 · mean -0.0349 · max -0.0182
t2r_reward/align: [0.0498, 0.0843, 0.207, 0.264, 0.291, 0.31, 0.318, 0.331, 0.338, 0.336]  min 0.0495 · mean 0.27 · max 0.346
t2r_reward/fall_penalty: [-0.0825, -0.0808, -0.0312, -0.0204, -0.0199, -0.0179, -0.0179, -0.018, -0.0186, -0.0186]  min -0.0955 · mean -0.0288 · max -0.0166
t2r_reward/gates: [0.11, 0.254, 0.359, 0.369, 0.407, 0.407, 0.402, 0.415, 0.41, 0.409]  min 0.11 · mean 0.374 · max 0.418
t2r_reward/handsfree: [1.7e-05, 1.2e-05, 0.000296, 0.00179, 0.00184, 0.00361, 0.00448, 0.0048, 0.0053, 0.00747]  min 0 · mean 0.00334 · max 0.00981
t2r_reward/nonfinite_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
t2r_reward/precision: [5e-05, 0.00151, 0.0204, 0.0347, 0.0429, 0.0549, 0.0607, 0.0669, 0.0693, 0.0766]  min 9e-06 · mean 0.0476 · max 0.0882
t2r_reward/release: [4.7e-05, 2.4e-05, 0.00123, 0.00693, 0.00671, 0.0133, 0.0169, 0.0165, 0.0209, 0.0273]  min 0 · mean 0.0129 · max 0.0371
t2r_reward/seat: [0.000262, 0.0101, 0.0917, 0.137, 0.159, 0.171, 0.18, 0.186, 0.187, 0.181]  min 0.000262 · mean 0.141 · max 0.191
t2r_reward/settle: [0.000497, 0.0319, 0.295, 0.437, 0.516, 0.549, 0.57, 0.598, 0.608, 0.585]  min 0.000497 · mean 0.452 · max 0.621
t2r_reward/stability: [0.000196, 0.000116, 0.00157, 0.0115, 0.0111, 0.0214, 0.0256, 0.0254, 0.0286, 0.0411]  min 0 · mean 0.0189 · max 0.0549
t2r_reward/success: [0, 0.0149, 0.104, 0.708, 0.858, 1.38, 1.77, 1.74, 2.01, 2.83]  min 0 · mean 1.28 · max 3.62
t2r_reward/total: [-0.0164, 0.254, 1, 1.92, 2.24, 2.87, 3.32, 3.34, 3.65, 4.47]  min -0.0164 · mean 2.54 · max 5.28
t2r_reward/withdraw: [6.1e-05, 1.8e-05, 0.000668, 0.00372, 0.00351, 0.00704, 0.00872, 0.00848, 0.0107, 0.0139]  min 0 · mean 0.00664 · max 0.0189

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. the success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness, or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: a component that grows while the task metrics do not may be paid for a behaviour that does not serve the task — gate it on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
All numbers below were measured, not estimated. Sources: the training logs of the policy trained by the
previous reward (349 epochs; the run ended there), a video of that policy watched by a person, and a
deterministic probe of the arm's rest posture.

WHAT THE PREVIOUS REWARD ACHIEVED (final logged epoch)

  sustained success 0.235     success within 5 cm 0.410     keypoint distance 0.157 m
  dropped 0.105               placed 0.045 (at the 0.03 m tolerance)
  resting 0.617               still 0.803                   released 0.882

Carrying the shoe to the rack, setting it down and letting go are learned. Do not rebuild them.

WHAT WENT WRONG: THE ARM IS LIFTED AFTER LETTING GO

A person watching the trained policy saw that, once the shoe is set down and released, the arm is raised
upward into a "hands up" pose instead of being brought back.

  - The previous reward had no term that depended on where the hand goes after letting go. Nothing in it
    read the hand's position relative to any rest location, and the context did not offer one.
  - `released` only asks that the palm be more than 0.15 m from the shoe, in any direction.
  - place/retreated (the hand ending within 0.15 m of where it started the episode) fell from 0.79 at the
    first epoch to 0.054 at the last.

WHAT CHANGED IN THE ENVIRONMENT SINCE THAT RUN

A fifth success condition, `ctx.home`, has been added (see the task description above). The hand must be back
at `ctx.home_palm_pos` — where the palm sits when the arm is in its default rest posture — for the steps to count.
The previous policy's final behaviour (arm raised) therefore now earns no success at all.

Measured facts about that rest position:

  home_palm_pos (env-local)                              (0.290, 0.380, 0.418) m, identical in every env
  distance from home_palm_pos to the target shoe centre  0.447 m   (so `released` and `home` can both hold)
  distance from the episode-start palm to home_palm_pos  0.225 m (median)
  driving the palm toward home_palm_pos with a simple proportional action from the episode start:
      within 0.05 m after 9 steps (median); final position error 0.010 m median, 0.032 m 90th percentile

Only the palm POSITION is judged. The arm's joint angles do not return to the rest posture when the palm does
(maximum joint error 1.01 rad median in the same probe), because the action commands the palm pose and the
7-joint arm has a free redundant direction. A reward term on joint angles would pay for something the policy
cannot control.

THE SUCCESS TERM IN THE PREVIOUS REWARD

It paid 40 + 12 per remaining step when success fired, and by the last epoch it was the largest component
(3.24 of a total of 4.89). With the home condition added, success can only fire after the hand has travelled
back about 0.45 m from the shoe, so it arrives later in the episode than before.
