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
    home_joint_tol: float          # arm_home_err at or below this counts as home (arm back in its rest posture) [rad]
    retract_steps: int             # steps the environment's scripted return of the arm to its rest posture takes
    retract_open_min: float        # grip open fraction (0 holding, 1 open) at or above which the environment takes over the arm

    # ---- palm: left Tesollo DG-5F hand on the 7-DOF arm -------------------------------------
    palm_pos: torch.Tensor         # (N,3) palm frame origin
    palm_quat: torch.Tensor        # (N,4) palm orientation quaternion (w,x,y,z)
    palm_normal: torch.Tensor      # (N,3) unit vector pointing out of the palm's grasping side
    home_palm_pos: torch.Tensor    # (N,3) where palm_pos sits when the arm is in its default rest posture; the same point for every env and every step
    palm_home_dist: torch.Tensor   # (N,) distance from palm_pos to home_palm_pos [m]
    arm_home_err: torch.Tensor     # (N,) largest absolute difference between arm_q and the rest-posture joint angles [rad]
    retracting: torch.Tensor       # (N,) bool, the environment has taken over the arm and is returning it to the rest posture (policy actions ignored)

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
    home: torch.Tensor             # (N,) bool, arm_home_err <= home_joint_tol this step
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
5. Placement success is computed by the environment and cannot be redefined by the reward. A step counts toward it when all five of these hold: `ctx.keypoint_dist <= 0.03` m (`ctx.placed`), `ctx.palm_shoe_dist > 0.15` m (`ctx.released`, the hand has let go), `abs(ctx.shoe_bottom_z - ctx.rack_top_z) <= ctx.resting_tol` (`ctx.resting`, the shoe rests on the rack rather than floating or sinking), `ctx.shoe_lin_vel.norm(dim=-1) < 0.05` m/s (`ctx.still`), and `ctx.arm_home_err <= 0.035` rad (`ctx.home`, every arm joint back in the rest posture). `ctx.stable_count` is how many of the last 30 steps had all five true — one bad step costs one count and does NOT reset it — and `ctx.success` is True on the step `ctx.stable_count` reaches 20.
   THE RETURN TO THE REST POSTURE IS NOT THE POLICY'S JOB. The first step the shoe is placed, resting and still while the grip is open at least 0.9 (open fraction `(ctx.grip_norm + 1) / 2`), the environment takes over the arm: it opens the hand fully and moves the arm joints to the rest posture over 20 steps, and from then on the policy's actions have no effect (`ctx.retracting` is True). The policy's task therefore ends at setting the shoe down still in the target pose and opening the hand; success follows if the shoe stays placed and still while the arm withdraws. Any reward paid while `ctx.retracting` is True cannot be influenced by the policy. You may add a bonus on `ctx.success` or on the individual conditions such as `ctx.placed` or `ctx.released`.
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
    """Place the held shoe on the rack: transport -> seat -> let go -> return home -> hold quiet.

    Changes vs the previous reward (only the return stage; learned terms are untouched):
      * home_return shape: a constant-slope linear ramp over 0.9 m dominates, so a 0.02 m step
        toward home pays ~0.037 at 0.6 m (was 0.003); tanh(0.15) + Gaussian(home_radius) finish;
      * home_return gate: `released` dropped (true even while holding), let-go ramp from 30% to 70%
        open, softer placement-quality factor (0.89 at 3 cm instead of 0.64).
    """

    # ---- shared quantities ---------------------------------------------------------------
    dist = ctx.keypoint_dist                       # (N,) mean keypoint error to the goal pose
    zero = torch.zeros_like(dist)

    tol = max(float(ctx.place_tolerance), 1e-3)        # 0.03 m placement tolerance
    rest_tol = max(float(ctx.resting_tol), 1e-3)
    still_sp = max(float(ctx.still_speed), 1e-3)
    rel_r = max(float(ctx.release_radius), 1e-3)
    home_r = max(float(ctx.home_radius), 1e-3)         # 0.05 m
    n_stable = max(int(ctx.stable_steps), 1)
    ep_steps = max(int(ctx.episode_steps), 1)

    # grip axis as an "open fraction": 0.0 = grip pose (holding), 1.0 = open pose
    open_frac = torch.clamp((ctx.grip_norm + 1.0) * 0.5, 0.0, 1.0)
    ep = torch.clamp(ctx.episode_progress, 0.0, 1.0)

    lin_sp = ctx.shoe_lin_vel.norm(dim=-1)
    ang_sp = ctx.shoe_ang_vel.norm(dim=-1)

    seated = ctx.placed & ctx.resting              # shoe at the goal pose, at rack height
    handed = seated & ctx.released                 # ... and the palm is away from it
    near_goal = dist <= (3.0 * tol)                # 0.09 m funnel for height/stillness/home terms

    # ---- 1. align: three length scales, w=1.5 (unchanged) ---------------------------------
    coarse = 1.0 - torch.tanh(dist / 0.20)
    mid = 1.0 - torch.tanh(dist / (2.0 * tol))
    fine = torch.exp(-((dist / (0.5 * tol)) ** 2))
    align = 1.5 * (0.30 * coarse + 0.35 * mid + 0.35 * fine)

    # ---- 2. precision: worst keypoint, w=0.8 (unchanged) ----------------------------------
    worst_err = ctx.keypoint_err.max(dim=-1).values
    precision = 0.8 * torch.exp(-((worst_err / (1.5 * tol)) ** 2))

    # ---- 3. seat: lowest point onto the rack top, w=0.5 (unchanged) -----------------------
    height_err = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    seat = 0.5 * torch.where(near_goal, 1.0 - torch.tanh(height_err / rest_tol), zero)

    # ---- 4. release: maintenance, w=0.8 (unchanged) ---------------------------------------
    release = 0.8 * torch.where(seated, open_frac, zero)

    # ---- 5. handsfree: margin inside tolerance once hand is off, w=0.6 (unchanged) ---------
    margin = torch.exp(-((dist / (0.7 * tol)) ** 2))
    handsfree = 0.6 * torch.where(handed, margin, zero)

    # ---- 6. withdraw: saturating at the release radius, w=0.4 (unchanged) -----------------
    withdraw = 0.4 * torch.where(
        seated, torch.clamp(ctx.palm_shoe_dist / rel_r, 0.0, 1.0), zero
    )

    # ---- 7. settle: real stillness margin near the goal, w=1.2 (unchanged) ----------------
    calm = 1.0 - torch.tanh(lin_sp / (0.4 * still_sp) + ang_sp / 3.0)
    settle = 1.2 * torch.where(ctx.resting & near_goal, calm, zero)

    # ---- 8. home_return: REWORKED, max 3.0 -------------------------------------------------
    # Gate: shoe resting within 9 cm of the goal (hard) x grip let go (ramp 30%..70% open)
    # x soft placement quality (0.89 at 3 cm, 0.73 at 5 cm, 0.37 at 9 cm).
    # `released` is not used: it is true even while holding (palm > 0.15 m from shoe origin).
    # Carrying the shoe home drags it out of the funnel and closes the gate by itself.
    hd = ctx.palm_home_dist
    let_go = torch.clamp((open_frac - 0.3) / 0.4, 0.0, 1.0)
    quality = torch.exp(-((dist / (3.0 * tol)) ** 2))
    home_gate = (ctx.resting & near_goal).float() * let_go * quality
    # Shape: 0.55 linear ramp over 0.9 m (constant slope where the return starts, 0.3..0.6 m),
    # 0.25 tanh at 0.15 m (last stretch), 0.20 Gaussian at home_radius (margin vs +-0.02 noise).
    # Values: 0.6 m -> 0.11, 0.45 m -> 0.22, 0.3 m -> 0.34, 0.1 m -> 0.58, 0 m -> 1.0 (x3.0).
    home_shape = (
        0.55 * torch.clamp(1.0 - hd / 0.9, 0.0, 1.0)
        + 0.25 * (1.0 - torch.tanh(hd / 0.15))
        + 0.20 * torch.exp(-((hd / home_r) ** 2))
    )
    home_return = 3.0 * home_gate * home_shape

    # ---- 9. home_hold: direct dense proxy of the fifth condition, w=1.0 (unchanged) --------
    home_hold = 1.0 * (ctx.placed & ctx.resting & ctx.released & ctx.home).float()

    # ---- 10. gates: smooth superlinear staircase over FIVE predicates, w=1.0 (unchanged) ---
    n_gates = (
        ctx.placed.float() + ctx.resting.float() + ctx.released.float()
        + ctx.still.float() + ctx.home.float()
    )
    gates = 1.0 * (n_gates / 5.0) ** 2

    # ---- 11. stability: direct success proxy, max 3.0 (unchanged) --------------------------
    frac = torch.clamp(ctx.stable_count.float() / n_stable, 0.0, 1.0)
    stability = 2.2 * frac + 0.8 * frac ** 2

    # ---- 12. success: paid PER REMAINING STEP, rate 16 (unchanged) --------------------------
    # Dense ceiling is ~13.8/step (home_return 2.5 -> 3.0); 16 per remaining step keeps
    # finishing strictly better than hovering, and sooner better than later.
    steps_left = torch.clamp(1.0 - ep, 0.0, 1.0) * float(ep_steps)
    success = torch.where(ctx.success, 40.0 + 16.0 * steps_left, zero)

    # ---- 13. fall: bounded guard against dropping the shoe below the table (unchanged) -----
    fall = torch.clamp(ctx.table_top_z - ctx.shoe_bottom_z, min=0.0)
    fall_penalty = -1.5 * torch.tanh(fall / 0.05)

    # ---- 14. action regularisation (unchanged; small so +-0.05 action noise cannot dominate) ---
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
        "home_return": home_return,
        "home_hold": home_hold,
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

episode_lengths/iter: [25, 150, 178, 199, 184, 182, 164, 158, 152, 149]  min 25 · mean 165 · max 199
episode_lengths/step: [40.1, 153, 171, 189, 179, 179, 172, 164, 149, 158]  min 40.1 · mean 165 · max 199
episode_lengths/time: [25, 150, 178, 199, 184, 182, 164, 158, 152, 149]  min 25 · mean 165 · max 199
iker/dropped: [1, 0.193, 0.117, 0.0652, 0.0844, 0.0915, 0.0865, 0.104, 0.0716, 0.101]  min 0 · mean 0.133 · max 1
iker/keypoint_distance_m: [0.626, 0.157, 0.15, 0.0987, 0.116, 0.131, 0.131, 0.147, 0.133, 0.153]  min 0.0174 · mean 0.271 · max 20.3
iker/success_5cm: [0, 0.505, 0.533, 0.731, 0.653, 0.633, 0.602, 0.571, 0.577, 0.562]  min 0 · mean 0.57 · max 1
place/home: [3.1e-05, 0, 0, 0.00133, 0.0273, 0.0633, 0.114, 0.139, 0.17, 0.19]  min 0 · mean 0.08 · max 0.205
place/placed: [0.0702, 0.291, 0.401, 0.53, 0.581, 0.461, 0.384, 0.306, 0.248, 0.223]  min 0.0702 · mean 0.361 · max 0.6
place/released: [0.365, 0.967, 0.798, 0.928, 0.849, 0.866, 0.878, 0.87, 0.869, 0.868]  min 0.365 · mean 0.868 · max 0.996
place/resting: [0.377, 0.771, 0.742, 0.83, 0.838, 0.766, 0.735, 0.709, 0.699, 0.685]  min 0.377 · mean 0.741 · max 0.864
place/retreated: [0.263, 0.0664, 0.205, 0.352, 0.358, 0.265, 0.211, 0.184, 0.177, 0.18]  min 0.00162 · mean 0.212 · max 0.436
place/still: [0.221, 0.923, 0.77, 0.907, 0.845, 0.836, 0.836, 0.824, 0.822, 0.823]  min 0.221 · mean 0.832 · max 0.961
rewards/iter: [-1.44, 508, 575, 956, 852, 871, 886, 924, 946, 1.02e+03]  min -1.44 · mean 811 · max 1.21e+03
rewards/step: [-0.371, 531, 630, 746, 792, 818, 972, 938, 1.08e+03, 940]  min -0.371 · mean 812 · max 1.21e+03
rewards/time: [-1.44, 508, 575, 956, 852, 871, 886, 924, 946, 1.02e+03]  min -1.44 · mean 811 · max 1.21e+03
t2r_reward/action_reg: [-0.0319, -0.0221, -0.021, -0.0173, -0.0176, -0.0168, -0.0158, -0.0155, -0.0151, -0.015]  min -0.0319 · mean -0.0178 · max -0.0148
t2r_reward/align: [0.25, 0.5, 0.57, 0.683, 0.708, 0.597, 0.537, 0.486, 0.459, 0.447]  min 0.249 · mean 0.538 · max 0.732
t2r_reward/fall_penalty: [-0.00887, -0.015, -0.0114, -0.00993, -0.00706, -0.0122, -0.0146, -0.0154, -0.016, -0.0171]  min -0.0195 · mean -0.0134 · max -0.00669
t2r_reward/gates: [0.109, 0.379, 0.362, 0.451, 0.454, 0.423, 0.415, 0.392, 0.383, 0.379]  min 0.109 · mean 0.395 · max 0.479
t2r_reward/handsfree: [0.0133, 0.07, 0.11, 0.167, 0.175, 0.128, 0.0978, 0.0712, 0.0524, 0.0478]  min 0.0133 · mean 0.0957 · max 0.188
t2r_reward/home_hold: [0, 0, 0, 0.00111, 0.0222, 0.0482, 0.0729, 0.0668, 0.0639, 0.0593]  min 0 · mean 0.0371 · max 0.0771
t2r_reward/home_return: [0.155, 0.561, 0.654, 0.858, 0.963, 0.972, 0.988, 0.953, 0.962, 0.95]  min 0.0453 · mean 0.826 · max 1.03
t2r_reward/nonfinite_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
t2r_reward/precision: [0.0598, 0.209, 0.282, 0.366, 0.389, 0.311, 0.268, 0.225, 0.192, 0.187]  min 0.0598 · mean 0.257 · max 0.404
t2r_reward/release: [0.053, 0.232, 0.319, 0.423, 0.463, 0.367, 0.306, 0.244, 0.197, 0.177]  min 0.053 · mean 0.288 · max 0.479
t2r_reward/seat: [0.0989, 0.237, 0.231, 0.261, 0.264, 0.238, 0.23, 0.218, 0.217, 0.209]  min 0.0989 · mean 0.229 · max 0.273
t2r_reward/settle: [0.21, 0.814, 0.756, 0.914, 0.901, 0.811, 0.779, 0.735, 0.73, 0.708]  min 0.21 · mean 0.774 · max 0.951
t2r_reward/stability: [0, 0, 0, 0.00273, 0.0677, 0.155, 0.212, 0.179, 0.159, 0.133]  min 0 · mean 0.102 · max 0.227
t2r_reward/success: [0, 0, 0, 0, 0.0549, 0.417, 1.22, 2.22, 2.71, 3.36]  min 0 · mean 1.27 · max 4.38
t2r_reward/total: [0.936, 3.08, 3.41, 4.31, 4.67, 4.62, 5.24, 5.88, 6.2, 6.72]  min 0.936 · mean 4.92 · max 7.6
t2r_reward/withdraw: [0.0275, 0.116, 0.158, 0.211, 0.23, 0.183, 0.153, 0.122, 0.0986, 0.0886]  min 0.0275 · mean 0.143 · max 0.238

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. the success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness, or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: a component that grows while the task metrics do not may be paid for a behaviour that does not serve the task — gate it on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
All numbers below were measured, not estimated. Sources: the training logs of the policy trained by the previous
reward (661 epochs, stopped on purpose), deterministic probes of that policy (64 environments, noise off), and a
person watching a video of it. Training for this round CONTINUES from that policy's weights.

WHAT THE PREVIOUS REWARD ACHIEVED

Logged at the last epoch: sustained success 0.277 (max 0.367), placed 0.212, resting 0.664, still 0.812, dropped
0.111. A deterministic probe of the epoch-500 policy: final keypoint error median 0.028 m (10th percentile 0.018,
90th percentile 0.100); the shoe ends 0.150 m (median) from the other shoe against a target of 0.136 m.
Carrying the shoe, setting it on the target and opening the hand are learned. Do not rebuild them.

WHY THE ENVIRONMENT CHANGED

The previous rounds judged "home" by the palm POSITION only. The trained policy put its palm at that position with
the arm in a completely different shape: with the palm within 0.05 m of home, the arm joints differed from the rest
posture by 34, 5, 54, 40, 25, 25 and 62 degrees (medians per joint) and the palm orientation by 126 degrees. A
person watching the video saw the arm splayed out in an unnatural pose. The policy cannot fix this: its action
commands the palm pose, and the 7-joint arm has a free redundant direction it does not control.

So the return is no longer learned. As the task description above states, the environment now takes over the arm
the first step the shoe is placed, resting and still with the grip open at least 0.9, opens the hand fully and
moves the arm joints to the rest posture over 20 steps; `ctx.home` now means every arm joint within 0.035 rad of
the rest posture; `ctx.retracting` is True from the takeover on, and the policy's actions have no effect then.

Consequences, all structural facts:
  - The previous reward's `home_return` and `home_hold` terms paid for moving the palm toward home. From now on the
    only way the arm gets home is the takeover; nothing the policy does after the takeover changes any reward.
  - The previous reward read `ctx.home_radius`, which no longer exists (it is now `ctx.home_joint_tol`).
  - The one thing the policy must do to trigger the return is: shoe placed (keypoint_dist <= 0.03 m), resting,
    still, AND grip open fraction (`(ctx.grip_norm + 1) / 2`) >= 0.9, all on the same step.

MEASURED UNDER THE NEW ENVIRONMENT (the previous policy, never trained with the takeover, zero reward, 64 envs)

  success (20 of the last 30 steps, all five conditions)   0.531 of episodes
  arm reached home at least once                            0.531 of episodes (every one of them succeeded)
  final palm distance to home                               0.0006 m median
  shoe placed on at least one step                          0.578 of episodes
  open fraction on steps where placed and resting held      0.985
  final keypoint error                                      0.019 / 0.030 / 0.109 m (10th / 50th / 90th percentile)

A separate probe of the takeover itself (64 envs): the shoe moved at most 0.0009 m while the arm withdrew, the palm
stayed at least 0.08 m from the shoe, and the arm ended within 0.13 degrees of the rest posture. The takeover does
not disturb a correctly placed shoe.

So the remaining failures are episodes where the shoe was never placed within 0.03 m and still with the hand open:
42% of episodes never reached `placed` at all.
