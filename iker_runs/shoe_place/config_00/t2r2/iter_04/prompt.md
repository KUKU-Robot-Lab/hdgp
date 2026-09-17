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
    """Place the held shoe on the rack: transport -> seat -> let go -> return home -> hold quiet.

    Changes vs the previous reward (driven by the measured rollouts and the new `home` condition):
      * NEW home_return: after the shoe rests near the goal AND the grip is open, pull the palm back
        to home_palm_pos (previously nothing depended on where the hand went; the arm was lifted);
      * NEW home_hold: flat pay when placed & resting & released & home hold together;
      * gates counts five predicates (home added);
      * success per remaining step raised 12 -> 16 to stay above the raised dense ceiling.
    Transport / seat / release / settle terms are kept as they were: they are learned.
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

    # ---- 8. home_return: NEW, the fix for the lifted arm, max 2.5 --------------------------
    # `released` alone is NOT proof of letting go (it was 0.995 at epoch 1, before any shoe
    # reached the rack: the palm is >0.15 m from the shoe origin even while holding). So the gate
    # also needs the grip opened past half and the shoe resting within 9 cm of the goal. Carrying
    # the shoe home would pull it off the rack and close its own gate.
    # Scaled by placement quality so "drop it roughly and go home" pays little:
    # 0.64 at 3 cm, 0.29 at 5 cm, ~0.02 at 9 cm.
    # Three length scales over the ~0.45 m return: 0.25 m long pull, 0.10 m middle, and a
    # Gaussian at home_radius for margin against +-0.02 m observation noise.
    hd = ctx.palm_home_dist
    let_go = torch.clamp((open_frac - 0.5) / 0.4, 0.0, 1.0)
    quality = torch.exp(-((dist / (1.5 * tol)) ** 2))
    home_gate = (ctx.resting & near_goal & ctx.released).float() * let_go * quality
    home_shape = (
        0.40 * (1.0 - torch.tanh(hd / 0.25))
        + 0.30 * (1.0 - torch.tanh(hd / 0.10))
        + 0.30 * torch.exp(-((hd / home_r) ** 2))
    )
    home_return = 2.5 * home_gate * home_shape

    # ---- 9. home_hold: NEW, direct dense proxy of the fifth condition, w=1.0 ---------------
    home_hold = 1.0 * (ctx.placed & ctx.resting & ctx.released & ctx.home).float()

    # ---- 10. gates: smooth superlinear staircase over FIVE predicates, w=1.0 ---------------
    # 3 -> 0.36, 4 -> 0.64, 5 -> 1.0: the last missing condition (now usually home) is worth most.
    n_gates = (
        ctx.placed.float() + ctx.resting.float() + ctx.released.float()
        + ctx.still.float() + ctx.home.float()
    )
    gates = 1.0 * (n_gates / 5.0) ** 2

    # ---- 11. stability: direct success proxy, max 3.0 (stable_count now includes home) -----
    frac = torch.clamp(ctx.stable_count.float() / n_stable, 0.0, 1.0)
    stability = 2.2 * frac + 0.8 * frac ** 2

    # ---- 12. success: paid PER REMAINING STEP, rate 12 -> 16 --------------------------------
    # Dense ceiling is now ~13.3/step (added home_return 2.5 + home_hold 1.0). 16 per remaining
    # step keeps finishing strictly better than hovering, and sooner better than later.
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

episode_lengths/iter: [24.8, 187, 92.4, 146, 156, 186, 183, 172, 166, 186]  min 24.8 · mean 163 · max 199
episode_lengths/step: [27.6, 188, 180, 140, 156, 179, 172, 174, 187, 177]  min 27.6 · mean 164 · max 199
episode_lengths/time: [24.8, 187, 92.4, 146, 156, 186, 183, 172, 166, 186]  min 24.8 · mean 163 · max 199
iker/dropped: [1, 0.0625, 0.27, 0.262, 0.259, 0.0717, 0.111, 0.14, 0.0876, 0.115]  min 0 · mean 0.214 · max 1
iker/keypoint_distance_m: [0.674, 0.27, 0.321, 0.292, 0.315, 0.148, 0.186, 0.186, 0.148, 0.152]  min 0.115 · mean 0.418 · max 26.7
iker/success_5cm: [0, 0, 0.00374, 0.055, 0.119, 0.265, 0.259, 0.339, 0.469, 0.525]  min 0 · mean 0.237 · max 0.633
place/home: [0.00198, 0, 0, 0, 0, 0, 0.000153, 0, 2.3e-05, 5.3e-05]  min 0 · mean 5.68e-05 · max 0.00198
place/placed: [0, 0, 0.00743, 0.0246, 0.0425, 0.072, 0.11, 0.188, 0.236, 0.329]  min 0 · mean 0.127 · max 0.392
place/released: [0.649, 0.995, 0.995, 0.81, 0.899, 0.952, 0.889, 0.893, 0.916, 0.895]  min 0.649 · mean 0.911 · max 1
place/resting: [0.0204, 0.00794, 0.15, 0.424, 0.569, 0.652, 0.675, 0.7, 0.725, 0.76]  min 0.00581 · mean 0.521 · max 0.793
place/retreated: [0.796, 0.0174, 0.00124, 0.0941, 0.0494, 0.0256, 0.0596, 0.0513, 0.0423, 0.048]  min 0.000214 · mean 0.0636 · max 0.796
place/still: [0.357, 0.949, 0.939, 0.673, 0.774, 0.88, 0.81, 0.813, 0.851, 0.842]  min 0.357 · mean 0.827 · max 0.981
rewards/iter: [-4.67, 10.6, 18, 97.7, 165, 326, 332, 327, 376, 505]  min -5.35 · mean 241 · max 638
rewards/step: [-4.65, 11.4, 33.9, 106, 177, 301, 243, 346, 447, 486]  min -5.35 · mean 242 · max 638
rewards/time: [-4.67, 10.6, 18, 97.7, 165, 326, 332, 327, 376, 505]  min -5.35 · mean 241 · max 638
t2r_reward/action_reg: [-0.095, -0.0682, -0.0531, -0.0476, -0.0436, -0.0391, -0.0367, -0.0337, -0.0311, -0.0289]  min -0.095 · mean -0.0436 · max -0.0261
t2r_reward/align: [0.0481, 0.0671, 0.113, 0.203, 0.275, 0.33, 0.368, 0.418, 0.457, 0.523]  min 0.0481 · mean 0.314 · max 0.56
t2r_reward/fall_penalty: [-0.0879, -0.0923, -0.0656, -0.03, -0.0242, -0.0182, -0.0163, -0.0141, -0.0146, -0.0117]  min -0.0957 · mean -0.032 · max -0.0103
t2r_reward/gates: [0.07, 0.155, 0.183, 0.197, 0.247, 0.29, 0.29, 0.318, 0.343, 0.372]  min 0.07 · mean 0.269 · max 0.396
t2r_reward/handsfree: [0, 0, 0.00135, 0.00456, 0.00866, 0.0149, 0.0233, 0.0424, 0.0551, 0.0836]  min 0 · mean 0.03 · max 0.0997
t2r_reward/home_hold: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
t2r_reward/home_return: [4e-06, 9e-06, 9.2e-05, 0.000585, 0.000866, 0.00236, 0.00383, 0.00334, 0.00279, 0.00345]  min 3e-06 · mean 0.00199 · max 0.0041
t2r_reward/nonfinite_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
t2r_reward/precision: [1.5e-05, 0.00014, 0.00829, 0.0258, 0.0475, 0.072, 0.0979, 0.141, 0.171, 0.226]  min 1.5e-05 · mean 0.0963 · max 0.263
t2r_reward/release: [0, 0, 0.00265, 0.0106, 0.0187, 0.0447, 0.0773, 0.138, 0.18, 0.258]  min 0 · mean 0.0941 · max 0.31
t2r_reward/seat: [0.000315, 0.0013, 0.0288, 0.0839, 0.141, 0.175, 0.19, 0.205, 0.218, 0.235]  min 0.000311 · mean 0.144 · max 0.246
t2r_reward/settle: [0.000439, 0.00333, 0.0929, 0.256, 0.452, 0.586, 0.621, 0.678, 0.734, 0.795]  min 0.000439 · mean 0.477 · max 0.844
t2r_reward/stability: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
t2r_reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
t2r_reward/total: [-0.064, 0.0664, 0.315, 0.714, 1.14, 1.49, 1.66, 1.97, 2.21, 2.59]  min -0.064 · mean 1.4 · max 2.82
t2r_reward/withdraw: [0, 0, 0.00296, 0.00984, 0.017, 0.0288, 0.0441, 0.0749, 0.0943, 0.131]  min 0 · mean 0.0506 · max 0.157

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. the success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness, or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: a component that grows while the task metrics do not may be paid for a behaviour that does not serve the task — gate it on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
All numbers below were measured or computed directly from the previous reward's code, not estimated.
The previous reward was trained for 218 epochs and stopped there on purpose. Training for this round
CONTINUES from that policy's weights, so everything it learned is kept; only the reward changes.

WHAT THE PREVIOUS REWARD ACHIEVED (last logged epoch; max over training in brackets)

  placed 0.378 (0.393)    success within 5 cm 0.564 (0.633)    keypoint distance 0.142 m
  resting 0.793           still 0.866                          dropped 0.093
  release component 0.299 (0.310), still rising

Carrying the shoe, setting it on the target and opening the hand are learned, faster than in any previous round.
Do not rebuild them.

WHAT DID NOT HAPPEN: THE HAND NEVER GOES HOME

  place/home          0.0001 last, 0.002 max over the whole run
  home_hold           0 every epoch
  success             0 every epoch
  home_return         0.0040 last, 0.0041 max — flat from epoch 150 on, while the release component that uses
                      a similar gate grew from 0.03 to 0.30 over the same epochs

WHY: THE RETURN TERM IS NEARLY FLAT WHERE THE HAND ACTUALLY IS

Geometry (measured): home_palm_pos is 0.447 m from the target shoe centre, and `released` requires the palm to be
more than 0.15 m from the shoe. So at the moment the return should begin, palm_home_dist is roughly 0.3 to 0.6 m.

The previous home_return shape was 0.40*(1 - tanh(d/0.25)) + 0.30*(1 - tanh(d/0.10)) + 0.30*exp(-(d/0.05)^2),
multiplied by 2.5 and by its gate. Its value, and how much it grows when the palm moves 0.02 m closer (the
largest move one step allows):

  palm_home_dist 0.60 m   value 0.016   gain for a 0.02 m step 0.003
  palm_home_dist 0.45 m   value 0.053   gain for a 0.02 m step 0.009
  palm_home_dist 0.30 m   value 0.170   gain for a 0.02 m step 0.028
  palm_home_dist 0.10 m   value 0.813   gain for a 0.02 m step 0.188

The gain is 20 to 60 times smaller over the first 0.3 m of the return than over the last 0.1 m — and the policy
never got close enough to feel the large part. Its gate also multiplied by a placement-quality factor and by the
grip being open past half, both below 1 in most states.

Measured reachability (unchanged): driving the palm toward home_palm_pos with a simple proportional action reaches
within 0.05 m in 9 steps (median); only the palm position is judged, not the joint angles.
