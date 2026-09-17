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
    """Place the held shoe on the rack: transport -> align -> seat -> open hand (env retracts arm).
    If the shoe was set down off target, go back and move it onto the target."""

    # ---- shared quantities ---------------------------------------------------------------
    dist = ctx.keypoint_dist                       # (N,) mean keypoint error to the goal pose
    zero = torch.zeros_like(dist)

    tol = max(float(ctx.place_tolerance), 1e-3)        # 0.03 m placement tolerance
    rest_tol = max(float(ctx.resting_tol), 1e-3)
    still_sp = max(float(ctx.still_speed), 1e-3)
    open_min = min(max(float(ctx.retract_open_min), 1e-3), 1.0)  # 0.9 takeover threshold
    n_stable = max(int(ctx.stable_steps), 1)
    ep_steps = max(int(ctx.episode_steps), 1)

    # grip axis as an "open fraction": 0.0 = grip pose (holding), 1.0 = open pose
    open_frac = torch.clamp((ctx.grip_norm + 1.0) * 0.5, 0.0, 1.0)
    open_enough = open_frac >= open_min
    ep = torch.clamp(ctx.episode_progress, 0.0, 1.0)
    active = ~ctx.retracting                       # policy still controls the arm

    lin_sp = ctx.shoe_lin_vel.norm(dim=-1)
    ang_sp = ctx.shoe_ang_vel.norm(dim=-1)

    seated = ctx.placed & ctx.resting              # shoe at the goal pose, at rack height
    ready = seated & ctx.still & open_enough       # exactly the environment's takeover trigger

    # closeness factor: 1 inside tolerance, ~0.29 at 4 cm, ~0 beyond 5 cm
    excess = torch.clamp(dist - tol, min=0.0)
    closeness = torch.exp(-((excess / (0.3 * tol)) ** 2))

    # ---- 1. align: three length scales, w=1.5 (unchanged, learned) -----------------------
    coarse = 1.0 - torch.tanh(dist / 0.20)
    mid = 1.0 - torch.tanh(dist / (2.0 * tol))
    fine = torch.exp(-((dist / (0.5 * tol)) ** 2))
    align = 1.5 * (0.30 * coarse + 0.35 * mid + 0.35 * fine)

    # ---- 2. precision: worst keypoint, w=0.8 (unchanged) ----------------------------------
    worst_err = ctx.keypoint_err.max(dim=-1).values
    precision = 0.8 * torch.exp(-((worst_err / (1.5 * tol)) ** 2))

    # ---- 3. lock: step on `placed` raised 0.7 -> 1.2, + 0.5 narrow Gaussian ----------------
    lock = 1.2 * ctx.placed.float() + 0.5 * torch.exp(-((dist / (0.4 * tol)) ** 2))

    # ---- 4. seat: height onto rack top, w=0.5, now scaled by closeness ---------------------
    # (was paid up to 9 cm off target -> paid the stranded state)
    height_err = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    seat = 0.5 * closeness * (1.0 - torch.tanh(height_err / rest_tol))

    # ---- 5. settle: stillness margin, w=1.2, now only near/at the target -------------------
    # (was the main reward for leaving a shoe motionless 4 cm off target)
    calm = 1.0 - torch.tanh(lin_sp / (0.4 * still_sp) + ang_sp / 3.0)
    settle = 1.2 * torch.where(ctx.resting, closeness * calm, zero)

    # ---- 6. release: w=1.2 (unchanged) -----------------------------------------------------
    release = 1.2 * torch.where(seated, torch.clamp(open_frac / open_min, 0.0, 1.0), zero)

    # ---- 7. ready: takeover condition, max 3.0 (unchanged) --------------------------------
    margin = torch.exp(-((dist / (0.7 * tol)) ** 2))
    ready_r = 3.0 * torch.where(ready, (2.0 + margin) / 3.0, zero)

    # ---- 8. gates: smooth superlinear staircase, w=1.0 (unchanged) -------------------------
    n_gates = (
        ctx.placed.float() + ctx.resting.float() + ctx.still.float()
        + open_enough.float() + ctx.home.float()
    )
    gates = 1.0 * (n_gates / 5.0) ** 2

    # ---- 9. stability: direct success proxy, max 3.0 (unchanged) --------------------------
    frac = torch.clamp(ctx.stable_count.float() / n_stable, 0.0, 1.0)
    stability = 2.2 * frac + 0.8 * frac ** 2

    # ---- 10. off_target: NEW, max -0.6 -----------------------------------------------------
    # Monotone in distance beyond tolerance (no cutoff to escape from), zero once placed,
    # masked during the scripted retract. Makes a stranded shoe cost reward every step.
    not_placed = active & ~ctx.placed
    off_target = -0.6 * torch.where(not_placed, torch.tanh(excess / tol), zero)

    # ---- 11. push: NEW, range [-1, 1] ------------------------------------------------------
    # Signed horizontal shoe velocity toward the target keypoint centroid while the shoe rests
    # on the rack off target. Signed, so back-and-forth nets ~0; pays moving it the right way.
    tgt_c = ctx.target_keypoints.mean(dim=1)       # (N,3)
    cur_c = ctx.keypoints.mean(dim=1)              # (N,3)
    to_tgt = (tgt_c - cur_c)[:, :2]
    to_tgt_dir = to_tgt / (to_tgt.norm(dim=-1, keepdim=True) + 1e-6)
    v_toward = (ctx.shoe_lin_vel[:, :2] * to_tgt_dir).sum(dim=-1)
    stranded = not_placed & ctx.resting & (dist < 0.15)
    push = 1.0 * torch.where(stranded, torch.tanh(v_toward / 0.05), zero)

    # ---- 12. reach: NEW, max 0.5 -----------------------------------------------------------
    # Palm back to the shoe surface when it rests off target (<0.12 m). Smaller than the
    # off_target penalty, so hovering without correcting never pays.
    reach_zone = stranded & (dist < 0.12)
    reach = 0.5 * torch.where(reach_zone, 1.0 - torch.tanh(ctx.palm_gap / 0.05), zero)

    # ---- 13. success: paid PER REMAINING STEP, rate 18 -------------------------------------
    # Dense ceiling ~14.9/step; 18 per remaining step keeps finishing strictly better.
    steps_left = torch.clamp(1.0 - ep, 0.0, 1.0) * float(ep_steps)
    success = torch.where(ctx.success, 40.0 + 18.0 * steps_left, zero)

    # ---- 14. fall: stronger guard now that the policy pushes the shoe (1.5 -> 5.0) ---------
    fall = torch.clamp(ctx.table_top_z - ctx.shoe_bottom_z, min=0.0)
    fall_penalty = -5.0 * torch.tanh(fall / 0.05)

    # ---- 15. action regularisation: small; masked while the env drives the arm -------------
    d_act = ctx.actions - ctx.prev_actions
    rate = torch.mean(d_act ** 2, dim=-1)
    mag = torch.mean(ctx.actions ** 2, dim=-1)
    arm_motion = torch.tanh(torch.mean(ctx.arm_qd ** 2, dim=-1) / 4.0)
    reg = -(0.08 * rate + 0.02 * mag + 0.03 * arm_motion)
    action_reg = torch.where(ctx.retracting, zero, reg)

    components = {
        "align": align,
        "precision": precision,
        "lock": lock,
        "seat": seat,
        "settle": settle,
        "release": release,
        "ready": ready_r,
        "gates": gates,
        "stability": stability,
        "off_target": off_target,
        "push": push,
        "reach": reach,
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

episode_lengths/iter: [25.4, 128, 131, 122, 123, 131, 117, 120, 131, 118]  min 25.4 · mean 121 · max 142
episode_lengths/step: [25.4, 128, 131, 122, 123, 131, 117, 120, 131, 118]  min 25.4 · mean 121 · max 142
episode_lengths/time: [25.4, 128, 131, 122, 123, 131, 117, 120, 131, 118]  min 25.4 · mean 121 · max 142
iker/dropped: [0.52, 0.167, 0.112, 0.0966, 0.105, 0.126, 0.0938, 0.112, 0.0992, 0.12]  min 0.0785 · mean 0.121 · max 0.52
iker/keypoint_distance_m: [0.364, 0.193, 0.153, 0.146, 0.146, 0.159, 0.358, 0.155, 0.148, 0.158]  min 0.0691 · mean 0.236 · max 8.62
iker/success_5cm: [0.48, 0.554, 0.571, 0.571, 0.563, 0.542, 0.583, 0.554, 0.576, 0.554]  min 0.468 · mean 0.579 · max 0.922
place/home: [0.00222, 0.0745, 0.0694, 0.0676, 0.0649, 0.0714, 0.0658, 0.0698, 0.0732, 0.0714]  min 0.00222 · mean 0.0686 · max 0.175
place/placed: [0.141, 0.138, 0.137, 0.142, 0.133, 0.14, 0.131, 0.138, 0.145, 0.145]  min 0.0725 · mean 0.139 · max 0.279
place/released: [0.289, 0.795, 0.819, 0.819, 0.805, 0.812, 0.811, 0.81, 0.804, 0.803]  min 0.289 · mean 0.807 · max 0.934
place/resting: [0.389, 0.604, 0.613, 0.622, 0.617, 0.614, 0.62, 0.61, 0.598, 0.6]  min 0.389 · mean 0.614 · max 0.681
place/retracting: [0.112, 0.13, 0.129, 0.133, 0.124, 0.132, 0.125, 0.132, 0.137, 0.137]  min 0.0716 · mean 0.13 · max 0.271
place/retreated: [0.412, 0.241, 0.233, 0.225, 0.245, 0.247, 0.243, 0.252, 0.251, 0.254]  min 0.209 · mean 0.245 · max 0.412
place/still: [0.222, 0.742, 0.766, 0.763, 0.758, 0.763, 0.76, 0.767, 0.755, 0.754]  min 0.222 · mean 0.755 · max 0.894
rewards/iter: [-22.7, 1.22e+03, 1.23e+03, 1.2e+03, 1.12e+03, 1.02e+03, 1.36e+03, 1.35e+03, 1.12e+03, 1.35e+03]  min -22.7 · mean 1.26e+03 · max 2.73e+03
rewards/step: [-22.7, 1.22e+03, 1.23e+03, 1.2e+03, 1.12e+03, 1.02e+03, 1.36e+03, 1.35e+03, 1.12e+03, 1.35e+03]  min -22.7 · mean 1.26e+03 · max 2.73e+03
rewards/time: [-22.7, 1.22e+03, 1.23e+03, 1.2e+03, 1.12e+03, 1.02e+03, 1.36e+03, 1.35e+03, 1.12e+03, 1.35e+03]  min -22.7 · mean 1.26e+03 · max 2.73e+03
t2r_reward/action_reg: [-0.0181, -0.0126, -0.0122, -0.0122, -0.0125, -0.0122, -0.0123, -0.0121, -0.0122, -0.0121]  min -0.0181 · mean -0.0122 · max -0.011
t2r_reward/align: [0.293, 0.378, 0.385, 0.388, 0.381, 0.384, 0.384, 0.382, 0.381, 0.382]  min 0.293 · mean 0.384 · max 0.462
t2r_reward/fall_penalty: [-0.0282, -0.0629, -0.0657, -0.0643, -0.0655, -0.0658, -0.0656, -0.0673, -0.0678, -0.0687]  min -0.0735 · mean -0.0653 · max -0.0282
t2r_reward/gates: [0.122, 0.265, 0.27, 0.271, 0.266, 0.27, 0.268, 0.268, 0.267, 0.266]  min 0.122 · mean 0.268 · max 0.36
t2r_reward/lock: [0.177, 0.173, 0.172, 0.178, 0.166, 0.175, 0.164, 0.172, 0.182, 0.182]  min 0.091 · mean 0.174 · max 0.352
t2r_reward/nonfinite_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
t2r_reward/off_target: [-0.46, -0.387, -0.382, -0.379, -0.384, -0.381, -0.381, -0.381, -0.383, -0.382]  min -0.46 · mean -0.382 · max -0.334
t2r_reward/precision: [0.0946, 0.131, 0.135, 0.135, 0.131, 0.133, 0.128, 0.131, 0.133, 0.134]  min 0.0946 · mean 0.132 · max 0.191
t2r_reward/push: [0.0246, 0.00685, 0.00574, 0.0067, 0.00659, 0.00781, 0.00852, 0.00746, 0.00821, 0.00772]  min 0.00359 · mean 0.00751 · max 0.0246
t2r_reward/reach: [0.00511, 0.00192, 0.00193, 0.00192, 0.00209, 0.00226, 0.00234, 0.00238, 0.00206, 0.00213]  min 0.00104 · mean 0.00216 · max 0.00511
t2r_reward/ready: [0.266, 0.297, 0.301, 0.306, 0.286, 0.306, 0.287, 0.3, 0.317, 0.315]  min 0.154 · mean 0.3 · max 0.645
t2r_reward/release: [0.165, 0.164, 0.163, 0.169, 0.159, 0.167, 0.156, 0.164, 0.173, 0.173]  min 0.0864 · mean 0.165 · max 0.334
t2r_reward/seat: [0.0617, 0.0824, 0.0868, 0.0873, 0.0853, 0.0867, 0.0851, 0.0867, 0.0888, 0.0874]  min 0.0617 · mean 0.086 · max 0.118
t2r_reward/settle: [0.174, 0.282, 0.301, 0.301, 0.294, 0.301, 0.296, 0.299, 0.306, 0.302]  min 0.174 · mean 0.297 · max 0.415
t2r_reward/stability: [0.000408, 0.106, 0.0966, 0.0901, 0.0899, 0.1, 0.0898, 0.0944, 0.0996, 0.1]  min 0.000408 · mean 0.094 · max 0.237
t2r_reward/success: [0, 9.9, 8.6, 7.64, 8.14, 8.74, 7.73, 8.42, 8.64, 8.98]  min 0 · mean 8.23 · max 18.2
t2r_reward/total: [0.877, 11.3, 10.1, 9.12, 9.55, 10.2, 9.14, 9.87, 10.1, 10.5]  min 0.877 · mean 9.68 · max 20.9

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. the success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness, or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: a component that grows while the task metrics do not may be paid for a behaviour that does not serve the task — gate it on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
All numbers below were measured, not estimated. Sources: training logs, and deterministic probes (64 environments,
noise off, first episode of each environment). The environment did not change.

HISTORY OF THE LAST TWO REWARDS

The reward before the previous one (call it R5) was trained 600 epochs. Its final policy (P5) is the best so far.
The previous reward (shown above as the previous code, R6) added `off_target`, `push` and `reach` and scaled `seat`
and `settle` by closeness. It was trained from P5 and STOPPED at epoch 200 because the deterministic probe got worse:

                                               P5 (epoch 600)    R6 policy (epoch 200)
  success                                       0.52 - 0.58       0.297
  failure: resting on rack, released, off target    0.36          0.39
  failure: still holding the shoe at the time limit 0.09          0.16
  failure: released, shoe not resting on the rack   0.03          0.13
  failure: dropped below the table                  0.00          0.03

Probe repeatability: the same P5 checkpoint probed twice gave 0.578 and 0.516, so differences under about 0.07 are
not meaningful. 0.297 is far outside that.

During the 200 epochs of R6 the training-log averages of `push` stayed at 0.006 - 0.009 and `reach` at 0.002 per
step, flat from the first epoch; `off_target` stayed at -0.38. The penalty was paid, but no behaviour that reduces it
was ever found.

WHY `reach` AND `push` NEVER FIRED (measured state of the off-target failures at the end of the episode)

  distance from the palm to the nearest point of the shoe (`ctx.palm_gap`), 10th / 50th / 90th percentile:
      P5:              0.238 / 0.297 / 0.347 m
      R6 policy:       0.159 / 0.255 / 0.409 m
  keypoint_dist of the same episodes: P5 0.033 / 0.043 / 0.078 m; R6 policy 0.034 / 0.056 / 0.084 m

After letting go of a shoe that is not placed, the policy moves its hand 0.25 - 0.30 m away from the shoe and keeps
it there. At that distance R6's `reach` = 0.5 * (1 - tanh(palm_gap / 0.05)) is below 0.0001, and `push` needs the
shoe to move, which needs the hand on it. So neither term gave any signal in the states the policy actually visits.

WHAT STAYS TRUE FROM BEFORE

  - Episodes in which the shoe is ever placed within 0.03 m almost all succeed; the scripted return works (final palm
    distance to home 0.0005 m median). Carrying, seating, opening the hand and triggering the takeover are learned.
  - The takeover requires `ctx.placed`. While the shoe is not placed the policy keeps control of the arm and grip for
    the rest of the episode.

TRAINING FOR THIS ROUND STARTS FROM P5's WEIGHTS (not from the R6 policy).

WHAT THE PERSON RUNNING THIS WANTS

After the shoe is set down, if it is not on the target, the policy should touch the shoe and move it onto the target
(adjust only the shoe it placed). Reaching the target with the first set-down is equally fine.
