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
    """Place the held shoe on the rack: transport -> align -> seat -> open hand (env retracts arm)."""

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

    lin_sp = ctx.shoe_lin_vel.norm(dim=-1)
    ang_sp = ctx.shoe_ang_vel.norm(dim=-1)

    seated = ctx.placed & ctx.resting              # shoe at the goal pose, at rack height
    ready = seated & ctx.still & open_enough       # exactly the environment's takeover trigger
    near_goal = dist <= (3.0 * tol)                # 0.09 m funnel for height/stillness terms

    # ---- 1. align: three length scales, w=1.5 (unchanged, learned) -----------------------
    coarse = 1.0 - torch.tanh(dist / 0.20)
    mid = 1.0 - torch.tanh(dist / (2.0 * tol))
    fine = torch.exp(-((dist / (0.5 * tol)) ** 2))
    align = 1.5 * (0.30 * coarse + 0.35 * mid + 0.35 * fine)

    # ---- 2. precision: worst keypoint, w=0.8 (unchanged) ----------------------------------
    worst_err = ctx.keypoint_err.max(dim=-1).values
    precision = 0.8 * torch.exp(-((worst_err / (1.5 * tol)) ** 2))

    # ---- 3. lock: NEW, max 1.2 -------------------------------------------------------------
    # 0.7 step on `placed` (value jump at the 3 cm boundary where the median policy stops)
    # + 0.5 narrow Gaussian (1.2 cm scale) for gradient well inside the tolerance.
    lock = 0.7 * ctx.placed.float() + 0.5 * torch.exp(-((dist / (0.4 * tol)) ** 2))

    # ---- 4. seat: lowest point onto the rack top, w=0.5 (unchanged) -----------------------
    height_err = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    seat = 0.5 * torch.where(near_goal, 1.0 - torch.tanh(height_err / rest_tol), zero)

    # ---- 5. settle: stillness margin near the goal, w=1.2 (unchanged) ---------------------
    calm = 1.0 - torch.tanh(lin_sp / (0.4 * still_sp) + ang_sp / 3.0)
    settle = 1.2 * torch.where(ctx.resting & near_goal, calm, zero)

    # ---- 6. release: RESHAPED, w=1.2 -------------------------------------------------------
    # Only when the shoe is seated on target; saturates at the takeover threshold (0.9 open),
    # so opening off-target earns nothing and the last bit to 0.9 is fully paid.
    release = 1.2 * torch.where(seated, torch.clamp(open_frac / open_min, 0.0, 1.0), zero)

    # ---- 7. ready: NEW, max 3.0 ------------------------------------------------------------
    # Paid on the takeover condition itself (placed & resting & still & open >= 0.9).
    # Cannot be hovered before the trigger (meeting it starts the takeover); afterwards it keeps
    # the value of triggering continuous while the shoe stays placed. The margin part (1.0)
    # favours shoes set well inside 3 cm, which stay placed while the arm withdraws.
    margin = torch.exp(-((dist / (0.7 * tol)) ** 2))
    ready_r = 3.0 * torch.where(ready, (2.0 + margin) / 3.0, zero)

    # ---- 8. gates: smooth superlinear staircase, w=1.0 -------------------------------------
    # `released` replaced by `open_enough` (released is true even while holding);
    # `home` stays since it is part of success and only reachable via the takeover.
    n_gates = (
        ctx.placed.float() + ctx.resting.float() + ctx.still.float()
        + open_enough.float() + ctx.home.float()
    )
    gates = 1.0 * (n_gates / 5.0) ** 2

    # ---- 9. stability: direct success proxy, max 3.0 (unchanged) --------------------------
    frac = torch.clamp(ctx.stable_count.float() / n_stable, 0.0, 1.0)
    stability = 2.2 * frac + 0.8 * frac ** 2

    # ---- 10. success: paid PER REMAINING STEP, rate 16 (unchanged) --------------------------
    # Dense ceiling is now ~13.4/step (1.5+0.8+1.2+0.5+1.2+1.2+3.0+1.0+3.0); 16 per remaining
    # step keeps finishing strictly better than hovering, and sooner better than later.
    steps_left = torch.clamp(1.0 - ep, 0.0, 1.0) * float(ep_steps)
    success = torch.where(ctx.success, 40.0 + 16.0 * steps_left, zero)

    # ---- 11. fall: bounded guard against dropping the shoe below the table (unchanged) -----
    fall = torch.clamp(ctx.table_top_z - ctx.shoe_bottom_z, min=0.0)
    fall_penalty = -1.5 * torch.tanh(fall / 0.05)

    # ---- 12. action regularisation: small; masked while the env drives the arm -------------
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

episode_lengths/iter: [25.2, 136, 128, 116, 121, 129, 120, 116, 127, 126]  min 25.2 · mean 123 · max 150
episode_lengths/step: [25.2, 136, 128, 116, 121, 129, 120, 116, 127, 126]  min 25.2 · mean 123 · max 150
episode_lengths/time: [25.2, 136, 128, 116, 121, 129, 120, 116, 127, 126]  min 25.2 · mean 123 · max 150
iker/dropped: [0.52, 0.126, 0.11, 0.118, 0.113, 0.117, 0.132, 0.113, 0.121, 0.106]  min 0.0711 · mean 0.114 · max 0.52
iker/keypoint_distance_m: [0.564, 0.163, 0.151, 0.158, 0.161, 0.159, 1.71, 0.153, 0.178, 0.142]  min 0.105 · mean 0.244 · max 9.3
iker/success_5cm: [0.48, 0.543, 0.556, 0.554, 0.549, 0.579, 0.539, 0.583, 0.573, 0.595]  min 0.451 · mean 0.57 · max 0.923
place/home: [0.00268, 0.055, 0.0587, 0.0632, 0.0692, 0.0715, 0.0681, 0.0731, 0.0716, 0.0684]  min 0.00268 · mean 0.0665 · max 0.154
place/placed: [0.128, 0.112, 0.12, 0.127, 0.142, 0.14, 0.138, 0.147, 0.145, 0.144]  min 0.0536 · mean 0.134 · max 0.236
place/released: [0.328, 0.836, 0.836, 0.829, 0.816, 0.809, 0.809, 0.81, 0.816, 0.807]  min 0.328 · mean 0.822 · max 0.954
place/resting: [0.42, 0.629, 0.627, 0.616, 0.614, 0.619, 0.608, 0.632, 0.61, 0.624]  min 0.42 · mean 0.621 · max 0.71
place/retracting: [0.102, 0.104, 0.112, 0.12, 0.133, 0.133, 0.13, 0.14, 0.135, 0.134]  min 0.0527 · mean 0.126 · max 0.23
place/retreated: [0.383, 0.207, 0.201, 0.242, 0.243, 0.264, 0.255, 0.272, 0.243, 0.254]  min 0.141 · mean 0.238 · max 0.383
place/still: [0.239, 0.78, 0.78, 0.772, 0.764, 0.757, 0.755, 0.763, 0.765, 0.757]  min 0.239 · mean 0.769 · max 0.918
rewards/iter: [-1.65, 1.02e+03, 1.11e+03, 1.19e+03, 1.33e+03, 1.15e+03, 1.16e+03, 1.31e+03, 1.13e+03, 1.17e+03]  min -1.65 · mean 1.19e+03 · max 2.49e+03
rewards/step: [-1.65, 1.02e+03, 1.11e+03, 1.19e+03, 1.33e+03, 1.15e+03, 1.16e+03, 1.31e+03, 1.13e+03, 1.17e+03]  min -1.65 · mean 1.19e+03 · max 2.49e+03
rewards/time: [-1.65, 1.02e+03, 1.11e+03, 1.19e+03, 1.33e+03, 1.15e+03, 1.16e+03, 1.31e+03, 1.13e+03, 1.17e+03]  min -1.65 · mean 1.19e+03 · max 2.49e+03
t2r_reward/action_reg: [-0.0181, -0.0129, -0.0127, -0.0125, -0.0123, -0.0122, -0.0123, -0.0121, -0.0122, -0.0121]  min -0.0181 · mean -0.0124 · max -0.0113
t2r_reward/align: [0.303, 0.378, 0.387, 0.381, 0.385, 0.387, 0.386, 0.388, 0.389, 0.385]  min 0.303 · mean 0.386 · max 0.455
t2r_reward/fall_penalty: [-0.00833, -0.0196, -0.0194, -0.0209, -0.0205, -0.0184, -0.0208, -0.0184, -0.0203, -0.0188]  min -0.0226 · mean -0.0197 · max -0.00833
t2r_reward/gates: [0.126, 0.263, 0.267, 0.266, 0.27, 0.271, 0.265, 0.277, 0.271, 0.272]  min 0.126 · mean 0.27 · max 0.353
t2r_reward/lock: [0.0966, 0.0844, 0.0899, 0.0971, 0.0995, 0.105, 0.112, 0.109, 0.112, 0.109]  min 0.0397 · mean 0.101 · max 0.177
t2r_reward/nonfinite_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
t2r_reward/precision: [0.0938, 0.129, 0.132, 0.134, 0.138, 0.14, 0.143, 0.14, 0.141, 0.139]  min 0.0938 · mean 0.137 · max 0.18
t2r_reward/ready: [0.241, 0.237, 0.255, 0.276, 0.307, 0.302, 0.299, 0.324, 0.317, 0.309]  min 0.111 · mean 0.29 · max 0.537
t2r_reward/release: [0.15, 0.133, 0.142, 0.151, 0.169, 0.167, 0.165, 0.175, 0.173, 0.171]  min 0.0639 · mean 0.16 · max 0.283
t2r_reward/seat: [0.117, 0.19, 0.191, 0.188, 0.187, 0.187, 0.183, 0.191, 0.186, 0.191]  min 0.117 · mean 0.189 · max 0.213
t2r_reward/settle: [0.262, 0.626, 0.629, 0.617, 0.609, 0.609, 0.596, 0.622, 0.603, 0.617]  min 0.262 · mean 0.617 · max 0.716
t2r_reward/stability: [0.000486, 0.0761, 0.0802, 0.0871, 0.0962, 0.096, 0.0938, 0.101, 0.1, 0.092]  min 0.000486 · mean 0.0908 · max 0.211
t2r_reward/success: [0, 5.93, 6.31, 6.98, 7.77, 7.58, 7.69, 7.91, 8.06, 7.27]  min 0 · mean 7.12 · max 14.8
t2r_reward/total: [1.36, 7.91, 7.91, 9.08, 9.17, 9.88, 10, 9.88, 10.3, 10.6]  min 1.36 · mean 9.33 · max 17.9

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. the success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness, or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: a component that grows while the task metrics do not may be paid for a behaviour that does not serve the task — gate it on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
All numbers below were measured, not estimated. Sources: the training log of the policy trained by the previous
reward (600 epochs, finished), and deterministic probes of that policy (64 environments, noise off, first episode of
each environment). Training for this round CONTINUES from that policy's weights. The environment did not change.

WHAT THE PREVIOUS REWARD ACHIEVED

Deterministic success (20 of the last 30 steps with all five conditions) by checkpoint of the previous run:

  epoch 200   0.531
  epoch 400   0.563
  epoch 600   0.578

For reference, the policy it started from scored 0.531 under the same probe. Training-log success rose from 0.37 to
0.41 (100-epoch averages) and was flat over the last 300 epochs.

At epoch 600: every episode in which the shoe was ever placed within 0.03 m went on to succeed except 3 of 64
(ever placed 0.625, success 0.578). The scripted return works: final palm distance to home 0.0005 m median.
Carrying, seating, opening the hand and triggering the takeover are learned. Do not rebuild them.

HOW THE REMAINING EPISODES FAIL (epoch 600, 27 of 64 episodes; state at the end of the episode)

  shoe resting on the rack, hand released, but keypoint_dist > 0.03 m      0.3125 of all episodes
      final keypoint_dist 0.031 / 0.040 / 0.081 m (10th / 50th / 90th percentile)
      smallest keypoint_dist during the episode 0.030 / 0.038 / 0.063 m
      these episodes run to the time limit (199 steps); the shoe stays where it was set down
  released, shoe not resting on the rack                                  0.031
  dropped below the table                                                 0.031
  still holding the shoe at the time limit                                0.047

At epoch 400 the same probe gave 0.375 for the first class with a median final error of 0.054 m, and 92 % of those
episodes never had keypoint_dist within 0.03 m at any step. So the dominant failure is: the shoe is set down a few
centimetres off the target and is never moved again.

STRUCTURAL FACTS ABOUT THAT STATE

  - The takeover requires `ctx.placed`. While keypoint_dist > 0.03 m the policy keeps control of the arm and the
    grip for the rest of the episode, so it can go back to the shoe and push or re-grasp it.
  - In the previous reward, `settle` (1.2 x calm) and `seat` (0.5) are paid whenever keypoint_dist <= 0.09 m and the
    shoe rests on the rack, whether or not it is placed. Over the last 50 epochs `settle` averaged 0.614 per step,
    the largest term after `success`; `seat` averaged 0.189. A shoe left motionless 0.04 m from the target keeps
    collecting both terms every step; touching it again makes it move, which lowers `settle`.

WHAT THE PERSON RUNNING THIS WANTS

After the shoe is set down, if it is not on the target, the policy should touch the shoe and move it onto the target
(adjust only the shoe it placed). Reaching the target with the first set-down is equally fine.
