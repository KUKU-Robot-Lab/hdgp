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
    stable_steps: int              # placed & released & resting & still must hold this many consecutive steps for success

    # ---- palm: left Tesollo DG-5F hand on the 7-DOF arm -------------------------------------
    palm_pos: torch.Tensor         # (N,3) palm frame origin
    palm_quat: torch.Tensor        # (N,4) palm orientation quaternion (w,x,y,z)
    palm_normal: torch.Tensor      # (N,3) unit vector pointing out of the palm's grasping side

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
    stable_count: torch.Tensor     # (N,) consecutive steps (up to this one) where placed & released & resting & still all held; 0 the moment any breaks
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
5. Placement success is computed by the environment and cannot be redefined by the reward. A step counts toward it when all four of these hold: `ctx.keypoint_dist <= 0.05` m (`ctx.placed`), `ctx.palm_shoe_dist > 0.15` m (`ctx.released`, the hand has let go), `abs(ctx.shoe_bottom_z - ctx.rack_top_z) <= ctx.resting_tol` (`ctx.resting`, the shoe rests on the rack rather than floating or sinking), and `ctx.shoe_lin_vel.norm(dim=-1) < 0.05` m/s (`ctx.still`). `ctx.stable_count` counts consecutive steps where all four hold (it drops to 0 the moment any one breaks, it is not cumulative) and `ctx.success` is True on the step `ctx.stable_count` reaches 20. You may add a bonus on `ctx.success` or on the individual conditions such as `ctx.placed` or `ctx.released`.
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
    """Place the held shoe on the rack: align -> seat -> release -> withdraw -> hold still."""

    # ---- shared quantities --------------------------------------------------------------
    dist = ctx.keypoint_dist                       # (N,) mean keypoint error to the goal pose
    zero = torch.zeros_like(dist)
    one = torch.ones_like(dist)

    # python-side constants, guarded against degenerate values
    tol = max(float(ctx.place_tolerance), 1e-3)
    rest_tol = max(float(ctx.resting_tol), 1e-3)
    still_sp = max(float(ctx.still_speed), 1e-3)
    n_stable = max(int(ctx.stable_steps), 1)

    # grip axis as an "open fraction": 0.0 = recorded grip pose (holding), 1.0 = open pose
    open_frac = torch.clamp((ctx.grip_norm + 1.0) * 0.5, min=0.0, max=1.0)

    # the state in which letting go is actually the right thing to do
    ready = ctx.placed & ctx.resting

    # ---- 1. align: carry the shoe onto the 4 target keypoints ---------------------------
    # two length scales: a long-range tanh pull (half value ~0.22 m) plus a sharp gaussian
    # that only lives inside the placement tolerance and supplies the final gradient.
    coarse = 1.0 - torch.tanh(dist / 0.20)
    fine = torch.exp(-((dist / tol) ** 2))
    align = 2.0 * (0.6 * coarse + 0.4 * fine)          # w=2.0: the dominant shaping term

    # ---- 2. progress: same error, normalised by this episode's own starting error -------
    init_err = (ctx.init_keypoints - ctx.target_keypoints).norm(dim=-1).mean(dim=-1)
    progress = 1.0 * torch.clamp(
        (init_err - dist) / torch.clamp(init_err, min=1e-3), min=0.0, max=1.0
    )                                                   # w=1.0, bounded [0,1]

    # ---- 3. seat: put the shoe's lowest point onto the rack top -------------------------
    height_err = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    seat_shape = 1.0 - torch.tanh(height_err / rest_tol)
    near_goal = dist <= (2.0 * tol)                     # only counts near the target, else the
    seat = 1.0 * torch.where(near_goal, seat_shape, zero)   # policy is paid to hover at rack height

    # ---- 4. release: open the fingers, but only once the shoe is placed and resting -----
    release = 1.5 * torch.where(ready, open_frac, zero)  # w=1.5, dense in the EMA opening

    # ---- 4b. premature release: the main degenerate strategy to price out ---------------
    early_open = torch.where(ready, zero, open_frac)     # opening anywhere else
    dropped = torch.where(ctx.released & (~ctx.placed), one, zero)   # shoe away from palm, not placed
    premature_release = -(0.5 * early_open + 1.0 * dropped)

    # ---- 5. withdraw: back the palm off past release_radius, after letting go -----------
    withdraw_frac = torch.clamp(
        ctx.palm_shoe_dist / (float(ctx.release_radius) + 0.10), min=0.0, max=1.0
    )
    can_withdraw = ready & (open_frac > 0.5)             # never rewarded while still gripping
    withdraw = 1.5 * torch.where(can_withdraw, withdraw_frac, zero)

    # ---- 6. still: set it down gently instead of throwing it ----------------------------
    lin_sp = ctx.shoe_lin_vel.norm(dim=-1)
    ang_sp = ctx.shoe_ang_vel.norm(dim=-1)
    calm = 1.0 - torch.tanh(lin_sp / still_sp + 0.5 * ang_sp)
    still = 0.75 * torch.where(ctx.placed, calm, zero)   # w=0.75, secondary to alignment

    # ---- 7. stability: dense credit for the 20-step success counter ---------------------
    stability = 1.0 * torch.clamp(ctx.stable_count.float() / n_stable, min=0.0, max=1.0)

    # ---- 8. success: the terminal bonus, scaled up for finishing early ------------------
    time_left = torch.clamp(1.0 - ctx.episode_progress, min=0.0, max=1.0)
    success = 20.0 * torch.where(ctx.success, 1.0 + 0.5 * time_left, zero)

    # ---- 9. fall: bounded cost for dropping the shoe below the TABLE (not the rack) -----
    # the transport path legitimately passes below rack height, so the table is the reference.
    fall = torch.clamp(ctx.table_top_z - ctx.shoe_bottom_z, min=0.0)
    fall_penalty = -2.0 * torch.tanh(fall / 0.05)

    # ---- 10. action regularisation: small, or the +-0.05 action noise kills exploration -
    d_act = ctx.actions - ctx.prev_actions
    rate = torch.mean(d_act ** 2, dim=-1)               # jitter is what breaks stable_count
    mag = torch.mean(ctx.actions ** 2, dim=-1)
    arm_motion = torch.tanh(torch.mean(ctx.arm_qd ** 2, dim=-1) / 4.0)
    action_reg = -(0.05 * rate + 0.02 * mag + 0.02 * arm_motion)

    components = {
        "align": align,
        "progress": progress,
        "seat": seat,
        "release": release,
        "premature_release": premature_release,
        "withdraw": withdraw,
        "still": still,
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

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components (t2r_reward/*) as well as task metrics computed by the environment (the placement flags place/placed, place/released, place/resting and place/retreated, the keypoint distance iker/keypoint_distance_m, the success rate iker/success_5cm, the drop rate iker/dropped, episode length and total reward) at 10 evenly spaced points during training, plus the min / mean / max encountered:

episode_lengths/iter: [24.8, 45.9, 64, 101, 113, 135, 125, 132, 134, 142]  min 24.8 · mean 112 · max 156
episode_lengths/step: [29.3, 61.4, 68.3, 97.5, 111, 128, 130, 137, 133, 147]  min 29.3 · mean 112 · max 156
episode_lengths/time: [24.8, 45.9, 64, 101, 113, 135, 125, 132, 134, 142]  min 24.8 · mean 112 · max 156
iker/dropped: [0.999, 0.894, 0.888, 0.735, 0.6, 0.526, 0.529, 0.487, 0.454, 0.456]  min 0.391 · mean 0.612 · max 1
iker/keypoint_distance_m: [0.692, 2, 0.818, 7.64, 0.603, 0.567, 0.546, 0.535, 0.523, 0.506]  min 0.453 · mean 1.55 · max 122
iker/success_5cm: [0.000781, 0.000453, 0.00101, 0.0047, 0.0294, 0.0508, 0.0678, 0.101, 0.12, 0.123]  min 0 · mean 0.0602 · max 0.164
place/placed: [0.000282, 0.000412, 0.00315, 0.0324, 0.0805, 0.0997, 0.144, 0.181, 0.184, 0.191]  min 0 · mean 0.108 · max 0.229
place/released: [0.643, 0.425, 0.246, 0.197, 0.153, 0.126, 0.126, 0.115, 0.107, 0.0978]  min 0.0911 · mean 0.201 · max 0.988
place/resting: [0.0245, 0.0612, 0.11, 0.216, 0.306, 0.3, 0.321, 0.358, 0.355, 0.352]  min 0.00808 · mean 0.26 · max 0.377
place/retreated: [0.786, 0.279, 0.294, 0.26, 0.21, 0.168, 0.166, 0.164, 0.163, 0.15]  min 0.144 · mean 0.213 · max 0.786
rewards/iter: [-13.7, -23.2, -3, 32.4, 70.5, 94.3, 108, 136, 126, 166]  min -116 · mean 74.9 · max 180
rewards/step: [-16.1, -33.3, -10.8, 37.9, 68.4, 97.8, 109, 107, 104, 158]  min -116 · mean 75.3 · max 180
rewards/time: [-13.7, -23.2, -3, 32.4, 70.5, 94.3, 108, 136, 126, 166]  min -116 · mean 74.9 · max 180
t2r_reward/action_reg: [-0.0634, -0.0543, -0.0483, -0.0418, -0.0365, -0.032, -0.0285, -0.0265, -0.0248, -0.0234]  min -0.0634 · mean -0.0354 · max -0.0222
t2r_reward/align: [0.132, 0.106, 0.185, 0.32, 0.424, 0.455, 0.501, 0.547, 0.547, 0.555]  min 0.101 · mean 0.407 · max 0.61
t2r_reward/fall_penalty: [-0.112, -0.0935, -0.0662, -0.0486, -0.0367, -0.0297, -0.0278, -0.0282, -0.0276, -0.0254]  min -0.127 · mean -0.0442 · max -0.0233
t2r_reward/nonfinite_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
t2r_reward/premature_release: [-0.894, -0.513, -0.29, -0.218, -0.164, -0.133, -0.131, -0.12, -0.111, -0.102]  min -1.18 · mean -0.228 · max -0.0943
t2r_reward/progress: [0.0144, 0.0253, 0.0914, 0.194, 0.264, 0.285, 0.306, 0.326, 0.324, 0.327]  min 0.0132 · mean 0.236 · max 0.353
t2r_reward/release: [0.000209, 6.5e-05, 0.000216, 0.000906, 0.000509, 0.000455, 0.000479, 0.000795, 0.000367, 0.000188]  min 0 · mean 0.000395 · max 0.00113
t2r_reward/seat: [0.00139, 0.00594, 0.0234, 0.0949, 0.172, 0.173, 0.201, 0.237, 0.24, 0.245]  min 0.00123 · mean 0.156 · max 0.269
t2r_reward/stability: [0.000105, 0.000101, 8e-05, 0.00031, 0.000185, 0.000373, 0.000787, 0.000412, 0.000239, 0.000186]  min 0 · mean 0.00028 · max 0.00107
t2r_reward/still: [0.000175, 0.000225, 0.0003, 0.00257, 0.00492, 0.00545, 0.0104, 0.0163, 0.0192, 0.0198]  min 0 · mean 0.00958 · max 0.0246
t2r_reward/success: [0.000433, 0.000201, 0.000205, 0.000212, 0.000324, 0.000753, 0.00131, 0.000166, 0.000187, 0.00017]  min 0 · mean 0.000414 · max 0.00175
t2r_reward/total: [-0.92, -0.522, -0.104, 0.304, 0.629, 0.724, 0.833, 0.952, 0.967, 0.996]  min -1.22 · mean 0.501 · max 1.11
t2r_reward/withdraw: [0.000173, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.23e-06 · max 0.000173

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. the success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness, or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: a component that grows while the task metrics do not may be paid for a behaviour that does not serve the task — gate it on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
All numbers below were measured, not estimated: a deterministic rollout (observation and action noise off)
of the trained policy from the previous reward, 64 environments, first episode of each, 173 steps on average.
The four predicate flags and the grip state were read directly out of the environment on every step.

Gate occupancy, as a fraction of all rollout steps (and the fraction of environments that reached it at least
once during their episode):

  placed   (keypoint_dist <= 0.05)                 0.330 of steps   0.719 of environments
  resting  (|shoe_bottom_z - rack_top_z| <= 0.01)  0.486 of steps   0.984 of environments
  placed AND resting                               0.299 of steps   0.719 of environments
  released (palm_shoe_dist > 0.15)                 0.043 of steps   0.219 of environments
  still    (lin. and ang. speed < 0.05)            0.017 of steps   0.047 of environments
  all four at once                                 0.000 of steps   0.000 of environments

The consecutive-success counter never left zero: its maximum over the whole rollout was 0 in every one of the
64 environments, against the 20 consecutive steps the task requires.

The grip axis, measured as open_frac = clamp((grip_norm + 1) / 2, 0, 1), where 0 is the grasp-bank grip pose
and 1 is the open pose:

  mean over all steps                              0.014
  mean over the steps where placed AND resting     0.000
  per-environment maximum, mean over environments  0.130
  per-environment maximum, 90th percentile         0.598

So the state in which the previous reward paid for opening the hand was reached on 30% of all steps and by 72%
of environments, and in those states the policy's grip stayed shut. The policy did not fail to reach the
release condition; it reached it constantly and did not open.

Two structural facts that follow from the definitions and hold regardless of the policy:

  1. `released` is palm-to-shoe distance above 0.15 m, so it cannot become true while the hand still holds the
     shoe. It is a consequence of opening the grip, not an independent condition the policy can satisfy.
  2. `still` requires the shoe's linear and angular speed to be below 0.05. A shoe held in a hand that is
     still moving is almost never still: this is why it sits at 0.017 while `resting` sits at 0.486.

Consequently the four-way conjunction has no path to becoming true until the grip opens, and the 20-step
counter cannot start. Over training, the components that pay without ever opening the hand (align and seat)
rose from 0.13 and 0.001 to 0.58 and 0.25, while release fell to 0.0001 and withdraw stayed at exactly 0.
Policy entropy fell from 9.87 to 1.83 over the same run.
