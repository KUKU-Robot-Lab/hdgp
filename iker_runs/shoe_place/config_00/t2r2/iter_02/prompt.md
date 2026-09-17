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
    stable_steps: int              # placed & released & resting & still must hold this many steps WITHIN the last window_steps for success
    window_steps: int              # length of that trailing window, in steps

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
    stable_count: torch.Tensor     # (N,) how many of the last window_steps steps had placed & released & resting & still all true; a single bad step costs 1, it does not reset the count
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
5. Placement success is computed by the environment and cannot be redefined by the reward. A step counts toward it when all four of these hold: `ctx.keypoint_dist <= 0.03` m (`ctx.placed`), `ctx.palm_shoe_dist > 0.15` m (`ctx.released`, the hand has let go), `abs(ctx.shoe_bottom_z - ctx.rack_top_z) <= ctx.resting_tol` (`ctx.resting`, the shoe rests on the rack rather than floating or sinking), and `ctx.shoe_lin_vel.norm(dim=-1) < 0.05` m/s (`ctx.still`). `ctx.stable_count` is how many of the last 30 steps had all four true — one bad step costs one count and does NOT reset it — and `ctx.success` is True on the step `ctx.stable_count` reaches 20. The policy may therefore set the shoe down, nudge it back into place, and let go again, all within that window. You may add a bonus on `ctx.success` or on the individual conditions such as `ctx.placed` or `ctx.released`.
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
    """Place the held shoe on the rack: align -> seat -> LET GO -> withdraw -> settle.

    Previous policy reached `placed & resting` on 30% of steps but held the grip shut
    (open_frac 0.000 there), so `released`/`still`/`stable_count` were unreachable.
    This version pays heavily for the grip opening in exactly that state, removes the
    permanent `dropped` penalty that made any imperfect release catastrophic, and
    un-gates `withdraw` from a condition that could never be satisfied.
    """

    # ---- shared quantities --------------------------------------------------------------
    dist = ctx.keypoint_dist                       # (N,) mean keypoint error to the goal pose
    zero = torch.zeros_like(dist)

    # python-side constants, guarded against degenerate values
    tol = max(float(ctx.place_tolerance), 1e-3)
    rest_tol = max(float(ctx.resting_tol), 1e-3)
    still_sp = max(float(ctx.still_speed), 1e-3)
    rel_r = max(float(ctx.release_radius), 1e-3)
    n_stable = max(int(ctx.stable_steps), 1)

    # grip axis as an "open fraction": 0.0 = recorded grip pose (holding), 1.0 = open pose
    open_frac = torch.clamp((ctx.grip_norm + 1.0) * 0.5, min=0.0, max=1.0)
    ep = torch.clamp(ctx.episode_progress, min=0.0, max=1.0)

    # the state in which letting go is the right thing to do: shoe at the goal pose,
    # sitting at rack height. Reached on 30% of steps by the previous policy.
    seated = ctx.placed & ctx.resting
    seated_f = seated.float()
    handed_over = seated & ctx.released            # ... and the hand is actually off it
    handed_f = handed_over.float()

    # ---- 1. align: carry the shoe onto the 4 target keypoints ---------------------------
    # unchanged: this worked (0.13 -> 0.555 over the previous run). Long-range tanh pull
    # plus a sharp gaussian inside the tolerance for the final gradient.
    coarse = 1.0 - torch.tanh(dist / 0.20)
    fine = torch.exp(-((dist / tol) ** 2))
    align = 2.0 * (0.6 * coarse + 0.4 * fine)          # w=2.0, the transport shaping term

    # ---- 2. progress: same error, normalised by this episode's own starting error -------
    init_err = (ctx.init_keypoints - ctx.target_keypoints).norm(dim=-1).mean(dim=-1)
    progress = 1.0 * torch.clamp(
        (init_err - dist) / torch.clamp(init_err, min=1e-3), min=0.0, max=1.0
    )                                                   # w=1.0, bounded [0,1]

    # ---- 3. seat: put the shoe's lowest point onto the rack top -------------------------
    height_err = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    seat_shape = 1.0 - torch.tanh(height_err / rest_tol)
    near_goal = dist <= (2.0 * tol)                     # only near the target, else the
    seat = 1.0 * torch.where(near_goal, seat_shape, zero)   # policy is paid to hover

    # ---- 4. release: THE FIX. w 1.5 -> 4.0, dense in the grip EMA -----------------------
    # one step of a=+1 moves open_frac by 0.4686, i.e. ~+1.9 reward immediately.
    release = 4.0 * torch.where(seated, open_frac, zero)

    # ---- 5. hold_cost: price the local optimum of never letting go ----------------------
    # -1.0 max, ramped with elapsed time so a careful early placement is not punished.
    # Together with `release` the grip axis is worth a 5.0/step swing while seated, yet
    # stays far below the ~4.6/step the policy already earns there, so `seated` stays
    # attractive and the policy has no reason to avoid reaching it.
    hold_cost = -1.0 * torch.where(seated, (1.0 - open_frac) * (0.5 + 0.5 * ep), zero)

    # ---- 6. handsfree: flat dense pay once the shoe stands on the rack alone ------------
    handsfree = 3.0 * handed_f                          # w=3.0, the state the task wants

    # ---- 7. withdraw: back the palm off; gated on `seated` only, NOT on the grip --------
    # the old `open_frac > 0.5` gate was never satisfiable, so this term paid exactly 0
    # for the whole previous run. Gating on `seated` is self-protecting: a hand that still
    # grips cannot retreat without dragging the shoe, which breaks placed/resting.
    withdraw_frac = torch.clamp(ctx.palm_shoe_dist / (rel_r + 0.05), min=0.0, max=1.0)
    clearance = torch.tanh(ctx.palm_gap / 0.12)         # finer, earlier signal than the above
    withdraw = 2.0 * torch.where(seated, 0.5 * withdraw_frac + 0.5 * clearance, zero)

    # ---- 8. settle: let the shoe come to rest once the hand is off it -------------------
    lin_sp = ctx.shoe_lin_vel.norm(dim=-1)
    ang_sp = ctx.shoe_ang_vel.norm(dim=-1)
    calm = 1.0 - torch.tanh(lin_sp / still_sp + 0.5 * ang_sp)
    settle = 2.0 * torch.where(handed_over, calm, zero)  # w=2.0, only meaningful after release

    # ---- 9. gates: smooth staircase over the four success predicates --------------------
    # superlinear so the 4th (missing) condition is worth more than the first three:
    # 2 gates -> 0.625, 3 -> 1.41, 4 -> 2.5.
    n_gates = (
        ctx.placed.float() + ctx.resting.float() + ctx.released.float() + ctx.still.float()
    )
    gates = 2.5 * (n_gates / 4.0) ** 2

    # ---- 10. stability: dense credit for the 20-step success counter, w 1.0 -> 3.0 ------
    stability = 3.0 * torch.clamp(ctx.stable_count.float() / n_stable, min=0.0, max=1.0)

    # ---- 11. success: terminal bonus, 20 -> 30, scaled up for finishing early -----------
    time_left = torch.clamp(1.0 - ep, min=0.0, max=1.0)
    success = 30.0 * torch.where(ctx.success, 1.0 + 0.5 * time_left, zero)

    # ---- 12. mishandle: the ONLY penalty on the grip axis, deliberately narrow ----------
    # the old -0.5*early_open froze actions[6] entirely (released 0.643 -> 0.098). This
    # fires only on opening that is clearly not a placement: off rack height AND far from
    # the goal pose. Opening near the rack is free to explore.
    bad_open = (~ctx.resting).float() * (dist > (2.0 * tol)).float()
    mishandle = -0.4 * open_frac * bad_open

    # ---- 13. fall: bounded cost for dropping the shoe below the TABLE (not the rack) ----
    # the transport path legitimately passes below rack height, so the table is the
    # reference. No permanent `released & ~placed` bleed any more: that cliff is what made
    # any imperfect release irrational compared with simply never opening the hand.
    fall = torch.clamp(ctx.table_top_z - ctx.shoe_bottom_z, min=0.0)
    fall_penalty = -2.0 * torch.tanh(fall / 0.05)

    # ---- 14. action regularisation: small, or the +-0.05 action noise kills exploration -
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
        "hold_cost": hold_cost,
        "handsfree": handsfree,
        "withdraw": withdraw,
        "settle": settle,
        "gates": gates,
        "stability": stability,
        "success": success,
        "mishandle": mishandle,
        "fall_penalty": fall_penalty,
        "action_reg": action_reg,
    }

    reward = torch.zeros_like(dist)
    for value in components.values():
        reward = reward + value

    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components (t2r_reward/*) as well as task metrics computed by the environment (the placement flags place/placed, place/released, place/resting and place/retreated, the keypoint distance iker/keypoint_distance_m, the success rate iker/success_5cm, the drop rate iker/dropped, episode length and total reward) at 10 evenly spaced points during training, plus the min / mean / max encountered:

episode_lengths/iter: [24.1, 199, 119, 134, 153, 171, 167, 178, 184, 174]  min 24.1 · mean 159 · max 199
episode_lengths/step: [27.5, 185, 190, 121, 154, 169, 178, 174, 183, 184]  min 27.5 · mean 160 · max 199
episode_lengths/time: [24.1, 199, 119, 134, 153, 171, 167, 178, 184, 174]  min 24.1 · mean 159 · max 199
iker/dropped: [0.996, 0.105, 0.215, 0.135, 0.125, 0.0982, 0.0985, 0.112, 0.0765, 0.0928]  min 0.0312 · mean 0.154 · max 1
iker/keypoint_distance_m: [0.677, 0.3, 0.2, 0.175, 0.183, 0.161, 0.164, 0.158, 0.132, 0.143]  min 0.13 · mean 1.29 · max 256
iker/success_5cm: [0.00404, 0.00159, 0.317, 0.307, 0.184, 0.215, 0.288, 0.364, 0.447, 0.469]  min 0 · mean 0.266 · max 0.494
place/placed: [0.000267, 0.000244, 0.00859, 0.0391, 0.0949, 0.19, 0.34, 0.414, 0.464, 0.48]  min 0 · mean 0.237 · max 0.511
place/released: [0.643, 0.975, 0.978, 0.908, 0.859, 0.83, 0.83, 0.841, 0.885, 0.876]  min 0.643 · mean 0.88 · max 1
place/resting: [0.0255, 0.0493, 0.426, 0.58, 0.632, 0.7, 0.716, 0.754, 0.753, 0.746]  min 0.00586 · mean 0.58 · max 0.767
place/retreated: [0.789, 0.0308, 0.0121, 0.0523, 0.121, 0.204, 0.322, 0.35, 0.366, 0.28]  min 0.0012 · mean 0.224 · max 0.789
rewards/iter: [-6.27, 125, 336, 378, 461, 670, 959, 1.34e+03, 1.46e+03, 1.37e+03]  min -6.27 · mean 831 · max 1.71e+03
rewards/step: [-3.73, 104, 342, 412, 482, 703, 1.07e+03, 1.26e+03, 1.57e+03, 1.62e+03]  min -5.36 · mean 835 · max 1.71e+03
rewards/time: [-6.27, 125, 336, 378, 461, 670, 959, 1.34e+03, 1.46e+03, 1.37e+03]  min -6.27 · mean 831 · max 1.71e+03
t2r_reward/action_reg: [-0.0634, -0.0523, -0.0453, -0.0413, -0.038, -0.0344, -0.0312, -0.0283, -0.0258, -0.0243]  min -0.0634 · mean -0.0359 · max -0.0221
t2r_reward/align: [0.133, 0.211, 0.451, 0.583, 0.679, 0.783, 0.904, 0.986, 1.03, 1.05]  min 0.131 · mean 0.738 · max 1.08
t2r_reward/fall_penalty: [-0.111, -0.105, -0.0509, -0.0319, -0.0254, -0.0209, -0.0213, -0.0164, -0.017, -0.0174]  min -0.127 · mean -0.036 · max -0.0151
t2r_reward/gates: [0.273, 0.627, 0.901, 0.971, 1.01, 1.11, 1.21, 1.28, 1.36, 1.35]  min 0.273 · mean 1.08 · max 1.4
t2r_reward/handsfree: [0.000801, 0.000732, 0.0252, 0.108, 0.199, 0.389, 0.813, 1.1, 1.33, 1.35]  min 0 · mean 0.632 · max 1.43
t2r_reward/hold_cost: [-7.2e-05, -5.4e-05, -0.00148, -0.00416, -0.00236, -0.00255, -0.00238, -0.00193, -0.00172, -0.00176]  min -0.0044 · mean -0.00192 · max 0
t2r_reward/mishandle: [-0.196, -0.0495, -0.0183, -0.0147, -0.0132, -0.0095, -0.00757, -0.00615, -0.00637, -0.00639]  min -0.202 · mean -0.0238 · max -0.00451
t2r_reward/nonfinite_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
t2r_reward/progress: [0.0147, 0.0624, 0.285, 0.391, 0.447, 0.506, 0.551, 0.592, 0.604, 0.607]  min 0.0113 · mean 0.442 · max 0.619
t2r_reward/release: [0.00056, 0.000691, 0.0246, 0.128, 0.363, 0.744, 1.34, 1.64, 1.84, 1.91]  min 0 · mean 0.932 · max 2.03
t2r_reward/seat: [0.0011, 0.0186, 0.197, 0.313, 0.366, 0.422, 0.45, 0.477, 0.478, 0.473]  min 0.0011 · mean 0.346 · max 0.487
t2r_reward/settle: [0.000438, 0.000472, 0.0142, 0.0588, 0.103, 0.204, 0.409, 0.524, 0.645, 0.664]  min 0 · mean 0.312 · max 0.71
t2r_reward/stability: [0.000236, 0.000144, 0.00904, 0.0381, 0.0519, 0.0804, 0.118, 0.121, 0.148, 0.138]  min 0 · mean 0.0818 · max 0.173
t2r_reward/success: [0, 0, 0.0142, 0.0459, 0.0462, 0.0395, 0.02, 0.00973, 0.0108, 0.0126]  min 0 · mean 0.0214 · max 0.0718
t2r_reward/total: [0.0538, 0.714, 1.82, 2.62, 3.34, 4.5, 6.29, 7.35, 8.21, 8.35]  min 0.0538 · mean 4.89 · max 8.67
t2r_reward/withdraw: [0.000492, 0.000486, 0.0166, 0.0712, 0.149, 0.293, 0.54, 0.684, 0.825, 0.838]  min 0 · mean 0.402 · max 0.887

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. the success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness, or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: a component that grows while the task metrics do not may be paid for a behaviour that does not serve the task — gate it on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
All numbers below were measured, not estimated. Source: deterministic rollouts (256 environments, first episode
of each) of the policy trained by the previous reward, plus the training logs of that run.

WHAT THE PREVIOUS REWARD ACHIEVED

The release chain worked. The policy now opens the hand, and it opens it in the right state:

  open_frac on the steps where placed AND resting held      0.97   (the reward before that one: 0.00)
  placed AND resting                                        0.446 of steps, 0.606 of environments
  released                                                  0.700 of steps
  all four conditions at once                               0.197 of steps, 0.539 of environments

So setting the shoe down, letting go, and backing off are all learned behaviours now. Do not rebuild them.

WHERE IT STILL FAILS

1. `still` is the least satisfied of the four conditions, and it is the one that gates the rest:

     placed   0.447 of steps        resting  0.711 of steps
     released 0.700 of steps        still    0.491 of steps

   `still` also fell over training while the other three rose. It was 0.811 at epoch 100 and 0.614 at epoch 250.

2. Final placement is short of the goal. The target sits 0.139 m (centre to centre) from the other shoe on the
   rack. Where the shoe actually came to rest, over 256 episodes:

     10th percentile  0.117 m      median  0.168 m      90th percentile  0.328 m

   The median is 0.029 m farther from the other shoe than the target is. Final keypoint error, same episodes:

     10th percentile  0.020 m      median  0.046 m      90th percentile  0.254 m

   The placement tolerance has since been tightened to 0.03 m, so that median of 0.046 m would now fail. Closing
   the last three centimetres is the main thing this reward has to buy that the previous one did not.

3. Reward mass concentrated away from finishing. In the previous reward's final epoch, the components paid after
   the hand lets go (release, handsfree, withdraw, settle, stability) summed to 5.17 of a total of 8.67 — 59.7%
   — while the success component averaged 0.020, or 0.23% of the total. Those components rose monotonically to
   the end of training while the task's own success metric did not.

4. One component contradicted its own name: `settle` rose from 0.126 to 0.699 over the same epochs in which the
   measured `still` occupancy fell from 0.811 to 0.614. Whatever it was gated on, it was not the shoe coming to
   rest.

WHAT CHANGED IN THE ENVIRONMENT SINCE THAT RUN

The success rule and the tolerance changed; both are already stated in the task description above, and the
context fields carry the new values. Two consequences worth stating plainly:

  - A single bad step no longer resets progress toward success, so the policy can set the shoe down, push it
    to correct its position, and let go again, all inside the same window.
  - 0.03 m is close to the 0.028 m gap the target leaves between the two shoes, so "succeeded" and "looks
    correctly placed" now mean nearly the same thing.
