You are an expert in robotics, reinforcement learning and code generation.

We control one 7-DOF OpenArm robot arm (the left arm) carrying a five-finger Tesollo DG-5F hand, standing at a table. A shoe lies on the table in front of the hand; a second shoe already stands on a raised rack beside the table, and the task is to put the loose shoe on the rack next to it. The rack's footprint is x in [0.11, 0.43] m and y in [-0.33, -0.02] m, and its top is at z = 0.325 m. Positions are in metres in each environment's local frame with +z up; the table top is at z = table_top_z.

At the start of an episode the arm stands in its rest posture and the hand is open; the shoe lies somewhere in the reachable area of the table, its position drawn fresh every episode (its resting orientation does not change). Some episodes instead start from a recorded later state — the shoe already in the hand, or already set down on the rack — so that the later parts of the task are practised too; `ctx.start_held` is True in those episodes.

The action space is a normalized `Box(-1, 1, (26,), float32)`, one action every 0.1 s, 360 steps per episode:
    actions[0:3] = change of the palm position in the robot base frame, 0.02 * a metres per step;
    actions[3:6] = change of the palm orientation in the robot base frame (axis-angle), 0.05 * a radians per step;
    actions[ 6] = hand joint  0: l_hj_thumb_1 range [-0.010, +0.010] rad  open +0.000  grip +0.000  locked
    actions[ 7] = hand joint  1: l_hj_thumb_2 range [+1.560, +1.580] rad  open +1.570  grip +1.570  locked
    actions[ 8] = hand joint  2: l_hj_thumb_3 range [-1.571, +0.000] rad  open +0.000  grip -1.800  movable
    actions[ 9] = hand joint  3: l_hj_thumb_4 range [-1.571, +0.000] rad  open +0.000  grip -1.800  movable
    actions[10] = hand joint  4: l_hj_index_1 range [-0.010, +0.010] rad  open +0.000  grip +0.000  locked
    actions[11] = hand joint  5: l_hj_index_2 range [+0.000, +2.007] rad  open +0.000  grip +1.900  movable
    actions[12] = hand joint  6: l_hj_index_3 range [+0.000, +1.571] rad  open +0.000  grip +1.800  movable
    actions[13] = hand joint  7: l_hj_index_4 range [+0.000, +1.571] rad  open +0.000  grip +1.800  movable
    actions[14] = hand joint  8: l_hj_middle_1 range [-0.010, +0.010] rad  open +0.000  grip +0.000  locked
    actions[15] = hand joint  9: l_hj_middle_2 range [+0.000, +2.007] rad  open +0.000  grip +1.900  movable
    actions[16] = hand joint 10: l_hj_middle_3 range [+0.000, +1.571] rad  open +0.000  grip +1.800  movable
    actions[17] = hand joint 11: l_hj_middle_4 range [+0.000, +1.571] rad  open +0.000  grip +1.800  movable
    actions[18] = hand joint 12: l_hj_ring_1 range [-0.010, +0.010] rad  open +0.000  grip +0.000  locked
    actions[19] = hand joint 13: l_hj_ring_2 range [+0.000, +1.920] rad  open +0.000  grip +1.900  movable
    actions[20] = hand joint 14: l_hj_ring_3 range [+0.000, +1.571] rad  open +0.000  grip +1.800  movable
    actions[21] = hand joint 15: l_hj_ring_4 range [+0.000, +1.571] rad  open +0.000  grip +1.800  movable
    actions[22] = hand joint 16: l_hj_pinky_1 range [-0.010, +0.000] rad  open +0.000  grip +0.000  locked
    actions[23] = hand joint 17: l_hj_pinky_2 range [-0.010, +0.000] rad  open +0.000  grip +0.000  locked
    actions[24] = hand joint 18: l_hj_pinky_3 range [+0.000, +1.571] rad  open +0.000  grip +1.800  movable
    actions[25] = hand joint 19: l_hj_pinky_4 range [+0.000, +1.571] rad  open +0.000  grip +1.800  movable
The palm pose change goes through damped least-squares inverse kinematics to arm joint position targets (PD, gravity compensated), clamped to the arm's joint limits. Each hand action is mapped to its joint's commandable range above (a = -1 the lower end, a = +1 the upper end) and then filtered: every step the joint target moves 0.4686 of the way from its previous value towards the commanded one.

Each environment draws once at startup a shoe mass scale of 0.3 to 2.0, a friction coefficient of 0.3 to 1.8, a restitution of 0 to 1 and a centre-of-mass offset of up to 2.5 cm per axis. During training the policy's observations carry uniform noise of +-0.02 (orientations up to 0.2 rad) and its actions uniform noise of +-0.05 before they are executed. Once the grasp has latched, random force and torque pulses act on the shoe: each step an environment fires with its own probability between 0.006 and 0.6, and a pulse has a normal random component per axis with a standard deviation of 2.7 N (0.27 N m) per kg of shoe mass.

Now I want you to help me write a reward function for reinforcement learning.

Typically, the reward function of a manipulation task consists of these parts (some are optional — include them only if really necessary):
1. the distance between the robot's hand and the target object
2. the difference between the object's current state and its goal state
3. regularization of the robot's action
4. [optional] extra constraints on the target object implied by the task
5. [optional] extra constraints on the robot implied by the task

The reward function receives a single argument `ctx`, an instance of this class (all positions env-local, metres; angles rad; speeds m/s; forces N):

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
```

Additional knowledge:
1. Everything is BATCHED: every tensor field has a leading dimension N (number of parallel environments). The function must return a reward tensor of shape (N,) — never a Python float — and a dict of named component tensors, each of shape (N,). Use only `torch` and `math`; do not import anything else.
2. Write staged rewards with tensor masks, not `if`/`else` on tensors, e.g. `staged = torch.where(ctx.held, r_carry, torch.zeros_like(r_carry))`.
3. Prefer bounded, smooth shaping such as `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead of raw negative distances, so that no single term dominates.
4. THE TASK HAS TWO HALVES IN ONE EPISODE and the policy must do both with the same action vector: first pick the shoe up off the table, then carry it to the rack, align it with the target keypoints, set it down and let go.
5. Grasp status computed by the environment: `ctx.held` is True on a step where the shoe is lifted at least 0.05 m clear of where it lay, the thumb has closed at least 0.05 rad, the palm is within 0.15 m of the shoe and the shoe slips less than 0.05 m/s relative to the palm. `ctx.hold_count` counts consecutive held steps, `ctx.latched` turns True once it reaches 3 and stays True, and `ctx.carried` stays True once the shoe has been lifted clear of the table at all. Carrying the shoe away from where it was picked up makes `ctx.dz_free` read 0 (it only measures a free lift near the pick spot), so `ctx.held` does NOT stay true across the transport — use the contact, slip and gap fields to tell whether the shoe is still in the hand.
6. Goal: `ctx.target_keypoints` holds 4 fixed target points on the rack, set once for the whole run by a vision model (not part of the reward). `ctx.keypoints` are the shoe's own 4 keypoints in its current pose, `ctx.init_keypoints` the same at the start of the episode, `ctx.keypoint_err` the per-keypoint distance to the target and `ctx.keypoint_dist` their mean.
7. Placement success is computed by the environment and cannot be redefined by the reward. A step counts toward it when all five of these hold: `ctx.keypoint_dist <= 0.03` m (`ctx.placed`), `ctx.palm_shoe_dist > 0.15` m (`ctx.released`, the hand has let go), `abs(ctx.shoe_bottom_z - ctx.rack_top_z) <= ctx.resting_tol` (`ctx.resting`), `ctx.shoe_lin_vel.norm(dim=-1) < 0.05` m/s (`ctx.still`), and `ctx.arm_home_err <= 0.035` rad (`ctx.home`, every arm joint back in the rest posture). `ctx.stable_count` is how many of the last 30 steps had all five true — one bad step costs one count and does NOT reset it — and `ctx.success` is True on the step `ctx.stable_count` reaches 20.
   THE RETURN TO THE REST POSTURE IS NOT THE POLICY'S JOB. The first step the shoe is placed, resting and still while the hand has opened at least 0.9 of the way to its open pose, the environment takes over the arm: it opens the hand fully and moves the arm joints to the rest posture over 20 steps, and from then on the policy's actions have no effect (`ctx.retracting` is True). Any reward paid while `ctx.retracting` is True cannot be influenced by the policy.
8. The episode ends on a success, when the shoe falls below z = 0.1 m, or after `ctx.episode_steps - 1` steps. Nothing is added to the reward outside your function.
9. Do not keep any state between calls (no globals, no attributes); the function must be pure. `ctx` is read-only — never assign to one of its fields (no `ctx.x = ...`) and never call an in-place op on one (no `.clamp_()`, no `x[:, 0] = ...`).
10. Each component you return is logged separately during training (as `t2r_reward/<name>`, the sum as `t2r_reward/total`) and may be shown back to you after training, so name components meaningfully (e.g. "approach", "grasp", "carry", "align", "release").

I want it to fulfil the following task: Pick the shoe up from the table, carry it to the rack, align it with the target keypoints next to the other shoe, set it down and let go.
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

Observations from watching the trained policy:
All numbers below were measured on this environment's two predecessors, not estimated. Sources: deterministic probes
(256 environments, training noise off, first episode of every environment) of the best stage-1 grasp policy and the best
stage-2 placement policy, and the training logs of those runs. This is the FIRST reward for the unified task: one policy
now does both halves, so neither of those policies can be reused and there is no previous reward to improve.

WHAT THE TWO SEPARATE POLICIES ACHIEVED, AND WHERE THEY FAILED

Stage 1 (grasp only, 120-step episodes, started from a pre-grasp pose 8-12 cm above the shoe with the hand open):

  success (the shoe lifted clear and held 20 steps)                       0.344
  the shoe was lifted and the hold conditions held at least once,
      but the 20-step hold never completed before the time limit          0.410
  the hold latched and the shoe then fell back to the table               0.207
  the shoe fell off the table                                             0.016
  lifted, but the hold conditions never held                              0.020
  the palm never came near the shoe                                       0.004

So reaching the shoe is solved and closing on it is solved; KEEPING it is not. 62 % of episodes reach a proper hold at
least once and only 34 % sustain it. In the unified task the shoe must stay in the hand for the whole carry to the rack,
which is far longer than 20 steps, so a grasp that merely touches and lifts is not enough.

Stage 2 (placement only, started with the shoe already in the hand):

  success from the states its own training used                           0.563
  success from the states the stage-1 policy actually hands over          0.285
  final keypoint error when it failed by leaving the shoe on the rack
      off target                                                          0.032 / 0.045 / 0.075 m (10th / 50th / 90th percentile)

The gap between 0.563 and 0.285 is the handover: the states its training bank contained had passed a re-verification of
the grasp, and the ones a real chain produces had not. Chained end to end the two policies finish about 0.12 of episodes.

Two more measured facts about the placement half:
  - Once the shoe is set down within 3 cm of the target and the hand opens, the environment's scripted return of the arm
    works: the arm ends 0.13 degrees from the rest posture and the shoe moves at most 0.9 mm while it withdraws.
  - The placement policy never learned to touch a shoe it had set down off target. Two generated rewards paid for going
    back to it and the terms stayed flat for 200 epochs each, because after letting go the policy parks the hand
    0.25-0.30 m away from the shoe and never returns.

WHAT THIS MEANS FOR THIS REWARD (facts, not instructions)

  - The policy holds the shoe for most of the episode. `ctx.dz_free` only measures a free lift near the pick spot, so it
    reads 0 as soon as the shoe is carried away; `ctx.held`, `ctx.hold_count` and `ctx.latched` are built on it and
    therefore describe the pick-up, not the carry. During the carry the facts available are the contact forces
    (`ctx.link_shoe_force`, `ctx.palm_shoe_force`), `ctx.palm_gap`, `ctx.palm_shoe_dist`, `ctx.slip_speed` and the shoe's
    height.
  - The shoe's position on the table is drawn fresh every episode, so the approach cannot be memorised.
  - A fraction of episodes starts from a recorded later state (shoe already held, or already set down off target);
    `ctx.start_held` marks them. They see the same reward.
