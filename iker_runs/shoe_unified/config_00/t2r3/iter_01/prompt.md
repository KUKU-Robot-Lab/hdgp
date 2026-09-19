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
```

Additional knowledge:
1. Everything is BATCHED: every tensor field has a leading dimension N (number of parallel environments). The function must return a reward tensor of shape (N,) — never a Python float — and a dict of named component tensors, each of shape (N,). Use only `torch` and `math`; do not import anything else.
2. Write staged rewards with tensor masks, not `if`/`else` on tensors, e.g. `staged = torch.where(ctx.held, r_carry, torch.zeros_like(r_carry))`.
3. Prefer bounded, smooth shaping such as `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead of raw negative distances, so that no single term dominates.
4. THE TASK HAS TWO HALVES IN ONE EPISODE and the policy must do both with the same action vector: first pick the shoe up off the table, then carry it to the rack, align it with the target keypoints, set it down and let go.
5. Grasp status computed by the environment: `ctx.held` is True on a step where the shoe is lifted at least 0.05 m clear of where it lay, the thumb has closed at least 0.05 rad, the palm is within 0.15 m of the shoe and the shoe slips less than 0.05 m/s relative to the palm. `ctx.hold_count` counts consecutive held steps, `ctx.latched` turns True once it reaches 3 and stays True, and `ctx.carried` stays True once the shoe has been lifted clear of the table at all. Carrying the shoe away from where it was picked up makes `ctx.dz_free` read 0 (it only measures a free lift near the pick spot), so `ctx.held` does NOT stay true across the transport — use the contact, slip and gap fields to tell whether the shoe is still in the hand.
6. Goal: `ctx.target_keypoints` holds 4 fixed target points on the rack, set once for the whole run by a vision model (not part of the reward). `ctx.keypoints` are the shoe's own 4 keypoints in its current pose, `ctx.init_keypoints` the same at the start of the episode, `ctx.keypoint_err` the per-keypoint distance to the target and `ctx.keypoint_dist` their mean.
7. Placement success is computed by the environment and cannot be redefined by the reward. A step counts toward it when all five of these hold: `ctx.keypoint_dist <= 0.03` m (`ctx.placed`), `ctx.palm_shoe_dist > 0.15` m (`ctx.released`, the hand has let go), `abs(ctx.shoe_bottom_z - ctx.rack_top_z) <= ctx.resting_tol` (`ctx.resting`), `ctx.shoe_lin_vel.norm(dim=-1) < 0.05` m/s (`ctx.still`), and `ctx.arm_home_err <= 0.035` rad (`ctx.home`, every arm joint back in the rest posture). `ctx.stable_count` is how many of the last 30 steps had all five true — one bad step costs one count and does NOT reset it — and `ctx.success` is True on the step `ctx.stable_count` reaches 20.
   THE RETURN TO THE REST POSTURE IS NOT THE POLICY'S JOB. The first step the shoe is placed, resting and still while `ctx.open_frac >= 0.9` (the hand has opened that far towards the open hand of a reset — this exact field, not any other measure of openness built from the finger joints), the environment takes over the arm: it opens the hand fully and moves the arm joints to the rest posture over 20 steps, and from then on the policy's actions have no effect (`ctx.retracting` is True). Any reward paid while `ctx.retracting` is True cannot be influenced by the policy.
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

The previous reward function was:
```python
import torch
import math


def _unit(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise unit vector; safe for zero rows."""
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def compute_reward(ctx) -> "tuple[torch.Tensor, dict[str, torch.Tensor]]":
    dev = ctx.palm_pos.device
    dt = ctx.palm_pos.dtype
    n = ctx.palm_pos.shape[0]
    zero = torch.zeros(n, device=dev, dtype=dt)

    # ---------------- weights -------------------------------------------------------------
    # Ordered so the task is strictly monotone in progress: every stage pays more than the one
    # before it, and the release/stable money is unreachable without a correct placement.
    W_APPROACH = 1.0    # cheapest: reaching was already solved by the predecessor policy
    W_GRASP = 1.5       # closing on the shoe
    W_LIFT = 1.5        # getting it off the table
    W_HOLD = 1.0        # paid EVERY carry step: the measured bottleneck is losing the grip
    W_TRANSPORT = 2.5   # coarse travel to the rack
    W_ALIGN = 2.0       # fine 6-DOF keypoint match (mean + worst keypoint)
    W_SETTLE = 1.5      # sitting at rack height, not moving
    W_RELEASE = 3.0     # opening and backing off, ONLY once the placement is already correct
    W_STABLE = 4.0      # every step that counts toward stable_count
    W_SUCCESS = 40.0    # one-shot terminal bonus
    W_PEN = 1.0         # drop / disturb / table / rack penalties (each sub-term <= ~0.5)

    # ---------------- hand closure from the normalised joint angles -----------------------
    # hand_q_norm is 0 at the lower limit and 1 at the upper limit. For the movable joints the
    # open pose sits at the LOWER limit, except the two thumb joints (_3, _4) whose open angle
    # 0.0 rad is the UPPER end of [-1.571, 0]; those two are flipped.
    movable = torch.tensor([2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 18, 19],
                           device=dev, dtype=torch.long)
    flip = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        device=dev, dtype=dt)
    q_n = ctx.hand_q_norm.index_select(1, movable).clamp(0.0, 1.0)
    t_n = ctx.hand_target_norm.index_select(1, movable).clamp(0.0, 1.0)
    curl = flip * (1.0 - q_n) + (1.0 - flip) * q_n          # 0 open, 1 fully gripped
    curl_cmd = flip * (1.0 - t_n) + (1.0 - flip) * t_n      # same, for the filtered targets
    close_frac = curl.mean(dim=-1)                          # (N,)
    open_cmd = 1.0 - curl_cmd.mean(dim=-1)                  # commanded open fraction (N,)

    # ---------------- contact / grasp facts (valid during the whole carry) -----------------
    finger_force = ctx.link_shoe_force.sum(dim=-1)                       # (N,5) per finger
    contact_score = torch.tanh(finger_force / 2.0).mean(dim=-1)          # bounded contact quality
    n_touch = (finger_force > 0.2).to(dt).sum(dim=-1) + (ctx.palm_shoe_force > 0.2).to(dt)
    tip_gap = ctx.link_shoe_gap[:, :, 2].mean(dim=-1)                    # fingertips to shoe
    all_gap = ctx.link_shoe_gap.mean(dim=(1, 2))                         # all finger links

    # Own "still in the hand" predicate: dz_free (and therefore held/latched) reads 0 once the
    # shoe is carried away from the pick spot, so the carry must be judged from contact,
    # distance and slip only. ctx.held is OR-ed in, it is never relied on alone.
    secure = (n_touch >= 2.0) & (ctx.palm_shoe_dist < ctx.hold_radius) \
        & (ctx.slip_speed < 3.0 * ctx.hold_slip_speed)
    in_hand = secure | ctx.held
    lifted = ctx.shoe_bottom_z > (ctx.table_top_z + 0.02)

    placed_ok = ctx.placed & ctx.resting          # a genuinely finished placement
    busy = ~ctx.retracting                        # the policy only controls the robot here
    shoe_speed = ctx.shoe_lin_vel.norm(dim=-1)

    # ---------------- 1. approach ----------------------------------------------------------
    # Off while the shoe is already correctly placed, so a good placement is never disturbed,
    # and off while the environment drives the arm.
    face = 0.5 * (1.0 + (ctx.palm_normal * _unit(ctx.shoe_pos - ctx.palm_pos)).sum(dim=-1))
    r_app = 0.45 * (1.0 - torch.tanh(5.0 * ctx.palm_gap)) \
        + 0.30 * (1.0 - torch.tanh(5.0 * tip_gap)) \
        + 0.25 * face.clamp(0.0, 1.0)
    m_app = (~in_hand) & (~placed_ok) & busy
    approach = W_APPROACH * torch.where(m_app, r_app, zero)

    # ---------------- 2. grasp -------------------------------------------------------------
    # Only within reach of the shoe, so the hand cannot farm it by closing in mid-air.
    thumb = (ctx.thumb_curl / 0.6).clamp(0.0, 1.0)          # deep curl, well past thumb_curl_min
    r_grasp = 0.50 * contact_score + 0.30 * thumb \
        + 0.20 * close_frac * (1.0 - torch.tanh(8.0 * all_gap))
    m_grasp = (ctx.palm_gap < 0.12) & (~placed_ok) & busy
    grasp = W_GRASP * torch.where(m_grasp, r_grasp, zero)

    # ---------------- 3. lift --------------------------------------------------------------
    # Height above the table works everywhere; dz_free only pays at the pick spot, which is
    # exactly where the free-lift signal is meaningful.
    h_tab = ((ctx.shoe_bottom_z - ctx.table_top_z) / 0.12).clamp(0.0, 1.0)
    h_free = (ctx.dz_free / max(ctx.lift_height, 1e-3)).clamp(0.0, 1.0)
    lift = W_LIFT * torch.where(in_hand, 0.6 * h_tab + 0.4 * h_free, zero)

    # ---------------- 4. hold (the anti-drop insurance, paid every carry step) -------------
    r_hold = 0.45 * contact_score \
        + 0.35 * (1.0 - torch.tanh(ctx.slip_speed / max(ctx.hold_slip_speed, 1e-3))) \
        + 0.20 * (1.0 - torch.tanh(8.0 * ctx.palm_gap))
    hold = W_HOLD * torch.where(in_hand & lifted, r_hold, zero)

    # ---------------- 5. transport ---------------------------------------------------------
    # Gated on the shoe being clear of the table (not on in_hand) so that letting go on target
    # does not knock this term out and create a cliff at the moment of release.
    transport = W_TRANSPORT * torch.where(
        lifted, 1.0 - torch.tanh(2.5 * ctx.keypoint_dist), zero)

    # ---------------- 6. align (position AND orientation) ----------------------------------
    # keypoint_dist is the mean; the worst keypoint punishes a shoe at the right spot but tilted.
    worst = ctx.keypoint_err.max(dim=-1).values
    r_align = 0.6 * (1.0 - torch.tanh(12.0 * ctx.keypoint_dist)) \
        + 0.4 * (1.0 - torch.tanh(8.0 * worst))
    align = W_ALIGN * torch.where(lifted, r_align, zero)

    # ---------------- 7. settle ------------------------------------------------------------
    dz_rest = (ctx.shoe_bottom_z - ctx.rack_top_z).abs()
    r_settle = 0.5 * (1.0 - torch.tanh(dz_rest / max(ctx.resting_tol, 1e-3))) \
        + 0.5 * (1.0 - torch.tanh(shoe_speed / max(ctx.still_speed, 1e-3)))
    m_settle = ctx.keypoint_dist < (3.0 * ctx.place_tolerance)
    settle = W_SETTLE * torch.where(m_settle, r_settle, zero)

    # ---------------- 8. release (hard-gated on an already correct placement) --------------
    # The measured failure of the predecessor was setting the shoe down off target and never
    # returning. Paying nothing at all for an off-target release removes that whole branch:
    # the only way to earn it is to be within place_tolerance and resting first.
    r_rel = 0.55 * open_cmd.clamp(0.0, 1.0) \
        + 0.45 * (ctx.palm_shoe_dist / max(ctx.release_radius, 1e-3)).clamp(0.0, 1.0)
    release = W_RELEASE * torch.where(placed_ok, r_rel, zero)

    # ---------------- 9. stable + success ---------------------------------------------------
    good_step = ctx.placed & ctx.released & ctx.resting & ctx.still & ctx.home
    stable = W_STABLE * good_step.to(dt)
    success = W_SUCCESS * ctx.success.to(dt)

    # ---------------- 10. penalties ---------------------------------------------------------
    # (a) dropping a shoe that had already been carried
    dropped = ctx.carried & (~in_hand) & (~placed_ok) \
        & (ctx.shoe_bottom_z < (ctx.table_top_z + 0.03))
    p_drop = torch.where(dropped, torch.full_like(ctx.shoe_bottom_z, 0.6), zero)
    # (b) shoving the shoe around the table instead of grasping it (not for start_held resets,
    #     where shoe_start_xy is where it started inside the hand)
    m_dist = (~ctx.carried) & (~in_hand) & (~ctx.start_held)
    p_dist = torch.where(m_dist, 0.4 * (ctx.shoe_shift_xy / 0.08).clamp(0.0, 1.0), zero)
    # (c) pressing the fingers into / through the table top
    p_tab = 0.4 * ((ctx.table_top_z - ctx.hand_z_min) / 0.05).clamp(0.0, 1.0)
    # (d) driving the shoe into the side of the rack instead of down onto its top surface
    inside = (ctx.shoe_pos[:, 0] > ctx.rack_x_min - 0.05) & (ctx.shoe_pos[:, 0] < ctx.rack_x_max + 0.05) \
        & (ctx.shoe_pos[:, 1] > ctx.rack_y_min - 0.05) & (ctx.shoe_pos[:, 1] < ctx.rack_y_max + 0.05)
    m_ram = inside & in_hand & ctx.carried & (ctx.keypoint_dist < 0.25)
    p_ram = torch.where(
        m_ram, 0.5 * (((ctx.rack_top_z - 0.01) - ctx.shoe_bottom_z) / 0.10).clamp(0.0, 1.0), zero)
    penalty = -W_PEN * torch.where(busy, p_drop + p_dist + p_tab + p_ram, zero)

    # ---------------- 11. action regularization ---------------------------------------------
    # Small on purpose (<= ~0.35/step against ~10/step of shaping at the end) and switched off
    # while the environment drives the arm, where the actions have no effect anyway.
    rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    mag = (ctx.actions[:, 0:6] ** 2).mean(dim=-1)
    armv = (ctx.arm_qd ** 2).mean(dim=-1).clamp(0.0, 25.0)
    action_reg = -torch.where(busy, 0.05 * rate + 0.02 * mag + 0.005 * armv, zero)

    reward = (approach + grasp + lift + hold + transport + align + settle
              + release + stable + success + penalty + action_reg)

    return reward, {
        "approach": approach,
        "grasp": grasp,
        "lift": lift,
        "hold": hold,
        "transport": transport,
        "align": align,
        "settle": settle,
        "release": release,
        "stable": stable,
        "success": success,
        "penalty": penalty,
        "action_reg": action_reg,
    }
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components (t2r_reward/*) as well as task metrics computed by the environment (the grasp flags grasp/held_frac, grasp/lost_frac, grasp/dz_free, grasp/thumb_curl, grasp/rel_speed, the placement flags place/placed, place/released, place/resting, place/still, place/home, the success rates place/success_table_start and place/success_held_start, the keypoint distance iker/keypoint_distance_m, the drop rate iker/dropped, episode length and total reward) at 10 evenly spaced points during training, plus the min / mean / max encountered:

episode_lengths/iter: [29.3, 341, 338, 347, 349, 342, 347, 346, 353, 347]  min 29.3 · mean 343 · max 358
episode_lengths/step: [29.3, 341, 338, 347, 349, 342, 347, 346, 353, 347]  min 29.3 · mean 343 · max 358
episode_lengths/time: [29.3, 341, 338, 347, 349, 342, 347, 346, 353, 347]  min 29.3 · mean 343 · max 358
grasp/dz_free: [0.027, 0.00292, 0.00316, 0.00346, 0.00311, 0.00293, 0.00306, 0.052, 0.00249, 0.00255]  min -0.00252 · mean 0.0756 · max 20.4
grasp/held_frac: [0.00265, 0.000294, 0.000366, 0.000484, 0.000458, 0.000504, 0.00037, 0.000393, 0.000443, 0.000404]  min 2.7e-05 · mean 0.000421 · max 0.00265
grasp/lost_frac: [0.000195, 0.000977, 0.00162, 0.00287, 0.00259, 0.00312, 0.00255, 0.00235, 0.00335, 0.00421]  min 0 · mean 0.00261 · max 0.00685
grasp/rel_speed: [0.255, 0.144, 0.115, 0.101, 0.0918, 0.0795, 0.0715, 0.202, 0.0703, 0.0658]  min 0.064 · mean 0.184 · max 12.5
grasp/thumb_curl: [0.538, 0.901, 0.863, 0.682, 0.621, 0.6, 0.575, 0.564, 0.579, 0.589]  min 0.384 · mean 0.673 · max 0.972
place/carried: [0.000572, 0.000977, 0.00168, 0.00294, 0.0026, 0.00314, 0.00258, 0.00237, 0.00337, 0.00426]  min 0 · mean 0.00266 · max 0.00697
place/home: [0.000469, 4e-06, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.13e-07 · max 0.000469
place/open_frac: [0.557, 0.403, 0.359, 0.382, 0.386, 0.397, 0.39, 0.396, 0.413, 0.426]  min 0.357 · mean 0.406 · max 0.557
place/placed: [0, 0.0351, 0.072, 0.0846, 0.087, 0.108, 0.0814, 0.0771, 0.092, 0.107]  min 0 · mean 0.0798 · max 0.119
place/released: [0.759, 0.127, 0.112, 0.147, 0.131, 0.0978, 0.108, 0.114, 0.178, 0.273]  min 0.0873 · mean 0.17 · max 0.948
place/resting: [0.0133, 0.169, 0.185, 0.203, 0.207, 0.226, 0.223, 0.219, 0.235, 0.242]  min 0.0028 · mean 0.206 · max 0.26
place/retracting: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.01e-07 · max 0.000122
place/still: [0.679, 0.27, 0.237, 0.2, 0.19, 0.179, 0.192, 0.178, 0.199, 0.205]  min 0.166 · mean 0.221 · max 0.912
place/success_held_start: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.1e-06 · max 0.00391
place/success_table_start: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
rewards/iter: [53.1, 637, 721, 775, 826, 824, 821, 839, 811, 973]  min 53.1 · mean 772 · max 1.11e+03
rewards/step: [53.1, 637, 721, 775, 826, 824, 821, 839, 811, 973]  min 53.1 · mean 772 · max 1.11e+03
rewards/time: [53.1, 637, 721, 775, 826, 824, 821, 839, 811, 973]  min 53.1 · mean 772 · max 1.11e+03
t2r_reward/action_reg: [-0.0631, -0.0425, -0.0288, -0.024, -0.0204, -0.0175, -0.0152, -0.0142, -0.0128, -0.0116]  min -0.0631 · mean -0.0222 · max -0.0112
t2r_reward/align: [0.00353, 0.158, 0.194, 0.211, 0.215, 0.247, 0.224, 0.227, 0.255, 0.277]  min 0.000933 · mean 0.217 · max 0.299
t2r_reward/approach: [0.358, 0.668, 0.634, 0.611, 0.628, 0.624, 0.627, 0.624, 0.617, 0.605]  min 0.326 · mean 0.624 · max 0.679
t2r_reward/grasp: [0.154, 0.579, 0.575, 0.56, 0.571, 0.561, 0.586, 0.581, 0.556, 0.561]  min 0.0477 · mean 0.561 · max 0.602
t2r_reward/hold: [0.0433, 0.044, 0.0678, 0.0764, 0.0783, 0.089, 0.0787, 0.0831, 0.0812, 0.0804]  min 0.00273 · mean 0.0724 · max 0.0913
t2r_reward/lift: [0.119, 0.0812, 0.12, 0.13, 0.132, 0.143, 0.128, 0.136, 0.131, 0.128]  min 0.00676 · mean 0.12 · max 0.147
t2r_reward/nonfinite_frac: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
t2r_reward/penalty: [-0.00296, -0.208, -0.166, -0.132, -0.122, -0.122, -0.0904, -0.0888, -0.0953, -0.113]  min -0.284 · mean -0.128 · max -0.00296
t2r_reward/release: [0, 0.0545, 0.109, 0.13, 0.132, 0.163, 0.124, 0.12, 0.144, 0.17]  min 0 · mean 0.124 · max 0.19
t2r_reward/settle: [0.000162, 0.119, 0.137, 0.147, 0.157, 0.177, 0.178, 0.181, 0.211, 0.232]  min 0.000162 · mean 0.168 · max 0.246
t2r_reward/stable: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.04e-07 · max 0.000305
t2r_reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.02e-07 · max 0.000153
t2r_reward/total: [0.776, 1.86, 2.1, 2.2, 2.28, 2.41, 2.37, 2.38, 2.45, 2.51]  min 0.309 · mean 2.23 · max 2.59
t2r_reward/transport: [0.163, 0.404, 0.462, 0.495, 0.509, 0.544, 0.529, 0.531, 0.562, 0.575]  min 0.0149 · mean 0.498 · max 0.615

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. the success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness, or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: a component that grows while the task metrics do not may be paid for a behaviour that does not serve the task — gate it on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
All numbers below were measured, not estimated. Sources: the training log of the policy trained by the previous reward
(3000 epochs, 8192 environments, 30 % of episodes starting with the shoe already in the hand), and deterministic probes
of its final checkpoint (256 environments, training noise off, first episode of every environment), run once with every
episode starting from the table and once with every episode starting with the shoe in the hand. Training for this round
CONTINUES from that policy's weights. The environment is unchanged except for ONE new context field, `ctx.open_frac`
(see below).

RESULT OF THE PREVIOUS REWARD: NO EPISODE EVER SUCCEEDED, FROM EITHER START. Two separate reasons.

1. THE SHOE IS NEVER PICKED UP FROM THE TABLE.

   Episodes starting from the table (256 of 256):
     the shoe never rose 1 cm above the table in any episode                       256 of 256
     finger links touching the shoe at the same time, most over the episode        1 / 2 / 3 (10th / 50th / 90th percentile) of 15
     closest the shoe came to the target (keypoint_dist)                           0.19 / 0.22 / 0.27 m
   Training log, whole run: the fraction of steps with `ctx.carried` true rose from 0.0004 to 0.0043 and no higher;
   `ctx.held` stayed at 0.0005.
   Frames of a recorded video of this policy show the hand going to the shoe and resting its fingertips on the TOP of the shoe, staying
   there; it does not reach around or under it.
   For comparison, when every episode started with the shoe already in the hand, the same policy's hand touched it with
   6 / 8 / 10 links (10th / 50th / 90th percentile) — that is what a real grasp of this shoe looks like in this hand.

   The previous reward's `grasp` term paid, whenever the palm was within 0.12 m of the shoe: 0.50 x the mean over fingers
   of tanh(force / 2), 0.30 x the thumb's closing (clamped at 0.6 rad), and 0.20 x the mean finger closing times a
   closeness factor. Over the last 100 epochs it averaged 0.546 per step, the second largest term, while nothing was
   ever lifted: closing the fingers on top of the shoe collects most of it.

2. THE HAND NEVER OPENS AFTER THE SHOE IS SET DOWN, SO THE SCRIPTED RETURN NEVER STARTS.

   Episodes starting with the shoe in the hand (256 of 256):
     episodes in which the shoe was placed (keypoint_dist <= 0.03 m) at least once  0.57
     closest the shoe came to the target in those episodes                        0.026 m median
   On the 15 452 steps where the shoe was placed:
     resting on the rack                                                          0.998 of those steps
     still                                                                        0.032
     open fraction as the takeover measures it                                    0.33 median, never >= 0.9
     released (palm_shoe_dist > release_radius)                                   0.005
   `ctx.retracting` was never true in the whole run, so `ctx.home` and `ctx.success` could never become true.

   Why: the takeover starts only when the hand's open fraction reaches 0.9. The previous reward measured openness with
   its own formula built from `hand_target_norm` and a per-joint direction guess; the quantity the takeover actually
   compares was not in the context. It now is: `ctx.open_frac` is EXACTLY that quantity, 0 at the grip pose, 1 at the
   open hand of a reset.

WHAT IS LEARNED AND MUST NOT BE LOST: approaching the shoe from the rest posture, and — once the shoe is in the hand —
carrying it to the rack and setting it within 3 cm of the target on the rack top (0.57 of such episodes).
