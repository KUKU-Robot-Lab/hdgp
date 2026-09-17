You are an expert in robotics, reinforcement learning and code generation.

We control a bimanual robot: two 7-DOF OpenArm arms, each carrying an Inspire RH56F1 five-finger \
hand, standing at a table. The RH56F1 is an UNDER-ACTUATED hand: only 6 joints per hand are driven \
(thumb abduction, thumb flexion, and one flexion drive for each of the index, middle, ring and pinky \
fingers); the remaining finger joints follow the driven ones through fixed mechanical couplings, so a \
finger always curls as a whole. The RIGHT arm is the "source" arm (src_*), the LEFT arm is the \
"receiver" arm (rcv_*). Two identical cups stand on the table: the source cup (in front of the right \
hand, filled with 20 small beads) and the receiver cup (in front of the left hand, empty). \
Positions are in metres in a frame whose origin is at the robot base; +z is up, +x is forward (away \
from the robot), +y is to the robot's left. The table top is at z = table_z.

The action space is a normalized `Box(-1, 1, (24,), float32)`:
  actions[0:6]   = source palm 6-DoF target offset (xyz + yaw/pitch/roll) — the arm is moved \
by a geometric-fabrics controller toward this target
  actions[6:12]  = source hand closure commands, one per driven joint \
(thumb abduction, thumb flexion, index, middle, ring, pinky; 0 = open, 1 = closed)
  actions[12:18] = receiver palm 6-DoF target offset
  actions[18:24] = receiver hand closure commands
The hand controller stops a finger automatically once one of its links touches its own cup \
(contact freeze); opening is always allowed. Fingers can only close when the palm is near its cup. \
The "cups" are slim shaker bodies about 6 cm in diameter and 11 cm tall, and the hand is small: a \
FINGERTIP (precision) grasp — thumb tip on one side, the tips of some fingers on the other — is the \
expected way to hold the cup; a full wrap-around power grasp is NOT required and usually not \
possible. The cups are light, so a finger brushing the cup easily knocks it over: approach slowly \
and touch it only with the fingertips.

Now I want you to help me write a reward function for reinforcement learning.

Typically, the reward function of a manipulation task consists of these parts (some are optional — include them only if really necessary):
1. the distance between the robot's hand and the target object (one term per arm)
2. the difference between the object's current state and its goal state
3. regularization of the robot's action
4. [optional] extra constraints on the target objects implied by the task (e.g. keep the receiver cup upright, do not spill)
5. [optional] extra constraints on the robot implied by the task

The reward function receives a single argument `ctx`, an instance of this class (all positions env-local, metres; angles rad; forces N):

```python
@dataclass(frozen=True)
class RewardContext:
    # ---- 상수 (에피소드 불변, python float) ---------------------------------------
    table_z: float            # 테이블 상면 높이 [m]
    cup_radius: float         # 컵 안쪽 반경 [m]
    cup_mouth_z: float        # 컵 원점(무게중심 부근) → 림(개구) 높이 [m]
    cup_bottom_z: float       # 컵 원점 → 바닥 높이 [m] (음수)
    num_beads: int            # 소스 컵에 든 비드 수

    # ---- 소스 팔 (비드가 든 컵을 잡아 붓는 팔) ------------------------------------
    src_palm_pos: torch.Tensor          # (N,3) 손바닥 위치
    src_palm_axes: torch.Tensor         # (N,6) 손바닥 회전행렬의 열 0(손바닥 법선)·열 1
    src_tips_pos: torch.Tensor          # (N,F,3) 손끝 위치 (F=손가락 수, 0번=엄지)
    src_hand_closure: torch.Tensor      # (N,) 실측 평균 손 폐쇄도 [0,1] (0=펴짐, 1=완전 파지)
    src_finger_force: torch.Tensor      # (N,F) 손가락별 소스 컵 접촉력 [N]
    src_palm_force: torch.Tensor        # (N,) 손바닥의 소스 컵 접촉력 [N]
    src_grasped: torch.Tensor           # (N,) bool 엄지 AND 다른 손가락이 동시에 소스 컵에 접촉
    src_arm_qd: torch.Tensor            # (N,A) 팔 관절 속도 [rad/s]
    # ---- 리시버 팔 (빈 컵을 잡아 받는 팔) -------------------------------------------
    rcv_palm_pos: torch.Tensor          # (N,3)
    rcv_palm_axes: torch.Tensor         # (N,6)
    rcv_tips_pos: torch.Tensor          # (N,F,3)
    rcv_hand_closure: torch.Tensor      # (N,)
    rcv_finger_force: torch.Tensor      # (N,F) 리시버 컵 접촉력
    rcv_palm_force: torch.Tensor        # (N,)
    rcv_grasped: torch.Tensor           # (N,) bool
    rcv_arm_qd: torch.Tensor            # (N,A)

    # ---- 소스 컵 (비드가 든 컵) ---------------------------------------------------
    src_cup_pos: torch.Tensor           # (N,3) 컵 원점 위치
    src_cup_quat: torch.Tensor          # (N,4) 자세 (w,x,y,z)
    src_cup_up: torch.Tensor            # (N,3) 컵의 위 방향 단위벡터 (직립이면 (0,0,1))
    src_cup_tilt: torch.Tensor          # (N,) 직립 대비 기울기 [rad] (0=직립, π/2=수평)
    src_cup_mouth_pos: torch.Tensor     # (N,3) 컵 개구(림) 중심 위치
    src_cup_lin_vel: torch.Tensor       # (N,3) [m/s]
    src_cup_ang_vel: torch.Tensor       # (N,3) [rad/s]
    src_cup_spawn_pos: torch.Tensor     # (N,3) 에피소드 시작 시 컵 위치(테이블 위)
    # ---- 리시버 컵 (빈 컵) --------------------------------------------------------
    rcv_cup_pos: torch.Tensor           # (N,3)
    rcv_cup_quat: torch.Tensor          # (N,4)
    rcv_cup_up: torch.Tensor            # (N,3)
    rcv_cup_tilt: torch.Tensor          # (N,)
    rcv_cup_mouth_pos: torch.Tensor     # (N,3)
    rcv_cup_lin_vel: torch.Tensor       # (N,3)
    rcv_cup_ang_vel: torch.Tensor       # (N,3)
    rcv_cup_spawn_pos: torch.Tensor     # (N,3)

    # ---- 비드 ----------------------------------------------------------------------
    bead_in_source_frac: torch.Tensor   # (N,) 소스 컵 안에 있는 비드 비율 [0,1]
    bead_in_target_frac: torch.Tensor   # (N,) 리시버 컵 안에 있고 소스 컵 밖인 비드 비율 [0,1] (소스 컵째 끼워 넣은 비드는 0)
    bead_spill_frac: torch.Tensor       # (N,) 어느 컵에도 없이 바닥/테이블로 떨어진 비율 [0,1]
    bead_centroid: torch.Tensor         # (N,3) 비드 무게중심 위치
    d_in_target: torch.Tensor           # (N,) 이번 스텝 bead_in_target_frac 증분 (Δ, 음수 가능)
    d_spill: torch.Tensor               # (N,) 이번 스텝 bead_spill_frac 증분

    # ---- 충돌 (실기 안전 — 09.14) ---------------------------------------------------------
    cup_cup_force: torch.Tensor         # (N,) 두 컵이 서로 부딪히는 접촉력 [N] (0 = 안 닿음)
    src_hand_foreign_force: torch.Tensor  # (N,) 소스 손이 **자기 컵 외**(상대 손·상대 컵·테이블)에 닿는 힘 [N]
    rcv_hand_foreign_force: torch.Tensor  # (N,) 리시버 손이 자기 컵 외에 닿는 힘 [N]

    # ---- 과제 판정 / 시간 ------------------------------------------------------------
    cups_nested: torch.Tensor           # (N,) bool 두 컵 원점 거리 < 9 cm — 소스 컵이 리시버 컵에 끼워져 있음(붓기가 아니라 성공 무효)
    success: torch.Tensor               # (N,) bool 성공 조건 충족 (env 가 판정, 보상이 바꿀 수 없음; cups_nested 면 항상 False)
    episode_progress: torch.Tensor      # (N,) 에피소드 진행도 [0,1]

    # ---- 액션 ------------------------------------------------------------------------
    actions: torch.Tensor               # (N,Dact) 이번 스텝 액션 [-1,1]
    prev_actions: torch.Tensor          # (N,Dact) 직전 스텝 액션
```

Additional knowledge:
1. Everything is BATCHED: every tensor field has a leading dimension N (number of parallel environments). The function must return a reward tensor of shape (N,) — never a Python float — and a dict of named component tensors, each of shape (N,). Use only `torch` and `math`; do not import anything else.
2. Write staged rewards with tensor masks, not `if`/`else` on tensors: `stage2 = torch.where(ctx.src_grasped, r_lift, torch.zeros_like(r_lift))`.
3. Prefer bounded, smooth shaping: `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead of raw negative distances, so that no single term dominates.
4. `ctx.src_grasped` / `ctx.rcv_grasped` are True when the thumb AND another finger both press on that hand's own cup — this is the grasp-established signal. `ctx.*_hand_closure` is the measured closure (0 = open, 1 = fully closed).
5. Cup tilt: `ctx.src_cup_tilt` is the angle between the cup's up axis and world +z (0 = upright, π/2 = horizontal). The source cup holds a fixed number of beads in this track. The first bead leaves at about 70° with the cup full, and a full cup has lost a fifth of its beads by 80° and half by 86°. `ctx.src_cup_mouth_pos` and `ctx.rcv_cup_mouth_pos` are the rim centres.
6. Bead bookkeeping: `bead_in_target_frac` rises as beads land in the receiver cup; `bead_spill_frac` counts beads lost outside both cups (permanent). `d_in_target` and `d_spill` are this step's increments — reward INCREMENTS of beads transferred rather than the level, otherwise the policy is paid for standing still with a filled cup.
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, the receiver cup held nearly upright — `ctx.rcv_cup_tilt` at most 20° — and the cups NOT nested). You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air.
8. Height above the table: `ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]` is how far the source cup has been lifted (0 while it rests on the table).
9. Do not keep any state between calls (no globals, no attributes); the function must be pure.
10. Each component you return is logged separately during training and shown back to you after training, so name them meaningfully (e.g. "approach_src", "grasp_rcv", "pour_delta").
11. This policy will be deployed on the real robot, so collisions are a safety hazard: `ctx.cup_cup_force` is the contact force between the two cups, and `ctx.src_hand_foreign_force` / `ctx.rcv_hand_foreign_force` are the forces each hand exerts on anything other than its own cup (the other hand, the other cup, the table). During a correct pour the cups do not touch each other and the hands touch only their own cup; penalise these forces (bounded, e.g. `torch.tanh(f / 5.0)`) so the policy learns to keep clearance. Cup poses seen by the policy are delayed and noisy as on the real perception system.

I want it to fulfil the following task: Using both arms, grasp each cup (a fingertip grasp is fine, no wrap-around needed): the right hand grasps the source cup (which contains the beads) and the left hand grasps the empty receiver cup. Lift both cups off the table, bring the mouth of the source cup over the mouth of the receiver cup, and tilt the source cup so that the beads pour into the receiver cup. Keep the receiver cup upright, do not drop either cup, and spill as few beads as possible. The task is finished when at least half of the beads are inside the receiver cup with little spill.
1. Please think step by step and tell me what this task means and which stages the robot must go through.
2. Then write a function with exactly this signature:
```python
def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, {"component_name": component_tensor, ...}
```
   where `reward` has shape (N,) and every dict value has shape (N,).
3. Take care of tensor shapes and types; never call methods that are not listed in the class definition above.
4. Put the final code in ONE ```python fenced block that starts with `import torch` and `import math` and contains only the function (plus small helper functions if needed). Add short comments explaining weights, e.g. `# weight 0.5 works together with the pour term 5.0 — pouring must dominate once grasped`.

The previous reward function was:
```python
import torch
import math


def _near(d: torch.Tensor, k: float) -> torch.Tensor:
    """Bounded closeness in (0, 1]: 1 at d = 0, never exactly 0 -> a gradient exists at ANY distance."""
    return torch.exp(-k * d)


def _band_err(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """0 inside [lo, hi], linear distance outside."""
    return torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)


def _cyl(points: torch.Tensor, cup_pos: torch.Tensor, cup_up: torch.Tensor):
    """Cup-cylinder coords of (N,K,3) points: axial height (N,K), radial vector (N,K,3), radius (N,K)."""
    rel = points - cup_pos[:, None, :]
    up = cup_up[:, None, :]
    axial = (rel * up).sum(dim=-1)
    rvec = rel - axial[..., None] * up
    radial = torch.norm(rvec, dim=-1)
    return axial, rvec, radial


def _axis_to_segments(rv_a: torch.Tensor, rv_b: torch.Tensor) -> torch.Tensor:
    """Distance (N,K) from the cup AXIS to each segment joining rv_a (N,1,3) to rv_b (N,K,3), measured in
    the plane normal to the axis. ~0 ONLY when the axis lies BETWEEN the two tips (= cup inside the jaw)."""
    seg = rv_b - rv_a
    u = torch.clamp(-(rv_a * seg).sum(dim=-1) / ((seg * seg).sum(dim=-1) + 1e-8), 0.0, 1.0)
    closest = rv_a + u[..., None] * seg
    return torch.norm(closest, dim=-1)


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, palm_force, closure,
                cup_pos, cup_up, grasped, a_hand):
    """ONE nested pre-grasp ladder + the contact terms for a single hand.

    Round 9 paid `approach` beside `spread` beside `open` as independent incomes, so the policy
    maximised whichever was cheapest and ignored the rest (12 cm away, fingers flat, 95 % of income).
    Here every pre-grasp condition is a FACTOR of one product chain: L_k = L_{k-1} * c_k. In a sum the
    optimum of a partial sum is reached by maxing the cheap terms; in a product the gradient always
    points at the WEAKEST factor, so the cheap-subset strategy that produced both plateaus is gone.
    """
    dtype = cup_pos.dtype
    r_in = ctx.cup_radius
    r_wall = r_in + 0.004                      # outer wall ~ inner radius + wall thickness
    w_cup = max(2.0 * r_wall, 0.055)           # the cup is ~57 mm across; floored so the target can't collapse
    mouth = ctx.cup_mouth_z
    bot = ctx.cup_bottom_z
    tip_lo = bot + 0.025                       # tips >= 2.5 cm above the cup bottom (table clearance)
    tip_hi = mouth - 0.012                     # tips >= 1.2 cm below the rim (no rim hooking)

    t_ax, t_rv, t_rad = _cyl(tips, cup_pos, cup_up)          # (N,F), (N,F,3), (N,F)
    t_dir = t_rv / (t_rad[..., None] + 1e-6)

    # ---- per-tip distance to the GRASPABLE BAND of the outer wall.
    #      NOTE the deliberate absence of any hard "too far" clamp: round 9's `r_ok` hit exactly 0
    #      beyond 8.2 cm, which is why the 12 cm plateau had no gradient at all out of it. Every
    #      "far" falloff here is an exp with global support.
    rad_out = torch.clamp(t_rad - (r_wall + 0.003), min=0.0)          # outside the wall
    rad_in = torch.clamp(r_in - t_rad, min=0.0) * 3.0                 # inside the bore reads as 3x the error
    h_err = _band_err(t_ax, tip_lo, tip_hi)
    surf = torch.sqrt(rad_out ** 2 + rad_in ** 2 + h_err ** 2 + 1e-8)  # (N,F)
    f_out = torch.clamp((t_rad - r_in) / 0.004, 0.0, 1.0)              # 0 for a tip dipped into the bore

    # ---- PAD DIRECTION, SIGNED (kept from round 9 -- a working gate).
    #      palm_axes[:, 0:3] is the palm normal, i.e. the direction the pad faces.
    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup = cup_pos - palm_pos
    u3 = to_cup / (torch.norm(to_cup, dim=-1, keepdim=True) + 1e-6)
    facing = (normal * u3).sum(dim=-1)                       # +1 pad at the cup, -1 back of the hand
    pad = torch.clamp(facing / 0.35, 0.0, 1.0)               # HARD gate, no floor: multiplies the WHOLE ladder
    back = torch.clamp(-facing, 0.0, 1.0)                    # 1 = the cup is behind the hand

    # ================= the six factors of the ladder, all per (thumb, finger j) PAIR =================
    th_surf = surf[:, 0:1]                                   # (N,1)
    fg_surf = surf[:, 1:]                                    # (N,F-1)
    # max(), not mean(): BOTH sides of the pinch must arrive. A wrist parked near the cup, or one
    # finger poking at it, leaves the other side far and the pair distance stays large.
    d_pair = torch.maximum(th_surf, fg_surf)                                     # (N,F-1)
    c1 = 0.5 * _near(d_pair, 4.0) + 0.5 * _near(d_pair, 25.0)   # reach: coarse pull + fine pull

    # c2 = pad (scalar per env, broadcast below)

    gap = torch.norm(tips[:, 0:1, :] - tips[:, 1:, :], dim=-1)                   # (N,F-1)
    # OPENING. Cubed sub-threshold ramp: a 45 mm hand keeps a strictly positive d(reward)/d(gap)
    # but collects only ~0.2 of the factor, so it can never be a resting place.
    c3 = (torch.clamp((gap - 0.020) / max(w_cup + 0.006 - 0.020, 1e-3), 0.0, 1.0) ** 3
          * torch.clamp((0.145 - gap) / 0.030, 0.0, 1.0))      # upper clamp: no hyperextension farming

    h_pair = torch.maximum(h_err[:, 0:1], h_err[:, 1:])
    c4 = _near(h_pair, 25.0)                                   # both tips at wall height

    # STRADDLE -- this is `cup_in_pocket`, the metric that was 0 for all of round 9 while the gap sat at
    # 84-93 mm. It is ~0 for an opening beside the cup and 1 only when the cup axis is between the tips.
    # This factor, not a separate `spread` income, is what makes opening pay: it is the geometry that
    # already had to be satisfied, so it cannot become a rung of its own.
    seg = _axis_to_segments(t_rv[:, 0:1, :], t_rv[:, 1:, :])                     # (N,F-1)
    adm = f_out[:, 0:1] * f_out[:, 1:]
    c5 = _near(seg, 45.0) * adm

    opp = -(t_dir[:, 1:, :] * t_dir[:, 0:1, :]).sum(dim=-1)                      # (N,F-1) in [-1,1]
    c6 = torch.clamp((opp - 0.3) / 0.5, 0.0, 1.0)       # HARD, no floor (kept): 107 deg -> 0, 120 deg -> 1
    c7 = (0.35 * _near(d_pair, 12.0) + 0.65 * _near(d_pair, 60.0)) * adm         # tips in contact range

    # ---- a hand laid ON the cup with the thumb tucked in among the fingers is not a jaw, however wide
    #      it measures (kept from round 9). Zeroes the whole ladder there.
    at_cup = _near(surf.min(dim=-1).values, 15.0)
    u_th = t_dir[:, 0, :]
    u_fm = t_dir[:, 1:3, :].mean(dim=1)
    u_fm = u_fm / (torch.norm(u_fm, dim=-1, keepdim=True) + 1e-6)
    opp_mid = -(u_th * u_fm).sum(dim=-1)                       # -1 = thumb among the fingers
    same_side = torch.clamp((-0.1 - opp_mid) / 0.5, 0.0, 1.0)
    jaw_ok = 1.0 - at_cup * same_side

    # ================= ONE nested ladder. L_k requires every condition of L_{k-1}. =================
    p = pad[:, None]
    L1 = c1 * p                      # arrived, pad toward the cup
    L2 = L1 * c3                     # ... opened wider than the cup
    L3 = L2 * c4                     # ... at wall height
    L4 = L3 * c5                     # ... the cup is between the tips   <- cup_in_pocket
    L5 = L4 * c6                     # ... genuinely opposed
    L6 = L5 * c7                     # ... on the wall: one flex from the grasp flag
    # Back-loaded weights (max 2.00). Everything payable WITHOUT the cup in the jaw is capped at 0.50,
    # and 1.10 of the 2.00 needs a fingertip in contact range: the best standing posture the ladder
    # admits is already the grasp precondition.
    j = jaw_ok[:, None]
    lad1 = (0.10 * L1 * j).max(dim=-1).values
    lad2 = (0.15 * L2 * j).max(dim=-1).values
    lad3 = (0.25 * L3 * j).max(dim=-1).values
    lad4 = (0.40 * L4 * j).max(dim=-1).values
    lad5 = (0.50 * L5 * j).max(dim=-1).values
    lad6 = (0.60 * L6 * j).max(dim=-1).values
    ready = (L6 * j).max(dim=-1).values

    # ================= contact: only an OPPOSING finger on the far wall counts =================
    thumb_low = torch.clamp((mouth - t_ax[:, 0]) / 0.015, 0.0, 1.0)
    opp_best = c6.max(dim=-1).values
    f_t = finger_force[:, 0]
    f_o = (finger_force[:, 1:] * c6 * adm).max(dim=-1).values     # force only counts where opposition does
    f_min = torch.minimum(f_t, f_o)
    # force sensors report contact in ANY direction, so without `pad` the back of the fingers reads as
    # grasp progress (round 8's failure). q is hard: no pad, no opposition, no thumb below the rim -> no pay.
    q_grasp = pad * opp_best * thumb_low * jaw_ok
    touch = 0.5 * ((1.0 - torch.exp(-f_t / 0.3)) + (1.0 - torch.exp(-f_o / 0.3))) * q_grasp
    grip = (0.5 * (1.0 - torch.exp(-f_min / 0.25)) + 0.5 * torch.clamp(f_min / 1.0, 0.0, 1.0)) * q_grasp
    grasped_f = grasped.to(dtype)
    grasp = grasped_f * q_grasp                                   # the env flag alone is not enough
    hold_soft = torch.clamp((f_min - 0.3) / 0.7, 0.0, 1.0)
    hold = torch.maximum(grasped_f, hold_soft) * q_grasp
    q = 0.3 + 0.7 * q_grasp

    # squeeze intent from the COMMANDED closure, only inside a pose that can actually trap the cup
    cmd = 0.5 * (torch.clamp(a_hand, -1.0, 1.0) + 1.0)
    cmd_close = (cmd[:, 0] + cmd[:, 1] + cmd[:, 2:6].mean(dim=-1)) / 3.0
    squeeze = ready * cmd_close

    # ---- failure postures
    # air pinch = closing on nothing AT the cup. Measured on the PINCH pair (thumb vs index/middle)
    # only: the thumb-to-pinky distance is naturally small, and taking the min over all four pairs
    # would pay the policy to splay every finger -- which is round 9's exact pathology.
    far = 1.0 - _near(surf.min(dim=-1).values, 20.0)
    gap_pinch = gap[:, 0:2].min(dim=-1).values
    air_pinch = torch.clamp((w_cup - 0.005 - gap_pinch) / 0.025, 0.0, 1.0) * (1.0 - far)
    curl_far = torch.clamp((closure - 0.3) / 0.4, 0.0, 1.0) * far
    over_rim = torch.clamp((t_ax[:, 0] - (mouth - 0.005)) / 0.015, 0.0, 1.0)
    rim_hook = over_rim * (t_rad[:, 0] < r_wall + 0.025).to(dtype)
    # fingers dipped INSIDE the bore (gripping the cup from the inside is not the task)
    inside = (torch.clamp((r_in - t_rad) / 0.006, 0.0, 1.0)
              * torch.clamp((mouth + 0.01 - t_ax) / 0.01, 0.0, 1.0)
              * torch.clamp((t_ax - bot) / 0.01, 0.0, 1.0))
    bore = inside.max(dim=-1).values
    # pressing a cup with a palm turned away, and loitering near it back-first
    dorsal = back * (0.15 * at_cup + 1.2 * torch.tanh((finger_force.sum(dim=-1) + palm_force) / 1.0))

    return {
        "lad1": lad1, "lad2": lad2, "lad3": lad3, "lad4": lad4, "lad5": lad5, "lad6": lad6,
        "ready": ready, "touch": touch, "grip": grip, "grasp": grasp, "hold": hold, "q": q,
        "squeeze": squeeze, "air_pinch": air_pinch, "curl_far": curl_far, "rim_hook": rim_hook,
        "bore": bore, "dorsal": dorsal,
    }


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)
    dtype = zero.dtype
    act = ctx.actions

    s = _hand_terms(ctx, ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                    ctx.src_palm_force, ctx.src_hand_closure, ctx.src_cup_pos, ctx.src_cup_up,
                    ctx.src_grasped, act[:, 6:12])
    r = _hand_terms(ctx, ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                    ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                    ctx.rcv_grasped, act[:, 18:24])
    hold_s, hold_r = s["hold"], r["hold"]
    q_s, q_r = s["q"], r["q"]

    # ------------------------------------------- stages 0-2: the single pre-grasp ladder (max 2.0/hand)
    lad1_src, lad1_rcv = s["lad1"], r["lad1"]
    lad2_src, lad2_rcv = s["lad2"], r["lad2"]
    lad3_src, lad3_rcv = s["lad3"], r["lad3"]
    lad4_src, lad4_rcv = s["lad4"], r["lad4"]
    lad5_src, lad5_rcv = s["lad5"], r["lad5"]
    lad6_src, lad6_rcv = s["lad6"], r["lad6"]
    # min(), not sum(): one hand improving while the other goes backwards must not pay.
    ready_both = 1.0 * torch.minimum(s["ready"], r["ready"])

    # ------------------------------------------- stage 3: contact and the grasp flag
    # 1.5 + 3.0 + 5.0 = 9.5 per hand, ~5x the ENTIRE ladder: reaching the top of the ladder is worth
    # far less than the single finger flex that turns it into a grasp.
    touch_src, touch_rcv = 1.5 * s["touch"], 1.5 * r["touch"]
    grip_src, grip_rcv = 3.0 * s["grip"], 3.0 * r["grip"]
    grasp_src, grasp_rcv = 5.0 * s["grasp"], 5.0 * r["grasp"]
    grasp_both = 4.0 * s["grasp"] * r["grasp"]
    squeeze_src, squeeze_rcv = 0.6 * s["squeeze"], 0.6 * r["squeeze"]
    air_pinch_pen = -0.5 * (s["air_pinch"] * (1.0 - hold_s) + r["air_pinch"] * (1.0 - hold_r))
    curl_far_pen = -0.3 * (s["curl_far"] + r["curl_far"])
    rim_hook_pen = -0.5 * (s["rim_hook"] + r["rim_hook"])
    bore_pen = -0.6 * (s["bore"] + r["bore"])
    dorsal_pen = -1.5 * (s["dorsal"] + r["dorsal"])     # the back of the hand on the cup is never progress

    # ------------------------------------------- stage 4: lift
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_src = 8.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.025)
    lift_rcv = 8.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.025)
    lifted = hold_s * (h_src > 0.03).to(dtype) * hold_r * (h_rcv > 0.03).to(dtype)

    # ------------------------------------------- stage 5: carry together (tilt-aware target)
    tilt = ctx.src_cup_tilt
    pour_prog = torch.clamp((tilt - 0.35) / 1.2, 0.0, 1.0)
    tgt_dxy = 0.10 * (1.0 - pour_prog)         # upright: mouths 10 cm apart, closing as the source tilts
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    dxy_err = torch.abs(mouth_dxy - tgt_dxy)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = _band_err(mouth_dz, 0.04, 0.12)                 # source mouth 4..12 cm higher
    carry_q = (0.5 * torch.clamp(1.0 - dxy_err / 0.30, 0.0, 1.0) + 0.5 * _near(dxy_err, 30.0)) * _near(dz_err, 15.0)
    align = 8.0 * lifted * q_s * carry_q

    # ------------------------------------------- stage 6: tilt and pour
    zone = _near(dxy_err, 40.0) * _near(dz_err, 40.0)
    tilt_r = 7.0 * lifted * q_s * zone * torch.clamp(tilt / 2.0, 0.0, 1.0)
    pour_delta = 200.0 * ctx.d_in_target       # +10 per transferred bead; increments, never the level
    spill_delta = -100.0 * ctx.d_spill         # -5 per spilled bead: costly, never worth refusing to pour

    # ------------------------------------------- stage 7: success
    success = 30.0 * ctx.success.to(dtype)

    # ------------------------------------------- constraints (guards, not shaping)
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    pre_tilt_pen = -1.0 * (1.0 - lifted * zone) * has_beads * torch.tanh(2.0 * torch.clamp(tilt - 0.8, min=0.0))
    rcv_upright_pen = -0.8 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.15, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    disturb_pen = -0.3 * (src_free * torch.tanh(3.0 * torch.clamp(ctx.src_cup_tilt - 0.3, min=0.0))
                          + rcv_free * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.3, min=0.0))) \
                  - 0.2 * (src_free * torch.tanh(torch.clamp(disp_src - 0.03, min=0.0) / 0.03)
                           + rcv_free * torch.tanh(torch.clamp(disp_rcv - 0.03, min=0.0) / 0.03))
    drop_pen = -1.0 * (src_free * (h_src > 0.015).to(dtype) * torch.tanh(torch.clamp(-ctx.src_cup_lin_vel[:, 2], min=0.0) / 0.5)
                       + rcv_free * (h_rcv > 0.015).to(dtype) * torch.tanh(torch.clamp(-ctx.rcv_cup_lin_vel[:, 2], min=0.0) / 0.5))
    cup_speed_pen = -0.2 * (torch.tanh(torch.clamp(torch.norm(ctx.src_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4)
                            + torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4))
    nested_pen = -2.0 * ctx.cups_nested.to(dtype)
    # real-robot safety: the cups never touch each other and each hand touches only its own cup
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    hand_foreign_pen = -2.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    # the palm is not part of a fingertip grasp: pushing the cup with it costs
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # ------------------------------------------- regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "lad1_reach_src": lad1_src, "lad1_reach_rcv": lad1_rcv,
        "lad2_open_src": lad2_src, "lad2_open_rcv": lad2_rcv,
        "lad3_height_src": lad3_src, "lad3_height_rcv": lad3_rcv,
        "lad4_straddle_src": lad4_src, "lad4_straddle_rcv": lad4_rcv,
        "lad5_oppose_src": lad5_src, "lad5_oppose_rcv": lad5_rcv,
        "lad6_ready_src": lad6_src, "lad6_ready_rcv": lad6_rcv,
        "ready_both": ready_both,
        "touch_src": touch_src, "touch_rcv": touch_rcv,
        "grip_src": grip_src, "grip_rcv": grip_rcv,
        "grasp_src": grasp_src, "grasp_rcv": grasp_rcv, "grasp_both": grasp_both,
        "squeeze_src": squeeze_src, "squeeze_rcv": squeeze_rcv,
        "air_pinch_pen": air_pinch_pen, "curl_far_pen": curl_far_pen,
        "rim_hook_pen": rim_hook_pen, "bore_pen": bore_pen, "dorsal_pen": dorsal_pen,
        "lift_src": lift_src, "lift_rcv": lift_rcv,
        "align": align, "tilt": tilt_r,
        "pour_delta": pour_delta, "spill_delta": spill_delta, "success": success,
        "pre_tilt_pen": pre_tilt_pen, "rcv_upright_pen": rcv_upright_pen,
        "disturb_pen": disturb_pen, "drop_pen": drop_pen, "cup_speed_pen": cup_speed_pen,
        "nested_pen": nested_pen, "cup_collision_pen": cup_collision_pen,
        "hand_foreign_pen": hand_foreign_pen, "palm_push_pen": palm_push_pen,
        "action_rate_pen": action_rate_pen, "arm_speed_pen": arm_speed_pen,
    }
    reward = zero
    for v in components.values():
        reward = reward + v
    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/spill: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/drop: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
done/mimic_runaway: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.44e-05 · max 0.000977
episode_lengths/step: [52.3, 68.5, 122, 863, 849, 861, 851, 891, 819, 817]  min 52.3 · mean 649 · max 891
reward/action_rate_pen: [-0.0167, -0.02, -0.021, -0.0191, -0.0204, -0.0188, -0.0212, -0.0145, -0.0207, -0.0213]  min -0.0219 · mean -0.02 · max -0.0145
reward/air_pinch_pen: [0, 0, 0, -8.3e-06, 0, -0.00188, -6.31e-05, -1.23e-05, -0.00256, -0.000115]  min -0.00579 · mean -0.000912 · max 0
reward/align: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/arm_speed_pen: [-0.0259, -0.00472, -0.00506, -0.0049, -0.00612, -0.005, -0.00585, -0.0462, -0.00563, -0.00738]  min -0.0462 · mean -0.0081 · max -0.00468
reward/bore_pen: [-0.00363, -2.43e-06, -0.000261, -0.000256, -0.000123, -0.000115, -0.000426, -0.000371, -0.00124, -0.000433]  min -0.00363 · mean -0.000454 · max 0
reward/cup_collision_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/cup_speed_pen: [-4.74e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -4.74e-05 · mean -1.23e-06 · max 0
reward/curl_far_pen: [0, -1.21e-05, -7.7e-05, 0, -5.16e-07, 0, -2.72e-07, 0, -1.01e-05, -1.17e-08]  min -0.000203 · mean -1.8e-05 · max 0
reward/disturb_pen: [-0.000512, -0.000122, -0.000119, 0, 0, -0.000319, 0, -2.1e-05, -0.000256, 0]  min -0.000584 · mean -6.94e-05 · max 0
reward/dorsal_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -7.61e-07 · mean -1.3e-08 · max 0
reward/drop_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_both: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.46e-05 · max 0.00303
reward/grip_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grip_src: [0, 0, 0, 6.32e-05, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.43e-05 · max 0.00105
reward/hand_foreign_pen: [-0.00433, 0, 0, 0, 0, -0.00255, 0, -0.00627, -8.56e-05, 0]  min -0.0934 · mean -0.00332 · max 0
reward/lad1_reach_rcv: [0.0238, 0.0251, 0.0249, 0.0255, 0.0249, 0.0265, 0.0252, 0.0256, 0.0258, 0.0254]  min 0.0238 · mean 0.0256 · max 0.027
reward/lad1_reach_src: [0.0836, 0.0966, 0.0975, 0.0921, 0.0944, 0.0959, 0.0961, 0.0654, 0.0961, 0.0974]  min 0.0435 · mean 0.0937 · max 0.0979
reward/lad2_open_rcv: [0.0356, 0.0372, 0.0374, 0.0352, 0.0367, 0.0364, 0.0344, 0.0384, 0.0351, 0.0346]  min 0.0344 · mean 0.036 · max 0.0389
reward/lad2_open_src: [0.125, 0.144, 0.146, 0.138, 0.141, 0.144, 0.144, 0.0981, 0.144, 0.146]  min 0.0652 · mean 0.14 · max 0.146
reward/lad3_height_rcv: [0.0545, 0.0553, 0.0563, 0.0531, 0.0471, 0.0602, 0.0418, 0.0552, 0.0486, 0.0415]  min 0.0415 · mean 0.0514 · max 0.062
reward/lad3_height_src: [0.207, 0.241, 0.243, 0.229, 0.235, 0.24, 0.24, 0.154, 0.24, 0.243]  min 0.0977 · mean 0.233 · max 0.244
reward/lad4_straddle_rcv: [0.000206, 2.38e-05, 2.47e-05, 2.83e-05, 1.96e-05, 6.95e-05, 1.74e-05, 8.32e-05, 2.94e-05, 2.09e-05]  min 1.67e-05 · mean 4.12e-05 · max 0.000206
reward/lad4_straddle_src: [0.292, 0.334, 0.336, 0.309, 0.324, 0.341, 0.318, 0.081, 0.323, 0.324]  min 0.0328 · mean 0.315 · max 0.341
reward/lad5_oppose_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lad5_oppose_src: [0.365, 0.418, 0.42, 0.386, 0.405, 0.426, 0.396, 0.0498, 0.402, 0.404]  min 0.0152 · mean 0.392 · max 0.426
reward/lad6_ready_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lad6_ready_src: [0.27, 0.435, 0.443, 0.411, 0.445, 0.444, 0.44, 0.03, 0.45, 0.452]  min 0.0163 · mean 0.417 · max 0.467
reward/lift_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.11e-06 · max 0.000212
reward/nested_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/palm_push_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pre_tilt_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.000626 · mean -1.39e-05 · max 0
reward/rcv_upright_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/ready_both: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/rim_hook_pen: [-0.00346, -0.000316, 0, -0.000809, -0.000337, -0.000488, -1.64e-06, -0.0237, -0.00104, -0.000488]  min -0.0372 · mean -0.00205 · max 0
reward/spill_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/squeeze_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/squeeze_src: [0.112, 0.213, 0.226, 0.173, 0.192, 0.174, 0.167, 0.0111, 0.17, 0.191]  min 0.00662 · mean 0.174 · max 0.227
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [1.55, 2.21, 2.18, 2.05, 2.15, 2.24, 2.08, 0.522, 2.15, 2.18]  min 0.292 · mean 2.07 · max 2.28
reward/touch_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/touch_src: [0.0402, 0.24, 0.172, 0.223, 0.228, 0.287, 0.21, 0.00634, 0.244, 0.252]  min 0.00347 · mean 0.228 · max 0.325
rewards/step: [-1.94, -6.71, 106, 1.76e+03, 1.72e+03, 1.7e+03, 1.68e+03, 1.77e+03, 1.62e+03, 1.61e+03]  min -6.71 · mean 1.26e+03 · max 1.8e+03
task/aim_dist: [0.318, 0.316, 0.317, 0.316, 0.316, 0.314, 0.317, 0.32, 0.317, 0.317]  min 0.313 · mean 0.316 · max 0.321
task/cup_collision_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/cups_center_dist: [0.319, 0.318, 0.318, 0.318, 0.318, 0.317, 0.318, 0.32, 0.318, 0.318]  min 0.316 · mean 0.318 · max 0.321
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/rcv_cup_lift: [8.1e-06, 7.94e-06, 8.05e-06, 8.8e-06, 8.33e-06, 8.16e-06, 7.97e-06, 9.8e-06, 8.37e-06, 7.91e-06]  min 7.83e-06 · mean 8.71e-06 · max 2.16e-05
task/rcv_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/rcv_hand_foreign_rate: [0.00293, 0, 0, 0, 0, 0, 0, 0.000977, 0, 0]  min 0 · mean 0.00235 · max 0.0898
task/src_cup_lift: [0.000402, 0.000667, 0.000456, 0.000615, 0.00058, 0.000819, 0.000365, 4.37e-05, 0.000539, 0.000481]  min 3.04e-05 · mean 0.000589 · max 0.00111
task/src_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.44e-05 · max 0.000977
task/src_hand_foreign_rate: [0, 0, 0, 0, 0, 0.00195, 0, 0.00293, 0, 0]  min 0 · mean 0.00035 · max 0.00293
task/src_tilt_deg: [1.46, 2.26, 1.54, 2.07, 1.96, 2.82, 1.22, 0.16, 1.84, 1.61]  min 0.0754 · mean 2.01 · max 3.94
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations for this round (previous reward function iter_09, trained from scratch for 665 epochs, 1024 envs, 3.14 h). Items 1-6 are facts: training metrics, the environment's own measurements and a rollout video. Item 7 records operator decisions.

1. **The source hand got the cup between its fingertips. This is the first time in four rounds that happened at all.** `task/*_cup_in_pocket_near` — the cup axis actually lying between the thumb tip and an opposing fingertip, both outside the cup wall — was 0 for every epoch of rounds 8, 9 and 10. Medians over epoch windows:

| metric (source hand) | ep 0-10 | ep 150-160 | ep 360-370 | ep 460-470 | ep 655-665 |
|---|---|---|---|---|---|
| `task/src_palm_to_cup` [m] | 0.235 | 0.252 | 0.205 | 0.100 | **0.094** |
| `task/src_near_rate` (palm < 12 cm) | 0.00 | 0.00 | 0.00 | 0.74 | **0.94** |
| `task/src_cup_in_pocket_near` | 0.00 | 0.00 | 0.00 | 0.993 | **0.997** |
| `task/src_thumb_oppose_near` | 0.06 | 0.00 | 0.00 | 0.914 | **0.922** |
| `task/src_tip_gap_mm_near` [mm] | 56.4 | – | – | 89.4 | **70.6** |
| `task/src_thumb_above_rim_mm_near` [mm] | 0.0 | – | – | −23.8 | **−25.0** |
| `contact/src_max` [N] | 0.17 | 0.00 | 0.00 | 0.062 | **0.276** |
| `task/src_closure` | 0.16 | 0.19 | 0.16 | 0.21 | 0.29 |

   - The hand opens wider than the 57 mm cup (89 mm), brings the palm in to 9.4 cm, puts the thumb 25 mm below the rim on the far wall, and then closes back to 71 mm — i.e. it opens, surrounds, and begins to squeeze.
   - Contact appeared at epoch ~460 and then quadrupled in the last 100 epochs (0.072 → 0.276 N). The run was still improving when the round ended.

2. **But no grasp ever registered, on either hand.** `task/src_grasped` and `task/rcv_grasped` were 0 for all 665 epochs, and `task/src_cup_lift` reached 0.09 cm. The environment's flag needs the thumb above 1 N AND another finger above 1 N at the same time; the best contact reached was 0.276 N on a single finger.

3. **The ladder behaved as designed: rungs lit in order, and none of them became a place to stand.** Source-hand terms at the end, with the previous round's failure mode for comparison:

| term | ep 150-160 | ep 360-370 | ep 655-665 |
|---|---|---|---|
| `reward/lad1_reach_src` | 0.026 | 0.029 | 0.093 |
| `reward/lad2_open_src` | 0.038 | 0.043 | 0.140 |
| `reward/lad3_height_src` | 0.058 | 0.069 | 0.230 |
| `reward/lad4_straddle_src` | 0.0001 | 0.0008 | 0.318 |
| `reward/lad5_oppose_src` | 0.000 | 0.000 | 0.397 |
| `reward/lad6_ready_src` | 0.000 | 0.000 | 0.423 |
| `reward/total` (both hands) | 0.184 | 0.256 | 2.159 |

   - Between epochs 150 and 370 the first three rungs were nearly flat (reach 0.026 → 0.029) and it looked like a plateau at 20 cm. It was not: at epoch ~460 the whole chain moved together. The nested product means a rung only pays once the rungs below it hold, so slow early progress is the expected shape, not a stall.
   - No rung saturated into a standing income. The last rung (`lad6_ready`, 0.423) is the pre-grasp posture itself, which is one finger flex from the environment's grasp flag.

4. **The receiver hand never started.** It ended 26.8 cm from its cup — further away than at epoch 150 — with zero contact, zero samples inside 12 cm, and its ladder stuck on the first three rungs (0.025 / 0.037 / 0.058, straddle 0.000).
   - `reward/ready_both` is `2.0 · min(src, rcv)` and was therefore 0 for the entire round. The two-hand coupling term paid nothing at any point, so nothing in the reward pulled the lagging hand forward while it was far behind.
   - The task is bimanual, but this round produced a one-handed policy.

5. **Rollout video** at the epoch-650 checkpoint (4 envs, 900 steps, `our_source/pour_t2r_rh_i09_ep650_0917.mp4`, frame sheet and 2.4x crops in `pour_t2r_rh_i09_ep650_0917_frames/`):
   - The source hand is down at its cup with the four fingers on the far side and the thumb on the near side: the cup body is genuinely between them, and the fingers reach past the cup's mid-height. This is the posture the metrics in item 1 describe, and it is the first round in which the video shows the hand around the cup at all.
   - The posture at step 350 and at step 880 is the same. The hand holds the straddle and never closes into a press; the cup stays upright on the mat and is never lifted. This matches contact stopping at 0.276 N and the grasp flag never firing.
   - The receiver arm is outside the crop in these frames, so its behaviour could not be judged from the video — only from item 4's metrics (26.8 cm away, no contact, ladder stuck on rungs 1-3).

6. **Physics stayed clean.** `ctrl/mimic_err_max` ended at 0.198 rad (it spiked to 1.4 rad in the first 50 epochs, then settled). No mimic runaway, no drop, no cup collision.

7. **Operator decisions** (after reviewing items 1-6 and the video):
   - **The round is extended rather than closed.** Training resumes from the epoch-650 checkpoint with this same reward function, because contact was still climbing when the round ended — 0.072 N to 0.276 N over the last 100 epochs — and the cheapest way to learn whether this reward reaches the 1 N grasp threshold is to let it run, not to redesign it.
   - **Stop rule:** if `contact/src_max` is not heading toward 1 N one tick (about 30 minutes) after the restart, the extension ends immediately and a new reward is generated.
   - **The extension ran 66 more epochs from the checkpoint and the stop rule fired.** Contact did not head for 1 N: `contact/src_max` was 0.293 N over the first ten extension epochs and 0.259 N over the last ten, with a peak of 0.518 N — flat, not climbing. `task/src_grasped` was non-zero in exactly 1 of 66 epochs, at 0.001.
   - The posture terms had saturated as well: over those 66 epochs every rung moved by less than 0.025 (`lad1` 0.095 → 0.097, `lad6` 0.427 → 0.451), and the receiver hand was unchanged at zero. `reward/total` 2.186 → 2.204.
   - So the ladder takes the hand to a stable pre-grasp straddle and stops there. Squeezing from 0.28 N to the 1 N flag is not something this reward function pays for on its own, and a new reward is generated.
   - **Two subjects for the next reward.** First, the squeeze: the hand is around the cup with the thumb opposed (oppose 0.95, pocket 99 %) and must now press both sides to 1 N. Second, the receiver hand, which never left the first three rungs — `ready_both = 2.0 · min(src, rcv)` pays nothing while one hand is at zero, so no term pulls the lagging hand forward.
   - **Keep:** the nested-product ladder (it solved the pocket problem that three previous rewards could not), the floorless opposition gate, the signed pad-facing condition, and the rule that dorsal contact is not progress.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
