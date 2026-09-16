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
5. Cup tilt: `ctx.src_cup_tilt` is the angle between the cup's up axis and world +z (0 = upright, π/2 = horizontal). Beads leave the source cup once it is tilted past roughly 110° (1.9 rad) with its mouth over the receiver cup's mouth. `ctx.src_cup_mouth_pos` and `ctx.rcv_cup_mouth_pos` are the rim centres.
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
    """Bounded closeness in (0, 1]: 1 at d = 0, ~0.37 at d = 1/k."""
    return torch.exp(-k * d)


def _band_err(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """0 inside [lo, hi], linear distance outside."""
    return torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)


def _cyl(points: torch.Tensor, cup_pos: torch.Tensor, cup_up: torch.Tensor):
    """Cup-cylinder coordinates of (N,K,3) points: axial height (N,K), radial vector (N,K,3), radial distance (N,K)."""
    rel = points - cup_pos[:, None, :]
    up = cup_up[:, None, :]
    axial = (rel * up).sum(dim=-1)
    rvec = rel - axial[..., None] * up
    radial = torch.norm(rvec, dim=-1)
    return axial, rvec, radial


def _axis_to_segment(rv_a: torch.Tensor, rv_b: torch.Tensor) -> torch.Tensor:
    """Distance (N,) from the cup axis to the segment joining two tips, in the plane perpendicular to the axis.
    ~0 when the axis lies BETWEEN the tips (cup inside the pinch); the nearest tip's radial distance otherwise."""
    seg = rv_b - rv_a
    u = torch.clamp(-(rv_a * seg).sum(dim=-1) / ((seg * seg).sum(dim=-1) + 1e-8), 0.0, 1.0)
    closest = rv_a + u[:, None] * seg
    return torch.norm(closest, dim=-1)


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, closure, cup_pos, cup_up, grasped, a_hand):
    """Per-arm approach / posture / wall closure / press / grasp signals, all (N,) in [0, 1]."""
    dtype = cup_pos.dtype
    mouth = ctx.cup_mouth_z
    r_in = ctx.cup_radius
    r_wall = r_in + 0.004                    # outer wall ~ inner radius + wall thickness
    tip_lo = ctx.cup_bottom_z + 0.03         # tips >= 3 cm above the cup bottom (table clearance)
    tip_hi = mouth - 0.015                   # tips >= 1.5 cm below the rim

    # ---- pre-grasp region for the PALM. UNCHANGED from iter_05: this posture was reached and must be kept;
    #      only its weight in compute_reward drops (1.5 -> 0.5 per hand) so that standing here is no longer the pay-off.
    p_ax, _, p_rad = _cyl(palm_pos[:, None, :], cup_pos, cup_up)
    rad_err = _band_err(p_rad[:, 0], r_wall + 0.01, 0.10)
    ht_err = _band_err(p_ax[:, 0], ctx.cup_bottom_z + 0.03, mouth + 0.01)
    d_pre = torch.sqrt(rad_err ** 2 + ht_err ** 2 + 1e-8)
    reach_lin = torch.clamp(1.0 - d_pre / 0.25, 0.0, 1.0)             # constant slope 4/m, no flat zone
    reach_fine = _near(d_pre, 25.0)                                    # sharp pull over the last few cm
    near_pre = _near(d_pre, 10.0)

    # ---- orientation: pad normal faces the cup horizontally (sign-agnostic), tips nearer the axis than the palm
    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup_xy = cup_pos[:, :2] - palm_pos[:, :2]
    d_palm_xy = torch.norm(to_cup_xy, dim=-1)
    u_xy = to_cup_xy / (d_palm_xy[:, None] + 1e-6)
    face = torch.abs((normal[:, :2] * u_xy).sum(dim=-1))
    tip_mid = tips[:, 0:3, :].mean(dim=1)                                        # thumb, index, middle
    d_tip_xy = torch.norm(tip_mid[:, :2] - cup_pos[:, :2], dim=-1)
    ahead = torch.clamp((d_palm_xy - d_tip_xy) / 0.03, 0.0, 1.0)
    orient = near_pre * (0.5 * face + 0.5 * ahead)

    # ---- fingertip -> outer wall distance (radial gap + height-band error), the quantity the policy must drive to 0.
    #      iter_05 measured this only inside tip_place (cap 0.4); here it carries weight 3.0 and a 3 mm tolerance.
    t_ax, t_rv, t_rad = _cyl(tips, cup_pos, cup_up)
    t_dir = t_rv / (t_rad[..., None] + 1e-6)
    rad_out = torch.clamp(t_rad - r_wall - 0.003, min=0.0)
    ht_bad = _band_err(t_ax, tip_lo, tip_hi)                                     # (N,F)
    surf = torch.sqrt(rad_out ** 2 + ht_bad ** 2 + 1e-8)
    s_thumb = surf[:, 0]
    s_finger = surf[:, 1:3].min(dim=-1).values                                   # thumb + index OR middle is enough
    # two scales: the wide one keeps a pull at 5 cm, the sharp one dominates the last centimetre
    # 5 cm -> 0.25, 3 cm -> 0.39, 1 cm -> 0.71, 3 mm -> 0.90, touching -> 1.00
    wall_t = 0.35 * _near(s_thumb, 12.0) + 0.65 * _near(s_thumb, 50.0)
    wall_f = 0.35 * _near(s_finger, 12.0) + 0.65 * _near(s_finger, 50.0)

    fdir = t_dir[:, 1:3, :].mean(dim=1)
    fdir = fdir / (torch.norm(fdir, dim=-1, keepdim=True) + 1e-6)
    opp = torch.clamp(-(t_dir[:, 0, :] * fdir).sum(dim=-1), 0.0, 1.0)            # 1 = thumb opposite the fingers

    # ---- thumb height w.r.t. the rim: low = on the wall (income allowed), over = hooked on the mouth
    thumb_ax = t_ax[:, 0]
    thumb_low = torch.clamp((mouth - thumb_ax) / 0.015, 0.0, 1.0)                # 1 at >= 1.5 cm below rim
    over_rim = torch.clamp((thumb_ax - (mouth - 0.005)) / 0.015, 0.0, 1.0)
    over_footprint = (t_rad[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = over_rim * over_footprint                                         # flat: independent of palm distance

    # ---- pinch posture gate: the cup axis lies BETWEEN the thumb tip and the index (or middle) tip,
    #      both tips at wall height and outside the opening. This is a POSTURE test only — it deliberately
    #      says nothing about the remaining gap, which is what `close` below measures.
    # radial window: tips outside the opening (lower ramp) AND not straddling from far away (upper ramp).
    # iter_05 had only the lower ramp, which saturates at r_in + 5 mm, so a wide-open hand whose tips sit
    # 20 cm either side of the axis at rim height scored a FULL pinch posture without approaching the cup.
    # Upper ramp: full credit out to r_wall + 3 cm (the posture actually reached this round), zero past r_wall + 5 cm.
    tip_ok = (_near(ht_bad, 40.0)
              * torch.clamp((t_rad - r_in) / 0.005, 0.0, 1.0)
              * torch.clamp((r_wall + 0.05 - t_rad) / 0.02, 0.0, 1.0))             # (N,F)
    pinch_i = _near(_axis_to_segment(t_rv[:, 0, :], t_rv[:, 1, :]), 40.0) * tip_ok[:, 1]
    pinch_m = _near(_axis_to_segment(t_rv[:, 0, :], t_rv[:, 2, :]), 40.0) * tip_ok[:, 2]
    pinch_geo = torch.maximum(pinch_i, pinch_m) * tip_ok[:, 0] * thumb_low

    # ---- NEW, the round's main term: drive both sides of the pinch onto the wall.
    #      The 0.35 floor keeps the signal alive before the posture exists (it replaces iter_05's ungated tip_place),
    #      the 0.65 share makes the correct straddling posture worth nearly 3x more.
    close = 0.5 * (wall_t + wall_f) * (0.35 + 0.65 * pinch_geo) * (0.5 + 0.5 * opp)

    # ---- closing on air: thumb tip within 3.5 cm of the nearest index/middle tip is impossible around a 6.4 cm cup
    gap = torch.minimum(torch.norm(tips[:, 0, :] - tips[:, 1, :], dim=-1),
                        torch.norm(tips[:, 0, :] - tips[:, 2, :], dim=-1))
    air_pinch = torch.clamp((0.035 - gap) / 0.02, 0.0, 1.0)

    # ---- squeeze intent: COMMANDED closure (measured closure is useless — contact freeze stops a real pinch at ~0.25
    #      while closing on air reaches 0.6). Now also requires the tips to be AT the wall, so commanding closure
    #      from 5 cm away pays ~0.3 of what commanding it in contact pays.
    cmd = 0.5 * (torch.clamp(a_hand, -1.0, 1.0) + 1.0)                          # (N,6) 0 = open, 1 = closed
    cmd_close = (cmd[:, 0] + cmd[:, 1] + cmd[:, 2:6].mean(dim=-1)) / 3.0
    squeeze = pinch_geo * cmd_close * (0.3 + 0.7 * 0.5 * (wall_t + wall_f))

    # ---- hand must be open while outside the pre-grasp region (unchanged)
    far = 1.0 - _near(torch.clamp(d_pre - 0.03, min=0.0), 20.0)
    curl_far = torch.clamp((closure - 0.3) / 0.4, 0.0, 1.0) * far

    # ---- forces. iter_05 multiplied the two sides, and since the thumb force was identically 0 the product
    #      was 0 everywhere and taught nothing. Here the sides are ADDED, and each side's force is paid only
    #      when the OPPOSITE side is already at the wall — so pressing one side and pushing the light cup away
    #      earns nothing, while squeezing both sides pays immediately.
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].max(dim=-1).values
    f_min = torch.minimum(f_t, f_o)
    s_t = 1.0 - torch.exp(-f_t / 0.3)                                            # 0.2 N -> 0.49, 1 N -> 0.96
    s_o = 1.0 - torch.exp(-f_o / 0.3)
    quality = thumb_low * (0.5 + 0.5 * opp)
    gate_q = 0.5 + 0.5 * quality                                                 # FLOORED: never zeroes a force gradient
    touch = (0.6 * s_t * wall_f + 0.4 * s_o * wall_t) * (0.4 + 0.6 * pinch_geo)  # thumb weighted: it never touched at all
    # half dense near zero (first simultaneous contact pays), half LINEAR to the env's 1 N flag so the
    # gradient does not saturate before the threshold that actually defines a grasp
    grip = (0.5 * (1.0 - torch.exp(-f_min / 0.25)) + 0.5 * torch.clamp(f_min / 1.0, 0.0, 1.0)) * gate_q

    # ---- grasp: env flag x quality (floored); soft hold bridges flag flicker around 1 N
    grasped_f = grasped.to(dtype)
    grasp = grasped_f * gate_q
    hold_soft = torch.clamp((f_min - 0.3) / 0.7, 0.0, 1.0)
    hold = torch.maximum(grasped_f, hold_soft) * thumb_low
    q = 0.3 + 0.7 * quality                                                      # multiplier on lift / carry / tilt

    return {
        "reach_lin": reach_lin, "reach_fine": reach_fine, "orient": orient,
        "pinch_geo": pinch_geo, "close": close, "squeeze": squeeze, "air_pinch": air_pinch,
        "curl_far": curl_far, "touch": touch, "grip": grip, "grasp": grasp, "hold": hold,
        "q": q, "rim_hook": rim_hook,
    }


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)
    dtype = zero.dtype
    act = ctx.actions

    s = _hand_terms(ctx, ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                    ctx.src_hand_closure, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_grasped, act[:, 6:12])
    r = _hand_terms(ctx, ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                    ctx.rcv_hand_closure, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_grasped, act[:, 18:24])
    hold_s, hold_r = s["hold"], r["hold"]
    q_s, q_r = s["q"], r["q"]

    # --------------------------------------------------------------- stage 0: approach (SOLVED — income cut 1.5 -> 0.5)
    # iter_05 paid 3.50 of its 3.71 total for standing here, which is more than every unreached stage combined.
    approach_src = 0.3 * s["reach_lin"] + 0.2 * s["reach_fine"]   # max 0.5 per hand
    approach_rcv = 0.3 * r["reach_lin"] + 0.2 * r["reach_fine"]
    orient_src = 0.15 * s["orient"]          # weight 0.15: helper only, halved with the rest of the plateau
    orient_rcv = 0.15 * r["orient"]

    # --------------------------------------------------------------- stage 1: pinch posture (keep it, do not pay to rest in it)
    pinch_geo_src = 0.8 * s["pinch_geo"]     # weight 0.8 < close 3.0: the posture is a gate, the gap is the goal
    pinch_geo_rcv = 0.8 * r["pinch_geo"]

    # --------------------------------------------------------------- stage 2: close the last centimetres (THE bottleneck)
    close_src = 3.0 * s["close"]             # weight 3.0 > the whole approach plateau (0.65/hand): moving in must dominate holding still
    close_rcv = 3.0 * r["close"]
    squeeze_src = 1.0 * s["squeeze"]         # weight 1.0 (was 0.6): command the fingers shut, but only in contact range
    squeeze_rcv = 1.0 * r["squeeze"]
    touch_src = 1.0 * s["touch"]             # one side pressing while the other is in place — a step, not a resting place
    touch_rcv = 1.0 * r["touch"]
    grip_src = 3.5 * s["grip"]               # weight 3.5: BOTH sides above 1 N is the state the env calls a grasp
    grip_rcv = 3.5 * r["grip"]
    grasp_src = 4.0 * s["grasp"]             # weight 4.0 > grip 3.5: the flag itself is still the target
    grasp_rcv = 4.0 * r["grasp"]
    grasp_both = 3.0 * s["grasp"] * r["grasp"]
    curl_far_pen = -0.3 * (s["curl_far"] + r["curl_far"])
    air_pinch_pen = -0.4 * (s["air_pinch"] * (1.0 - hold_s) + r["air_pinch"] * (1.0 - hold_r))
    rim_hook_pen = -0.5 * (s["rim_hook"] + r["rim_hook"])

    # --------------------------------------------------------------- stage 3: lift (independent per arm, x quality)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # tanh(h/2.5cm): 5 mm -> 0.20, 3 cm -> 0.83; weight 5.0 > grasp 4.0 so holding still on the table is not the end
    lift_src = 5.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.025)
    lift_rcv = 5.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.025)
    lifted = hold_s * (h_src > 0.03).to(dtype) * hold_r * (h_rcv > 0.03).to(dtype)

    # --------------------------------------------------------------- stage 4: carry together (tilt-aware target)
    tilt = ctx.src_cup_tilt
    # upright: mouths 10 cm apart (bodies side by side, no nesting); the gap closes to 0 as tilt -> ~90 deg,
    # because in a real pour the source ORIGIN stays ~5 cm aside and the tilted mouth swings over the receiver.
    pour_prog = torch.clamp((tilt - 0.35) / 1.2, 0.0, 1.0)
    tgt_dxy = 0.10 * (1.0 - pour_prog)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    dxy_err = torch.abs(mouth_dxy - tgt_dxy)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = _band_err(mouth_dz, 0.04, 0.12)                                     # source mouth 4..12 cm higher
    carry_q = (0.5 * torch.clamp(1.0 - dxy_err / 0.30, 0.0, 1.0) + 0.5 * _near(dxy_err, 30.0)) * _near(dz_err, 15.0)
    align = 6.0 * lifted * q_s * carry_q     # weight 6.0 > lift 5.0: carrying toward the receiver must pay

    # --------------------------------------------------------------- stage 5: tilt and pour
    zone = _near(dxy_err, 40.0) * _near(dz_err, 40.0)                            # 1 cm error -> 0.67
    tilt_r = 5.0 * lifted * q_s * zone * torch.clamp(tilt / 2.0, 0.0, 1.0)       # tilt pays only on the schedule
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads); increments, never the level
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # --------------------------------------------------------------- stage 6: success
    success = 25.0 * ctx.success.to(dtype)   # 25/step on top of all held income: keep the goal state

    # --------------------------------------------------------------- constraints (guards, not shaping)
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    pre_tilt_pen = -1.0 * (1.0 - lifted * zone) * has_beads * torch.tanh(2.0 * torch.clamp(tilt - 0.8, min=0.0))
    # receiver upright only while held; -0.8 max stays far below the receiver's grasp income (a pinch tilts a light cup a little)
    rcv_upright_pen = -0.8 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.15, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    # knocking a cup over and shoving it across the table are the same failure (pressing one side only) — one term
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
    # palm is not part of a fingertip grasp: pushing the cup with it costs
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # --------------------------------------------------------------- regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "orient_src": orient_src,
        "orient_rcv": orient_rcv,
        "pinch_geo_src": pinch_geo_src,
        "pinch_geo_rcv": pinch_geo_rcv,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "squeeze_src": squeeze_src,
        "squeeze_rcv": squeeze_rcv,
        "touch_src": touch_src,
        "touch_rcv": touch_rcv,
        "grip_src": grip_src,
        "grip_rcv": grip_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "grasp_both": grasp_both,
        "curl_far_pen": curl_far_pen,
        "air_pinch_pen": air_pinch_pen,
        "rim_hook_pen": rim_hook_pen,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "align": align,
        "tilt": tilt_r,
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
        "success": success,
        "pre_tilt_pen": pre_tilt_pen,
        "rcv_upright_pen": rcv_upright_pen,
        "disturb_pen": disturb_pen,
        "drop_pen": drop_pen,
        "cup_speed_pen": cup_speed_pen,
        "nested_pen": nested_pen,
        "cup_collision_pen": cup_collision_pen,
        "hand_foreign_pen": hand_foreign_pen,
        "palm_push_pen": palm_push_pen,
        "action_rate_pen": action_rate_pen,
        "arm_speed_pen": arm_speed_pen,
    }
    reward = zero
    for v in components.values():
        reward = reward + v
    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/spill: [9.77e-05, 0, 4.88e-05, 0, 4.88e-05, 0, 0, 0, 0, 0]  min 0 · mean 9.76e-05 · max 0.00181
done/drop: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000164 · max 0.00977
done/mimic_runaway: [0.00293, 0, 0, 0, 0.000977, 0, 0, 0, 0, 0]  min 0 · mean 0.00015 · max 0.00293
episode_lengths/step: [114, 846, 828, 840, 843, 854, 854, 849, 828, 887]  min 114 · mean 817 · max 898
reward/action_rate_pen: [-0.0509, -0.0485, -0.0445, -0.0423, -0.041, -0.0402, -0.0399, -0.0399, -0.0389, -0.0371]  min -0.0517 · mean -0.0415 · max -0.0354
reward/air_pinch_pen: [0, -0.00548, -0.000776, -0.0443, -0.00742, -0.0017, -0.00241, -7.84e-05, -0.000289, 0]  min -0.0598 · mean -0.00671 · max 0
reward/align: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/approach_rcv: [0.157, 0.232, 0.301, 0.309, 0.32, 0.312, 0.327, 0.347, 0.312, 0.199]  min 0.137 · mean 0.285 · max 0.354
reward/approach_src: [0.171, 0.272, 0.315, 0.362, 0.401, 0.398, 0.424, 0.419, 0.437, 0.483]  min 0.161 · mean 0.386 · max 0.485
reward/arm_speed_pen: [-0.0561, -0.0252, -0.0171, -0.0123, -0.0108, -0.00991, -0.00948, -0.00899, -0.0105, -0.00965]  min -0.0561 · mean -0.0141 · max -0.00859
reward/close_rcv: [0.0496, 0.0838, 0.132, 0.15, 0.173, 0.183, 0.205, 0.23, 0.209, 0.062]  min 0.0373 · mean 0.149 · max 0.246
reward/close_src: [0.0523, 0.121, 0.229, 0.495, 0.602, 0.6, 0.635, 0.701, 0.997, 2.12]  min 0.0457 · mean 0.799 · max 2.36
reward/cup_collision_pen: [-0.000513, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00415 · mean -2.88e-05 · max 0
reward/cup_speed_pen: [-0.0055, 0, 0, 0, -0.000195, 0, -0.000195, 0, 0, 0]  min -0.00944 · mean -0.000242 · max 0
reward/curl_far_pen: [0, -0.0689, -0.0289, -0.00213, -0.000777, -0.000663, -0.000437, -0.000342, -0.00146, -7.94e-06]  min -0.087 · mean -0.0123 · max 0
reward/disturb_pen: [-0.0465, -0.000608, -0.000692, -0.000315, -0.000888, 0, -0.000511, -0.000462, -0.00276, 0]  min -0.0758 · mean -0.00228 · max 0
reward/drop_pen: [-0.0011, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0085 · mean -0.00024 · max 0
reward/grasp_both: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000102 · max 0.00586
reward/grip_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grip_src: [0, 0, 0, 0, 0.00317, 0.00303, 0.00337, 0.00108, 0.0019, 0]  min 0 · mean 0.00159 · max 0.0109
reward/hand_foreign_pen: [-0.013, -0.00044, -0.0157, -0.0224, -0.0205, -0.0172, -0.0162, -0.00492, -0.0114, -0.00312]  min -0.0491 · mean -0.0126 · max 0
reward/lift_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.05e-05 · max 0.00428
reward/nested_pen: [-0.00391, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0234 · mean -0.000394 · max 0
reward/orient_rcv: [0.034, 0.053, 0.0616, 0.0627, 0.0656, 0.0677, 0.0686, 0.0755, 0.0673, 0.0437]  min 0.0286 · mean 0.0612 · max 0.0763
reward/orient_src: [0.0337, 0.0568, 0.0614, 0.0762, 0.09, 0.0873, 0.0946, 0.0889, 0.0881, 0.0909]  min 0.0337 · mean 0.0801 · max 0.0975
reward/palm_push_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pinch_geo_rcv: [0, 1.94e-05, 1.83e-05, 0, 3.01e-06, 0.000137, 9.44e-05, 0.00105, 0.0152, 0]  min 0 · mean 0.00107 · max 0.0201
reward/pinch_geo_src: [2.82e-05, 0.000148, 0.00733, 0.14, 0.158, 0.164, 0.169, 0.2, 0.266, 0.608]  min 0 · mean 0.213 · max 0.672
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00977 · mean -1.73e-05 · max 0
reward/pre_tilt_pen: [-0.00842, -0.000732, 0, 0, 0, 0, 0, -0.000149, -0.000557, 0]  min -0.0183 · mean -0.00057 · max 0
reward/rcv_upright_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/rim_hook_pen: [-0.318, -0.041, -0.0427, -0.0124, -0.0182, -0.0194, -0.0139, -0.0252, -0.039, -0.0244]  min -0.318 · mean -0.0301 · max -0.00808
reward/spill_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0146 · mean -0.000216 · max 0.0146
reward/squeeze_rcv: [0, 5.04e-06, 4.78e-06, 0, 5.51e-07, 2.28e-05, 2.35e-05, 0.000276, 0.00472, 0]  min 0 · mean 0.00032 · max 0.00618
reward/squeeze_src: [1.62e-05, 7.21e-05, 0.00395, 0.104, 0.131, 0.133, 0.14, 0.161, 0.194, 0.342]  min 0 · mean 0.144 · max 0.402
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [-0.0196, 0.615, 0.951, 1.56, 1.85, 1.87, 2.01, 2.15, 2.5, 3.88]  min -0.0451 · mean 2.01 · max 4.55
reward/touch_rcv: [0.000366, 0, 2.4e-05, 5.42e-05, 8.79e-05, 6.66e-05, 0.000175, 0.00086, 0.00125, 0]  min 0 · mean 0.000242 · max 0.00166
reward/touch_src: [0.000773, 8.6e-05, 0.000515, 0.00821, 0.0197, 0.0197, 0.034, 0.0174, 0.0237, 0.022]  min 0 · mean 0.0221 · max 0.0964
rewards/step: [-6.97, 480, 774, 1.28e+03, 1.55e+03, 1.65e+03, 1.69e+03, 1.77e+03, 1.95e+03, 3.48e+03]  min -6.97 · mean 1.66e+03 · max 3.93e+03
task/aim_dist: [0.31, 0.319, 0.32, 0.317, 0.316, 0.319, 0.414, 0.317, 0.317, 0.318]  min 0.31 · mean 0.331 · max 1
task/cup_collision_rate: [0.000977, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.46e-05 · max 0.00488
task/cups_center_dist: [0.313, 0.32, 0.32, 0.318, 0.317, 0.319, 0.414, 0.318, 0.318, 0.319]  min 0.313 · mean 0.332 · max 1
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.00195, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000197 · max 0.0117
task/rcv_cup_lift: [0.000969, 2.59e-05, 4.41e-05, 3.52e-05, 8e-05, 5.23e-05, 0.0348, 0.000184, 0.000191, 1.7e-05]  min -7.63e-05 · mean 0.0013 · max 0.154
task/rcv_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/rcv_hand_foreign_rate: [0.00586, 0, 0.00195, 0.000977, 0, 0, 0.000977, 0, 0.00195, 0]  min 0 · mean 0.000781 · max 0.0166
task/src_cup_lift: [0.000619, 1.6e-05, 9.85e-05, 0.000409, 0.000337, 0.000238, 0.000268, 0.000165, 0.000213, 0.000236]  min -7.12e-05 · mean 0.00115 · max 0.0444
task/src_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.28e-05 · max 0.00195
task/src_hand_foreign_rate: [0.00195, 0, 0.0107, 0.0156, 0.0137, 0.0107, 0.00781, 0.00488, 0.00586, 0.00195]  min 0 · mean 0.00773 · max 0.0342
task/src_tilt_deg: [3.21, 0.208, 0.341, 1.52, 1.15, 0.807, 0.906, 0.615, 0.836, 0.765]  min 0.0483 · mean 1.07 · max 4.4
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations for this round (previous reward function iter_06, trained from scratch, stopped early at epoch 565 by operator decision). Items 1-6 are facts: a rollout video the operator watched, two checks run against the reward code and the robot model, training metrics, and a new logging-only measurement added to the environment. Item 7 records operator decisions.

1. **The hand was never around the cup. The operator watched the previous round's video and found the parts in the wrong order.** Looking along the hand: index finger — THUMB — cup. The thumb sits between the fingers and the cup instead of on the far wall, so closing the fingers can never trap the cup. The tip sensors that have to meet (thumb tip against the index/middle/ring/pinky tips) never face each other.

2. **The previous reward pays for that unusable posture.** `pinch_geo` measures the distance from the cup axis to the segment joining the thumb tip and the index (or middle) tip, both taken in the plane normal to the cup axis. Evaluated directly on synthetic postures with a 57 mm cup:

| posture | cup axis to segment | `pinch_geo` |
|---|---|---|
| true opposition (thumb 0°, index 180°) | 0.0 mm | 1.00 |
| the posture in the video (index — thumb — cup, same side) | 34.5 mm | 0.19 |
| thumb and index side by side on the same wall | 34.0 mm | 0.26 |
| thumb and index 90° apart | 24.4 mm | 0.38 |

   The previous round (iter_05) ended with `pinch_geo` at 0.21, which is exactly this band. A term that reads as "one fifth achieved" was in fact paying for a posture that cannot grasp.
   The opposition factor `opp` exists inside the reward (`0.5 + 0.5·opp` multiplies `close`), but it is not logged, so no metric distinguished the two cases.
   `close` also has a floor: with `pinch_geo = 0` and `opp = 0` it still pays `0.35 × 0.5 = 0.175` of its weight 3.0 for tips merely near the wall.

3. **The hand itself can grasp this cup — the failure is not kinematic.** Forward kinematics on the robot's own URDF, sweeping the thumb opposition joint `r_hj_thumb_1` over its full range (0 to 2.09 rad):
   - open hand: thumb tip to index tip 95-107 mm; the profile's measured thumb-to-four-finger normal gap is 83.7 mm at 1.57 rad, its maximum.
   - grip pose (`thumb_1` 1.20, four fingers 1.08/0.85): thumb tip to index tip 43.8 mm, to middle tip 54.0 mm.
   - The cup is 57 mm across. It fits in the open pocket and the grip pose closes past it, so a real opposition grasp is reachable.

4. **Training metrics up to the stop (medians over epoch windows).** The source hand improved fast on the reward's own terms; the receiver hand never engaged.

| metric | ep 0-10 | ep 150-160 | ep 300-310 | ep 400-410 | ep 555-565 |
|---|---|---|---|---|---|
| `reward/close_src` (max 3.0) | 0.05 | 0.46 | 0.63 | 0.72 | 2.30 |
| `reward/close_rcv` (max 3.0) | 0.05 | 0.14 | 0.20 | 0.24 | 0.18 |
| `reward/pinch_geo_src` (max 0.8) | 0.00 | 0.126 | 0.166 | 0.200 | 0.619 |
| `reward/pinch_geo_rcv` (max 0.8) | 0.00 | 0.000 | 0.000 | 0.002 | 0.000 |
| `reward/grip_src` / `reward/grip_rcv` | 0 / 0 | 0 / 0 | 0.004 / 0 | 0.002 / 0 | 0 / 0 |
| `reward/touch_src` | 0.001 | 0.007 | 0.026 | 0.018 | 0.084 |
| `contact/src_max` [N] | 0.25 | 0.08 | 0.09 | 0.07 | 0.21 |
| `contact/rcv_max` [N] | 0.77 | 0.000 | 0.006 | 0.035 | 0.002 |
| `task/src_grasped` / `task/rcv_grasped` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `task/src_cup_lift` [m] | 0.013 | 0.0004 | 0.0003 | 0.0002 | 0.0007 |
| `task/src_palm_to_cup` / `task/rcv_palm_to_cup` [m] | 0.294 / 0.283 | 0.137 / 0.166 | 0.124 / 0.153 | 0.122 / 0.146 | 0.095 / 0.161 |
| `task/src_near_rate` / `task/rcv_near_rate` | 0 / 0.002 | 0 / 0 | 0.001 / 0 | 0.001 / 0 | 0.917 / 0 |
| `task/src_thumb_above_rim_mm_near` [mm] | – | – | – | −31.5 | −40.4 |
| `task/src_closure` / `task/rcv_closure` | 0.14 / 0.13 | 0.47 / 0.24 | 0.47 / 0.27 | 0.42 / 0.27 | 0.29 / 0.26 |

   - Reward split over the last 20 epochs, `reward/total` 4.44: close_src 2.30, pinch_geo_src 0.63, approach_src 0.48, squeeze_src 0.39, approach_rcv 0.30, close_rcv 0.16, orient_src 0.09, touch_src 0.08. Everything downstream of contact (grip, grasp, lift, align, tilt, pour, success) is 0.
   - So the source hand collected 3.4 of 4.4 from posture terms while its finger force stayed at 0.21 N, one fifth of the 1 N the environment's grasp flag needs, and `grip_src` (which needs BOTH the thumb and an opposing finger pressing) fell back to 0.
   - The receiver hand went backwards: it ended 16.1 cm from its cup with zero contact and `pinch_geo` 0.
   - Physics stayed clean: `ctrl/mimic_err_max` 0.25 rad, no runaway, no drop, no cup collision.

5. **New logging-only measurements were added to the environment for the next round** (not in the reward, not in the observations; fixed by a contract test). Over envs whose palm is within 12 cm of the cup:
   - `task/{src,rcv}_thumb_oppose_near` — how opposite the thumb tip is to the mean of the index/middle tips, in the plane normal to the cup axis. 1 = exactly opposite, 0 = same side. This is the quantity that was invisible in every round so far.
   - `task/{src,rcv}_tip_gap_mm_near` — thumb tip to nearest index/middle tip, in mm. Compare against the 57 mm cup.
   - `task/{src,rcv}_cup_in_pocket_near` — fraction of envs where the cup axis actually lies between the two tips and both tips are outside the cup's inner radius.

6. **Previous rounds, for context.**
   - Round 7 (iter_05): `pinch_geo` reached 0.21, contact stalled at 0.20 N, no grasp. The video showed the index — thumb — cup order described in item 1.
   - Round 6 (iter_04): the approach was solved (palms beside the cups, thumbs below the rim) but the fingers never closed on the cup.
   - Round 5 (iter_03): both palms stopped 16-17 cm away with zero contact.

7. **Operator decisions** (after reviewing items 1-6 and the previous round's video):
   - This draft is approved as the feedback for this round, and a new reward is to be generated. Training starts from scratch, because the reward's meaning changes.
   - The round was stopped early on purpose. The reason is item 1-2: the posture terms were rewarding a hand that cannot grasp, so letting the round finish would only have refined that posture.
   - What the next reward must fix: a posture only counts when the thumb is on the OPPOSITE wall from the index/middle tips, with the cup body between them. A thumb beside the fingers, a thumb between the fingers and the cup, or tips merely near the wall must not pay. The floor in `close` (0.175 of its weight with no opposition and no pinch) is part of this problem.
   - The goal after that is unchanged: the thumb and at least one opposing finger both above 1 N at the same time (the environment's grasp flag), then lifting.
   - The three new measurements in item 5 are for judging the next round without a video; they must stay out of the reward and the observations.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
