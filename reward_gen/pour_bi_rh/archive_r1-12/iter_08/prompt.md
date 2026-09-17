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
    """Cup-cylinder coords of (N,K,3) points: axial height (N,K), radial vector (N,K,3), radial distance (N,K)."""
    rel = points - cup_pos[:, None, :]
    up = cup_up[:, None, :]
    axial = (rel * up).sum(dim=-1)
    rvec = rel - axial[..., None] * up
    radial = torch.norm(rvec, dim=-1)
    return axial, rvec, radial


def _axis_to_segments(rv_a: torch.Tensor, rv_b: torch.Tensor) -> torch.Tensor:
    """Distance (N,K) from the cup AXIS to each segment joining rv_a (N,1,3) to rv_b (N,K,3),
    measured in the plane normal to the axis. ~0 only when the axis lies BETWEEN the two tips."""
    seg = rv_b - rv_a
    u = torch.clamp(-(rv_a * seg).sum(dim=-1) / ((seg * seg).sum(dim=-1) + 1e-8), 0.0, 1.0)
    closest = rv_a + u[..., None] * seg
    return torch.norm(closest, dim=-1)


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, closure, cup_pos, cup_up, grasped, a_hand):
    """Per-arm approach / opposition / pocket / contact signals, all (N,) in [0, 1]."""
    dtype = cup_pos.dtype
    mouth = ctx.cup_mouth_z
    r_in = ctx.cup_radius
    r_wall = r_in + 0.004                    # outer wall ~ inner radius + wall thickness
    d_out = 2.0 * r_wall                     # outer diameter ~ 6.5 cm for a 5.7 cm bore
    tip_lo = ctx.cup_bottom_z + 0.03         # tips >= 3 cm above the cup bottom (table clearance)
    tip_hi = mouth - 0.015                   # tips >= 1.5 cm below the rim

    # ---- palm pre-grasp region (unchanged; this stage is solved and stays cheap)
    p_ax, _, p_rad = _cyl(palm_pos[:, None, :], cup_pos, cup_up)
    rad_err = _band_err(p_rad[:, 0], r_wall + 0.01, 0.10)
    ht_err = _band_err(p_ax[:, 0], ctx.cup_bottom_z + 0.03, mouth + 0.01)
    d_pre = torch.sqrt(rad_err ** 2 + ht_err ** 2 + 1e-8)
    reach_lin = torch.clamp(1.0 - d_pre / 0.25, 0.0, 1.0)
    reach_fine = _near(d_pre, 25.0)
    near_pre = _near(d_pre, 10.0)

    # ---- orientation: pad normal faces the cup, tips nearer the axis than the palm
    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup_xy = cup_pos[:, :2] - palm_pos[:, :2]
    d_palm_xy = torch.norm(to_cup_xy, dim=-1)
    u_xy = to_cup_xy / (d_palm_xy[:, None] + 1e-6)
    face = torch.abs((normal[:, :2] * u_xy).sum(dim=-1))
    tip_mid_all = tips[:, 0:3, :].mean(dim=1)
    d_tip_xy = torch.norm(tip_mid_all[:, :2] - cup_pos[:, :2], dim=-1)
    ahead = torch.clamp((d_palm_xy - d_tip_xy) / 0.03, 0.0, 1.0)
    orient = near_pre * (0.5 * face + 0.5 * ahead)

    # ---- fingertips in cup-cylinder coordinates
    t_ax, t_rv, t_rad = _cyl(tips, cup_pos, cup_up)
    t_dir = t_rv / (t_rad[..., None] + 1e-6)

    # ---- OPPOSITION (the quantity every previous round was blind to).
    #      +1 = this finger sits on the far wall from the thumb, -1 = same side as the thumb.
    u_th = t_dir[:, 0, :]
    opp_all = (t_dir[:, 1:, :] * u_th[:, None, :]).sum(dim=-1) * (-1.0)      # (N,F-1) in [-1, 1]
    opp_best = torch.clamp(opp_all.max(dim=-1).values, 0.0, 1.0)             # logged as oppose_*
    # HARD gate, no floor: 90 deg apart -> 0, 120 deg -> 1.  A thumb beside or inside the fingers scores 0.
    opp_gate_all = torch.clamp((opp_all - 0.3) / 0.5, 0.0, 1.0)
    opp_gate = opp_gate_all.max(dim=-1).values

    # ---- tip admissibility: at wall height and outside the bore (not dipped into the cup, not far outside)
    ht_bad = _band_err(t_ax, tip_lo, tip_hi)                                  # (N,F)
    h_ok = _near(ht_bad, 40.0)
    r_ok = (torch.clamp((t_rad - r_in) / 0.005, 0.0, 1.0)
            * torch.clamp((r_wall + 0.05 - t_rad) / 0.02, 0.0, 1.0))
    tip_ok = h_ok * r_ok                                                      # (N,F)

    # ---- thumb height w.r.t. the rim: below the rim = on the wall, above = hooked on the mouth
    thumb_ax = t_ax[:, 0]
    thumb_low = torch.clamp((mouth - thumb_ax) / 0.015, 0.0, 1.0)

    # ---- POCKET: is the cup body actually between the thumb tip and the finger tips?
    #      midpoint of the thumb tip and the index/middle mean must sit ON the cup axis, the opening must be
    #      wider than the cup and narrower than a spread hand, and both sides must be at wall height.
    tip_f_mid = tips[:, 1:3, :].mean(dim=1)
    mid = 0.5 * (tips[:, 0, :] + tip_f_mid)
    _, rv_mid, d_pocket = _cyl(mid[:, None, :], cup_pos, cup_up)
    d_pocket = d_pocket[:, 0]
    gap_mid = torch.norm(tips[:, 0, :] - tip_f_mid, dim=-1)
    # full credit for an opening of 6.5..11 cm (cup is 6.5 cm outside, the open hand reaches ~10 cm)
    gap_ok = (torch.clamp((gap_mid - (d_out - 0.015)) / 0.015, 0.0, 1.0)
              * torch.clamp((0.135 - gap_mid) / 0.025, 0.0, 1.0))
    # soft opposition factor: exactly 0 when the thumb is on the same side as the fingers
    u_fm = t_dir[:, 1:3, :].mean(dim=1)
    u_fm = u_fm / (torch.norm(u_fm, dim=-1, keepdim=True) + 1e-6)
    opp_mid = -(u_th * u_fm).sum(dim=-1)
    opp_soft = torch.clamp((opp_mid + 0.1) / 0.6, 0.0, 1.0)
    h_pocket = h_ok[:, 0] * h_ok[:, 1:3].max(dim=-1).values
    pocket = ((0.35 * _near(d_pocket, 15.0) + 0.65 * _near(d_pocket, 45.0))
              * gap_ok * h_pocket * opp_soft * thumb_low)

    # ---- POSE: the strict, per-finger straddle test. The cup axis lies between the thumb tip and THIS
    #      finger's tip, both tips admissible, and the pair genuinely opposed. No floor anywhere.
    seg_d = _axis_to_segments(t_rv[:, 0:1, :], t_rv[:, 1:, :])                # (N,F-1)
    pair_geo = _near(seg_d, 40.0) * opp_gate_all * tip_ok[:, 1:] * tip_ok[:, 0:1]
    pose = pair_geo.max(dim=-1).values

    # ---- distance of each tip to the outer wall (radial gap + height-band error)
    rad_out = torch.clamp(t_rad - r_wall - 0.003, min=0.0)
    surf = torch.sqrt(rad_out ** 2 + ht_bad ** 2 + 1e-8)
    wall_all = 0.35 * _near(surf, 12.0) + 0.65 * _near(surf, 50.0)
    wall_t = wall_all[:, 0]
    wall_f = wall_all[:, 1:].max(dim=-1).values
    wall_pair = 0.5 * (wall_t + wall_f)

    # ---- thumb hooked over the rim instead of pinching the wall
    over_rim = torch.clamp((thumb_ax - (mouth - 0.005)) / 0.015, 0.0, 1.0)
    over_footprint = (t_rad[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = over_rim * over_footprint

    # ---- close the last centimetres: ONLY inside a pose that can trap the cup
    close = pose * wall_pair

    # ---- closing on air
    gap_min = torch.norm(tips[:, 0:1, :] - tips[:, 1:, :], dim=-1).min(dim=-1).values
    air_pinch = torch.clamp((d_out - 0.02 - gap_min) / 0.02, 0.0, 1.0)

    # ---- squeeze intent from the COMMANDED closure
    cmd = 0.5 * (torch.clamp(a_hand, -1.0, 1.0) + 1.0)
    cmd_close = (cmd[:, 0] + cmd[:, 1] + cmd[:, 2:6].mean(dim=-1)) / 3.0
    squeeze = pose * cmd_close * wall_pair

    # ---- hand must be open while outside the pre-grasp region
    far = 1.0 - _near(torch.clamp(d_pre - 0.03, min=0.0), 20.0)
    curl_far = torch.clamp((closure - 0.3) / 0.4, 0.0, 1.0) * far

    # ---- forces. Only a finger that OPPOSES the thumb counts as the far side of the pinch.
    f_t = finger_force[:, 0]
    f_o = (finger_force[:, 1:] * opp_gate_all).max(dim=-1).values
    f_min = torch.minimum(f_t, f_o)
    s_t = 1.0 - torch.exp(-f_t / 0.3)
    s_o = 1.0 - torch.exp(-f_o / 0.3)
    q_grasp = opp_gate * thumb_low                                            # hard: no opposition, no pay
    touch = (0.6 * s_t * wall_f + 0.4 * s_o * wall_t) * q_grasp
    grip = (0.5 * (1.0 - torch.exp(-f_min / 0.25)) + 0.5 * torch.clamp(f_min / 1.0, 0.0, 1.0)) * q_grasp
    grasped_f = grasped.to(dtype)
    grasp = grasped_f * q_grasp
    hold_soft = torch.clamp((f_min - 0.3) / 0.7, 0.0, 1.0)
    # opposition gates the HOLD too: the env grasp flag can fire with the thumb and a finger both
    # pressing the SAME wall, which shoves the cup instead of trapping it — that must not pay for lifting.
    hold = torch.maximum(grasped_f, hold_soft) * thumb_low * opp_gate
    q = 0.3 + 0.7 * q_grasp

    return {
        "reach_lin": reach_lin, "reach_fine": reach_fine, "orient": orient,
        "oppose": opp_best * near_pre, "pocket": pocket, "pose": pose,
        "close": close, "squeeze": squeeze, "air_pinch": air_pinch, "curl_far": curl_far,
        "touch": touch, "grip": grip, "grasp": grasp, "hold": hold, "q": q, "rim_hook": rim_hook,
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

    # ---------------------------------------------------- stage 0: approach (solved; kept deliberately cheap)
    approach_src = 0.2 * s["reach_lin"] + 0.1 * s["reach_fine"]      # max 0.3 per hand
    approach_rcv = 0.2 * r["reach_lin"] + 0.1 * r["reach_fine"]
    orient_src = 0.10 * s["orient"]
    orient_rcv = 0.10 * r["orient"]

    # ---------------------------------------------------- stage 1: get the cup BETWEEN thumb and fingers
    oppose_src = 0.3 * s["oppose"]        # logged diagnostic + gentle pull; 0 for a same-side thumb
    oppose_rcv = 0.3 * r["oppose"]
    pocket_src = 0.8 * s["pocket"]        # dense "cup inside the opening" signal
    pocket_rcv = 0.8 * r["pocket"]
    pose_src = 1.2 * s["pose"]            # strict straddle; the gate for everything below
    pose_rcv = 1.2 * r["pose"]
    # min(), not sum(): last round the source hand improved while the receiver went backwards, which a
    # per-hand sum rewards. A minimum cannot be raised by the good hand alone.
    pose_both = 2.0 * torch.minimum(s["pose"], r["pose"])

    # ---------------------------------------------------- stage 2: onto the wall and squeeze
    close_src = 2.0 * s["close"]          # weight 2.0 but NO floor: 0 unless the pose can trap the cup
    close_rcv = 2.0 * r["close"]
    squeeze_src = 0.8 * s["squeeze"]      # command the fingers shut, only inside a valid pose in contact range
    squeeze_rcv = 0.8 * r["squeeze"]
    touch_src = 1.0 * s["touch"]          # one side pressing while the other is in place — a step, not a rest
    touch_rcv = 1.0 * r["touch"]
    grip_src = 3.0 * s["grip"]            # weight 3.0: both sides above 1 N is what the env calls a grasp
    grip_rcv = 3.0 * r["grip"]
    grasp_src = 4.0 * s["grasp"]          # weight 4.0 > grip 3.0: the flag itself is the target
    grasp_rcv = 4.0 * r["grasp"]
    grasp_both = 3.0 * s["grasp"] * r["grasp"]
    curl_far_pen = -0.3 * (s["curl_far"] + r["curl_far"])
    air_pinch_pen = -0.5 * (s["air_pinch"] * (1.0 - hold_s) + r["air_pinch"] * (1.0 - hold_r))
    rim_hook_pen = -0.5 * (s["rim_hook"] + r["rim_hook"])

    # ---------------------------------------------------- stage 3: lift
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # weight 8.0 per hand > the whole grasp plateau's marginal value: a finished grasp on the table is not the end
    lift_src = 8.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.025)
    lift_rcv = 8.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.025)
    lifted = hold_s * (h_src > 0.03).to(dtype) * hold_r * (h_rcv > 0.03).to(dtype)

    # ---------------------------------------------------- stage 4: carry together (tilt-aware target)
    tilt = ctx.src_cup_tilt
    # upright: mouths 10 cm apart (bodies side by side, no nesting); the gap closes as tilt -> ~90 deg,
    # because in a real pour the source ORIGIN stays aside and the tilted mouth swings over the receiver.
    pour_prog = torch.clamp((tilt - 0.35) / 1.2, 0.0, 1.0)
    tgt_dxy = 0.10 * (1.0 - pour_prog)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    dxy_err = torch.abs(mouth_dxy - tgt_dxy)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = _band_err(mouth_dz, 0.04, 0.12)                     # source mouth 4..12 cm higher
    carry_q = (0.5 * torch.clamp(1.0 - dxy_err / 0.30, 0.0, 1.0) + 0.5 * _near(dxy_err, 30.0)) * _near(dz_err, 15.0)
    align = 8.0 * lifted * q_s * carry_q     # weight 8.0 = lift: carrying toward the receiver must keep paying

    # ---------------------------------------------------- stage 5: tilt and pour
    zone = _near(dxy_err, 40.0) * _near(dz_err, 40.0)            # 1 cm error -> 0.67
    tilt_r = 7.0 * lifted * q_s * zone * torch.clamp(tilt / 2.0, 0.0, 1.0)   # tilt pays only on the schedule
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads); increments, never the level
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # ---------------------------------------------------- stage 6: success
    success = 30.0 * ctx.success.to(dtype)   # 30/step on top of all held income: keep the goal state

    # ---------------------------------------------------- constraints (guards, not shaping)
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    pre_tilt_pen = -1.0 * (1.0 - lifted * zone) * has_beads * torch.tanh(2.0 * torch.clamp(tilt - 0.8, min=0.0))
    rcv_upright_pen = -0.8 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.15, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    # knocking a cup over and shoving it across the table are the same failure (pressing one side only)
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

    # ---------------------------------------------------- regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "approach_src": approach_src, "approach_rcv": approach_rcv,
        "orient_src": orient_src, "orient_rcv": orient_rcv,
        "oppose_src": oppose_src, "oppose_rcv": oppose_rcv,
        "pocket_src": pocket_src, "pocket_rcv": pocket_rcv,
        "pose_src": pose_src, "pose_rcv": pose_rcv, "pose_both": pose_both,
        "close_src": close_src, "close_rcv": close_rcv,
        "squeeze_src": squeeze_src, "squeeze_rcv": squeeze_rcv,
        "touch_src": touch_src, "touch_rcv": touch_rcv,
        "grip_src": grip_src, "grip_rcv": grip_rcv,
        "grasp_src": grasp_src, "grasp_rcv": grasp_rcv, "grasp_both": grasp_both,
        "curl_far_pen": curl_far_pen, "air_pinch_pen": air_pinch_pen, "rim_hook_pen": rim_hook_pen,
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
bead/spill: [4.88e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.14e-05 · max 0.00161
done/drop: [0.000977, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000153 · max 0.00977
done/mimic_runaway: [0.00586, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000106 · max 0.00586
episode_lengths/step: [113, 840, 852, 868, 870, 885, 890, 886, 880, 865]  min 113 · mean 847 · max 898
reward/action_rate_pen: [-0.0514, -0.044, -0.0424, -0.0396, -0.0361, -0.0345, -0.0338, -0.0313, -0.0304, -0.0288]  min -0.0518 · mean -0.036 · max -0.0278
reward/air_pinch_pen: [0, -6.69e-05, -0.000376, -7.79e-05, -0.00519, -0.00176, -0.000842, -0.00195, -0.00131, -0.00104]  min -0.0178 · mean -0.00184 · max 0
reward/align: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/approach_rcv: [0.101, 0.147, 0.188, 0.219, 0.25, 0.264, 0.272, 0.282, 0.283, 0.278]  min 0.04 · mean 0.236 · max 0.285
reward/approach_src: [0.107, 0.167, 0.227, 0.265, 0.268, 0.274, 0.28, 0.284, 0.279, 0.283]  min 0.0538 · mean 0.25 · max 0.286
reward/arm_speed_pen: [-0.0565, -0.0246, -0.0133, -0.0112, -0.0103, -0.00885, -0.00778, -0.0073, -0.0073, -0.00685]  min -0.0565 · mean -0.0126 · max -0.0062
reward/close_rcv: [0.000423, 0, 0, 4.41e-09, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.32e-06 · max 0.0011
reward/close_src: [0.000165, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.54e-06 · max 0.000289
reward/cup_collision_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0048 · mean -7.36e-05 · max 0
reward/cup_speed_pen: [-0.0089, 0, -1.98e-05, 0, 0, 0, 0, 0, -0.000181, 0]  min -0.0089 · mean -0.00017 · max 0
reward/curl_far_pen: [0, -0.0116, -0.00506, -0.00233, -0.00104, -2.11e-05, -1.47e-05, -0.000102, -6.02e-05, -5.68e-06]  min -0.0265 · mean -0.00267 · max 0
reward/disturb_pen: [-0.0475, -0.00118, -0.000741, -0.000492, -0.000946, -5.04e-06, -3.49e-05, -0.000295, -0.000461, -0.000149]  min -0.0678 · mean -0.00189 · max 0
reward/drop_pen: [-0.00306, 0, 0, 0, 0, 0, 0, 0, -0.000524, 0]  min -0.00759 · mean -0.000112 · max 0
reward/grasp_both: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grip_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grip_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/hand_foreign_pen: [-0.00483, -0.000201, -0.00195, 0, -0.00314, -0.00195, 0, 0, 0, 0]  min -0.0233 · mean -0.00101 · max 0
reward/lift_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/nested_pen: [-0.00977, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0293 · mean -0.000611 · max 0
reward/oppose_rcv: [0.0143, 2.8e-05, 0, 0.000123, 0, 1.08e-05, 0, 0, 0, 0]  min 0 · mean 0.000206 · max 0.0143
reward/oppose_src: [0.0145, 0, 2.88e-05, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000129 · max 0.0145
reward/orient_rcv: [0.0224, 0.0292, 0.0382, 0.0425, 0.0428, 0.0414, 0.0487, 0.0543, 0.0578, 0.0594]  min 0.0111 · mean 0.0457 · max 0.0609
reward/orient_src: [0.022, 0.0273, 0.041, 0.0417, 0.0492, 0.0579, 0.0606, 0.0599, 0.062, 0.0629]  min 0.0114 · mean 0.0498 · max 0.0651
reward/palm_push_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pocket_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.49e-07 · max 0.000303
reward/pocket_src: [1.23e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.8e-08 · max 1.23e-05
reward/pose_both: [1.97e-06, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.21e-08 · max 6.2e-06
reward/pose_rcv: [0.000456, 0, 0, 2.53e-08, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.31e-06 · max 0.0009
reward/pose_src: [0.000184, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.61e-06 · max 0.000253
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pre_tilt_pen: [-0.00601, -0.00035, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0222 · mean -0.000278 · max 0
reward/rcv_upright_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/rim_hook_pen: [-0.314, -0.00488, -0.0083, -0.0083, -0.0146, -0.0117, -0.0107, -0.0083, -0.00537, -0.00977]  min -0.314 · mean -0.0153 · max -0.00146
reward/spill_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0146 · mean -6.45e-05 · max 0.00488
reward/squeeze_rcv: [8.03e-05, 0, 0, 5.28e-10, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.42e-06 · max 0.000206
reward/squeeze_src: [2.62e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.49e-07 · max 8.32e-05
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [-0.221, 0.283, 0.421, 0.504, 0.535, 0.576, 0.606, 0.632, 0.631, 0.636]  min -0.245 · mean 0.509 · max 0.654
reward/touch_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.04e-06 · max 0.00034
reward/touch_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
rewards/step: [-28.3, 216, 348, 425, 489, 512, 537, 544, 551, 549]  min -57.3 · mean 444 · max 572
task/aim_dist: [0.309, 0.319, 0.32, 0.32, 0.32, 0.322, 0.321, 0.321, 0.322, 0.324]  min 0.309 · mean 0.325 · max 0.911
task/cup_collision_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000125 · max 0.00781
task/cups_center_dist: [0.311, 0.319, 0.32, 0.32, 0.32, 0.322, 0.32, 0.32, 0.321, 0.322]  min 0.311 · mean 0.325 · max 0.912
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.00488, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000305 · max 0.0146
task/rcv_cup_lift: [0.001, 1.77e-05, 3.9e-05, 0.000121, 7.46e-05, 0.000156, 0.00025, 0.000287, 0.000166, 0.000242]  min -0.000198 · mean 0.000581 · max 0.0494
task/rcv_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/rcv_hand_foreign_rate: [0.00293, 0, 0, 0, 0.00195, 0, 0, 0, 0, 0]  min 0 · mean 0.000376 · max 0.0156
task/src_cup_lift: [0.000935, 5.72e-06, 6.1e-05, 0.000102, 8.76e-05, 0.000147, 0.00012, 0.000132, 0.000256, 0.000572]  min -6.69e-05 · mean 0.000489 · max 0.0865
task/src_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/src_hand_foreign_rate: [0, 0, 0.000977, 0, 0, 0.000977, 0, 0, 0, 0]  min 0 · mean 0.000283 · max 0.00391
task/src_tilt_deg: [3.13, 0.108, 0.218, 0.4, 0.279, 0.471, 0.379, 0.431, 0.721, 1.98]  min 0.015 · mean 0.578 · max 4.85
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations for this round (previous reward function iter_07, trained from scratch for 680 epochs, 1024 envs, 3.47 h). Items 1-6 are facts: training metrics, the previous reward function's own terms, new environment measurements and a rollout video. Item 7 records operator decisions.

1. **The change made last round did what it was meant to do: the reward no longer pays for a posture that cannot grasp.** Last round's finding was that the posture term paid 0.19-0.38 for a hand with the thumb on the same side as the fingers. This round's reward gated everything on real opposition and removed the floors. The result is that the posture terms paid essentially nothing for the whole run:

| term | ep 0-10 | ep 200-210 | ep 400-410 | ep 670-680 | positive epochs in the last 150 |
|---|---|---|---|---|---|
| `reward/oppose_src` | 0.0066 | 0.0000 | 0.0000 | 0.0000 | 0 / 150 |
| `reward/oppose_rcv` | 0.0080 | 0.00002 | 0.0000 | 0.0000 | 6 / 150 (max 2.5e-05) |
| `reward/pose_src` / `pose_rcv` | 0.00006 / 0.0002 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 150 · 0 / 150 |
| `reward/close_src` / `close_rcv` | 0.00005 / 0.0002 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 150 · 0 / 150 |
| `reward/pocket_src` / `pocket_rcv` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 150 · 0 / 150 |
| `reward/grip_src` / `grip_rcv` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 150 · 0 / 150 |

2. **With no income past the approach, the policy settled on the approach term and stopped there.**
   - `reward/total` 0.64 at the end, of which `approach_src` 0.282 and `approach_rcv` 0.280 — 88 % of all income.
   - `task/src_palm_to_cup` 0.231 → 0.128 (ep 200) → 0.120 (ep 400) → 0.121 m; `task/rcv_palm_to_cup` 0.363 → 0.149 → 0.126 → 0.121 m. Both palms have been pinned at 12 cm since about epoch 400, which is where the approach term saturates.
   - `task/src_near_rate` (palm within 10 cm) stayed at 0-1 %, so the environment's rim/oppose measurements, which only count envs inside 12 cm, had almost no sample all round.

3. **The hand closed to narrower than the cup and stayed there. This is the mechanism that blocks the pocket.**

| metric | ep 0-10 | ep 200-210 | ep 400-410 | ep 670-680 |
|---|---|---|---|---|
| `task/src_tip_gap_mm_near` [mm] | 55.4 | 52.2 | 50.5 | 48.3 |
| `task/rcv_tip_gap_mm_near` [mm] | 66.6 | – | 48.9 | 49.5 |
| `task/src_closure` / `task/rcv_closure` | 0.15 / 0.14 | 0.26 / 0.36 | 0.32 / 0.44 | 0.36 / 0.29 |
| `task/{src,rcv}_cup_in_pocket_near` | 0 / 0.17 | 0 / 0 | 0 / 0 | 0 / 0 |

   - `tip_gap` is the distance from the thumb tip to the nearer of the index/middle tips. The cup is 57 mm across.
   - At reset the hand is 55-67 mm wide, which is wide enough. Over training it narrowed monotonically to 48-50 mm, i.e. the policy actively closed the hand to less than the cup's width and kept it there while standing 12 cm away.
   - Consequently the cup was never between the tips: `cup_in_pocket` was 0 in every one of the last 150 epochs.
   - Nothing in the reward asks the hand to be open wider than the cup near the cup. `curl_far` only penalises curling while FAR from the pre-grasp region, and every term that would pay for opening (`pocket`, `pose`, `close`) is gated behind an opposition that the closed hand can never reach.

4. **Contact, grasp and lift.**

| metric | ep 0-10 | ep 200-210 | ep 400-410 | ep 670-680 |
|---|---|---|---|---|
| `contact/src_max` / `contact/rcv_max` [N] | 0.16 / 0.66 | 0.02 / 0.04 | 0.06 / 0.08 | 0.12 / 0.09 |
| `task/src_grasped` / `task/rcv_grasped` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `task/src_cup_lift` / `task/rcv_cup_lift` [m] | 0.001 / 0.013 | 0.0001 / 0.0001 | 0.0001 / 0.0002 | 0.0003 / 0.0003 |
| `task/episode_success` | 0 | 0 | 0 | 0 |

   - 0.09-0.12 N with opposition at zero is a graze, not a press. The environment's grasp flag needs the thumb above 1 N AND another finger above 1 N at the same time.
   - The thumb height did come down: `task/src_thumb_above_rim_mm_near` reached −15.6 mm mid-round, and `task/{src,rcv}_thumb_over_rim_near` was 0 all round, so the old rim-hook failure is gone. Height is solved; the left-right placement is not.

5. **Rollout video** at the epoch-650 checkpoint (4 envs, 900 steps, `our_source/pour_t2r_rh_i07_ep650_0916.mp4`, frame sheet and 2.4x crops in `pour_t2r_rh_i07_ep650_0916_frames/`). The question asked of the video was whether the hand arrives already closed or opens and then closes again. The answer is the first:
   - At step 60, while the arm is still travelling, the four fingers are ALREADY curled downward. The hand passes beside the cup in that shape.
   - At step 500 the hand has settled next to the cup in the same shape and stays there to the end of the episode. The fingertips point down at the mat, not around the cup wall, and the cup stands to one side of the palm rather than between the fingers and the thumb.
   - **Operator observation (takes precedence over the frame reading above): both hands put the BACK of the index-to-pinky fingers against the cup and then hold still.** The dorsal side of the fingers is what meets the cup, not the pads. The finger force sensors report whatever touches the link, in any direction, so the 0.09-0.12 N in item 4 is contact on the wrong surface entirely — the palmar side never faces the cup.
   - At no point in the episode does the hand open wider and then close. So the narrow `tip_gap` in item 3 is not a closing motion that overshoots; the hand simply never opens, which matches a reward that contains no term asking it to.

5b. **The reward cannot tell the palm side from the back of the hand.** In the previous reward the orientation term is

```
face = |normal_xy · u_xy|        # absolute value
orient = near_pre * (0.5*face + 0.5*ahead)
```

   where `normal` is the palm-pad normal (`palm_axes[:, 0:3]`) and `u_xy` points from the palm to the cup. Because of the absolute value, a palm facing the cup and a palm facing exactly away from it score identically. Nothing else in the reward looks at pad direction either: `pose`, `pocket` and `close` use tip POSITIONS only, and `ctx.*_finger_force` is the net force on the sensor link, which is direction-agnostic. So the posture the operator saw — the back of the fingers laid against the cup — costs nothing and reads as legitimate contact. Any fix has to make the pad side facing the cup a signed requirement, and must not count force arriving on the dorsal side as progress.

6. **Physics stayed clean and was the best of any round.** `ctrl/mimic_err_max` 1.35 at start → 0.12 rad at the end. No mimic runaway, no drop, no cup collision.

7. **Operator decisions** (after reviewing items 1-6 and the video):
   - This draft is approved as the feedback for this round, and a new reward is to be generated. Training starts from scratch.
   - **Keep what worked.** The opposition gate with no floors (item 1) did its job: no posture that cannot grasp was paid anything, all round. Do not reintroduce a floor that pays for standing near the cup.
   - **Three things the next reward must add**, because the round showed each of them missing:
     1. *The pad side must face the cup, as a signed condition.* Replace the absolute value in `face` (item 5b). A palm turned away from the cup must score zero, not full marks.
     2. *Opening the hand near the cup must be rewarded before opposition is reached.* The thumb-to-finger gap must exceed the cup's 57 mm while the palm is near the cup. This is the step with no gradient today, and the reason the hand closed to 48-50 mm and stayed there (item 3).
     3. *Contact arriving on the back of the fingers must not count as progress.* Force on a finger whose pad does not face the cup is not a grasp in the making.
   - The goal after that is unchanged: the thumb and at least one opposing finger both above 1 N at the same time (the environment's grasp flag), then lifting.
   - The three measurements added in the previous round stay logging-only, and `task/{src,rcv}_tip_gap_mm_near` is the metric to watch for requirement 2.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
