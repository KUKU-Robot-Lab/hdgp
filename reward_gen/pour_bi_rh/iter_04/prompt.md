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


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted)."""
    ARRIVE_RISE = 0.045    # arrive 45 mm above grasp height (0/336 knock-overs at +45 mm) ...
    FUNNEL_R0 = 0.01       # ... and be at grasp height once the cup is centred within 1 cm
    FUNNEL_R1 = 0.04
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z        # pinch height above cup origin (~13 mm, inside the 0-25 mm lift group)
    thumb_hi = 0.5 * cup_mouth_z        # thumb band 0..27 mm above origin (no lift with thumb >= 50 mm)

    up = cup_up.unsqueeze(1)                                   # (N,1,3)
    rel = tips_pos - cup_pos.unsqueeze(1)                      # (N,F,3)
    h_tip = (rel * up).sum(dim=-1)                             # (N,F) height along the cup axis
    rad_vec = rel - h_tip.unsqueeze(-1) * up                   # (N,F,3)
    rad = rad_vec.norm(dim=-1)                                 # (N,F)
    # gap is zero only at the measured contact radius (tip frame 26-28 mm, cup outer 28.5 mm);
    # the old +10 mm pad made 33-37 mm count as "on the wall" and killed the gradient
    gap = torch.clamp(rad - tip_r, min=0.0)                    # (N,F)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)        # (N,)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values   # nearest non-thumb tip, (N,)

    # opposition: 1 when the thumb and the other fingers sit on opposite sides of the cup axis
    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))                # (N,)

    # pinch centre and the descending funnel target
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))  # (N,3)
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    rise = ARRIVE_RISE * torch.clamp((rad_m - FUNNEL_R0) / (FUNNEL_R1 - FUNNEL_R0), 0.0, 1.0)
    d_funnel = torch.sqrt(rad_m ** 2 + (h_m - grasp_h - rise) ** 2)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp,
            "d_funnel": d_funnel, "d_grip": d_grip, "thumb_band": thumb_band}


def _hand_cup_terms(tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd, cup_pos, cup_up,
                    cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok, lift_target):
    """Unweighted per-arm terms, each in [0, 1]. pose_ok (N,) = cup attitude is acceptable for this stage."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)

    # A. reach: long-range funnel term + per-side fingertip-on-wall term.
    #    Each side is scored on its own (sum form) so the thumb keeps its full gradient while the finger
    #    side is already done, and measured force completes a side: reach saturates only with contact
    #    force on BOTH sides (the parked pose of i01 had thumb 0.00 N and reach 0.97).
    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh(f_o / 0.5))
    reach_tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach = 0.4 * reach_far + 0.6 * reach_tips

    # B. grasp: balanced two-sided pinch. Before the flag, finger-first touch earns 0.15 and any thumb force
    #    added to it can only raise the value (0.5 N -> +0.19); thumb-only still earns nothing.
    #    Flag boundary (1 N / 1 N): pre_grasp 0.355 -> flagged 0.75, a step up, never a valley.
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)          # lifts had thumb ~2.3 N vs fingers ~2.7 N
    pre_grasp = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    contact_q = torch.where(g, 0.4 + 0.6 * pinch, pre_grasp)
    grasp = contact_q * (0.1 + 0.9 * geo["thumb_band"]) * pose_ok

    # closure shaping: measured grasps close to 0.68-0.98, the parked hand sat at 0.19 with the closure
    # channels driven open; pays only once the pinch centre is at the grasp pose (contact freeze makes
    # closing there harmless), saturates at 0.7 so over-squeezing is not rewarded
    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # C. lift: measured cup elevation only, clamped, and only while really held (explosion-proof)
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)                # 5 mm dead band: tipping on the rim is not lifting
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    held = g_f * torch.exp(-(geo["d_grip"] / 0.08) ** 2) * calm
    lift_prog = 0.5 * torch.tanh(h_pos / 0.02) + 0.5 * torch.clamp(h_pos / lift_target, max=1.0)
    lift = held * pose_ok * lift_prog
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)               # 0 on the table, 1 from 2.5 cm up
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)            # "held and at least 5 cm up"

    # penalties (unweighted, bounded)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)       # 1 N dead band: a slightly stronger thumb is free
    finger_excess = torch.clamp(f_o - f_t, min=0.0)             # finger-only contact: adding thumb force lowers this
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)   # knock-overs came at ~0.09 m/s hand speed
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"reach": reach, "grasp": grasp, "close": close, "lift": lift, "up_w": up_w, "slide": slide,
            "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def _pour_geometry(ctx, outer_r):
    """Distance of the source pouring lip (lowest rim point) from a point above the receiver rim centre."""
    POUR_CLEARANCE = 0.05   # lip 5 cm above the receiver rim: no cup-cup contact, origins stay > 9 cm apart
    ez = torch.tensor([0.0, 0.0, 1.0], device=ctx.src_cup_up.device, dtype=ctx.src_cup_up.dtype)
    # (up * up_z - ez) has norm sin(tilt): zero when upright, straight down by one radius when horizontal
    lip = ctx.src_cup_mouth_pos + outer_r * (ctx.src_cup_up * ctx.src_cup_up[:, 2:3] - ez)
    delta = lip - (ctx.rcv_cup_mouth_pos + POUR_CLEARANCE * ez)
    d_xy = delta[:, :2].norm(dim=-1)
    d_align = torch.sqrt(d_xy ** 2 + (0.5 * delta[:, 2]) ** 2)          # height error counts half
    align_gate = torch.exp(-(d_align / 0.06) ** 2)                       # wide: tilting is allowed here
    pour_gate = torch.exp(-(d_align / max(ctx.cup_radius, 0.015)) ** 2)  # narrow: beads would land inside
    return d_align, align_gate, pour_gate


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Stage ladder (per step, cumulative): reach 1 < grasp 2 < lift 3 per arm, then align 4 < tilt 5.
    # A later stage keeps every earlier term, so progress never lowers the reward.
    W_REACH, W_GRASP, W_LIFT = 1.0, 2.0, 3.0
    W_CLOSE = 0.3         # closure shaping at the grasp pose; small next to grasp 2.0, only a gradient helper
    W_ALIGN, W_TILT = 4.0, 5.0
    W_POUR = 150.0        # x d_in_target: 7.5 per bead with 20 beads, one-time, dominates the dense terms at that step
    W_SPILL = 60.0        # x d_spill: 3 per bead, below the pour pay so an imperfect pour is still worth trying
    W_SUCCESS = 20.0      # per step while the env reports success
    W_ALIVE = 1.0         # with reach it outweighs all persistent penalties, so ending the episode never pays
    W_SLIDE, W_TILT_PEN = 0.5, 0.3
    W_ONE_SIDED, W_PALM, W_RUSH, W_FLING = 0.3, 0.2, 0.2, 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05     # x mean squared action change (<= 4), regularisation only
    CUP_OUTER_R = 0.0285  # cup wall, used for the pouring lip
    TIP_R = 0.027         # tip-frame radius at measured two-sided contact (thumb 27.9 mm, nearest finger 26.2 mm)
    UPRIGHT_TOL = 0.35    # rad, = the 20 deg success limit on the receiver cup
    TILT_FULL = math.radians(100.0)   # half the beads are out by 86 deg, reward saturates a little past that
    LIFT_TARGET_SRC, LIFT_TARGET_RCV = 0.15, 0.08   # source ends above the receiver

    dtype = ctx.src_cup_tilt.dtype
    d_align, align_gate, pour_gate = _pour_geometry(ctx, CUP_OUTER_R)
    upright_src = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = align_gate + (1.0 - align_gate) * upright_src   # source may tilt only over the receiver
    pose_ok_rcv = upright_rcv

    src = _hand_cup_terms(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                          ctx.src_grasped, ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up,
                          ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_src, LIFT_TARGET_SRC)
    rcv = _hand_cup_terms(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                          ctx.rcv_grasped, ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                          ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_rcv, LIFT_TARGET_RCV)

    nested = ctx.cups_nested.to(dtype)
    both_up = src["up_w"] * rcv["up_w"] * (1.0 - nested)
    align = both_up * (0.5 * (1.0 - torch.tanh(4.0 * d_align)) + 0.5 * torch.exp(-25.0 * d_align))
    pour_tilt = both_up * pour_gate * torch.clamp(ctx.src_cup_tilt / TILT_FULL, max=1.0)

    tilt_pen_src = (1.0 - align_gate) * torch.tanh(ctx.src_cup_tilt / UPRIGHT_TOL)
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    act_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)

    comps = {
        "alive": W_ALIVE * torch.ones_like(ctx.src_cup_tilt),
        "reach_src": W_REACH * src["reach"],
        "reach_rcv": W_REACH * rcv["reach"],
        "close_src": W_CLOSE * src["close"],
        "close_rcv": W_CLOSE * rcv["close"],
        "grasp_src": W_GRASP * src["grasp"],
        "grasp_rcv": W_GRASP * rcv["grasp"],
        "lift_src": W_LIFT * src["lift"],
        "lift_rcv": W_LIFT * rcv["lift"],
        "align_mouths": W_ALIGN * align,
        "pour_tilt": W_TILT * pour_tilt,
        "pour_delta": W_POUR * ctx.d_in_target,
        "spill_delta": -W_SPILL * ctx.d_spill,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype),
        "cup_disturb_src": -(W_SLIDE * src["slide"] + W_TILT_PEN * tilt_pen_src),
        "cup_disturb_rcv": -(W_SLIDE * rcv["slide"] + W_TILT_PEN * tilt_pen_rcv),
        "contact_push": -(W_ONE_SIDED * (src["one_sided"] + rcv["one_sided"])
                          + W_PALM * (src["palm_push"] + rcv["palm_push"])),
        "rush_near_cup": -W_RUSH * (src["rush"] + rcv["rush"]),
        "cup_fling": -W_FLING * (src["fling"] + rcv["fling"]),
        "cup_collision": -W_CUP_CUP * torch.tanh(ctx.cup_cup_force / 5.0),
        "hand_foreign": -W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                      + torch.tanh(ctx.rcv_hand_foreign_force / 5.0)),
        "nested": -W_NESTED * nested,
        "action_rate": -W_ACT_RATE * act_rate,
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0, 0, 4.88e-05, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.45e-05 · max 0.000244
bead/spill: [0, 0.00186, 0.00356, 0.0165, 0.00303, 0.00249, 0.000684, 0.00205, 0.00166, 0.000928]  min 0 · mean 0.00386 · max 0.0271
done/drop: [0, 0, 0.000977, 0.000977, 0.000977, 0, 0, 0, 0, 0]  min 0 · mean 0.000138 · max 0.00391
done/mimic_runaway: [0, 0, 0, 0, 0, 0, 0.000977, 0, 0, 0]  min 0 · mean 0.000127 · max 0.00293
episode_lengths/step: [116, 834, 783, 524, 802, 795, 860, 846, 843, 847]  min 116 · mean 812 · max 899
reward/action_rate: [-0.0181, -0.0302, -0.0335, -0.0362, -0.0377, -0.0331, -0.0312, -0.0292, -0.029, -0.0304]  min -0.0391 · mean -0.0315 · max -0.0181
reward/align_mouths: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.7e-07 · max 0.000546
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/close_rcv: [0.0302, 0.0783, 0.0878, 0.05, 0.0101, 0.0152, 0.00678, 0.00181, 0.00311, 0.00296]  min 0.00152 · mean 0.0308 · max 0.112
reward/close_src: [8.88e-06, 0.126, 0.119, 0.0946, 0.069, 0.0754, 0.0774, 0.0891, 0.0734, 0.0825]  min 8.88e-06 · mean 0.0913 · max 0.153
reward/contact_push: [-0.000288, -0.0276, -0.0307, -0.0351, -0.00244, -0.00537, -0.00122, -0.000797, -0.00155, -0.000482]  min -0.0696 · mean -0.0116 · max -2.04e-05
reward/cup_collision: [0, -0.000736, 0, -0.0014, 0, 0, -0.000508, 0, 0, 0]  min -0.00282 · mean -8.6e-05 · max 0
reward/cup_disturb_rcv: [-0.0381, -0.0606, -0.117, -0.142, -0.0784, -0.0894, -0.0603, -0.0427, -0.0467, -0.0524]  min -0.242 · mean -0.0751 · max -0.0381
reward/cup_disturb_src: [-0.0349, -0.217, -0.206, -0.169, -0.0786, -0.07, -0.0548, -0.0401, -0.0512, -0.0492]  min -0.302 · mean -0.105 · max -0.0349
reward/cup_fling: [0, -0.00105, -0.00275, -0.00713, -0.00268, -0.00168, -0.00185, -0.00144, -0.00143, -0.00148]  min -0.00944 · mean -0.00213 · max 0
reward/grasp_rcv: [0.000206, 0.00879, 0.0503, 0.0104, 0.000345, 0.000297, 8.87e-05, 4.72e-05, 0.000372, 0]  min 0 · mean 0.0039 · max 0.0931
reward/grasp_src: [0, 0.288, 0.306, 0.836, 1.39, 1.41, 1.55, 1.56, 1.53, 1.57]  min 0 · mean 1.09 · max 1.63
reward/hand_foreign: [0, -0.000409, -0.00152, -0.00222, -0.000447, -0.000102, -0.00142, -5.91e-07, -0.000846, -0.000339]  min -0.00579 · mean -0.000622 · max 0
reward/lift_rcv: [0, 0, 0, 2.81e-05, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.23e-05 · max 0.00308
reward/lift_src: [0, 0.000109, 0, 0.648, 1.6, 1.65, 1.97, 2.06, 1.97, 2.08]  min 0 · mean 1.27 · max 2.16
reward/nested: [0, -0.00391, 0, -0.00391, -0.00195, 0, -0.00195, 0, 0, 0]  min -0.0195 · mean -0.000797 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0146 · mean 3.48e-05 · max 0.022
reward/pour_tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.11e-21 · max 5.24e-17
reward/reach_rcv: [0.55, 0.698, 0.75, 0.56, 0.266, 0.391, 0.299, 0.255, 0.276, 0.27]  min 0.144 · mean 0.415 · max 0.782
reward/reach_src: [0.339, 0.844, 0.852, 0.761, 0.664, 0.666, 0.664, 0.689, 0.649, 0.678]  min 0.327 · mean 0.713 · max 0.896
reward/rush_near_cup: [-0.268, -0.238, -0.224, -0.162, -0.0782, -0.118, -0.087, -0.054, -0.06, -0.0661]  min -0.268 · mean -0.125 · max -0.033
reward/spill_delta: [0, -6.98e-10, -0.0322, -0.00879, -0.0117, 0, 0, -0.00293, 0.00879, -0.00293]  min -0.0732 · mean -0.00329 · max 0.0498
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [1.56, 2.42, 2.45, 3.31, 4.65, 4.84, 5.29, 5.44, 5.27, 5.44]  min 1.37 · mean 4.22 · max 5.59
rewards/step: [131, 1.99e+03, 2.01e+03, 1.77e+03, 3.73e+03, 3.88e+03, 4.49e+03, 4.57e+03, 4.49e+03, 4.55e+03]  min 131 · mean 3.47e+03 · max 4.84e+03
task/aim_dist: [0.321, 0.309, 1.17, 2.02, 0.268, 0.274, 0.336, 0.32, 0.477, 0.334]  min 0.238 · mean 0.51 · max 4.01
task/cup_collision_rate: [0, 0.00195, 0, 0.00195, 0, 0, 0.000977, 0, 0, 0]  min 0 · mean 0.000121 · max 0.00391
task/cups_center_dist: [0.321, 0.311, 1.18, 2.01, 0.268, 0.275, 0.337, 0.32, 0.478, 0.334]  min 0.239 · mean 0.51 · max 4
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.00195, 0, 0.00195, 0.000977, 0, 0.000977, 0, 0, 0]  min 0 · mean 0.000399 · max 0.00977
task/rcv_cup_lift: [1.4e-05, 8.71e-05, 0.284, 0.00103, 0.00139, 0.000263, 0.0037, 3.19e-05, 0.0186, 5.81e-05]  min -0.000542 · mean 0.00984 · max 0.414
task/rcv_grasped: [0, 0, 0.00391, 0.00293, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000294 · max 0.0195
task/rcv_hand_foreign_rate: [0, 0, 0.00391, 0.00195, 0.000977, 0.000977, 0.00293, 0, 0, 0.000977]  min 0 · mean 0.00116 · max 0.0195
task/src_cup_lift: [2.71e-06, 0.00127, 0.0481, 0.819, 0.148, 0.147, 0.221, 0.223, 0.23, 0.238]  min 2.42e-06 · mean 0.179 · max 1.11
task/src_grasped: [0, 0.0156, 0.0195, 0.447, 0.744, 0.753, 0.834, 0.832, 0.828, 0.851]  min 0 · mean 0.556 · max 0.881
task/src_hand_foreign_rate: [0, 0.000977, 0, 0.00391, 0.000977, 0, 0.000977, 0, 0.00293, 0]  min 0 · mean 0.000807 · max 0.0127
task/src_tilt_deg: [0.0153, 2.33, 3.81, 8.42, 4.42, 3.68, 2.69, 2.25, 2.63, 2.82]  min 0.0146 · mean 3.29 · max 11.2
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Operator notes for round 4 (iter_03 result). Every number below was measured in this simulation from run t2r_rh2_i03 (iter_03 reward, 1024 envs, started from the i01 epoch-1250 checkpoint, stopped by the operator at epoch 7350 after 41 h). Nothing here is a hypothesis unless it says so. Env unchanged: hand-cup friction 2.0, contact-freeze threshold 1.0 N, grasp flag = thumb > 1.0 N AND another finger > 1.0 N.

## Outcome
task/episode_success, reward/align_mouths, reward/pour_tilt, reward/success_bonus were 0.000 at every epoch. rewards/iter plateaued (4554 at e5000, 4697 at e7186). The run is judged failed: the source hand learned grasp and lift, the receiver hand abandoned its cup.

## Source hand (worked)
10-epoch means at e2000 / e2500 / e3000 / e7186:
- reward/grasp_src 0.42 / 0.76 / 1.37 / 1.61, reward/lift_src 0.005 / 0.73 / 1.62 / 2.09, reward/reach_src 0.81 / 0.64 / 0.67 / 0.66.
- At e7186: task/src_grasped 0.85, contact/src_max median 3.6 N, source cup lifted. The iter_03 two-sided reach + force-completed grasp terms are what produced this; keep their form for the source hand.
- The source cup is lifted but never brought over the receiver: task/cups_center_dist 0.34 m at e7186, align_mouths 0.

## Receiver hand (collapsed between e2000 and e2500)
10-epoch means at e1000 / e2000 / e2250 / e2500 / e3000 / e7186:
- task/rcv_near_rate 0.92 / 0.87 / 0.30 / 0.19 / 0.16 / 0.05.
- reward/reach_rcv 0.73 / 0.71 / 0.49 / 0.22 / 0.27 / 0.20.
- reward/grasp_rcv 0.014 / 0.009 / 0.006 / 0.001 / 0.001 / 0.000. task/rcv_grasped <= 0.007 at every epoch, 0.000 from e2500.
- contact/rcv_max 3.3 / 0.54 / 0.19 / 0.05 / 0.06 / 0.01 N. task/rcv_tip_gap_mm_near 66 / 74 / 71 / 77 / 77 / 86 mm (never closed on the cup; the source reached < 45 mm).
- reward/close_rcv 0.088 / 0.102 / 0.037 / 0.012 / 0.009 / 0.002.
- fabric/rcv_rot_err_deg 13 at e2000, 102 at e2500, 91 at e2750, 30 at e3000, 18 at e7186.
- In the e7150 play video one arm stays folded near the torso for the whole episode.

Timing: the receiver retreat (e2000 -> e2500) coincides with lift_src switching on (0.005 -> 0.73).

Penalties while the receiver was still near its cup (e2000): rush_near_cup -0.23 (both hands summed), contact_push -0.047, cup_disturb_rcv -0.13, cup_disturb_src -0.25. After the retreat (e2500): rush -0.10, contact_push -0.013, cup_disturb_rcv -0.076. So near its cup the receiver collected reach about 0.7 with grasp about 0.01 and paid roughly 0.2 to 0.3 in penalties; away from the cup it kept reach about 0.2 and paid almost none. Net loss from leaving was about 0.3 per step, while the source side gained about 3.0 per step over the same epochs.

Measured cup fact (scripted probe, 09.17): the cup (base diameter 37.8 mm, 0.134 kg) tips over at a one-sided contact of 0.3 to 0.4 N. Two-sided simultaneous contact does not tip it.

## Hypotheses (not measured)
- H1: with a shared scalar reward and one policy, the receiver's weak positive signal (reach only, no grasp ever achieved) was outweighed by its near-cup penalties once the optimiser focused on the large source lift gain.
- H2: the receiver fingers stop at 66 to 77 mm tip gap because any one-sided touch tips the cup and triggers slide/tilt penalties, so the policy learned to avoid touching.

## What the operator wants from iter_04
1. The receiver must keep approaching and must grasp. Make receiver progress a precondition for the big source pay: e.g. cap or gate lift_src (and later align/tilt) by receiver stage (reach_rcv, then rcv grasp), so that lifting the source while the receiver is away is worth little.
2. Near-cup penalties for the receiver (rush, slide, tilt, palm push, one-sided) must in total stay clearly below what the receiver earns by being in the grasp pose, and leaving the cup must cost clearly more than staying. Do not remove knock-over protection; rescale it.
3. Keep the source-hand reach/grasp/lift formulation that worked.
4. After both cups are held, there must be a gradient that brings the source mouth over the receiver (cups_center_dist stayed 0.34 m with align 0).

## Launch plan
iter_04 starts from the i03 epoch-2000 checkpoint (receiver near_rate 0.87, source grasp just forming), 1024 envs.
