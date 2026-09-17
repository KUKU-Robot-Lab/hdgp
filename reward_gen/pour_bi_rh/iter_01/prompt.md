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


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, outer_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted)."""
    PAD = 0.010            # tip frame -> pad surface allowance [m]; nothing is paid for pressing deeper
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
    gap = torch.clamp(rad - (outer_r + PAD), min=0.0)          # (N,F) distance left to the cup wall

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


def _hand_cup_terms(tips_pos, finger_force, palm_force, grasped, arm_qd, cup_pos, cup_up,
                    cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, outer_r, pose_ok, lift_target):
    """Unweighted per-arm terms, each in [0, 1]. pose_ok (N,) = cup attitude is acceptable for this stage."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, outer_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)

    # A. reach: long-range funnel term + short-range fingertip-on-wall term
    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    reach_tips = torch.exp(-30.0 * 0.5 * (geo["d_thumb"] + geo["d_oth"])) * (0.3 + 0.7 * geo["opp"])
    reach = 0.5 * reach_far + 0.5 * reach_tips

    # B. grasp: balanced two-sided pinch; fingers-first touch gets a small credit, thumb-only gets none
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)          # lifts had thumb ~2.3 N vs fingers ~2.7 N
    touch_first = 0.15 * torch.tanh(f_o / 1.0) * (1.0 - torch.tanh(f_t / 0.3))
    contact_q = torch.where(g, 0.4 + 0.6 * pinch, touch_first)
    grasp = contact_q * (0.1 + 0.9 * geo["thumb_band"]) * pose_ok

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
    thumb_excess = torch.clamp(f_t - f_o, min=0.0)
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)   # knock-overs came at ~0.09 m/s hand speed
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"reach": reach, "grasp": grasp, "lift": lift, "up_w": up_w, "slide": slide,
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
    W_ALIGN, W_TILT = 4.0, 5.0
    W_POUR = 150.0        # x d_in_target: 7.5 per bead with 20 beads, one-time, dominates the dense terms at that step
    W_SPILL = 60.0        # x d_spill: 3 per bead, below the pour pay so an imperfect pour is still worth trying
    W_SUCCESS = 20.0      # per step while the env reports success
    W_ALIVE = 1.0         # with reach it outweighs all persistent penalties, so ending the episode never pays
    W_SLIDE, W_TILT_PEN = 0.5, 0.3
    W_ONE_SIDED, W_PALM, W_RUSH, W_FLING = 0.3, 0.2, 0.2, 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05     # x mean squared action change (<= 4), regularisation only
    CUP_OUTER_R = 0.0285
    UPRIGHT_TOL = 0.35    # rad, = the 20 deg success limit on the receiver cup
    TILT_FULL = math.radians(100.0)   # half the beads are out by 86 deg, reward saturates a little past that
    LIFT_TARGET_SRC, LIFT_TARGET_RCV = 0.15, 0.08   # source ends above the receiver

    dtype = ctx.src_cup_tilt.dtype
    d_align, align_gate, pour_gate = _pour_geometry(ctx, CUP_OUTER_R)
    upright_src = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = align_gate + (1.0 - align_gate) * upright_src   # source may tilt only over the receiver
    pose_ok_rcv = upright_rcv

    src = _hand_cup_terms(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_grasped,
                          ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel,
                          ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z, CUP_OUTER_R,
                          pose_ok_src, LIFT_TARGET_SRC)
    rcv = _hand_cup_terms(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_grasped,
                          ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel,
                          ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z, CUP_OUTER_R,
                          pose_ok_rcv, LIFT_TARGET_RCV)

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
bead/in_target: [0, 4.88e-05, 4.88e-05, 4.88e-05, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.4e-05 · max 9.77e-05
bead/spill: [9.77e-05, 0.00342, 0.00576, 0.00386, 0.00254, 0.00146, 0, 0, 0, 0]  min 0 · mean 0.00143 · max 0.00576
done/drop: [0.000977, 0.00586, 0.00293, 0.00195, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00103 · max 0.00586
done/mimic_runaway: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
episode_lengths/step: [112, 174, 188, 247, 289, 718, 881, 899, 871, 878]  min 112 · mean 595 · max 899
reward/action_rate: [-0.0516, -0.0509, -0.0509, -0.0512, -0.0502, -0.0502, -0.0499, -0.0499, -0.0507, -0.0501]  min -0.0517 · mean -0.0504 · max -0.0494
reward/align_mouths: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/contact_push: [-0.0105, -0.00693, -0.00338, -0.0013, -0.000244, 0, 0, 0, 0, 0]  min -0.0105 · mean -0.00136 · max 0
reward/cup_collision: [-3.49e-05, -0.000261, -0.00122, -0.000523, -0.000107, -0.000322, -8.83e-06, 0, 0, 0]  min -0.00199 · mean -0.000322 · max 0
reward/cup_disturb_rcv: [-0.117, -0.109, -0.0868, -0.0646, -0.0595, -0.0491, -0.0439, -0.038, -0.0366, -0.0376]  min -0.117 · mean -0.0588 · max -0.0365
reward/cup_disturb_src: [-0.11, -0.126, -0.101, -0.0744, -0.0606, -0.0528, -0.044, -0.0381, -0.0377, -0.0365]  min -0.126 · mean -0.0615 · max -0.0354
reward/cup_fling: [-0.0107, -0.0102, -0.00537, -0.00296, -0.000284, 0, 0, 0, -8.22e-05, 0]  min -0.0112 · mean -0.00216 · max 0
reward/grasp_rcv: [0.000618, 0.000255, 0.000189, 6.04e-05, 2.81e-05, 0, 0, 0, 0, 0]  min 0 · mean 7.21e-05 · max 0.000618
reward/grasp_src: [0.000562, 0.000521, 0.000192, 6.46e-05, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00012 · max 0.00145
reward/hand_foreign: [-0.000957, -0.000548, -0.00215, -0.00058, -3.78e-05, -0.000588, -0.000632, -0.00037, 0, -7.01e-05]  min -0.00264 · mean -0.000685 · max 0
reward/lift_rcv: [0, 0, 9.27e-12, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.65e-13 · max 9.27e-12
reward/lift_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/nested: [-0.00391, -0.0137, -0.00781, -0.00586, -0.00391, -0.00391, -0.00195, 0, 0, 0]  min -0.0156 · mean -0.00396 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pour_tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/reach_rcv: [0.231, 0.223, 0.218, 0.205, 0.181, 0.169, 0.171, 0.174, 0.177, 0.21]  min 0.167 · mean 0.2 · max 0.233
reward/reach_src: [0.239, 0.24, 0.223, 0.215, 0.202, 0.19, 0.188, 0.192, 0.2, 0.201]  min 0.183 · mean 0.209 · max 0.241
reward/rush_near_cup: [-0.0417, -0.0337, -0.0282, -0.0252, -0.018, -0.0107, -0.00964, -0.00996, -0.0119, -0.0172]  min -0.0417 · mean -0.0204 · max -0.00958
reward/spill_delta: [0, -0.00293, -0.00879, 0, -0.00586, 0, 0, 0, 0, 0]  min -0.00879 · mean -0.000335 · max 0.0176
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [1.1, 0.892, 0.979, 1.12, 1.17, 1.17, 1.16, 1.14, 1.2, 1.26]  min 0.881 · mean 1.14 · max 1.27
rewards/step: [81, 153, 165, 239, 284, 756, 1e+03, 1.06e+03, 1.03e+03, 1.04e+03]  min 81 · mean 674 · max 1.07e+03
task/aim_dist: [0.308, 0.779, 0.467, 0.898, 0.317, 0.318, 0.319, 0.319, 0.32, 0.32]  min 0.308 · mean 0.401 · max 0.898
task/cup_collision_rate: [0, 0, 0.000977, 0.00195, 0, 0.000977, 0, 0, 0, 0]  min 0 · mean 0.00053 · max 0.00391
task/cups_center_dist: [0.311, 0.779, 0.468, 0.898, 0.318, 0.318, 0.319, 0.319, 0.32, 0.32]  min 0.311 · mean 0.401 · max 0.898
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.00195, 0.00684, 0.00391, 0.00293, 0.00195, 0.00195, 0.000977, 0, 0, 0]  min 0 · mean 0.00198 · max 0.00781
task/rcv_cup_lift: [0.00086, 0.071, 0.000702, 0.000268, -1.4e-05, 3.56e-05, 3.34e-05, 2.84e-05, 1.34e-05, 7.74e-06]  min -2.1e-05 · mean 0.00412 · max 0.071
task/rcv_grasped: [0, 0, 0.000977, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.79e-05 · max 0.000977
task/rcv_hand_foreign_rate: [0.00195, 0.00195, 0.00293, 0.000977, 0, 0.000977, 0.000977, 0.000977, 0, 0]  min 0 · mean 0.00134 · max 0.00586
task/src_cup_lift: [0.00138, 0.02, 0.0421, 0.0368, 1.63e-05, 4.71e-05, 3.53e-05, 2.91e-05, 1.71e-05, 6.88e-06]  min -2.08e-05 · mean 0.00698 · max 0.0632
task/src_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.79e-05 · max 0.000977
task/src_hand_foreign_rate: [0.000977, 0, 0.00195, 0, 0, 0.000977, 0.000977, 0, 0, 0]  min 0 · mean 0.000474 · max 0.00293
task/src_tilt_deg: [3.46, 4.11, 2.98, 1.16, 0.416, 0.187, 0.11, 0.0582, 0.0599, 0.0297]  min 0.0297 · mean 1.07 · max 4.92
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Operator notes for round 2 (iter_01). Every number below was measured in this simulation (same robot, same cups, same hand controller) or is a recorded observation of a trained policy. Nothing here is a hypothesis unless it says so.

## What happened to iter_00
- The iter_00 run (t2r_rh2_i00) was stopped by the operator after 35 epochs (about 2.3 M frames) to free the GPU for probes. reward/reach fell from 0.24 to 0.21 (source) and 0.23 to 0.20 (receiver), grasp stayed near 0. That is too short to judge the iter_00 reward. The changes asked for below come from the new hand measurement and from a sibling track, not from a failure of iter_00.

## New measurement: hand geometry in the palm frame (probe_rh_hand_geometry, 09.17)
Palm frame: origin = `*_palm_pos`; n = palm normal = `*_palm_axes[:, 0:3]`; f = finger-length direction = `*_palm_axes[:, 3:6]`; w = across-fingers direction = cross of those two (its sign is opposite on the two hands, so use it only through |w . cup_up| or through n and f).
- At reset both hands are already oriented for a side grasp: w is vertical (index finger on top, pinky at the bottom), n and f are horizontal. The cup origin sits at f = +138 mm, n = +129 mm, 106 mm below the palm origin (right hand) and f = +141 mm, n = +150 mm, 108 mm below (left hand). The grasp place is reached by TRANSLATION ONLY: about 70 mm along f, 85 to 100 mm along n, and about 100 mm down. No wrist rotation is needed: scripted trials at the reset orientation lifted at least as often (13 of 128) as any of the six +/-15 deg wrist variants (2 to 11 of 128 each).
- Open hand (mean closure 0.11 to 0.14): thumb tip at (f 43, n 72) mm, index tip at (f 100, n 10) mm. The opening between thumb tip and index tip, seen along the cup axis, is 84 mm (right) and 93 mm (left). The cup is 57 mm wide at the rim and about 51 mm at pinch height.
- The pinch centre (midpoint of the thumb tip and the mean of the other tips) of the open hand is at (f 71, n 40) mm right, (f 69, n 41) mm left. With the cup axis on that pinch centre, every open fingertip is 41 to 49 mm from the axis, i.e. 12 to 20 mm clear of the 28.6 mm rim radius all round.
- Fingertips stack along the cup axis about 17 to 20 mm apart: in scripted grasps with the thumb tip 53 to 60 mm above the cup origin, index was at +42 to +49, middle +23 to +26, ring +6 to +12, pinky -3 to -13 mm. Relative to the thumb tip the other tips are therefore about 10 mm (index), 32 mm (middle), 47 mm (ring) and 65 mm (pinky) lower. The cup bottom and the table are 59.9 mm below the cup origin, so with the thumb tip 5 mm above the cup origin the pinky tip is already at table height, and with the thumb at +25 mm the pinky is 20 mm above the table.
- While closing, the thumb tip and the index/middle tips reach a cup of radius 25 to 28.6 mm at the same time at mean closure 0.28 to 0.38, with the cup axis at (f 64 to 69, n 46 to 51) mm. Full closure is never needed to hold this cup.
- At mean closure 0.69 the thumb tip and index tip are 5 mm apart. A hand that is more than about 0.35 closed cannot admit the cup between thumb and fingers at all: it must ARRIVE OPEN and close only once the cup axis is at the pinch centre.
- At the reset height the lowest open fingertip (pinky) is 72 mm above the cup origin, 18 mm above the rim, so the open hand can move horizontally over the cup and then descend with the cup passing between thumb and fingers.

## New measurement: friction and restitution are not the blocker (probe_rh_grasp_lift, 6 conditions x 896 trials, 09.17)
- Hand friction 1.0 -> 2.0 and 3.5, table friction 1.0 -> 0.5, restitution -> 0, alone and combined: the number of trials where the cup was still upright with two-sided contact at the end of closing, and the number of lifts above 2 cm, did not change beyond trial-to-trial noise. The cup tips over at 0.3 to 0.4 N of one-sided force whatever the friction (static tipping limit m*g*r_base/h with base radius 18.9 mm).

## Observation of a trained policy on the sibling single-arm track (grasp_fj_rand iter_00, other hand, same reward generator, 2378 epochs)
- What worked there: a staged reward. First an approach pose with an OPEN hand (palm beside the cup, facing it, right height, not touching), then closing. Approach was reached in 0.94 of episodes by epoch 750 and the closed grasp in 0.76 by epoch 900, with the cup placed anywhere on the table.
- What failed there: holding the grasped cup on the table paid about 0.70 per step, lifting it added about 0.05 and brought tilt, table and arm-speed penalties. The policy closed its hand and stayed; in the recording the hand sank towards the table after the grasp instead of rising. The cup was airborne 0.021 of the time after 2378 epochs.
- What was decided there, and is asked for here too:
  1. Grasp terms must saturate once the grasp exists. They are the entry ticket to lifting, not the income.
  2. After the grasp the income must come from cup height above its own spawn height, dense from the first millimetres and rising steadily, and then from carrying. Upward motion of the held cup should pay immediately.
  3. Holding the cup and moving the hand or cup DOWN must pay clearly less than holding still, and holding still clearly less than rising. A policy that closes and never lifts must end up clearly worse off than one that lifts.
  4. Penalties that grow when the cup is lifted (tilt, arm speed, cup speed, table) must stay small against the lift income, so that a slow steady lift always beats no lift. They shape how to lift, they must not be the reason not to lift.
  5. The approach income must not drop once grasp income appears (there it fell from 0.94 to about 0.6).

## Carried over from round 1 (unchanged measurements)
(Round-1 notes, written for iter_00, kept verbatim.) Every number below was recomputed from the raw per-trial results of an open-loop scripted grasp-and-lift test in the same simulation (same robot, same cups, same hand controller). 1344 trials: 2 hands x 3 values of the env's per-finger contact-freeze force (0.3 / 1.0 / 4.0 N) x 224 palm offsets relative to the cup. The script moves the palm sideways onto the cup, closes the hand at the env's maximum closing rate, then raises the palm 10 cm. The training env uses a freeze force of 4.0 N: a finger stops closing once it feels that force. These are measurements of one scripted motion, not proof of what a learned policy can or cannot do.

History: rounds 1-12 of this track are archived. In the last one (reward iter_10, 677 epochs, 1024 envs) neither hand reached its cup: `task/src_grasped` was 0 in all 677 epochs, and `task/src_cup_lift` / `task/rcv_cup_lift` were 0.0001 / 0.007 in epochs 0-10 and 0 afterwards.

Cup: mass 0.134 kg, outer diameter 57 mm, rim at `cup_mouth_z` = +53.9 mm above the cup origin.

1. **The approach knocks the cup over when the palm comes in low.** Before closing started, the cup had already moved more than 10 mm or tilted more than 5 degrees in 120 of 336 trials with the grasp pocket 10 mm below nominal height, 59 of 336 at +10 mm, 13 of 336 at +27 mm and 0 of 336 at +45 mm. All 192 happened during the final sideways move onto the cup, with a median finger force of 1.8-2.5 N at a hand speed of 0.086-0.099 m/s. Items 2-5 use only the 1152 trials whose approach left the cup undisturbed.

2. **Closing the hand moves the cup almost every time, at a very small force.** The cup moved during closing in 94-100 % of trials in all six hand/threshold groups (1132 of 1152). At the instant it first moved, the largest single finger force was 0.30-0.43 N (group medians) and `hand_closure` was 0.31-0.33. Contact at that instant was one-sided in 84-125 trials per group and two-sided in 29-69. Thumb-only contact (62-84 per group) outnumbered fingers-only contact (18-48) by 1.6x to 3.7x.

3. **Which finger touches first goes with whether the cup survives closing.** Thumb first: 519 trials, 380 ended closing fallen over (tilt > 45 deg), 66 upright (tilt < 15 deg). Middle finger first: 573 trials, 303 fallen, 196 upright. Index first: 33 trials, 12 fallen, 13 upright.

4. **A higher freeze force leaves fewer cups standing.** Fallen at the end of closing, source hand: 69/180 (0.3 N), 118/181 (1.0 N), 144/180 (4.0 N). Receiver hand: 123/205, 118/206, 141/200. Upright with thumb AND an opposing finger both in contact (the condition behind `src_grasped` / `rcv_grasped`): source 70 / 37 / 7, receiver 49 / 47 / 8. At the training value of 4.0 N that is 15 of 380.

5. **`grasped` true and upright almost never meant the cup could be lifted.** 218 trials ended closing upright with the grasped condition true. Only 4 of them lifted the cup more than 2 cm when the palm rose 10 cm (the cup rose 6.9-8.6 cm). In the other 214 the cup rose at most 0.3 cm, tilted (median peak tilt 15 deg) and the hand slid off.
   - All 4 lifts were the receiver hand at the 1.0 N freeze force. There were 0 lifts with the source hand, 0 at 0.3 N and 0 at the training value of 4.0 N.
   - The 4 lifts had thumb force 2.08-2.52 N, index force 1.05-2.66 N, and three of them middle-finger force 0.97-1.63 N. Summed finger force 4.76-5.18 N. Palm force was 0 in all four.
   - Force size alone did not separate lifts from failures: the non-lifting trials at 1.0 N had a median summed force of 5.2 N, and those at 4.0 N had 6.1 N with 0 lifts. At 0.3 N (119 trials, median sum 2.2 N: thumb 0.87 N, other fingers about 0.4 N each) nothing lifted.
   - Thumb tip height (`tips_pos[:, 0]` z minus cup origin z) was the clearest split found, on small counts: thumb 0-25 mm above the cup origin, 3 lifts of 52; 25-50 mm, 1 lift of 53 (that thumb was at 49.5 mm); 50 mm or higher, i.e. at or above the rim, 0 lifts of 113, although that group had the largest thumb force (median 2.5 N). With summed force above 4 N: thumb below 50 mm, 4 lifts of 29; thumb at 50 mm or higher, 0 of 63.
   - Even in the best group found, 25 of 29 trials failed to lift. No tested condition lifted reliably.

6. **Items 1 and 5 pull against each other.** A low pocket gives the thumb height at which the lifts happened but hits the cup on the way in; a pocket at +45 mm arrives cleanly but puts the thumb near the rim.

7. **Contact explosions are common.** By the end of closing the cup was more than 1 m away horizontally in 330 of 1344 trials (more than 10 m in 177). In 41 trials the cup was more than 1 m above its rest height during the hold. A reward that pays for raw cup height or cup speed would pay for these. In training the env ends the episode when the cup drops.

Operator decisions: no reward audit on this track. The env is not changed for this round; only the reward function changes.
