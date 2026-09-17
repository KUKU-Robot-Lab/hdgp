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
bead/in_target: [4.88e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.05e-07 · max 9.77e-05
bead/spill: [0.000195, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.64e-05 · max 0.00767
done/drop: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.83e-05 · max 0.00781
done/mimic_runaway: [0.00195, 0, 0, 0, 0, 0, 0, 0, 0, 0.000977]  min 0 · mean 7.83e-06 · max 0.00293
episode_lengths/step: [113, 896, 898, 893, 898, 894, 899, 899, 894, 892]  min 113 · mean 890 · max 899
reward/action_rate: [-0.0516, -0.036, -0.0218, -0.021, -0.0237, -0.0241, -0.0251, -0.0252, -0.0245, -0.0248]  min -0.0516 · mean -0.0263 · max -0.0189
reward/align_mouths: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/contact_push: [-0.0144, 0, 0, -0.00043, -0.00152, -0.00386, -0.00523, -0.00854, -0.0287, -0.076]  min -0.136 · mean -0.0215 · max 0
reward/cup_collision: [-0.000999, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00396 · mean -1.14e-05 · max 0
reward/cup_disturb_rcv: [-0.121, -0.0361, -0.0383, -0.0391, -0.0453, -0.0451, -0.0437, -0.0402, -0.043, -0.0387]  min -0.121 · mean -0.0427 · max -0.0352
reward/cup_disturb_src: [-0.11, -0.0346, -0.0356, -0.0366, -0.04, -0.043, -0.044, -0.0516, -0.0693, -0.0932]  min -0.122 · mean -0.0558 · max -0.0332
reward/cup_fling: [-0.0181, 0, 0, 0, -6.03e-05, 0, -5.46e-05, 0, 0, -0.00027]  min -0.0181 · mean -0.000106 · max 0
reward/grasp_rcv: [0.00081, 0, 0, 7.11e-05, 0.000387, 0, 0.00105, 4.99e-05, 0.00021, 0.000389]  min 0 · mean 0.000572 · max 0.00701
reward/grasp_src: [0.000816, 0, 0, 4.13e-05, 0.000953, 0.00431, 0.00455, 0.0238, 0.0645, 0.152]  min 0 · mean 0.0405 · max 0.233
reward/hand_foreign: [-0.00243, 0, 0, 0, -0.000376, -0.000232, 0, -0.000713, -0.00022, -0.00049]  min -0.00942 · mean -0.00039 · max 0
reward/lift_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.27e-09 · max 3.33e-06
reward/lift_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.34e-08 · max 6.09e-05
reward/nested: [-0.0137, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0215 · mean -9.47e-05 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pour_tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/reach_rcv: [0.23, 0.317, 0.372, 0.569, 0.807, 0.906, 0.923, 0.912, 0.919, 0.916]  min 0.171 · mean 0.724 · max 0.939
reward/reach_src: [0.232, 0.346, 0.374, 0.527, 0.804, 0.897, 0.925, 0.913, 0.924, 0.929]  min 0.189 · mean 0.722 · max 0.944
reward/rush_near_cup: [-0.0414, -0.0827, -0.104, -0.221, -0.239, -0.219, -0.21, -0.194, -0.199, -0.203]  min -0.265 · mean -0.179 · max -0.00884
reward/spill_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0527 · mean -9.95e-05 · max 0.0146
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [1.07, 1.41, 1.48, 1.74, 2.21, 2.43, 2.49, 2.47, 2.49, 2.52]  min 0.847 · mean 2.11 · max 2.57
rewards/step: [82.7, 1.29e+03, 1.35e+03, 1.51e+03, 1.98e+03, 2.17e+03, 2.21e+03, 2.23e+03, 2.24e+03, 2.22e+03]  min 82.7 · mean 1.88e+03 · max 2.27e+03
task/aim_dist: [0.368, 0.32, 0.319, 0.321, 0.32, 0.32, 0.322, 0.32, 0.32, 0.319]  min 0.317 · mean 0.326 · max 1.38
task/cup_collision_rate: [0.00195, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.64e-05 · max 0.00684
task/cups_center_dist: [0.37, 0.32, 0.319, 0.321, 0.32, 0.32, 0.322, 0.32, 0.32, 0.32]  min 0.317 · mean 0.326 · max 1.38
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.00684, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.73e-05 · max 0.0107
task/rcv_cup_lift: [0.0153, 1.47e-05, 2.39e-05, 1.93e-05, 4.42e-05, 4.5e-05, 3.05e-05, 2.13e-05, 3.2e-05, 1.98e-05]  min -4.89e-05 · mean 0.000682 · max 0.128
task/rcv_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.45e-07 · max 0.000977
task/rcv_hand_foreign_rate: [0.00586, 0, 0, 0, 0.000977, 0.000977, 0, 0.00195, 0.000977, 0.000977]  min 0 · mean 0.00106 · max 0.0244
task/src_cup_lift: [0.00322, 7.19e-06, 1.36e-05, 9.94e-06, 4.75e-05, 3.53e-05, 4.12e-05, 7.62e-05, 0.000146, 0.000462]  min -2.75e-05 · mean 0.000522 · max 0.0974
task/src_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.27e-05 · max 0.00195
task/src_hand_foreign_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.68e-05 · max 0.00293
task/src_tilt_deg: [3.65, 0.015, 0.0252, 0.0376, 0.0811, 0.0995, 0.127, 0.241, 0.463, 0.971]  min 0.0137 · mean 0.38 · max 5.44
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Operator notes for round 3 (iter_02). Every number below was measured in this simulation (same robot, same cups, same hand controller). Nothing here is a hypothesis unless it says so.

## What iter_01 learned (run t2r_rh2_i01, 1024 envs, fresh start, stopped by the operator at epoch 2615)

Env changes that were in force for this run and stay in force: hand-cup friction 2.0 (robot and cup materials), contact-freeze threshold 1.0 N, grasp flag = thumb > 1.0 N AND another finger > 1.0 N.

Training curves (epoch 10 / 1000 / 2570):
- reward/reach_src 0.22 / 0.75 / 0.92, task/src_cup_in_pocket_near 0.06 / 0.98 / 0.95, task/src_thumb_oppose_near 0.06 / 0.93 / 0.90.
- reward/grasp_src 0.000 / 0.002 / 0.220, contact/src_max 0.28 / 0.03 / 1.95 N.
- task/src_grasped and task/rcv_grasped 0.0000 at every epoch. task/src_cup_lift <= 0.0004.
- task/src_closure 0.24 / 0.12 / 0.19, task/rcv_closure 0.22 / 0.05 / 0.18.
- losses/entropy 33.9 / 15.9 / 6.7. No terminations (done/drop, mimic, arm all 0.0000 after epoch 100).

## Deterministic play trace of the epoch-2600 checkpoint (play --trace_steps 900, 64 envs)

The hand arrives in about 60 steps and then parks for the rest of the 900-step episode:
- Source hand: palm 86-90 mm from the cup origin, 60-66 mm above it. The cup axis is between the thumb and the four fingers in 64/64 envs (thumb tip about -36 mm lateral, finger tips +32 to +52 mm lateral).
- Hand closure stays at 0.19 (median), 0.25 (p99). Hand actions are mostly negative (open): thumb abduction channel goes to -1.0, thumb flexion -0.8, middle -0.9.
- Tip-frame distance from the cup axis (median over steps 100-890): source thumb 37.2 mm, index 57.6, middle 33.5, ring 48.1, pinky 60.0 mm. Receiver thumb 36.4, index 44.3, middle 37.8, ring 42.9, pinky 48.8 mm.
- Forces: source index 2.7 N median (4.1 N p90), a non-thumb finger above 1.0 N in 90 % of env-steps. Source thumb mean 0.00 N; only 4/64 envs ever had thumb force above 0.05 N (max 0.81 N). Receiver fingers 0.00 N. Grasp flag true in 0/64 envs on both hands.
- Reward per step while parked: reach_src 0.97, reach_rcv 0.94, grasp_src 0.26-0.28, grasp_rcv 0.00, contact_push -0.16, rush_near_cup -0.18, cup_disturb_src -0.10, cup_disturb_rcv -0.04, lift 0.

## Where real two-sided contact sits (scripted probe, 3 friction conditions x 896 trials, trials that ended upright with thumb > 1 N and another finger > 1 N, n = 239)

- Thumb tip frame 27.9 mm from the cup axis (p10-p90 25.3-31.9 mm). Nearest other tip 26.2 mm (23.3-29.0 mm).
- Hand closure 0.86 (p10-p90 0.68-0.98).

## What in the iter_01 reward produced the parked pose (read from the reward code, checked against the trace)

1. `PAD = 0.010` with `CUP_OUTER_R = 0.0285`: the tip gap is zero from 38.5 mm inward. The parked tips (thumb 37.2 mm, middle 33.5 mm) already count as "on the wall", so reach is 0.97 with no gradient left, while real contact needs 26-28 mm.
2. `touch_first = 0.15 * tanh(f_o / 1.0) * (1 - tanh(f_t / 0.3))`: the parked pose earns exactly its maximum (2.0 x 0.15 x 0.99 x 0.92 = 0.27, measured 0.26-0.28). Any thumb force below the 1.0 N flag removes it (f_t = 0.3 N -> 0.06, f_t = 0.9 N -> 0.00) and only f_t > 1.0 N jumps to about 1.4. The thumb has to cross a stretch where reward is lower than not touching at all.

## Changes made by the operator for iter_02 (hand patch, no regeneration)

- `PAD = 0.0`.
- `reach_tips` keeps the long scale and adds a short one: `0.5 * exp(-30 d) + 0.5 * exp(-160 d)`, d = mean of thumb gap and nearest-finger gap.
- `touch_first = 0.15 * tanh(f_o / 1.0) + 0.25 * tanh(min(f_t, f_o) / 0.5)`: thumb-only still earns nothing, adding thumb force to a finger contact never lowers the reward.
- Everything else (weights, lift, pour, penalties) unchanged.
