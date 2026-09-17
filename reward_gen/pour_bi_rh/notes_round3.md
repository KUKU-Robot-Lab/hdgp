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
