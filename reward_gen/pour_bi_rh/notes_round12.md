Observations for this round (previous reward function iter_10, trained from scratch for 677 epochs, 1024 envs, 1.96 h). Items 1-6 are facts: training metrics, a synthetic measurement of the reward itself, and a rollout video. Item 7 records operator decisions.

1. **Neither hand ever reached its cup.** Every metric that is gated on the palm being within 12 cm had no samples for the whole run.

| metric | ep 0-10 | ep 355 | ep 463 | ep 570 | ep 667-677 |
|---|---|---|---|---|---|
| `task/src_palm_to_cup` [m] | 0.240 | 0.243 | 0.239 | 0.232 | **0.231** |
| `task/rcv_palm_to_cup` [m] | 0.320 | 0.306 | 0.299 | 0.298 | **0.299** |
| `task/src_near_rate` / `task/rcv_near_rate` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **0 / 0** |
| `task/{src,rcv}_cup_in_pocket_near` | – | – | – | – | no samples |
| `contact/src_max` / `contact/rcv_max` [N] | 0.089 / 0.304 | 0 / 0 | 0 / 0 | 0 / 0 | **0 / 0** |
| `task/src_grasped` | 0 | 0 | 0 | 0 | **0 (677/677 epochs)** |
| `task/src_cup_lift` / `task/rcv_cup_lift` | 0.0001 / 0.007 | 0 | 0 | 0 | 0 / 0 |

   - The non-zero values in the first ten epochs are the random initial policy brushing the cups, not learned behaviour.
   - `reward/total` settled at 0.53 and moved by less than 0.02 over the last 300 epochs.

2. **The ladder stopped at rung 3 on the source side, and rung 3 on the receiver side.** End-of-round values, with round 11 (iter_09) at the same epoch for comparison:

| term | iter_10 (ep 677) | iter_09 (ep 665) |
|---|---|---|
| `lad1_reach_src` | 0.069 | 0.093 |
| `lad2_open_src` | 0.092 | 0.140 |
| `lad3_height_src` | 0.119 | 0.230 |
| `lad4_straddle_src` | **0.0003** | **0.318** |
| `lad5_oppose_src` | 0 | 0.397 |
| `lad6_ready_src` | 0 | 0.423 |
| `lad1_reach_rcv` | **0.059** | 0.025 |
| `lad2_open_rcv` | **0.079** | 0.037 |
| `lad3_height_rcv` | **0.101** | 0.058 |
| `lag_pull_rcv` | 0.034 | (term did not exist) |
| `reward/total` | 0.53 | 2.16 |

   - **The handicap worked on its own terms.** The receiver hand, which sat at exactly zero for all of round 11, now earns 2-2.4x more than round 11's receiver on every rung it reached, and `lag_pull_rcv` is live at 0.034.
   - **But the source hand collapsed.** Round 11's source hand was at the straddle and one flex from the grasp flag; this round's never got there. The two hands ended up symmetric in the middle of the ladder instead of one of them finishing it.
   - Penalties explain nothing: the largest penalty in either run is `action_rate_pen` at −0.011 (iter_10) and −0.019 (iter_09), and `bore_pen` was 0.

3. **The obvious explanation — that the new reward pays less for approaching — is wrong, and was measured to be wrong.** Both reward functions were run on identical synthetic postures (same cup, same palm distance, same tip geometry, same forces):

| synthetic posture | iter_09 `lad1/lad2/lad3` | iter_10 `lad1/lad2/lad3` |
|---|---|---|
| palm 24 cm, hand closed to 40 mm | 0.030 / 0 / 0 | 0.068 / 0 / 0 |
| palm 12 cm, gap 70 mm, oppose 0.9 | 0.078 / 0.027 / 0.045 | **0.149 / 0.076 / 0.101** |
| palm 9 cm, gap 70 mm, 0.28 N | 0.078 / 0.030 / 0.050 | **0.149 / 0.085 / 0.114** |
| palm 9 cm, gap 64 mm, 1.2 N both sides (total) | 2.93 | **5.11** |

   - iter_10 pays roughly twice as much as iter_09 at every approach posture tested, and nearly twice as much for a completed squeeze. Whatever stopped this round, it is not a weaker approach gradient.
   - Caveat on this measurement: the synthetic tip geometry did not satisfy either function's straddle test, so rungs 4-6 read 0 for both and could not be compared this way.

4. **What actually differs, then, is where the policy spent its effort.** One actor drives both arms. In round 11 the source hand ran far ahead and broke through the straddle alone while the receiver stayed at zero. In round 12 the handicap gave the receiver hand its own income from the first epochs, both hands advanced together through rungs 1-3, and neither accumulated enough to clear rung 4. The round produced two half-climbs instead of one full climb.

5. **Rollout video** at the epoch-650 checkpoint (4 envs, 900 steps, `our_source/pour_t2r_rh_i10_ep650_0917.mp4`, frame sheet and 2.2x crops in `pour_t2r_rh_i10_ep650_0917_frames/`):
   - The arm hangs above and behind the mat with the hand at roughly chest height, well above and short of the cups. The fingers are half-curled and still. Both cups stand untouched on the mat, clearly separated from the hand.
   - The posture does not change through the episode: the arm never descends to cup height and never extends over the mat. This is the 23 cm in item 1 seen directly — the hand is not beside the cup at all, it is hovering away from the table.
   - Compare round 11's video at the same checkpoint index, where the hand was down at the cup with the fingers on either side of it.

6. **Physics stayed clean.** `ctrl/mimic_err_max` 0.020 rad at the end, the lowest of any round. No runaway, no drop, no cup collision.

7. **Operator decisions** (to be filled in after reviewing items 1-6 and the video).

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
