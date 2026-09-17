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
