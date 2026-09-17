Observations for this round (previous reward function iter_08, trained from scratch for 675 epochs, 1024 envs, 2.43 h). Items 1-6 are facts: training metrics, the previous reward function's own terms, the environment's own measurements and a rollout video. Item 7 records operator decisions.

1. **The hand now opens wider than the cup. This was the round's first question and it is answered.** The previous reward added a `spread` term paying only above 60 mm of thumb-to-finger gap, full at 75 mm, ungated on opposition. Medians over epoch windows (the cup is 57 mm across):

| metric | ep 0-10 | ep 150-160 | ep 300-310 | ep 450-460 | ep 665-675 |
|---|---|---|---|---|---|
| `task/src_tip_gap_mm_near` [mm] | 54.9 | 83.5 | 80.4 | 78.4 | **83.9** |
| `task/rcv_tip_gap_mm_near` [mm] | 64.0 | – | 84.9 | 90.7 | **92.7** |
| `task/src_thumb_above_rim_mm_near` [mm] | 0.0 | +3.7 | +5.5 | −10.1 | **−8.7** |
| `task/rcv_thumb_above_rim_mm_near` [mm] | +8.5 | – | +3.2 | −3.0 | **−10.8** |
| `task/{src,rcv}_thumb_over_rim_near` | 0 / 1.0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

   - The previous round ended at 48-50 mm and narrowing. One reward term reversed that inside 150 epochs, and it held for the rest of the run.
   - Both thumbs are also below the rim now, and the old rim-hook posture never returned.
   - So two preconditions that had never held together before — hand wider than the cup, thumb at wall height — now hold together on both hands.

2. **But the hands never touched their cups. Not once in the last 200 epochs, on either side.**

| metric | last 200 epochs |
|---|---|
| `contact/src_max` non-zero | **0 / 200** |
| `contact/rcv_max` non-zero | **0 / 200** |
| `reward/dorsal_pen` non-zero | 0 / 200 |
| `reward/oppose_src` / `oppose_rcv` non-zero | 1 / 200 · 1 / 200 (max 0.0000) |
| `reward/pocket_src`, `pose_src`, `close_*`, `grip_*` | 0 / 200 |
| `task/{src,rcv}_cup_in_pocket_near` | 0 / 200 |
| `task/{src,rcv}_grasped`, `task/*_cup_lift`, `task/episode_success` | 0 |

   - Question 2 of this round (does the dorsal-contact penalty fall as the palm turns toward the cup?) could not be answered at all: the penalty stayed at zero because there was no contact of any kind to penalise.
   - Question 3 (does opposition follow?) never started.

3. **Where the reward came from, and why the policy stopped.** Per-step medians over the last 10 epochs, `reward/total` 3.61:
   - `spread_src` 1.136 + `spread_rcv` 1.091 = 2.227
   - `open_src` 0.326 + `open_rcv` 0.313 = 0.639
   - `approach_src` 0.286 + `approach_rcv` 0.283 = 0.569
   - Together **3.43 of 3.61 — 95 %** — and every one of those terms is payable **without the hand ever touching the cup**.
   - Everything downstream (oppose, pocket, pose, close, touch, grip, grasp, lift, align, tilt, pour, success) is exactly 0.
   - `task/src_closure` fell 0.17 → 0.10 and `task/rcv_closure` 0.17 → 0.10: the hands ended the run held almost flat open.
   - `task/src_palm_to_cup` 0.234 → 0.119 m and `task/rcv_palm_to_cup` 0.252 → 0.119 m, pinned at about 12 cm from epoch 150 onward.
   - The near-rate moved the wrong way as the spread income grew: `task/src_near_rate` peaked at 0.22 around epoch 300 and ended at 0.018; `task/rcv_near_rate` peaked at 0.15 and ended at 0.063. Standing back and holding the hand open pays more than closing the last 12 cm.

4. **This is the same failure shape as the previous round, one rung further up the ladder.** In round 9 the policy settled on the approach term (88 % of income, palm pinned at 12 cm). In round 10 it settled on the opening terms (95 % of income, palm pinned at 12 cm). Each time, the rung that was added to unblock the ladder became a place to stand, because it pays a standing posture rather than a transition. The rungs themselves worked: approach was solved in round 6, opening in round 10. What is missing is that reaching a rung must stop paying once the next one is available.

5. **Rollout video** at the epoch-650 checkpoint (4 envs, 900 steps, `our_source/pour_t2r_rh_i08_ep650_0916.mp4`, frame sheet and 2.4x crops in `pour_t2r_rh_i08_ep650_0916_frames/`). The question was whether the hand holds still with the fingers spread or approaches and retreats. It holds still:
   - At step 60 the hand is already beside the cup with all five fingers spread wide and flat, laid out over the mat next to the cup rather than around it. The thumb is spread away from the cup as well.
   - At step 700 the posture is unchanged. Nothing approaches, nothing closes, and no part of the hand reaches the cup wall at any point in the episode.
   - **Operator observation (takes precedence over the frame reading above): what the arm brings close to the cup is the WRIST — the j7 motor end of the arm. The palm side never comes close at all.**
   - That observation and the logged numbers agree, and together they show how the plateau is built. `spread` is already gated on the signed pad direction (`spread = near_pre · pad · gap_ok · jaw_ok`), and it was paying 1.136 of its 1.5 weight, i.e. a raw 0.76 — so the pad was *pointing* at the cup. But pointing and approaching are separate conditions in the reward: `approach` is computed from palm distance alone and is worth at most 0.3 per hand, while `spread` is worth 1.5. The cheapest way to collect is therefore to park the arm with the wrist nearest the cup, aim the pad at it from 12 cm away, and spread the fingers. Everything that would require the palm itself to arrive — pocket, pose, close, touch, grip — needs the cup between the fingertips, and stayed at 0.
   - The fingers are spread in the plane of the mat, beside the cup, not on either side of it. So the wide `tip_gap` in item 1 is real but it is not a pocket around the cup — the opening happens in the wrong place, which is exactly what `cup_in_pocket` at 0 / 200 says.

6. **Physics was the cleanest of any round.** `ctrl/mimic_err_max` 1.52 at start → **0.025 rad** at the end. No mimic runaway, no drop, no cup collision, no foreign contact.

7. **Operator decisions** (after reviewing items 1-6 and the video):
   - This draft is approved as the feedback for this round, and a new reward is to be generated. Training starts from scratch.
   - **How to fix item 3 and item 4 is left to the reward designer.** The operator is not prescribing a mechanism this time. What is required is that the facts above be taken as binding: a hand standing 12 cm away with the fingers spread flat beside the cup currently collects 95 % of the available reward, and that has now happened twice in a row with a different term each time (approach in round 9, opening in round 10).
   - **Keep the three gates that are working**, all introduced in the last two rounds and none of them implicated in this failure: the opposition gate with no floors, the signed pad-facing condition, and the rule that force on the back of the fingers is not progress. Round 10 produced no ungraspable posture that was paid, and no dorsal contact.
   - The goal is unchanged: the thumb and at least one opposing finger both above 1 N at the same time (the environment's grasp flag), then lifting, then the rest of the task.
   - `task/{src,rcv}_cup_in_pocket_near` is the metric that separates a useful opening from this one. It was 0 for the entire round while `tip_gap` sat at 84-93 mm.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
