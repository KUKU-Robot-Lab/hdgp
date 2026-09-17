Observations for this round. Items 1-5 are facts (measured training metrics and the previous reward function's own formulas); item 6 is what the operator saw; item 7 records operator decisions.

1. Between the previous round and this one the robot model was corrected, and the SAME previous reward function was retrained from scratch:
   - The hand collision shapes were fixed. Duplicated overlapping fingertip shapes were removed, and the palm shells and the robot body no longer have inflated shapes. With self-collision ON, no hand link touches another hand link or the body during any finger sweep.
   - The finger drives were made stiffer. A commanded finger position is now followed within 0.01 rad; before, the finger lagged about 0.5 rad behind the command.
   - Finger contact force is still reported per finger, and the observations and actions keep their meaning.
   - The operator judged the model to be fine now. The failure below is therefore attributed to the reward function.

2. Both hands stop about 16-17 cm from their cups, never touch them, and never grasp. The pattern is the same as in the previous round with the old model at the same epochs (medians over the given epoch windows):

| metric | this round, ep 63-83 | this round, ep 116-146 | previous round, ep 116-146 | previous round, ep 280-310 |
|---|---|---|---|---|
| `task/src_palm_to_cup` [m] | 0.214 | 0.170 | 0.164 | 0.127 |
| `task/rcv_palm_to_cup` [m] | 0.210 | 0.164 | 0.190 | 0.166 |
| `contact/src_max` [N] | 0.000 | 0.000 | 0.001 | 0.138 |
| `task/src_grasped` / `task/rcv_grasped` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `task/src_closure` / `task/rcv_closure` | 0.33 / 0.39 | 0.32 / 0.42 | 0.44 / 0.33 | 0.51 / 0.22 |
| `reward/close_src` / `reward/close_rcv` | 0.040 / 0.054 | 0.078 / 0.118 | 0.098 / 0.061 | 0.177 / 0.060 |
| `reward/curl_far_pen` | −0.336 | −0.283 | −0.315 | −0.192 |
| `reward/rim_hook_pen` | −0.029 | −0.059 | −0.020 | −0.024 |
| `reward/approach_src` | 0.480 | 0.605 | 0.631 | 0.806 |

3. Approach and rim-hook terms of the previous function, per hand:
   - Approach, `exp(−4 d) + 0.5 exp(−15 max(d − 0.06, 0))`, pays 0.60 at d = 17 cm and 1.29 at d = 6 cm.
   - The rim-hook penalty is multiplied by `exp(−10 max(d − 0.08, 0))`. That factor is 0.41 at 17 cm and 1.0 at 8 cm or closer. So any thumb-over-rim posture costs 2.4 times more once the palm closes the last 9 cm, up to −1.0 per hand.
   - As the palms came from about 21 cm to about 17 cm, `reward/rim_hook_pen` grew from −0.029 to −0.059. The thumb tip is already at rim height over the cup opening while the palm is still outside 8 cm.

4. The closing term `close_near = closure · exp(−15 · mean tip-to-wall distance)` has no palm-distance condition. `reward/close_src` / `reward/close_rcv` rose from 0.015 / 0.011 (epochs 0-10) to 0.078 / 0.118. Over the same epochs the palm stayed about 16-17 cm away and finger contact stayed 0. The hands are 32-42 % closed while far, which also costs `reward/curl_far_pen` about −0.28 (both hands).

5. Physics blow-ups of the underactuated hand stayed small with the stiffer drives: `ctrl/mimic_err_max` median 0.14 rad over the last 30 epochs, versus about 0.77 rad median in the previous round.

6. Operator observations:
   - This round: the policy approaches so that the thumb gets caught on the rim, and without a proper approach no grasp is possible.
   - Previous round (video at epoch 800, old model): the right hand held its cup with the thumb over the cup mouth and the four fingers on the near side; the cup was never lifted. The left hand hovered above and behind its cup and never wrapped it.

7. Operator decisions:
   - The model problems are fixed, so the reward must change: generate a new reward and retrain from scratch.
   - The judgement from earlier rounds still stands: once both hands reliably approach their cups and grasp them with the thumb on the cup wall opposite the fingers (not over the rim), the later stages are expected to follow.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
