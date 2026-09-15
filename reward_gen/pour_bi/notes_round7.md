Observations from the loop operator: rollout of the policy trained with the previous function (final checkpoint, 8 environments, per-step trace), the operator's own viewing of the video, and the environment changes made for this round. Facts only, no design instructions.

1. What the previous policy achieved: step-averaged `task/episode_success` about 0.81 with the physics/perception randomisation at its maximum (`adr/progress` 1.0). In the recorded rollout about 19 of 20 beads ended in the receiver cup (in-target 95.6–97.5 %), spill 2.5–4.4 %, all 8 environments succeeded. Hand–hand contact stayed at 2–4 %, cup–cup contact about 1 %, no nesting. The right hand formed a wrap grasp and the pour was a real pour through the air.

2. Right (source) hand grasp collapses while holding after the pour (operator saw it in the video; measured from step 300 to step 880 of the rollout, mean over envs, index/middle/ring): the command for the proximal flexion joint `_2` went down from 1.07 to 0.49 rad target while the command for the middle/distal joints went up (target 1.26 → 1.48 rad). The finger straightened at the knuckle and curled at the tip, so the cup was pressed by the fingertips: palm contact force fell from 7.0 N to 2.2 N and the distal-link force rose from about 7 N to 11 N. The left hand kept its grasp (palm force 4.0 → 3.1 N).

3. Arm vibration (operator saw it; measured): the policy switched the sign of two source palm rotation commands on 84 % and 81 % of consecutive steps (about +0.8 ↔ −0.5…−1.0), while the third rotation command stayed at −1. The palm target jumped about 68° per step, the arm joints vibrated with a dominant frequency of 4–5 Hz. The receiver arm did not do this.

4. Receiver cup tilt (operator saw it; measured): while the source cup was tilted past 80°, the receiver cup was tilted by a median of 46° and up to 55°. The operator's requirement: lift the receiver cup moderately, keep its mouth pointing to the sky and tilt it only slightly.

5. Environment changes for this round (all active in training; the policy is trained from scratch because the action space changed):
   - The action space is now 18-dimensional with the layout given in the robot description: per hand one thumb-opposition command, one thumb-closure command and ONE four-finger closure command that drives all flexion joints of index, middle, ring and pinky together. A finger can no longer straighten its knuckle while curling its tip.
   - The palm 6-DoF commands are low-pass filtered by the environment before the arm controller (exponential moving average, new = 0.25·command + 0.75·previous, at 60 Hz). `ctx.actions` / `ctx.prev_actions` are the raw policy outputs before this filter.
   - `ctx.success` now additionally requires the receiver cup tilt `ctx.rcv_cup_tilt` ≤ 20°. Holding the receiver at 46° as before is no longer a success.

6. How to read the feedback table above: it was recorded with the old 42-dimensional action space and the old success definition (no receiver-tilt condition). Under the new definition the previous policy's success would have been much lower, because the receiver was tilted about 46° during the pour. `task/rcv_tilt_deg` in the table is a step average over the whole episode (including the time before the pour).

7. Reminder of an earlier correction that still holds: with contact freeze the measured `hand_closure` saturates at about 0.35–0.43 on these cups whatever the grasp type, so closure above ~0.45 is not reachable and does not distinguish a hook from a wrap.

The policy must complete the full task (wrap-grasp both cups → lift → bring together without contact → tilt the source cup → beads in the receiver, receiver held nearly upright, no drop, little spill) and hold the grasp stably after pouring.
