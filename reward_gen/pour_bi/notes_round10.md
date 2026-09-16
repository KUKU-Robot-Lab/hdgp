Observations from the loop operator about the policy trained with the previous function, one environment fix made for this round, and the operator's requirements. Facts only, apart from the explicitly marked operator requirements.

1. The previous function was stopped early, after 159 epochs (1.3 h), because the source hand never grasped its cup. The feedback table above therefore covers only these epochs. Comparison at epoch 160 with the two earlier converged runs (this run / round-8 run / round-6 run):

| metric | this run | round 8 | round 6 |
|---|---|---|---|
| source grasped | 0.000 | 0.872 | 0.871 |
| receiver grasped | 0.775 | 0.875 | 0.866 |
| source cup lift [m] | 0.000 | 0.166 | 0.207 |
| receiver cup lift [m] | 0.222 | 0.192 | 0.143 |
| source tilt [deg] | 0.1 | 42.6 | 33.3 |
| receiver hand–foreign contact | 10.8 % | 4.0 % | 2.8 % |
| reward/total | 2.8 | 8.0 | 10.2 |

   The receiver side learned normally (grasp 0.78, lift 0.22 m, receiver held 8.9° from upright). The source hand approached its cup (`reward/approach_src` 0.44) but did not close (`task/src_closure` 0.12) and never grasped; the source cup stayed untouched and upright.

2. Why (measured, first epoch → epoch 159): at epoch 1 the premature-tilt latch fired in 19.8 % of environments while source grasping was 0.000 — random exploration knocked the source cup over without grasping it, and the environment latched those episodes (the latch had no grasp condition). The previous function paid `premature_latch` −0.5 per step for the rest of every latched episode. The policy removed the latch by avoiding the source cup: latch rate 19.8 % → 0.06 %, source hand–foreign contact 14.9 % → 2.0 %, source tilt 16.6° → 0.1°, source grasp 0.000 throughout. This is the same mechanism as the round-7 receiver avoidance (always-on receiver-upright penalty), now on the source side.

3. Environment fix for this round: `ctx.premature_tilt` now latches only when the source cup is GRASPED (thumb and another finger in contact) at the moment it exceeds 30° while its mouth is more than 0.10 m (xy) from the receiver's mouth. A cup knocked over without being grasped no longer latches. Tilting a grasped cup on the table before bringing the mouths together — what the round-8 policy did (20° at a mouth distance of 0.26 m with the cup still on the table) — still latches and still invalidates success. Everything else is unchanged from the previous round: beads 30 mm, 6–20 per episode (upper bound 12 → 20 with success), `ctx.bead_fill_level`, receiver ≤ 20° for success. The policy is trained from scratch.

4. Spill of 0.8 % appeared although the source cup was never moved: it is the bead-spawn/settling floor of this environment (measured 0.6–6 % in probes), not a policy behaviour.

5. OPERATOR REQUIREMENTS (unchanged):
   a. The source cup stays upright while grasped, lifted and carried; it may tilt only once its mouth is close to the receiver's mouth (success invalid beyond 30° while farther than 0.10 m, when grasped).
   b. The receiver cup is lifted and held at a steady position during the pour; small tilt acceptable, less movement better (`task/rcv_palm_speed`).
   c. All beads are to be transferred whatever the fill level.

6. Unchanged facts from earlier rounds: with contact freeze the measured `hand_closure` saturates at about 0.35–0.43 on these cups; hand–hand and cup–cup contact remain safety hazards on the real robot; measured spill onset with the 30 mm beads is about 70° for a full cup and about 90° for a few beads.

The policy must complete the full task (wrap-grasp both cups → lift both, source upright → bring the mouths together → tilt the source cup only then → beads in the receiver, receiver held steady and nearly upright, no drop, little spill) and hold both grasps stably after pouring.
