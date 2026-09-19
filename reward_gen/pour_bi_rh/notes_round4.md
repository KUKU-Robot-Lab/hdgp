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
