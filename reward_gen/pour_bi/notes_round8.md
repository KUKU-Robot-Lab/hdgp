Observations from the loop operator about the policy trained with the previous function, and one operator requirement. Facts only, apart from the explicitly marked operator requirement.

1. This round was stopped early, after 208 epochs (1.7 h), because the receiver branch never started. The environment is unchanged from the previous round: 18-dimensional action space (per hand thumb opposition, thumb closure, one four-finger closure), palm commands low-pass filtered (EMA 0.25), and `ctx.success` requires the receiver tilt ≤ 20°. The feedback table above therefore covers only these 208 epochs.

2. Source side learned normally; the receiver side never did. Comparison at the same epochs with the round before the previous function (same task, older environment, a policy that did learn the receiver), shown as earlier-run / this-run:

| metric | epoch 20 | epoch 50 | epoch 145 | epoch 208 |
|---|---|---|---|---|
| source grasped | 0.06 / 0.06 | 0.57 / 0.47 | 0.87 / 0.88 | 0.88 / 0.88 |
| source lift [m] | 0.004 / 0.005 | 0.026 / 0.025 | 0.206 / 0.163 | 0.217 / 0.196 |
| receiver grasped | 0.16 / 0.00 | 0.64 / 0.00 | 0.86 / 0.00 | 0.87 / 0.00 |
| receiver lift [m] | 0.037 / 0.010 | 0.102 / 0.000 | 0.134 / 0.000 | 0.152 / 0.000 |
| reward/approach_rcv | 0.50 / 0.45 | 0.57 / 0.46 | 0.64 / 0.44 | 0.67 / 0.45 |
| cup distance `task/aim_dist` [m] | 0.44 / 0.34 | 0.71 / 0.41 | 0.34 / 0.47 | 0.21 / 0.48 |

   In this run the receiver hand stayed almost open (`task/rcv_closure` 0.01–0.02) and the receiver arm did not move toward its cup. The source cup was tilted to 53° while the cups stayed 0.48 m apart, so no pour and no success (0.0).

3. Early epochs, receiver upright penalty and receiver cup tilt (earlier-run / this-run):

| metric | epoch 1 | epoch 5 | epoch 10 | epoch 20 | epoch 50 |
|---|---|---|---|---|---|
| reward/upright_rcv | −0.10 / −0.36 | −0.08 / −0.22 | −0.06 / −0.11 | −0.07 / −0.014 | −0.09 / −0.001 |
| `task/rcv_tilt_deg` | 19 / 15 | 15 / 10 | 13 / 5 | 13 / 0.7 | 18 / 0.1 |
| reward/grasp_rcv | 0.02 / 0.01 | 0.02 / 0.01 | 0.04 / 0.02 | 0.21 / 0.01 | 0.73 / 0.01 |

   During early random exploration both runs bumped the receiver cup (15–19° tilt). With the previous function this cost −0.36 per step at epoch 1 instead of −0.10; within 20 epochs the policy stopped touching the receiver cup at all (tilt 0.1–0.7°, i.e. the cup standing untouched on the table) and receiver grasping never began. In the previous function `upright_rcv` is applied whenever the receiver cup is tilted more than about 7°, regardless of whether the receiver cup is grasped or lifted.

4. OPERATOR REQUIREMENT: the penalty on receiver tilt must differ between phases. While the receiver hand approaches and grasps the receiver cup on the table, tilt caused by hand contact must not be penalised. The receiver-upright penalty (and any upright-dependent scaling) should apply only after the receiver cup has been grasped and lifted, and especially during the pour, where the receiver must be held with its mouth up and only slightly tilted (the environment success still requires ≤ 20°).

5. Unchanged facts from earlier rounds: with contact freeze the measured `hand_closure` saturates at about 0.35–0.43 on these cups whatever the grasp type; hand–hand and cup–cup contact remain safety hazards on the real robot; the policy is trained from scratch again.

The policy must complete the full task (wrap-grasp both cups → lift → bring together without contact → tilt the source cup → beads in the receiver, receiver held nearly upright during the pour, no drop, little spill) and hold both grasps stably after pouring.
