Observations from the operator after watching the trained iter_01 policy and from the sim-to-real requirements (this round's environment changes):

1. In the rollout video the two cups often BUMP INTO EACH OTHER early — right after both are lifted, the policy drags them together fast and they collide before the source cup is tilted. On the real robot this is a collision hazard. The environment now exposes `ctx.cup_cup_force`, `ctx.src_hand_foreign_force` and `ctx.rcv_hand_foreign_force` (see the class definition and knowledge item 11); these are new since the previous reward. Also the previous metrics table shows `task/cup_collision_rate` / `task/*_hand_foreign_rate` were not yet logged, so treat them as unknown-but-observed-high.

2. The receiver cup is lifted less than the source cup (~15 cm vs ~30 cm) and the source pours from high above; a gentler pour (source rim only a few cm above the receiver rim, slower tilt) spills less. Spill was 7% at the end of training.

3. The environment now adds perception delay/noise on the cup poses seen by the policy and randomizes cup mass, joint gains and cup friction as training progresses (curriculum keyed on success). Rewards that depend on very precise instantaneous cup geometry (sharp exp(-k·d) with large k) will be noisier; prefer tolerances of a few centimetres.

The policy must still complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
