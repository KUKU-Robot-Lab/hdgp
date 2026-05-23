# Adaptive Grasp Checkpoint Comparison

Protocol: deterministic RL-Games rollout with fixed bead-count bins 0/10/20/30 and equal episode count per bin.

Interpretation notes:
- v7 is a `synergy_forced_mass_baseline`: the cup mass is forced to 0.17/0.27/0.37/0.47kg, but the policy was not trained with bead or mass observations.
- v8 uses the same 5D synergy grip interface as v7, but is mass-aware through bead-conditioned training.
- v9 and v10 use 20D full-joint grip control; compare their grip metric as a curl-joint/full-hand intensity summary rather than a direct action-space equivalent to v7/v8.
- v10 should be judged by combined evidence: success across mass, sufficient force ratio, stable contact, lower slip proxy, and full-joint adaptive grip behavior.

| version | label | bead | mass_kg | n | success_rate | grip | contacts | force_ratio | slip_proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v10 | full_joint | 20 | 0.370 | 4 | 1.000 | 0.973 | 5.000 | 2.688 | 0.000 |
