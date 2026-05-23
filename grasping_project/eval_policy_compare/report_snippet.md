# Adaptive Grasp Checkpoint Comparison

Protocol: each version/bead-count bin was evaluated in an isolated Isaac process, then merged.

Interpretation notes:
- v7 is a `synergy_forced_mass_baseline`: the cup mass is forced to 0.17/0.27/0.37/0.47kg, but the policy was not trained with bead or mass observations.
- v8 uses the same 5D synergy grip interface as v7, but is mass-aware through bead-conditioned training.
- v9 and v10 use 20D full-joint grip control; compare their grip metric as a curl-joint/full-hand intensity summary rather than a direct action-space equivalent to v7/v8.
- v10 should be judged by combined evidence: success across mass, sufficient force ratio, stable contact, lower slip proxy, and full-joint adaptive grip behavior.

| version | label | bead | mass_kg | n | success_rate | grip | contacts | force_ratio | slip_proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v7 | synergy_forced_mass_baseline | 0 | 0.170 | 4 | 0.250 | 0.050 | 1.250 | 1.820 | nan |
| v7 | synergy_forced_mass_baseline | 10 | 0.270 | 4 | 0.000 | 0.000 | 1.000 | 1.284 | nan |
| v7 | synergy_forced_mass_baseline | 20 | 0.370 | 4 | 0.000 | 0.000 | 1.000 | 0.948 | nan |
| v7 | synergy_forced_mass_baseline | 30 | 0.470 | 4 | 0.000 | 0.000 | 1.000 | 1.394 | nan |
| v8 | synergy_mass_aware | 0 | 0.170 | 4 | 1.000 | 0.728 | 3.250 | 5.931 | nan |
| v8 | synergy_mass_aware | 10 | 0.270 | 4 | 1.000 | 0.718 | 3.000 | 4.133 | nan |
| v8 | synergy_mass_aware | 20 | 0.370 | 4 | 1.000 | 0.749 | 3.250 | 3.198 | nan |
| v8 | synergy_mass_aware | 30 | 0.470 | 4 | 1.000 | 0.627 | 2.750 | 2.232 | nan |
| v9 | full_joint | 0 | 0.170 | 4 | 0.000 | 1.000 | 4.000 | 1.297 | 0.000 |
| v9 | full_joint | 10 | 0.270 | 4 | 0.750 | 0.974 | 4.750 | 2.771 | 0.000 |
| v9 | full_joint | 20 | 0.370 | 4 | 0.500 | 0.987 | 3.250 | 1.309 | 0.000 |
| v9 | full_joint | 30 | 0.470 | 4 | 1.000 | 0.985 | 5.000 | 1.080 | 0.000 |
| v10 | full_joint | 0 | 0.170 | 4 | 0.750 | 0.741 | 3.750 | 2.494 | 0.000 |
| v10 | full_joint | 10 | 0.270 | 4 | 1.000 | 0.991 | 5.000 | 2.046 | 0.000 |
| v10 | full_joint | 20 | 0.370 | 4 | 1.000 | 0.973 | 5.000 | 2.688 | 0.000 |
| v10 | full_joint | 30 | 0.470 | 4 | 1.000 | 0.989 | 5.000 | 1.787 | 0.000 |
