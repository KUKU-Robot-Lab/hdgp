# Adaptive Grasp Checkpoint Comparison

Protocol: each version/bead-count bin was evaluated in an isolated Isaac process, then merged.

Interpretation notes:
- v7 is a `synergy_forced_mass_baseline`: the cup mass is forced to 0.17/0.27/0.37/0.47kg, but the policy was not trained with bead or mass observations.
- v8 uses the same 5D synergy grip interface as v7, but is mass-aware through bead-conditioned training.
- v9 and v10 use 20D full-joint grip control; compare their grip metric as a curl-joint/full-hand intensity summary rather than a direct action-space equivalent to v7/v8.
- v10 should be judged by combined evidence: success across mass, sufficient force ratio, stable contact, lower slip proxy, and full-joint adaptive grip behavior.

| version | label | bead | mass_kg | n | success_rate | grip | contacts | force_ratio | slip_proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v7 | synergy_forced_mass_baseline | 10 | 0.270 | 4 | 0.000 | 0.000 | 1.000 | 1.284 | nan |
