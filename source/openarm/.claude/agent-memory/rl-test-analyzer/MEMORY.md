# RL Test Analyzer Agent Memory

## Projects
- [v9 20D Action Space Training Failure](v9_analysis_findings.md) — 5g_grasp_right_v9 seed divergence, enclosure decoy, force target misdirection

## Key Findings Library
- **20D action space curse**: Combinatorial explosion of hand configs; <0.1% functional → extreme seed sensitivity
- **Enclosure reward trap**: Geometric proximity rewards without force contact → policies learn to stay near cup without gripping
- **Force target penalty bug**: Linear negative penalty on force difference → perverse incentive to minimize grip force
- **Phase-reward coupling bug**: Lift signal only in lift phase; grasp phase has zero gradient for lifting actions → exploration doesn't discover lift
