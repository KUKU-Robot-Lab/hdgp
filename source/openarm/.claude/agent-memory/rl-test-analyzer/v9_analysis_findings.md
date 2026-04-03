---
name: v9 20D Action Space Training Failure Analysis
description: Root cause analysis of 5g_grasp_right_v9 (20D per-joint delta) training divergence and failures
type: project
---

## Summary

**5g_grasp_right_v9** training shows extreme seed sensitivity due to 20D action space complexity:
- **test1**: Episode success **0.0%** (trapped in local minimum: hand near cup, fingers open, enclosure=1.6)
- **test2**: Episode success **0.32%** (lucky seed, found lift basin by epoch ~700)
- **Gap**: 3900+ total reward in test2 vs 990 in test1 (3.95x difference)

## Root Causes (Ranked)

### 1. **20D Action Space Too High-Dimensional** (PRIMARY)
- v8 used **5D synergy lerp** (bounded hand pose interpolation)
- v9 uses **20D per-joint deltas** (±0.30 rad/step × 20 joints)
- Result: Combinatorial explosion of hand configs; <0.1% are functional
- Test1 explores but lands in non-grasping basin; test2 luckily explores toward grasping
- **Evidence**: 
  - Test1 grasp_quality_lift = 0.00027 (never discovers lift)
  - Test1 num_contacts = 0.00049 (hand barely touches)
  - Entropy increases to 208 but reward plateaus (exploration doesn't help)

### 2. **Enclosure Reward is Geometric Decoy** (SECONDARY)
- v9 enclosure metric = fingertip proximity to circle (purely geometric)
- **But actual grasping requires contact forces** (not in v9 observations for actor)
- Test1: enclosure ≈ 1.6 (high) while grip_force ≈ 0 (no force) and num_contacts ≈ 0 (no contact)
- Policy settles: "move hand near cup, keep semi-open" = reward without grasping
- **No escape**: Enclosure weight = 3.0 (significant), rewards path without lifting

### 3. **Force Target Reward Misdirected** (SECONDARY)
- Current formula: `R_force_target = -10 * |F_total - F_target| / F_target`
- Both tests have `F_total << F_target(mass)` → reward always **negative**
- **Perverse incentive**: Policy learns to **minimize force** to reduce penalty
- Test1 escalates: reduces grip force → hand slips off → enclosure via geometry alone
- Test2 still negative (-0.727 final) despite lifting

### 4. **Lift Reward Only Gated in Lift Phase** (TERTIARY)
- lift_reward = 0 if `object_z ≤ z_init` (binary gate)
- Grasp phase (480 steps): **zero gradient** for lifting (no signal to explore lift)
- Only lift phase (120 steps) activates lift, **but only after successful grasp**
- Result: No intermediate milestone; exploration never tries lifting actions in grasp phase
- Test1 never grasps → never enters lift phase → never sees lift signal

### 5. **Stochastic Initialization Creates Basin Trap** (TERTIARY)
- Actor network weights randomly initialized (different per seed)
- Both tests have identical policy architecture, reward, env → divergence is purely from weight init
- **Mechanism**: Different parameter initializations create different loss landscapes
- Test1's landscape: local min at "hand near cup, open" (reward ≈800)
- Test2's landscape: passes through "grasping" basin at epoch 700 (reward ≈3900)

## Evidence Summary

| Metric | Test1 | Test2 | Ratio | Interpretation |
|--------|-------|-------|-------|---|
| lift_reward (final) | 0.00032 | 1.02846 | **3214x** | Test1 never lifts |
| num_contacts | 0.00049 | 0.88037 | **1803x** | Test1: no sustained contact |
| total_tip_force | 0.000004 | 0.06068 | **15,427x** | Test1: no grip force |
| episode_success | 0% | 0.32% | ∞ | Test2 barely succeeds |
| enclosure (false) | 1.61 | 0.71 | 0.44x | Test1's reward trap |

## Why Test2 Escapes Failure

1. **Lucky exploration**: Initial actor weights favor finger-closing actions (rare, stochastic)
2. **Slip reward activation** (epoch ~650): Contact → slip_reward signal appears (weight=8.0, strong)
3. **Positive feedback**: Contact → lift attempt → lift_reward fires (weight=6.0, strong) → policy commits
4. **Scales dominance**: slip_reward alone contributes ≈3.8 to total reward (8.0 weight × 0.475 mean value)

## Recommended Fixes (Priority Order)

### Fix 1: Reduce Action Space to 10-11D
- Revert to **6D palm + 5D synergy** (v8 style) OR
- Use **4D PCA basis** of hand joint velocities
- **Why**: Tractable exploration space, learned interpolation between functional poses
- **Expected Impact**: Both tests should find lift basin by epoch 400-500

### Fix 2: Implement Phase-Aware Reward
```python
if step < 480:  # Grasp phase
    R = palm_approach + force_balance + multi_phalanx + slip
    # NO enclosure, NO lift_reward (removes decoy signal)
else:  # Lift phase
    R = lift_reward + grasp_quality_lift + slip
    # Penalize enclosure drop (maintain during lift)
```
- **Why**: Separates grasp and lift objectives, prevents geometric trap
- **Expected Impact**: Both tests converge toward grasping basin

### Fix 3: Asymmetric Force Target Penalty
```python
force_ratio = F_total / F_target(mass)
if force_ratio >= 1.0:
    R = -0.5 * (force_ratio - 1.0)^2  # Penalize over-gripping
else:
    R = -2.0 * (1.0 - force_ratio)^2  # Penalize under-gripping (2x weight)
```
- Change weight from 10.0 → 2.0
- **Why**: Removes perverse incentive to minimize force
- **Expected Impact**: Test1 no longer learns "reduce force to reduce penalty"

### Fix 4: Add Curriculum Reward Shaping
```python
# Epoch-based curriculum
if epoch < 500:
    enclosure_weight *= 2.0
    grasp_quality_weight *= 0.0  # No lift yet
elif epoch < 1000:
    enclosure_weight *= 1.0
    grasp_quality_weight *= 1.0  # Introduce lift
else:
    enclosure_weight *= 0.5  # De-emphasize
    grasp_quality_weight *= 2.0  # Emphasize lift
```
- **Why**: Guides exploration toward lift basin, removes reliance on lucky seed
- **Expected Impact**: Consistent >50% success across seeds

## Files to Modify

1. **grasp_right_env.py**
   - Line 663: Action space (finger_delta computation)
   - Reward section: Phase-aware reward logic
   
2. **grasp_right_env_cfg.py**
   - Lines 151-152: Action space configuration
   - Lines 188-191: force_target parameters (weight, base, scale)
   - Lines 196-209: slip/efficiency/smooth weights

3. **grasp_right_constants.py**
   - Line for NUM_ACTIONS (if synergy basis, set to 11 instead of 26)

## Why v8 Likely Worked Better

- **5D synergy space**: Bounded, learned interpolation between hand_open and hand_grasp poses
- **Continuous grasp control**: All actions map to valid, grasping-adjacent hand shapes
- **No geometric trap**: Synergy forces finger closure (C0 continuity in pose space)

## Test2's Fragility

Despite reaching 0.32% success, test2 is not production-ready:
- **Mass sensitivity**: Fixed grip force (~1.3 N) fails on heavier loads (bead variation 0-0.20 kg)
- **Orientation loss**: force_balance ≈ 0 (asymmetric grip) → tipping risk
- **Slip not managed**: slip_reward maxes out (4.8) but suggests jerky hand transitions
- **Low reproducibility**: Only marginal signal on real robots (0.32% ≈ 1 success per 300 episodes)

**Conclusion**: Need to implement fixes 1-3 (at least) before test2's approach is viable.

---

**Last Updated**: 2026-04-03 (analysis_report_20260403.md)
**Analysis Confidence**: Very High (based on 2800+ iterations, 143 TensorBoard metrics, code inspection)
