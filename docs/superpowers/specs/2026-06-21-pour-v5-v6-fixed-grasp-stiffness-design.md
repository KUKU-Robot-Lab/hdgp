# Pour v5/v6 Fixed-Grasp Stiffness Design

## Goal

Strengthen the fixed Tesollo grasp in `pour_v5` and `pour_v6` without adding finger policy actions or changing the warmstart hand pose.

## Change

Apply the same actuator settings in both task configs:

- `tesollo_hand_curl`: stiffness `100.0 -> 150.0`, damping `18.0 -> 22.0`
- `tesollo_hand_pip`: stiffness `100.0 -> 150.0`, damping `18.0 -> 22.0`
- `tesollo_hand_dip`: stiffness `100.0 -> 150.0`, damping `18.0 -> 22.0`

Keep abduction settings, joint targets, warmstart cache poses, action dimensions, and rewards unchanged. The higher damping approximately preserves the existing damping ratio after the stiffness increase.

## Expected Behavior

The initial joint target remains equal to the cached grasp pose, so the change does not issue an additional closing step at reset. When contact forces or palm motion push fingers away from that target, the implicit actuator applies stronger restoring torque.

## Verification

1. Add a focused static config test asserting identical curl/PIP/DIP gains in pour-v5 and pour-v6 and unchanged abduction gains.
2. Run the focused test and Python syntax checks for both config files.
3. In the next matched training run, compare cup relative drift, `Reward/hold`, episode length, and observed drop/ejection behavior. Do not change `r_hold` in the same experiment.

## Non-Goals

- No tighter grasp pose or post-reset squeeze ramp.
- No finger policy channel.
- No reward-weight or termination changes.
- No actuator changes outside pour-v5 and pour-v6.
