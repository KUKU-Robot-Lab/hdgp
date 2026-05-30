# 구조 변경.

## 1 PPO + Behavior Cloning (BC) auxiliary loss 타당성

### 1.1 Demonstration‑augmented policy gradients

- **DAPG (Demonstration‑Augmented Policy Gradient)** is a well‑known approach for combining demonstrations with reinforcement learning. In the DAPG framework, the policy is initialized by behavior cloning and then optimized with a PPO‑style policy gradient plus an additional BC loss term. A survey of robotic in‑hand manipulation notes that DAPG “uses pre‑training with BC to initialize the policy and an **augmented loss function** to reduce ongoing bias toward the demonstration,” and that the resulting policies “acquire more human‑like motion compared to RL from scratch” while improving robustness and sample efficiency.
- The method decays the weight of the BC term over training. This dynamic weighting allows the policy to take advantage of demonstrations during early learning and later rely more on the reward function. Using a decreasing BC weight prevents the policy from being stuck near the demonstration and reduces policy collapse.

### 1.2 Auxiliary behavior‑cloning loss in actor‑critic methods

- The DDPG+Demonstrations (DDPGfD) paper shows that adding a **BC loss on demonstration transitions** to the actor gradient, weighted by a constant, significantly accelerates learning on sparse‑reward tasks. The paper also introduces a **Q‑filter** so that BC loss is only applied when the demonstrator’s action has a higher Q‑value than the current actor’s action. This prevents the demonstrator from degrading the policy after it exceeds expert performance.
- Other works, such as action‑chunked PPO with self behavior cloning, add a BC auxiliary loss computed on self‑collected high‑quality trajectories. This auxiliary loss encourages the policy to mimic successful past experiences and adaptively changes its weight during training.
- In summary, **adding a BC auxiliary loss** to PPO is a common and empirically supported way to leverage demonstrations. It is crucial to (1) anneal the BC weight over training, (2) optionally use a critic‑based filter to avoid learning from worse‑than‑policy demonstrations, and (3) ensure that the BC loss does not dominate the PPO objective.

### 1.3 Guidelines for v5

1. **Warm‑start with BC**: initialize the policy by pre‑training with BC on HDF5 demonstrations to give a reasonable starting point.
2. **Auxiliary BC loss during PPO**: add a BC loss term to the PPO objective during RL training. Begin with a relatively high BC weight to accelerate early learning and gradually decay it. Weight schedules used in DAPG (e.g., exponential decay) or DDPGfD can be adapted.
3. **Q‑filter or gating**: apply BC loss only when the critic predicts that the demonstrator action is better than the current policy, or use a recurrent gate to ramp in the LSTM/BC loss over time.
4. **Diagnostic metrics**: monitor BC loss and its weight (e.g., `bc/loss_real_demo`, `bc/weight_real_demo`) to ensure that the BC term helps early training but does not prevent exploration.

## 2 LSTM PPO 타당성

### 2.1 Why use LSTMs?

Manipulation tasks often exhibit **partial observability**: the agent observes only limited proprioceptive and visual information at each step.  Recurrent policies can maintain a hidden state that summarizes past observations.  The `ICLR Blog Track` article on PPO implementation details provides practical guidelines for LSTM policies:

- LSTM weights are initialized with standard deviation 1 and biases to zero; hidden and cell states are initialized to zero.
- **Reset LSTM states at episode boundaries**: the hidden states must be reset to zero whenever an episode ends to avoid leakage between episodes.
- **Sequential minibatches**: when training LSTM PPO, one must preserve the temporal order of roll‑outs when forming minibatches. Hidden states should be reconstructed for each sequence during back‑propagation through time.

Stable Baselines’ Recurrent PPO documentation similarly notes that one must pass the previous hidden state and episode start flags to the policy at every step; failing to do so leads to incorrect LSTM state updates.

### 2.2 Sequence length and mini‑batch choices

For RL‑Games, the **`seq_length`** parameter controls how many time steps are batched together for the recurrent network.  Good practice is to set `seq_length` such that `horizon_length` (the rollout length per environment) is divisible by `seq_length`, ensuring that each mini‑batch contains whole sequences.  Typical `seq_length` values range from **32–128** depending on the environment and computational budget.  The mini‑batch size should equal the number of environments (`num_envs`) times the horizon length divided by the number of mini‑epochs, or use RL‑Games’ `minibatch_size_per_env` to simplify.

### 2.3 Asymmetric actor–critic and central value

RL‑Games allows an asymmetric actor–critic architecture where the **actor** observes only partial information (e.g., proprioception, tactile features) while the **critic** (called the *central value* network) has access to full state information.  The `README` notes that `use_central_value` can be enabled, after which the observation must be a dictionary with keys `'obs'` (actor inputs) and `'state'` (critic inputs).  RL‑Games also provides a `before_mlp` flag that controls whether the RNN is applied before the MLP.  Applying the RNN before the MLP allows the recurrent layer to learn temporal structure from low‑level features.

When combining an LSTM policy with a central critic, ensure that **both the actor and the central critic** are recurrent or that the critic receives the actor’s hidden state.  RL‑Games supports concatenating the actor’s hidden state into the critic input; `concat_input` and `layer_norm` options can also affect stability.  Hidden states must be reset and carried consistently between the actor and critic.

### 2.4 Guidelines for v5

1. **Choose a sequence length** such that `horizon_length` is divisible by it (e.g., if using `horizon_length=4096` and 128 environments, use `seq_length=64` or `128`).
2. **Reset hidden states** at episode boundaries within the environment; set `zero_rnn_on_done=True` in the RL‑Games config to automatically zero the hidden state when an environment resets.
3. **Configure `rnn.before_mlp` and `concat_input`**: apply the LSTM before the MLP to learn temporal features and optionally concatenate the previous hidden state. Enable `layer_norm` inside the LSTM to stabilize training.
4. **Central value considerations**: if using asymmetric actor–critic, ensure the critic’s input dimension matches the environment’s full state and that hidden states are treated consistently.

## 3 HDF5 데모 action supervision

### 3.1 Single‑arm policy from bimanual demonstrations

Bimanual demonstrations capture object interactions that may require both arms, which can introduce *covariate shift* when training a single‑arm policy.  A Japanese robotics study discusses that human demonstrators often only use one hand even when tasks usually require two hands; such “unnatural” single‑hand demonstrations can reduce effectiveness.  To address this, they propose a dual‑arm demonstration based single‑arm motion planning algorithm.  The method identifies the point at which both hands are closest, concatenates the motion segments before and after this point, smooths the trajectory with a low‑pass filter and converts it to the robot’s configuration space.  Through this process, a single‑arm robot learns a trajectory from dual‑arm demonstrations.  Experiments showed higher success rates and improved path quality compared with existing methods.

This result suggests that extracting the **right‑arm trajectory** from bimanual demonstrations can be effective if the demonstration is processed to isolate the relevant segment and smoothed.  However, if the left hand’s actions significantly influence the task (e.g., setting up the object), simply ignoring them may lead to distribution mismatch.

### 3.2 Mapping human actions to simulator actions

Converting a human’s 18D teleoperation action (joint velocities or pose changes) to a robot’s 11D action space requires careful remapping.  It is advisable to create a **linear mapping function** that projects the demonstration actions onto the simulator action space and to verify that the mapped actions reproduce similar motion when replayed in simulation.  When action data are unavailable, one can derive actions by differentiating successive joint poses, but noise may require smoothing.

### 3.3 Guidelines for v5

1. **Pre‑process dual‑arm demonstrations**: identify the segment where only the right arm manipulates the cup and extract that portion. Low‑pass filter the extracted motion to remove discontinuities.
2. **Action remapping**: convert 18D teleoperation actions to the 11D simulator actions by selecting the relevant joints and scaling velocities; check consistency by replaying demonstration trajectories.
3. **Check for covariate shift**: examine whether left‑hand motions influence the object’s state in ways not captured in the right‑arm state; if so, consider including object state features in the policy input or using domain randomization.

## 4 Demo pose reward vs BC loss 역할 분리

The current v5 environment uses a **nearest‑neighbor pose reward**: the agent’s right‑arm joint pose is compared to a bank of demonstration poses; the nearest demonstration frame plus a look‑ahead offset is selected and pose/hand alignment rewards are computed.  While effective for guiding the policy to follow the demonstration, this reward has notable limitations:

- **Temporal ambiguity**: nearest‑neighbor matching ignores the temporal order of the demonstration. The demonstration may be composed of approach, pre‑pour and pouring phases; nearest‑neighbor reward can match a later phase even when the agent is still approaching, leading to unnatural transitions.
- **Gate and warm‑up mechanisms**: the current implementation activates the demo reward only near the goal (`demo_pose_near_gate_xy`) and applies a warm‑up to avoid early bias. Removing these gates means the demo reward influences the agent throughout the episode, necessitating careful re‑tuning of the reward weights.

Recent work on latent nearest‑neighbor reward shaping indicates that the reward is computed as the distance between the agent’s state and the demonstration trajectories in an embedding space; this encourages the agent to stay close to demonstrations.  However, such methods still rely on distance alone and may not enforce monotonic progress along the demonstration.

In contrast, an **auxiliary BC loss** directly supervises the policy’s actions using demonstration actions.  BC enforces temporal order and can capture subtle motion patterns that nearest‑neighbor rewards miss.  Combining a pose reward and BC can provide complementary guidance: the pose reward pulls the state towards demonstration poses, while the BC loss aligns the action distribution.  Yet, if both use the same demonstration data, there is a risk of over‑constraining the policy and reducing exploration.  It is therefore advisable to **decouple the weights**: treat the pose reward as a shaping reward with a relatively small weight and rely on BC as the primary demonstrator supervision, decaying its weight over training.

## 5 Phase‑aware vs phase‑free demo matching

### 5.1 Issues with phase‑free matching

Using a single nearest‑neighbor bank for the entire demonstration ignores **phase information**, causing the agent to potentially match a pouring pose during the approach phase or vice versa.  This can misalign the policy’s progress and lead to unstable behavior.  Without a monotonic progress variable, the agent may oscillate between phases.

### 5.2 Phase‑aware alternatives

- **Progress variables or phase estimators**: incorporate a phase variable (e.g., normalized time index or progress fraction) into the reward. The demonstration can be segmented into approach, pre‑pour and pour phases; the agent receives rewards for matching the corresponding segment based on its progress.
- **Temporal alignment methods**: use dynamic time warping, dynamic movement primitives (DMPs) or recurrent imitation learning to align the agent’s trajectory with the demonstration. These methods preserve temporal order and can provide phase‑conditioned targets.
- **Goal‑conditioned imitation**: treat the current goal (e.g., cup tilt angle, cup position) as the conditioning input and compute demonstration pose rewards relative to that goal.

### 5.3 Guidelines for v5

1. **Set `demo_pose_phase="all"`** but disable `demo_pose_near_gate_xy` to allow the demo reward throughout the episode.
2. **Reduce weights of removal terms**: set the dense reward weights (approach XY/Z, pre‑pour alignment, directional tilt, cup upright) to zero, but continue computing them for diagnostics.
3. **Adjust `demo_pose_warmup_steps`**: increase the warm‑up period so that the demo reward ramps in gradually and does not dominate early exploration.
4. **Consider phase‑aware matching**: if nearest‑neighbor matching causes confusion, implement a simple progress variable that increments with steps and uses it to select a demonstration segment. Alternatively, adopt a monotonic distance metric along the demonstration.

## 6 RL‑Games LSTM + central value 조합 검토

RL‑Games supports various network configurations:

- **`before_mlp`**: when set to `True`, the RNN processes the observation before it is passed into the MLP; otherwise the MLP processes the observation first. The README notes this parameter under the network options. Applying the RNN before the MLP allows the recurrent layer to learn temporal patterns directly from raw inputs, which is beneficial for partial‑observable tasks.
- **`use_central_value`**: enabling this flag uses an asymmetric actor–critic where the critic receives additional state information (e.g., full robot pose, environment state). The environment must return a dictionary with keys `obs` and `state`. When using LSTMs, ensure that the central value network is also recurrent or that its hidden state is concatenated with the actor’s hidden state. RL‑Games recently added `concat_output` support to combine the actor’s hidden state with the critic input.

For the v5 pouring task:

1. **Asymmetric critic**: using a central critic with full object and robot state may improve value estimation and accelerate learning. The actor should still observe partial information (e.g., hand joint angles, palm pose). Make sure the environment returns a dictionary with `obs` and `state` and that LSTM hidden states are handled consistently.
2. **Sequence length**: choose `seq_length` such that `horizon_length` is divisible by it.
3. **`zero_rnn_on_done`**: set this to `True` so that hidden states are reset at episode end.
4. **`before_mlp=True` and `concat_output=True`**: process inputs with the LSTM before the MLP and concatenate the hidden state with the critic input if using a central value network.

## 7 우선 권장 조치

Based on the literature and code inspection, the following steps are recommended for the v5 *5g_pour_right_v5* task:

1. **Fix task registration**: ensure that `config/__init__.py` registers the v5 gym IDs (`5g_pour_right-v5` and `5g_pour_right-play-v5`) and points to the correct entry point.
2. **Reward modifications**: set `demo_pose_phase="all"`, remove gates (`demo_pose_near_gate_xy=1.0`), and set the weights of approach/pre‑pour/directional tilt/cup upright rewards to zero. Keep the terms for diagnostics but multiply their contribution by zero. Increase `demo_pose_warmup_steps` to avoid early over‑shaping.
3. **LSTM + BC agent**: port the v4 LSTM+BC implementation. Use a policy network with LSTM before the MLP and a central value network. Initialize the agent by BC pre‑training on the right‑arm demonstrations and add a BC auxiliary loss with a decaying weight during PPO. Use a recurrent gate to gradually ramp in the LSTM.
4. **HDF5 demonstration handling**: extract the right‑arm portion from the bimanual demonstration using the method described above. Convert 18D teleoperation actions to 11D simulator actions and verify the mapping.
5. **Phase awareness**: monitor whether nearest‑neighbor pose rewards cause phase confusion; if so, implement a progress variable or segment the demonstration.
6. **Diagnostics**: track BC loss (`bc/loss` and `bc/loss_real_demo`), BC weight, recurrent gate alpha, demonstration pose errors, capture and cross rewards, success rate and spill ratio. These metrics will help determine whether the auxiliary BC loss and LSTM contribute positively.

Implementing these recommendations should allow v5 to leverage demonstrations more effectively and provide a foundation for exploring LSTM+BC+PPO in robotic pouring tasks.