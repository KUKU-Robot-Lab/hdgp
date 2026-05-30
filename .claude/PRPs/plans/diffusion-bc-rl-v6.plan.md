# Plan: Diffusion BC Pre-train → skrl RL Fine-tune (5g_pour_right_v6)

## Summary
Phase 1에서 ConditionalUNet1D 기반 Diffusion BC를 데모(양팔 FK obs)로 선학습하고,
Phase 2에서 skrl PPO의 Actor로 사용해 sparse task reward로 fine-tune한다.
왼팔은 kinematic 유지 + 에피소드마다 domain randomization으로 target cup 위치를 변화시켜 robustness를 확보한다.
모든 코드는 5g_pour_right_v6 디렉터리 내에 추가한다.

## User Story
As a robot learning engineer,
I want a diffusion policy trained from limited demos to know the pour motion,
So that RL can fine-tune robustness against varied cup positions without movement reward conflicts.

## Problem → Solution
**현재**: v6에 BC가 auxiliary loss로 붙어 있어 "랜덤 정책"에서 RL 탐색이 시작됨 →
  sparse reward로는 bead_in_target을 발견 못함 (v5 실패 재현).

**목표**: Diffusion BC 선학습으로 "이미 물 붓는 정책"에서 RL 시작 →
  sparse reward (bead capture + spill)로 domain robust 학습 가능.

## Metadata
- **Complexity**: XL
- **Source PRD**: N/A
- **Estimated New Files**: 8
- **Estimated Modified Files**: 4

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: BC Pre-train (standalone, ~5000 epoch)            │
│                                                             │
│  HDF5 demos (pour_v1_a11~a20.hdf5)                        │
│       ↓ _load_episode_fk() [NEW]                           │
│  obs(52D) = 양팔 joint + 양팔 EEF FK (datagen_info 없음)   │
│  action(11D) = right palm delta(6) + finger lerp(5)        │
│       ↓ DDPM noise prediction loss                         │
│  DiffusionBCNet checkpoint (.pt)                           │
│                                                             │
│  DiffusionBCNet:                                           │
│    obs_encoder: Linear(52→256)→Mish→Linear(256→256)        │
│    noise_net: ConditionalUNet1D(input=11, cond=256,        │
│               down_dims=[256,512,1024], chunk=16)          │
│    noise_scheduler: DDPM(T=100, squaredcos_cap_v2)         │
└─────────────────────────────────────────────────────────────┘
            ↓ checkpoint 로드
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: RL Fine-tune (skrl PPO + asymmetric A2C)          │
│                                                             │
│  DiffusionActor (skrl GaussianMixin):                      │
│    act(): DDIM K=5 steps → μ(11D)                         │
│    log_std: 학습 가능 파라미터 (탐색 분산)                  │
│    BC aux loss: DDPM loss (diffusion model 함께 fine-tune)  │
│                                                             │
│  AsymmetricCritic (skrl): 143D full sim state              │
│                                                             │
│  Reward: r_capture_spill + r_success (sparse, task only)   │
│                                                             │
│  Domain Random:                                            │
│    - source cup spawn ±10cm XY                            │
│    - target cup (left arm) ±3cm XY, ±2cm Z [NEW]          │
│    - pour tilt target 100~130°                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Obs 설계 (52D, 양팔 FK, datagen_info cup 제거)

```
[0:7]   right_arm_joint_pos       7  — from HDF5 right_arm_joint_pos
[7:14]  right_arm_joint_vel       7  — from HDF5 right_joint_vel[:7]
[14:19] right_hand_summary        5  — finger lerp progress
[19:26] left_arm_joint_pos        7  — from HDF5 left_arm_joint_pos [양팔!]
[26:33] left_arm_joint_vel        7  — zeros (HDF5에 없음, BC서 중요도 낮음)
[33:36] right_palm_pos            3  — from eef_pose/right (FK, = source cup 위치)
[36:40] right_palm_quat           4  — from eef_pose/right (FK, = source cup 자세)
[40:43] left_palm_pos             3  — from eef_pose/left (FK, = target cup 위치)
[43:46] pour_vec                  3  — left_palm_pos - right_palm_pos (방향벡터)
[46:52] last_palm_actions         6  — action[t-1][:6]
Total: 52D
```

**기존 60D와의 차이**: bead/flow/spill(8D) 제거, 왼팔 joint(14D) 추가, left EEF FK(3D) 추가.

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---------|------|-------|-----|
| P0 | `demo_bc_buffer.py` | 330~432 | `_load_episode()` 구조 — FK 버전의 기반 |
| P0 | `demo_pose_reference.py` | 350~410 | `left_arm_joint_pos`, `eef_pose/left` 로딩 패턴 |
| P0 | `pour_right_env.py` | 2243~2300 | `_get_left_cup_attached_pose()` — domain random 추가 위치 |
| P1 | `pour_right_env_cfg.py` | 382~436 | 물체 spawn 설정 — left arm random 파라미터 추가 위치 |
| P1 | `diffusion_policy/model/diffusion/conditional_unet1d.py` | 전체 | UNet1D API |
| P1 | `pour_chunk_bc_agent.py` | 35~70 | chunk_head 구조 참조 (대체 대상) |
| P2 | `hdgp/scripts/reinforcement_learning/skrl/train.py` | 전체 | skrl Runner 사용 패턴 |
| P2 | `pour_right_constants.py` | 전체 | NUM_OBSERVATIONS 변경 대상 (60→52) |

---

## Patterns to Mirror

### DEMO_LOAD_PATTERN
```python
# SOURCE: demo_bc_buffer.py:332~345
with h5py.File(path, "r") as h5:
    demo = h5["data"]["demo_0"]
    s = slice(start, None, stride)
    arm_pos = torch.as_tensor(demo["obs"]["right_arm_joint_pos"][s], ...)
    eef_mat = torch.as_tensor(demo["obs"]["datagen_info"]["eef_pose"]["right"][s], ...)
    # left EEF: demo["obs"]["datagen_info"]["eef_pose"]["left"][s]
```

### LEFT_EEF_PATTERN
```python
# SOURCE: demo_pose_reference.py:385~390
target_cup = _left_eef_matrix_to_cup_pose(
    np.asarray(demo["obs/datagen_info/eef_pose/left"][selector])
)
# → eef/left 4×4 mat → pos(3) + quat(4)
```

### DDPM_TRAIN_PATTERN
```python
# SOURCE: diffusion_policy/diffusion_unet_lowdim_policy.py (참조)
noise = torch.randn_like(actions)
t = torch.randint(0, T, (B,))
noisy = scheduler.add_noise(actions, noise, t)
pred = noise_net(noisy, t, global_cond=obs_cond)
loss = F.mse_loss(pred * mask, noise * mask)
```

### DDIM_INFERENCE_PATTERN
```python
# K-step DDIM (diffusion_policy DDIM scheduler)
from diffusers import DDIMScheduler
scheduler = DDIMScheduler(num_train_timesteps=100)
scheduler.set_timesteps(K)  # K=5 inference steps
x = torch.randn(N, chunk_size, 11)
for t in scheduler.timesteps:
    pred_noise = noise_net(x, t, global_cond=cond)
    x = scheduler.step(pred_noise, t, x).prev_sample
return x[:, 0, :]  # 첫 action 실행
```

### SKRL_GAUSSIAN_ACTOR_PATTERN
```python
# SOURCE: skrl GaussianMixin 패턴
from skrl.models.torch import GaussianMixin, Model
class DiffusionActor(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device, **kw):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=True,
                               clip_log_std=True, min_log_std=-4, max_log_std=2)
    def compute(self, inputs, role):
        mu = self.diffusion_sample(inputs["states"])
        return mu, self.log_std_parameter, {}
```

### LEFT_ARM_DOMAIN_RANDOM_PATTERN
```python
# SOURCE: pour_right_env.py:2283~2292 (_set_left_demo_reference)
# 현재: demo 값 그대로 사용
self.left_arm_target_pos[env_ids_t] = target_left  # demo에서

# 수정: noise 추가
noise_joints = torch.randn_like(target_left) * cfg.left_arm_domain_noise_scale
self.left_arm_target_pos[env_ids_t] = target_left + noise_joints.clamp(-cfg.left_arm_noise_clip, ...)
```

---

## Files to Change

| File | Action | Justification |
|------|--------|---------------|
| `diffusion_bc/__init__.py` | CREATE | 패키지 |
| `diffusion_bc/net.py` | CREATE | DiffusionBCNet (obs_encoder + UNet1D) |
| `diffusion_bc/buffer.py` | CREATE | DemoBCBufferV2: 52D FK obs, datagen cup 제거 |
| `diffusion_bc/trainer.py` | CREATE | Phase 1 DDPM 학습 루프 |
| `diffusion_bc/ddim_sampler.py` | CREATE | DDIM K=5 inference |
| `diffusion_actor_skrl.py` | CREATE | skrl GaussianMixin Actor |
| `train_bc.py` | CREATE | Phase 1 launch script (Isaac 없이 실행) |
| `config/agents/skrl_diffusion_ppo_cfg.yaml` | CREATE | Phase 2 PPO config |
| `pour_right_constants.py` | UPDATE | NUM_OBSERVATIONS: 60→52, obs layout 주석 |
| `pour_right_env_cfg.py` | UPDATE | left arm domain random 파라미터, BC obs flag |
| `pour_right_env.py` | UPDATE | `_reset_idx` 내 left arm domain random 추가 |
| `demo_bc_buffer.py` | UPDATE | `_load_episode_fk()` 추가 (선택적 대체 함수) |

## NOT Building

- rl_games 코드 수정 없음 (Phase 2는 skrl 전용)
- obs 60D LSTM PPO 경로 제거 안 함 (기존 경로 유지, 새 경로 추가)
- 양팔 동시 policy 제어 없음 (왼팔 kinematic 유지)
- Transformer diffusion 없음 (UNet1D로 충분)
- Real robot deployment 코드 없음

---

## Step-by-Step Tasks

### Task 1: DemoBCBufferV2 — 양팔 FK obs (datagen cup 제거)
- **ACTION**: `diffusion_bc/buffer.py` 신규 작성
- **IMPLEMENT**:
  ```python
  def _load_episode_fk(path, stride, device):
      """HDF5 → 52D FK obs + 11D action.
      datagen_info cup pos 사용 안 함. EEF FK만 사용."""
      with h5py.File(path, "r") as h5:
          demo = h5["data"]["demo_0"]
          s = slice(0, None, stride)
          arm_pos   = demo["obs"]["right_arm_joint_pos"][s]       # (T, 7)
          arm_vel   = demo["obs"]["right_joint_vel"][s][:, :7]    # (T, 7)
          hand_pos  = demo["obs"]["right_hand_joint_pos"][s]      # (T, 20)
          left_arm  = demo["obs"]["left_arm_joint_pos"][s]        # (T, 7)
          r_eef_mat = demo["obs"]["datagen_info"]["eef_pose"]["right"][s]  # (T,4,4)
          l_eef_mat = demo["obs"]["datagen_info"]["eef_pose"]["left"][s]   # (T,4,4)
          tgt_mat   = demo["obs"]["datagen_info"]["target_eef_pose"]["right"][s]
          pour_start= demo["obs"]["datagen_info"]["subtask_start_signals"]["pour_start"][s]
      
      r_palm_pos  = r_eef_mat[:, :3, 3]                    # (T, 3)
      r_palm_quat = _quat_xyzw_from_matrix(r_eef_mat[:, :3, :3])  # (T, 4)
      l_palm_pos  = l_eef_mat[:, :3, 3]                    # (T, 3)
      pour_vec    = l_palm_pos - r_palm_pos                  # (T, 3)
      left_vel    = np.zeros_like(left_arm)                  # (T, 7) zeros
      finger_prog = _finger_grasp_progress(hand_pos)         # (T, 5)
      last_actions = np.zeros((T, 6), dtype=np.float32)
      last_actions[1:] = palm_actions[:-1]                   # shift-1
      
      obs = np.concatenate([
          arm_pos,       # 7
          arm_vel,       # 7
          finger_prog,   # 5
          left_arm,      # 7
          left_vel,      # 7
          r_palm_pos,    # 3
          r_palm_quat,   # 4
          l_palm_pos,    # 3
          pour_vec,      # 3
          last_actions,  # 6
      ], axis=-1)  # 52D
      
      # action: delta from episode-base palm pose (기존 demo_bc_buffer.py 방식 동일)
      # palm_actions (T, 6), finger_act = ones (T, 5)
      return DemoEpisodeFk(obs=obs, actions=actions, pour_mask=pour_mask)
  ```
- **MIRROR**: DEMO_LOAD_PATTERN, LEFT_EEF_PATTERN
- **GOTCHA**: `eef_pose/left` key가 HDF5에 있는지 사전 확인 필요.
  demo_pose_reference.py:385에서 이미 사용 중이므로 존재 확인됨.
  `right_joint_vel`은 27D(arm+hand); `[:7]`이 arm vel.
- **VALIDATE**: `obs.shape == (T, 52)`, `actions.shape == (T, 11)` assert

### Task 2: DiffusionBCNet — UNet1D + obs encoder
- **ACTION**: `diffusion_bc/net.py` 신규 작성
- **IMPLEMENT**:
  ```python
  import sys
  from pathlib import Path
  # diffusion_policy repo 경로 추가 (hdgp에서의 경로)
  _DP_SRC = Path("/home/user/rl_ws/repo/diffusion_policy")
  if str(_DP_SRC) not in sys.path:
      sys.path.insert(0, str(_DP_SRC))
  
  from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
  from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
  from diffusers.schedulers.scheduling_ddim import DDIMScheduler
  
  BC_OBS_DIM    = 52
  BC_ACTION_DIM = 11
  BC_CHUNK_SIZE = 16   # H=16 steps @ 60Hz = 0.27s
  BC_COND_DIM   = 256
  
  class DiffusionBCNet(nn.Module):
      def __init__(self):
          self.obs_encoder = nn.Sequential(
              nn.Linear(BC_OBS_DIM, 256), nn.Mish(),
              nn.Linear(256, BC_COND_DIM),
          )
          self.noise_net = ConditionalUnet1D(
              input_dim=BC_ACTION_DIM,
              global_cond_dim=BC_COND_DIM,
              diffusion_step_embed_dim=128,
              down_dims=[256, 512, 1024],
              kernel_size=3,
              n_groups=8,
          )
          self.ddpm_scheduler = DDPMScheduler(
              num_train_timesteps=100,
              beta_schedule="squaredcos_cap_v2",
              clip_sample=True,
          )
          self.ddim_scheduler = DDIMScheduler(
              num_train_timesteps=100,
              beta_schedule="squaredcos_cap_v2",
              clip_sample=True,
          )
      
      def encode_obs(self, obs):
          return self.obs_encoder(obs)  # (B, 256)
      
      def compute_loss(self, obs, action_chunk, mask):
          """DDPM noise prediction loss."""
          B = obs.shape[0]
          cond = self.encode_obs(obs)
          t = torch.randint(0, 100, (B,), device=obs.device).long()
          noise = torch.randn_like(action_chunk)
          noisy = self.ddpm_scheduler.add_noise(action_chunk, noise, t)
          pred = self.noise_net(noisy, t, global_cond=cond)  # (B, H, 11)
          m = mask.unsqueeze(-1).float()
          return (F.mse_loss(pred * m, noise * m, reduction='sum') /
                  m.sum().clamp(min=1.0))
      
      @torch.no_grad()
      def sample(self, obs, n_steps=5):
          """DDIM K-step inference → first action."""
          B = obs.shape[0]
          cond = self.encode_obs(obs)
          self.ddim_scheduler.set_timesteps(n_steps)
          x = torch.randn(B, BC_CHUNK_SIZE, BC_ACTION_DIM, device=obs.device)
          for t in self.ddim_scheduler.timesteps:
              pred = self.noise_net(x, t.expand(B), global_cond=cond)
              x = self.ddim_scheduler.step(pred, t, x).prev_sample
          return x[:, 0, :]  # (B, 11)
  ```
- **MIRROR**: DDPM_TRAIN_PATTERN, DDIM_INFERENCE_PATTERN
- **IMPORTS**: `diffusers` (pip install diffusers 필요), `conditional_unet1d.py` (repo path)
- **GOTCHA**:
  1. `ConditionalUnet1D.forward(sample, timestep, global_cond)` — sample shape: `(B, H, 11)` → UNet은 내부에서 `(B, 11, H)`로 rearrange
  2. `DDPMScheduler.add_noise(original, noise, timesteps)` — timesteps는 `(B,)` int tensor
  3. `diffusers` 버전 확인: `pip show diffusers` — 0.20+ 필요
- **VALIDATE**: `python3 -c "from diffusion_bc.net import DiffusionBCNet; m=DiffusionBCNet(); print('OK')`

### Task 3: DiffusionBCTrainer — Phase 1 학습 루프
- **ACTION**: `diffusion_bc/trainer.py` 신규 작성
- **IMPLEMENT**:
  ```python
  def train_bc(
      demo_paths,          # pour_v1_a11~a20.hdf5 경로들
      output_dir,          # checkpoint 저장 위치
      num_epochs=5000,
      batch_size=256,
      lr=1e-4,
      chunk_size=16,
      stride=2,
  ):
      dataset = DemoBCDatasetV2(demo_paths, stride=stride, chunk_size=chunk_size)
      loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
      model   = DiffusionBCNet().to("cuda")
      # EMA model for stable BC inference
      ema     = EMAModel(model, power=0.75)
      optim   = torch.optim.Adam(model.parameters(), lr=lr)
      
      for epoch in range(num_epochs):
          for obs, actions, mask in loader:
              loss = model.compute_loss(obs, actions, mask)
              optim.zero_grad(); loss.backward(); optim.step()
              ema.step(model)
          
          if epoch % 500 == 0:
              # EMA 모델로 체크포인트 저장
              torch.save({"model": ema.averaged_model.state_dict(),
                          "epoch": epoch}, f"{output_dir}/bc_{epoch}.pt")
  ```
- **MIRROR**: DDPM_TRAIN_PATTERN
- **IMPORTS**: `diffusion_bc.buffer.DemoBCDatasetV2`, `diffusion_bc.net.DiffusionBCNet`
- **GOTCHA**: EMA 필수 — DDPM 없이는 inference 불안정. `diffusion_policy/model/diffusion/ema_model.py` 참조.
- **VALIDATE**: 학습 loss가 1000 epoch 이내 0.1 이하로 감소하는지 확인

### Task 4: train_bc.py — Phase 1 launch script
- **ACTION**: `train_bc.py` 신규 작성 (Isaac Sim 없이 실행 가능)
- **IMPLEMENT**:
  ```python
  #!/usr/bin/env python3
  """Phase 1: Diffusion BC pre-training. Isaac Sim 불필요."""
  import argparse
  from pathlib import Path
  from diffusion_bc.trainer import train_bc
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--demo_dir", default="/home/user/rl_ws/datasets")
  parser.add_argument("--output_dir", default="/home/user/rl_ws/hdgp/log/diffusion_bc/v6")
  parser.add_argument("--epochs", type=int, default=5000)
  parser.add_argument("--batch_size", type=int, default=256)
  args = parser.parse_args()
  
  demo_paths = [Path(args.demo_dir) / f"pour_v1_a{i}.hdf5" for i in range(11, 21)]
  train_bc(demo_paths, args.output_dir, num_epochs=args.epochs)
  ```
- **VALIDATE**: `python3 train_bc.py --epochs 10` (smoke test, Isaac 없이)

### Task 5: skrl DiffusionActor
- **ACTION**: `diffusion_actor_skrl.py` 신규 작성
- **IMPLEMENT**:
  ```python
  from skrl.models.torch import GaussianMixin, Model
  from diffusion_bc.net import DiffusionBCNet
  
  class DiffusionActor(GaussianMixin, Model):
      def __init__(self, observation_space, action_space, device,
                   bc_checkpoint_path, n_ddim_steps=5, freeze_bc=False):
          Model.__init__(self, observation_space, action_space, device)
          GaussianMixin.__init__(self,
              clip_actions=True, clip_log_std=True,
              min_log_std=-4.0, max_log_std=0.0,
          )
          self.diffusion_bc = DiffusionBCNet().to(device)
          ckpt = torch.load(bc_checkpoint_path, map_location=device)
          self.diffusion_bc.load_state_dict(ckpt["model"])
          if freeze_bc:
              for p in self.diffusion_bc.parameters():
                  p.requires_grad_(False)
          
          self.n_ddim_steps = n_ddim_steps
          # 탐색 분산 — PPO가 학습함
          self.log_std_parameter = nn.Parameter(
              -2.0 * torch.ones(action_space.shape[0])
          )
      
      def compute(self, inputs, role):
          obs = inputs["states"]  # (N, 52) — 새 52D actor obs
          mu = self.diffusion_bc.sample(obs, n_steps=self.n_ddim_steps)
          return mu, self.log_std_parameter, {}
      
      def bc_loss(self, obs, action_chunk, mask):
          """BC fine-tune loss — PPO update와 함께 사용."""
          return self.diffusion_bc.compute_loss(obs, action_chunk, mask)
  ```
- **MIRROR**: SKRL_GAUSSIAN_ACTOR_PATTERN
- **GOTCHA**:
  1. `GaussianMixin.act()` 내부에서 `compute()`를 호출해 `(mu, log_std, _)` 받음.
  2. `log_std_parameter`는 scalar broadcast → `(N, 11)` 되어야 함. `expand_as(mu)` 불필요,
     GaussianMixin이 내부 처리.
  3. skrl PPO는 `get_log_prob(actions, states)`를 내부 호출 → GaussianMixin이 처리.
  4. `freeze_bc=True` 옵션: 처음엔 diffusion 고정, 나중에 fine-tune 여부 결정.

### Task 6: pour_right_constants.py — obs dim 업데이트
- **ACTION**: `pour_right_constants.py` 수정
- **IMPLEMENT**:
  ```python
  # 기존 (60D):
  # NUM_OBSERVATIONS = 60  # Actor: ...
  
  # 수정 (52D):
  NUM_OBSERVATIONS = 52   # DiffusionActor obs: 7+7+5+7+7+3+4+3+3+6
  # [0:7]  right_arm_joint_pos
  # [7:14] right_arm_joint_vel
  # [14:19] right_hand_summary (finger lerp)
  # [19:26] left_arm_joint_pos
  # [26:33] left_arm_joint_vel (zeros)
  # [33:36] right_palm_pos (FK)
  # [36:40] right_palm_quat (FK)
  # [40:43] left_palm_pos (FK = target cup)
  # [43:46] pour_vec (left-right)
  # [46:52] last_palm_actions
  ```
- **GOTCHA**: `NUM_CRITIC_OBSERVATIONS = 143` 유지 — 크리틱 obs 변경 없음.
  `pour_right_env.py`의 `_get_observations()` 반환값도 52D가 되도록 확인.
- **VALIDATE**: `grep NUM_OBSERVATIONS pour_right_env.py` → 사용처 모두 52D 확인

### Task 7: pour_right_env.py — actor obs 재계산 + left arm domain random
- **ACTION**: `pour_right_env.py` 두 곳 수정

**수정 7-1: `_get_observations()` 내 actor obs 52D로 재조립**
```python
# 기존 60D obs 조립 부분을 대체
# 새 52D: right_arm(14) + right_hand(5) + left_arm(14) + right_palm_FK(7) + left_palm_pos(3) + pour_vec(3) + last_actions(6)

left_arm_pos = self.robot.data.joint_pos[:, self.left_arm_dof_indices[:7]]  # (N,7)
left_arm_vel = torch.zeros_like(left_arm_pos)                                # (N,7)
right_palm_pos = self.palm_center_pos  # (N, 3), 이미 계산됨
right_palm_quat = self.robot.data.body_quat_w[:, self.palm_body_index]      # (N, 4)

# left hand body → left palm pos (target cup 위치 추정)
left_palm_pos = self.robot.data.body_pos_w[:, self.left_hand_body_index] - env_origins  # (N, 3)
pour_vec = left_palm_pos - right_palm_pos   # (N, 3)

actor_obs = torch.cat([
    arm_joint_pos,           # 7
    arm_joint_vel,           # 7
    finger_grasp_progress,   # 5
    left_arm_pos,            # 7
    left_arm_vel,            # 7
    right_palm_pos,          # 3
    right_palm_quat,         # 4
    left_palm_pos,           # 3
    pour_vec,                # 3
    last_palm_actions,       # 6
], dim=-1)  # 52D
```

**수정 7-2: `_set_left_demo_reference()` 내 domain randomization**
```python
# 기존 (demo 값 그대로):
self.left_arm_target_pos[env_ids_t] = target_left

# 수정 (domain random 추가):
if self.cfg.enable_left_arm_domain_random:
    noise = torch.randn_like(target_left) * self.cfg.left_arm_domain_noise_scale
    noise = noise.clamp(-self.cfg.left_arm_domain_noise_clip,
                         self.cfg.left_arm_domain_noise_clip)
    self.left_arm_target_pos[env_ids_t] = target_left + noise
else:
    self.left_arm_target_pos[env_ids_t] = target_left
```
- **MIRROR**: LEFT_ARM_DOMAIN_RANDOM_PATTERN
- **GOTCHA**: `left_arm_dof_indices`는 7D arm + 2D gripper = 9D. `[:7]`만 joint pos.
  `left_hand_body_index`는 env.__init__에서 이미 계산됨 (line 267).

### Task 8: pour_right_env_cfg.py — left arm domain random 파라미터
- **ACTION**: `pour_right_env_cfg.py` 파라미터 추가
- **IMPLEMENT**:
  ```python
  # -----------------------------------------------------------------------
  # Left arm domain randomization (Phase 2 RL)
  # target cup 위치를 에피소드마다 변화시켜 robustness 확보
  # -----------------------------------------------------------------------
  enable_left_arm_domain_random: bool = False  # Phase 1 BC: False, Phase 2 RL: True
  # joint 단위 noise (rad). 관절 1rad ≈ 수cm EEF 이동
  left_arm_domain_noise_scale: float = 0.03   # σ ≈ 0.03 rad/joint → EEF ~3cm
  left_arm_domain_noise_clip: float = 0.06    # ±0.06 rad clip
  ```
- **VALIDATE**: `python3 -c "from pour_right_env_cfg import PourRightEnvCfg; c=PourRightEnvCfg(); print(c.enable_left_arm_domain_random)"` → `False`

### Task 9: skrl PPO config YAML
- **ACTION**: `config/agents/skrl_diffusion_ppo_cfg.yaml` 신규 작성
- **IMPLEMENT**:
  ```yaml
  agent:
    class: PPO
    rollouts: 16
    learning_epochs: 8
    mini_batches: 4
    discount_factor: 0.998
    lambda_value: 0.95
    learning_rate: 2.0e-4
    learning_rate_scheduler: KLAdaptiveRL
    learning_rate_scheduler_kwargs:
      kl_threshold: 0.013
    grad_norm_clip: 1.0
    ratio_clip: 0.2
    value_clip: 0.2
    clip_predicted_values: True
    entropy_loss_scale: 0.0015
    value_loss_scale: 0.5
    # Asymmetric A2C: 별도 critic
    shared_observation_space: False
    
    # BC aux loss 설정 (DiffusionActor.bc_loss)
    bc_aux_weight: 0.5          # PPO loss에 추가할 BC 비율
    bc_freeze_after_epoch: 3000  # 이후 diffusion 고정
  
  models:
    policy:
      class: DiffusionActor
      bc_checkpoint_path: "/home/user/rl_ws/hdgp/log/diffusion_bc/v6/bc_5000.pt"
      n_ddim_steps: 5
      freeze_bc: false
    value:
      class: AsymmetricCritic  # 기존 143D 크리틱 구조 유지
  ```

---

## Testing Strategy

### Phase 1 Smoke Test (Isaac Sim 없이)
```bash
# 1. buffer 테스트
python3 -c "
from diffusion_bc.buffer import DemoBCDatasetV2
from pathlib import Path
ds = DemoBCDatasetV2([Path('/home/user/rl_ws/datasets/pour_v1_a11.hdf5')])
obs, act, mask = ds[0]
assert obs.shape == (52,), f'obs {obs.shape}'
assert act.shape == (16, 11), f'act {act.shape}'
print('Buffer OK')
"

# 2. net 테스트
python3 -c "
from diffusion_bc.net import DiffusionBCNet
import torch
net = DiffusionBCNet()
obs = torch.randn(4, 52)
act = torch.randn(4, 16, 11)
mask = torch.ones(4, 16, dtype=torch.bool)
loss = net.compute_loss(obs, act, mask)
print(f'Loss OK: {loss.item():.4f}')
sample = net.sample(obs)
assert sample.shape == (4, 11)
print('Sample OK')
"

# 3. Phase 1 단축 학습 (10 epoch)
python3 train_bc.py --epochs 10 --batch_size 16
```

### Phase 2 Validation (학습 전 구조 검증)
```bash
# obs dim 검증
python3 -c "
import sys; sys.path.insert(0, '.')
from pour_right_constants import NUM_OBSERVATIONS
assert NUM_OBSERVATIONS == 52, f'expected 52, got {NUM_OBSERVATIONS}'
print('Obs dim OK')
"

# skrl actor 구조 검증 (BC checkpoint 없이)
python3 -c "
from diffusion_actor_skrl import DiffusionActor
# mock checkpoint으로 구조만 확인
import gymnasium as gym
import torch
obs_space = gym.spaces.Box(low=-5, high=5, shape=(52,))
act_space = gym.spaces.Box(low=-1, high=1, shape=(11,))
# ... init 검증
"
```

### Phase 2 RL TFEvents 핵심 지표

| 지표 | 기대 (2000 step 기준) | 의미 |
|-----|-------------------|------|
| `Episode/log/bead_in_target` | > 0 | IL prior가 pour 위치 탐색 성공 |
| `Episode/log/cup_center_xy_dist` | 점진적 감소 | diffusion이 transport 학습 |
| `policy/std` (skrl) | 0.05~0.3 | 탐색 분산 (너무 크면 BC prior 무력화) |
| `losses/policy_loss` | 감소 추세 | PPO 정상 학습 |

---

## Validation Commands

```bash
# 1. Buffer 단위 테스트
cd /home/user/rl_ws/hdgp/source/openarm/.../5g_pour_right_v6
python3 -m pytest tests/ -x -v -k "v2 or fk or diffusion" 2>&1 | tail -20

# 2. Phase 1 smoke test
python3 train_bc.py --epochs 50

# 3. Phase 1 full train
python3 train_bc.py --epochs 5000 \
  --output_dir /home/user/rl_ws/hdgp/log/diffusion_bc/v6

# 4. Phase 2 RL (skrl, Isaac Sim 필요)
python3 /home/user/rl_ws/hdgp/scripts/reinforcement_learning/skrl/train.py \
  --task OpenArm-Pour-Right-v6 \
  --agent skrl_diffusion_ppo_cfg \
  --num_envs 128
```

---

## Acceptance Criteria
- [ ] `DemoBCDatasetV2`: obs 52D, left arm joint 포함, datagen cup 미사용
- [ ] `DiffusionBCNet.compute_loss()`: loss 정상 감소 (50 epoch < 0.5)
- [ ] `DiffusionBCNet.sample()`: shape `(B, 11)`, 값 range `[-1, 1]`
- [ ] `train_bc.py`: Isaac Sim 없이 실행 가능
- [ ] `DiffusionActor`: skrl PPO 루프에서 `act()` / `get_log_prob()` 정상 호출
- [ ] `pour_right_constants.py`: `NUM_OBSERVATIONS == 52`
- [ ] `pour_right_env.py`: actor obs 52D 반환, left arm domain random 동작
- [ ] Phase 2 RL: `bead_in_target > 0` (최소 몇 에피소드 성공)

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `eef_pose/left` key가 일부 HDF5에 없음 | Low | High | demo_pose_reference.py:385에서 확인됨; 없으면 fallback to left_arm FK 계산 |
| DDIM K=5 inference가 너무 느림 (128 envs) | Medium | Medium | K=3으로 줄이거나 float16 연산 |
| left arm noise가 너무 커서 target cup이 source cup 밑으로 | Medium | Medium | noise_clip=0.06 rad 제한으로 EEF ~3cm 이내 |
| GaussianMixin log_std가 너무 작아서 탐색 부족 | Medium | High | log_std init=-2.0 (σ≈0.13). 부족하면 -1.5로 올림 |
| Phase 1 BC가 demo 특정 자세에 overfitting | Low | Medium | stride=2 + 10개 demo + EMA로 완화 |
| diffusion_policy repo 의존성 (경로 하드코딩) | Medium | Low | `_DP_SRC` 환경변수로 override 가능하게 |

---

## Notes

### diffusion_policy 의존성 관리
```python
# diffusion_bc/net.py 상단
import os, sys
_DP_PATH = os.environ.get(
    "DIFFUSION_POLICY_SRC",
    "/home/user/rl_ws/repo/diffusion_policy"
)
if _DP_PATH not in sys.path:
    sys.path.insert(0, _DP_PATH)
```

### Phase 1 → Phase 2 전환 체크리스트
```
1. Phase 1 완료 확인:
   - log/diffusion_bc/v6/bc_5000.pt 존재
   - BC loss < 0.05 (500 epoch 이후)

2. Phase 2 시작 전:
   - pour_right_env_cfg.py: enable_left_arm_domain_random = True
   - pour_right_constants.py: NUM_OBSERVATIONS = 52 확인
   - skrl_diffusion_ppo_cfg.yaml: bc_checkpoint_path 경로 확인
   - RL reward: weight_pour_xy=0.0, weight_dist_to_target=0.0 (task only)
```

### left arm domain random 값 근거
```
noise_scale = 0.03 rad/joint (7 joints)
EEF 민감도 ≈ 0.3 m/rad (팔 길이 ~0.5m 기준)
EEF 이동 ≈ 0.03 × 0.3 × 7^0.5 = ~24mm ≈ 2.4cm
→ target cup이 약 ±2~3cm 범위에서 변화 (컵 반경 4.1cm의 약 절반)
noise_clip = 0.06 rad → 최대 ~4.8cm (컵 반경 이내)
```
