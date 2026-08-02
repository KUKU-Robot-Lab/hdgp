"""right/grasp_v1 체크포인트 → left warmstart 미러 변환 (Y-미러, XZ평면 반사).

obs(114D)·action(11D)는 canonical이 아니라 raw 물리 프레임이므로, right 정책 가중치를
차원별 미러 부호(s_o, s_a)로 변환해야 left에서 동일 행동이 재현된다.

수학: a_L = s_a ⊙ net_R(s_o ⊙ o_L)  (o_L = left obs, a_L = left action)
  ↳ running_mean *= s_o
  ↳ actor_mlp.0.weight  cols(114) *= s_o
  ↳ rnn.weight_ih_l0    cols[512:626] *= s_o   (concat_input: [mlp_out512, obs114])
  ↳ mu.weight cols[1024:1138] *= s_o, 이어서 mu.weight/bias rows *= s_a
  ↳ value.weight cols[1024:1138] *= s_o
  sigma·layer_norm·rnn.weight_hh·mlp.2 등 canonical 경로는 불변.
"""
import sys
import torch

SRC = sys.argv[1]
DST = sys.argv[2]

# --- 미러 부호 (preset grasp_left_preset.py 와 동일 유도) ---
HAND_SIGN = [
    -1, -1, -1, -1,   # thumb  (X,Z,X,X)
    -1,  1,  1,  1,   # index  (X,Y,Y,Y)
    -1,  1,  1,  1,   # middle
    -1,  1,  1,  1,   # ring
    -1, -1,  1,  1,   # pinky  (Z,X,Y,Y)
]
ARM_SIGN = [-1, -1, -1, 1, -1, -1, -1]
# action: palm[x,y,z, ez,ey,ex] + finger[5]. Y-미러: y·ez·ex 반전, x·z·ey 유지, finger 불변.
SA_PALM = [1, -1, 1, -1, 1, -1]
SA_FING = [1, 1, 1, 1, 1]
S_A = SA_PALM + SA_FING                       # 11
VEC3 = [1, -1, 1]                             # world 3-vec Y-미러

S_O = (
    ARM_SIGN + ARM_SIGN                       # arm pos7 + vel7
    + HAND_SIGN + HAND_SIGN                    # finger pos20 + vel20
    + VEC3                                     # palm_center 3
    + VEC3 * 5                                 # fingertip_rel_palm 15
    + VEC3                                     # palm_to_cup 3
    + VEC3 * 5                                 # cup_to_fingertip 15
    + [1] * 5                                  # binary_contact 5
    + S_A                                      # last_actions 11
    + [1] * 8                                  # object onehot 8
)
assert len(S_O) == 114, len(S_O)
assert len(S_A) == 11, len(S_A)

ck = torch.load(SRC, map_location="cpu", weights_only=False)
sd = ck["model"]
so = torch.tensor(S_O, dtype=torch.float32)
sa = torch.tensor(S_A, dtype=torch.float32)

# 사전 shape 검증
assert sd["running_mean_std.running_mean"].shape[0] == 114
assert sd["a2c_network.actor_mlp.0.weight"].shape[1] == 114
assert sd["a2c_network.rnn.rnn.weight_ih_l0"].shape[1] == 626
assert sd["a2c_network.mu.weight"].shape[1] == 1138
assert sd["a2c_network.mu.weight"].shape[0] == 11

sd["running_mean_std.running_mean"] *= so
sd["a2c_network.actor_mlp.0.weight"] *= so.unsqueeze(0)
sd["a2c_network.rnn.rnn.weight_ih_l0"][:, 512:626] *= so.unsqueeze(0)
sd["a2c_network.mu.weight"][:, 1024:1138] *= so.unsqueeze(0)
sd["a2c_network.mu.weight"] *= sa.unsqueeze(1)
sd["a2c_network.mu.bias"] *= sa
sd["a2c_network.value.weight"][:, 1024:1138] *= so.unsqueeze(0)

torch.save(ck, DST)
print(f"[OK] mirrored actor saved → {DST}")
print(f"  s_o flips: {(so < 0).sum().item()}/114,  s_a flips: {(sa < 0).sum().item()}/11")
