# Phase 2 — radial-압축 fragile damage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development 또는 superpowers:executing-plans. Steps use `- [ ]` checkboxes.
>
> **⚠️ RL 실험 루프(`hdgp/CLAUDE.md`).** 최종 검증은 GPU 학습+TFEvents. 정적 pytest는 "코드가 안 깨짐"만 보장. reward/gate 변경은 **reward-audit 통과** 필수. 학습은 사용자가 server GPU0에서 실행.

**Goal:** grasp_adapt에 radial 압축 좌굴 virtual damage를 도입해, 감싸기(radial↑)를 물리적으로 벌하고 **손끝-only 정밀 파지가 reward 구조에서 자연히 나오게** 한다. Phase 1의 geometry envelope penalty는 제거한다.

**Architecture:** rigid cup 유지(실제 변형 없음). 접촉력(tip world force + middle net force)의 컵 중심축 inward 성분 합 = `radial_compression`. `r_damage = -w·hold·relu(radial − F_safe)` 순간 penalty + `radial > F_buckle` 파손 종료. spec: `docs/superpowers/specs/2026-07-27-grasp-adapt-phase2-fragile-radial-damage-design.md`.

**Tech Stack:** Isaac Lab DirectRLEnv, PyTorch, rl_games, pytest(정적), TFEvents(`parse_tfevents.py`).

## Global Constraints

- **손끝-only (하드웨어 제약, 전 Phase 공통):** Tesollo는 손끝에만 6축 F/T. 감싸기(중간마디 접촉)는 tactile로 못 읽어 무효. Phase 2는 이를 radial damage 물리로 유도한다.
- **대상 폴더만 수정:** `hdgp/source/openarm/openarm/tesollo/right/grasp_adapt/`. 공유 코어(`openarm/common/*`)는 건드리지 않는다(Phase 2는 grasp_adapt 국소).
- **차원 불변:** obs/action 차원 변경 없음(reward/gate/done만). 단 성공·종료 조건 변경 → fresh 재학습.
- **손가락 인덱스:** tip/middle/distal 텐서 index 0=엄지(`r_hl_thumb_*`), 1~4=검지/중지/약지/소지.
- **접촉 임계:** `CONTACT_FORCE_THRESHOLD=0.1N`. radial 계산은 접촉 binary로 마스킹(자기접촉/노이즈 배제).
- **로그 먼저:** F_safe/F_buckle 초기값은 학습 초기 `task/radial_compression` 분포를 보고 보정. 수치 근거 없이 재튜닝 금지.
- **학습 실행:** server `oem@10.102.101.240` GPU0, `source ~/miniforge3/etc/profile.d/conda.sh && conda activate proj-hdgp-py311 && source ~/rl_ws/IsaacLab/_isaac_sim/setup_conda_env.sh && CUDA_VISIBLE_DEVICES=0 setsid bash -c './train.sh <task> <label> --num_envs 2048 >log 2>&1' </dev/null &`.

## File Structure

- `grasp_right_utils.py`: 신규 순수 함수 `compute_radial_compression` (테스트 가능, torch만 의존).
- `grasp_right_env_cfg.py`: 신규 cfg 필드(`damage_penalty_weight`, `f_safe`, `f_buckle`, `buckle_penalty`), `envelope_penalty_weight=0`.
- `grasp_right_env.py`: `_get_rewards`(radial 계산 배선 + r_damage + envelope 제거 + 로깅), `_get_dones`(buckle 종료), groups 로깅 매핑.
- `tests/test_radial_compression.py`: 순수 함수 계약(감싸기→radial↑, 손끝→radial↓, inward만).

---

### Task 1: radial 압축 순수 함수 + 계약 테스트

**Files:**
- Modify: `.../grasp_adapt/grasp_right_utils.py`
- Test: `.../grasp_adapt/tests/test_radial_compression.py`

**Interfaces:**
- Produces: `compute_radial_compression(contact_pos, contact_force, cup_center, cup_axis, contact_mask) -> torch.Tensor`. 입력 `contact_pos`/`contact_force` `(N,K,3)` world, `cup_center` `(N,3)`, `cup_axis` `(N,3)` 정규화 컵 up축, `contact_mask` `(N,K)` {0,1}. 출력 `(N,)` radial inward 성분 합(≥0).

- [ ] **Step 1: 실패 테스트 작성**

Create `.../grasp_adapt/tests/test_radial_compression.py` (grasp_v1 관행: 파일 직접 로드):
```python
"""radial 압축 순수 함수 계약: 감싸기→radial↑, 손끝(축방향)→radial↓, inward만 양수."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "grasp_right_utils.py"
SPEC = importlib.util.spec_from_file_location("grasp_adapt_utils_radial", MODULE_PATH)
_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = _mod
SPEC.loader.exec_module(_mod)

compute_radial_compression = _mod.compute_radial_compression

# 컵: 중심 원점, up축 +z, 반경 0.045
CENTER = torch.tensor([[0.0, 0.0, 0.0]])
AXIS = torch.tensor([[0.0, 0.0, 1.0]])
R = 0.045


def test_envelope_high_radial():
    # 4접촉이 컵 옆면 사방(+x,-x,+y,-y)에서 안으로 미는 감싸기 → radial 큼
    pos = torch.tensor([[[R, 0, 0], [-R, 0, 0], [0, R, 0], [0, -R, 0]]])
    force = torch.tensor([[[-5.0, 0, 0], [5.0, 0, 0], [0, -5.0, 0], [0, 5.0, 0]]])  # 모두 inward
    mask = torch.ones(1, 4)
    out = compute_radial_compression(pos, force, CENTER, AXIS, mask)
    assert float(out[0]) > 15.0  # 4×5N inward ≈ 20


def test_fingertip_axial_low_radial():
    # 손끝이 컵 위 테두리를 축방향(-z)으로 누름 → radial 성분 거의 0
    pos = torch.tensor([[[R, 0, 0.08], [-R, 0, 0.08]]])
    force = torch.tensor([[[0, 0, -5.0], [0, 0, -5.0]]])  # 축방향, radial 0
    mask = torch.ones(1, 2)
    out = compute_radial_compression(pos, force, CENTER, AXIS, mask)
    assert float(out[0]) < 1.0


def test_outward_force_not_counted():
    # 바깥으로 미는(당기는) 힘은 inward relu로 0
    pos = torch.tensor([[[R, 0, 0]]])
    force = torch.tensor([[[5.0, 0, 0]]])  # outward
    mask = torch.ones(1, 1)
    out = compute_radial_compression(pos, force, CENTER, AXIS, mask)
    assert float(out[0]) < 1e-5


def test_mask_excludes_contact():
    pos = torch.tensor([[[R, 0, 0]]])
    force = torch.tensor([[[-5.0, 0, 0]]])  # inward
    mask = torch.zeros(1, 1)  # 마스크로 배제
    out = compute_radial_compression(pos, force, CENTER, AXIS, mask)
    assert float(out[0]) < 1e-5
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `cd /home/user/rl_ws/hdgp && python3 -m pytest source/openarm/openarm/tesollo/right/grasp_adapt/tests/test_radial_compression.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'compute_radial_compression'`.

- [ ] **Step 3: 순수 함수 구현**

`grasp_right_utils.py`에 추가:
```python
def compute_radial_compression(
    contact_pos: torch.Tensor,
    contact_force: torch.Tensor,
    cup_center: torch.Tensor,
    cup_axis: torch.Tensor,
    contact_mask: torch.Tensor,
) -> torch.Tensor:
    """접촉력의 컵 중심축 방향 inward(radial 압축) 성분 합.

    감싸기(사방 마디가 벽을 안으로 압박)는 radial↑, 손끝 국부/축방향 파지는 radial↓.
    종이컵 좌굴은 radial 압축이 임계를 넘을 때 발생.

    Args:
        contact_pos: (N,K,3) 접촉점 world 좌표.
        contact_force: (N,K,3) 접촉력 world 벡터.
        cup_center: (N,3) 컵 중심 world.
        cup_axis: (N,3) 컵 up축(정규화 가정).
        contact_mask: (N,K) 유효 접촉 {0,1}.
    Returns:
        (N,) radial inward 성분 합(≥0).
    """
    rel = contact_pos - cup_center.unsqueeze(1)                      # (N,K,3)
    axis = cup_axis.unsqueeze(1)                                     # (N,1,3)
    axial = (rel * axis).sum(dim=-1, keepdim=True) * axis            # 축방향 성분
    radial_vec = rel - axial                                        # 축에 수직(바깥 방향)
    radial_out = radial_vec / radial_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    # 힘의 inward 성분(바깥 방향의 반대)만 양수로
    inward = (-(contact_force * radial_out).sum(dim=-1)).clamp(min=0.0)  # (N,K)
    return (inward * contact_mask).sum(dim=-1)                       # (N,)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/user/rl_ws/hdgp && python3 -m pytest source/openarm/openarm/tesollo/right/grasp_adapt/tests/test_radial_compression.py -q`
Expected: 4 passed.

- [ ] **Step 5: 커밋**

```bash
cd /home/user/rl_ws/hdgp
git add source/openarm/openarm/tesollo/right/grasp_adapt/grasp_right_utils.py source/openarm/openarm/tesollo/right/grasp_adapt/tests/test_radial_compression.py
git commit -m "feat(grasp_adapt): Phase 2 radial 압축 순수 함수 + 계약 테스트"
```

---

### Task 2: cfg 신규 필드 (damage 파라미터, envelope 제거)

**Files:**
- Modify: `.../grasp_adapt/grasp_right_env_cfg.py`

**Interfaces:**
- Produces: cfg 필드 `damage_penalty_weight: float`, `f_safe: float`, `f_buckle: float`, `buckle_penalty: float`, `envelope_penalty_weight: float = 0.0`.

- [ ] **Step 1: cfg 필드 추가/수정**

`grasp_right_env_cfg.py`의 `envelope_penalty_weight` 정의를 찾아 0.0으로 바꾸고 damage 필드를 그 아래 추가:
```python
    # Phase 1 envelope penalty 제거(radial damage로 일원화). 필드는 존치하되 0.
    envelope_penalty_weight: float = 0.0
    # Phase 2 radial-압축 fragile damage (하드웨어 제약: 손끝-only를 물리로 유도)
    #   radial_compression = 접촉력의 컵 중심 inward 성분 합.
    #   r_damage = -damage_penalty_weight · hold_gate · relu(radial - f_safe)
    #   radial > f_buckle → 파손 종료(buckle) + buckle_penalty.
    # f_safe/f_buckle은 종이컵 추정 placeholder(설계 §4: f_safe≈0.6~0.8·F_yield).
    # ★ 학습 초기 task/radial_compression 분포를 보고 보정할 것(로그 먼저).
    damage_penalty_weight: float = 3.0
    f_safe:   float = 8.0    # N, 안전 radial 압축 상한(초과분 penalty)
    f_buckle: float = 15.0   # N, 좌굴(파손) radial 압축 임계
    buckle_penalty: float = 10.0   # 파손 종료 시 음의 보상 크기
```
(위치: 기존 `envelope_penalty_weight` 라인을 대체. 없으면 `post_lift_contact_loss_weight` 아래.)

- [ ] **Step 2: AST 검증**

Run: `cd /home/user/rl_ws/hdgp && python3 -c "import ast; ast.parse(open('source/openarm/openarm/tesollo/right/grasp_adapt/grasp_right_env_cfg.py').read()); print('AST OK')"`
Expected: `AST OK`.

- [ ] **Step 3: 커밋**

```bash
git add source/openarm/openarm/tesollo/right/grasp_adapt/grasp_right_env_cfg.py
git commit -m "feat(grasp_adapt): Phase 2 cfg — damage 파라미터 추가, envelope_penalty 0"
```

---

### Task 3: env — radial 계산 배선 + r_damage + envelope 제거 + 로깅

**Files:**
- Modify: `.../grasp_adapt/grasp_right_env.py`

**Interfaces:**
- Consumes: Task 1 `compute_radial_compression`, Task 2 cfg 필드. 기존 버퍼 `contact_force_xyz_raw`(N,5,3 tip world), `middle_sensor`/`fingertip_body_indices`/`middle3_body_indices`, `middle_binary_contact_buf`, `binary_contact_buf`, `object_pos`, `object_rot`, `hold_gate`.
- Produces: `self._radial_compression_buf` `(N,)`, reward term `r_damage`, 로깅 태그.

- [ ] **Step 1: import에 compute_radial_compression 추가**

`from .grasp_right_utils import compute_precision_grasp_mask, to_torch` 를
`from .grasp_right_utils import compute_precision_grasp_mask, compute_radial_compression, to_torch` 로 수정.

- [ ] **Step 2: radial 계산 헬퍼 추가 (`_get_rewards` 내, hold_gate 계산 이후)**

`hold_gate = adaptive_terms["hold_gate"]` 다음에, 기존 envelope penalty 블록(`middle_contact_frac`/`distal_contact_frac`/`r_envelope`)을 **radial 계산 + r_damage로 교체**:
```python
        # 손끝-only 자연 유도(하드웨어 제약): radial 압축 좌굴 fragile damage.
        # 감싸기(사방 마디가 벽 안으로 압박)=radial↑=파손, 손끝 국부 파지=radial↓.
        env_origins = self.scene.env_origins
        # 컵 up축(world)
        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_axis = quat_apply(self.object_rot, z_local)              # (N,3)
        cup_center = self.object_pos                                 # (N,3) env-local
        # tip: world force + world pos(env-local) + Cup 접촉 마스크
        tip_pos = self.fingertip_pos                                 # (N,5,3) env-local
        tip_force = self.contact_force_xyz_raw                       # (N,5,3) world
        tip_mask = self.binary_contact_buf.float()                  # (N,5)
        # middle: net force 벡터 + body pos + 접촉 마스크(자기접촉 배제)
        middle_pos = (
            self.robot.data.body_pos_w[:, self.middle3_body_indices, :] - env_origins.unsqueeze(1)
        )                                                            # (N,5,3)
        middle_force = self._middle_sensor.data.net_forces_w        # (N,5,3) world
        middle_force = torch.nan_to_num(middle_force, nan=0.0, posinf=0.0, neginf=0.0)
        middle_mask = self.middle_binary_contact_buf.float()        # (N,5)
        contact_pos = torch.cat([tip_pos, middle_pos], dim=1)        # (N,10,3)
        contact_force = torch.cat([tip_force, middle_force], dim=1)  # (N,10,3)
        contact_mask = torch.cat([tip_mask, middle_mask], dim=1)     # (N,10)
        radial_compression = compute_radial_compression(
            contact_pos, contact_force, cup_center, cup_axis, contact_mask
        )                                                            # (N,)
        self._radial_compression_buf.copy_(radial_compression)
        # 순간 penalty: hold 구간 F_safe 초과분
        r_damage = (
            -float(self.cfg.damage_penalty_weight)
            * hold_gate
            * torch.relu(radial_compression - float(self.cfg.f_safe))
        )
        # 진단(로깅용): middle/distal 접촉률 유지
        middle_contact_frac = middle_mask.sum(dim=-1) / float(NUM_MIDDLE_SENSORS)
        distal_contact_frac = (
            self.distal_binary_contact_buf.float().sum(dim=-1) / float(NUM_DISTAL_SENSORS)
        )
```
**주의:** 기존 `r_envelope`/`middle_contact_frac`/`distal_contact_frac` 정의를 위 블록으로 대체(중복 정의 금지). `quat_apply`는 이미 env에서 import됨(확인: `grep quat_apply` — `_get_dones`에서 사용 중).

- [ ] **Step 3: `_radial_compression_buf` 버퍼 초기화 (`__init__`/버퍼 선언부)**

`self.middle_binary_contact_buf = ...` 선언 근처에 추가:
```python
        self._radial_compression_buf = torch.zeros(self.num_envs, device=self.device)
```

- [ ] **Step 4: total에 r_damage 반영 (r_envelope 대체)**

기존 `total = ... + r_objective + r_envelope,` 를
`total = ... + r_objective + r_damage,` 로 수정.

- [ ] **Step 5: 로깅 교체 (reward/envelope → reward/damage + radial)**

기존 로깅 블록에서:
```python
        self.extras["reward/envelope"] = r_envelope.mean()
```
를
```python
        self.extras["reward/damage"] = r_damage.mean()
        self.extras["task/radial_compression"] = radial_compression.mean()
        self.extras["task/radial_compression_hold"] = (
            radial_compression * hold_gate
        ).sum() / hold_gate.sum().clamp(min=1.0)
```
로 교체. `task/middle_contact_rate`·`task/distal_contact_rate` 로깅은 유지.

- [ ] **Step 6: groups 매핑 갱신**

groups 딕셔너리에서 `"reward/envelope": (...)` 를
```python
            "reward/damage": ("reward/summary", "damage"),
            "task/radial_compression": ("task/damage", "radial"),
            "task/radial_compression_hold": ("task/damage", "radial_hold"),
```
로 교체.

- [ ] **Step 7: 정적 검증**

Run:
```bash
cd /home/user/rl_ws/hdgp
python3 -c "import ast; ast.parse(open('source/openarm/openarm/tesollo/right/grasp_adapt/grasp_right_env.py').read()); print('AST OK')"
grep -n "r_envelope" source/openarm/openarm/tesollo/right/grasp_adapt/grasp_right_env.py || echo "r_envelope orphan 없음 ✓"
python3 -m pytest source/openarm/openarm/tesollo/right/grasp_adapt/tests/ -q
```
Expected: AST OK, r_envelope 잔존 없음, 모든 테스트 pass.

- [ ] **Step 8: 커밋**

```bash
git add source/openarm/openarm/tesollo/right/grasp_adapt/grasp_right_env.py
git commit -m "feat(grasp_adapt): Phase 2 radial damage 배선 — r_damage penalty + envelope 제거 + 로깅"
```

---

### Task 4: _get_dones — buckle 파손 종료

**Files:**
- Modify: `.../grasp_adapt/grasp_right_env.py`

**Interfaces:**
- Consumes: `self._radial_compression_buf`(Task 3), cfg `f_buckle`.
- Produces: `buckle` 종료 조건, `task/buckle_rate` 로깅.

- [ ] **Step 1: buckle 종료 조건 추가 (`_get_dones`)**

`terminated = out_x | out_y | fallen | tipped | final_success_held` 를:
```python
        buckle = self._radial_compression_buf > float(self.cfg.f_buckle)
        self.extras["task/buckle_rate"] = buckle.float().mean()
        terminated = out_x | out_y | fallen | tipped | final_success_held | buckle
```
로 수정. (`_radial_compression_buf`는 `_get_rewards`에서 갱신되고 Isaac Lab은 rewards→dones 순이므로 최신값 — 순서 확인: 기존 `num_contacts_buf`도 동일 패턴.)

- [ ] **Step 2: buckle_penalty 반영 (파손 시 음의 보상)**

`_get_rewards`의 total 조립 직후(또는 r_damage 블록에)에 파손 종료 보상 추가:
```python
        # 파손(좌굴) 순간 음의 보상: radial이 f_buckle 초과 시
        buckle_now = radial_compression > float(self.cfg.f_buckle)
        total = total - float(self.cfg.buckle_penalty) * buckle_now.float()
```
(total nan_to_num 이후, return 전.)

- [ ] **Step 3: groups 매핑에 buckle_rate 추가**

```python
            "task/buckle_rate": ("task/damage", "buckle_rate"),
```

- [ ] **Step 4: 정적 검증**

Run: `cd /home/user/rl_ws/hdgp && python3 -c "import ast; ast.parse(open('source/openarm/openarm/tesollo/right/grasp_adapt/grasp_right_env.py').read()); print('AST OK')" && python3 -m pytest source/openarm/openarm/tesollo/right/grasp_adapt/tests/ -q`
Expected: AST OK, 테스트 pass.

- [ ] **Step 5: 커밋**

```bash
git add source/openarm/openarm/tesollo/right/grasp_adapt/grasp_right_env.py
git commit -m "feat(grasp_adapt): Phase 2 buckle 파손 종료 + buckle_penalty"
```

---

### Task 5: reward-audit + 학습 스냅샷 + GPU 학습

- [ ] **Step 1: reward-audit 통과**

신규 `r_damage`(radial penalty) + `buckle` 종료 + envelope 제거에 대해 reward-audit 스킬 실행. 예상 지표: radial_compression 하락, middle_contact_rate 하락(손끝-only), success 유지(손끝 파지 가능 시), buckle_rate 낮음. Check 3(secure 충돌)은 radial이 secure와 직교(감싸기만 벌하고 손끝 secure는 온전)라 Phase 1과 다름 — 이 논거를 기록. ACCEPT 후 진행.

- [ ] **Step 2: 전체 정적 검증 + push**

```bash
cd /home/user/rl_ws/hdgp
python3 -m pytest source/openarm/openarm/tesollo/right/grasp_adapt/tests/ -q
python3 -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('source/openarm/openarm/tesollo/right/grasp_adapt/**/*.py', recursive=True)]; print('AST OK')"
git push origin pour
```

- [ ] **Step 3: server 재학습 (GPU0, label=damage_test1)**

```bash
# server pull 후 (Global Constraints의 실행 커맨드)
CUDA_VISIBLE_DEVICES=0 NOTE="Phase2 radial damage (f_safe=8 f_buckle=15 w=3, envelope 제거)" \
  ./train.sh open-tesol_r_grasp_adapt-lstm damage_test1 --num_envs 2048 --headless
```

### Phase 2 검증 게이트 (TFEvents 근거) — Phase 3 진입 exit 기준
`parse_tfevents.py`로 확인, `analysis.md` 누적:
- **손끝-only 확립:** `task/middle_contact_rate` 낮게 수렴 + `task/radial_compression_hold`가 `f_safe`(8) 아래로 수렴.
- **파지·리프트 성립:** `task/success_rate` >0.5 + `cup/height_delta` 10cm + `task/buckle_rate` 낮음(<5%).
- `play.py` 육안: 손끝으로 집고 컵 안 찌그러뜨림.
- **실패 대응(로그 먼저):**
  - radial 안 내려감 → `damage_penalty_weight`↑(3→) 또는 `f_safe`↓.
  - radial 초기 분포가 f_safe/f_buckle과 안 맞음(전부 초과 or 전혀 미달) → 물성 placeholder 재보정.
  - buckle_rate 과다(학습 정체) → `f_buckle`↑ 또는 파손 종료 커리큘럼 도입(hold 진입 후만 판정).
  - success 붕괴(손끝만으로 물리적 파지 불가) → 컵 무게·마찰·크기 재검토(물성/물체 선행 조정).

---

## Self-Review (spec 대비 커버리지)

- **spec §1 radial 계산(tip+middle inward):** Task 1(순수 함수) + Task 3(배선). distal은 진단 로깅(Task 3 middle/distal_contact_rate 유지).
- **spec §2 reward/종료:** r_damage penalty(Task 3), buckle 종료+penalty(Task 4), envelope 제거(Task 2/3).
- **spec §3 물성 placeholder:** Task 2 cfg(f_safe=8·f_buckle=15, "로그 보고 보정" 명시). 파손 종료 커리큘럼은 검증 게이트 실패 대응에 배치(초기 단순화, YAGNI).
- **spec §4 로깅/검증:** Task 3/4 로깅 + Phase 2 검증 게이트.
- **spec 리스크:** radial 구분(Task 1 테스트), 손끝 파지 가능성(검증 게이트 실패 대응), middle net force 오염(Task 3 middle_mask 마스킹).
- **Placeholder 스캔:** f_safe/f_buckle은 의도적 물성 placeholder(spec 명시). 그 외 실제 코드/명령/기대출력 포함.
- **Type 일관성:** `compute_radial_compression` 시그니처가 Task 1 정의와 Task 3 호출 일치. `_radial_compression_buf` Task 3 선언→Task 4 소비 일치.
