# distillation — teacher(state) → student(vision) 증류

state 기반 teacher(PPO/LSTM, 물체 pose를 privileged obs로 받음)를
**RealSense D435i RGB-D만 보는 student**로 DAgger 증류한다.

DEXTRAH `dextrah_lab/distillation/` 이식. 알고리즘 본체는
`source/openarm/openarm/distillation/dagger.py`에 있고, 이 디렉토리는 진입점만 담는다.

지원 태스크: `open-tesol_r_grasp_v2-distill`, `open-tesol_l_grasp_v2-distill`

---

## 학습: `distill.sh`

Dagger가 DDP 위에서 돌게 짜여 있어 **GPU가 하나여도 torchrun으로 띄워야 한다.**
`isaaclab.sh -p run_distillation.py`로 바로 실행하면 `WORLD_SIZE` 부재로 즉시 에러가 난다.
`distill.sh` 가 torchrun 기동·GPU 지정·포트 격리를 처리하므로 이걸 쓴다.

```bash
cd hdgp

# GPU0 — right
GPU=0 ./distill.sh open-tesol_r_grasp_v2-distill test1 \
    log/rl_games/open-tesol/right/grasp-v2/lstm_test12/nn/last_....pth

# GPU1 — left (동시 실행 가능)
GPU=1 ./distill.sh open-tesol_l_grasp_v2-distill test1 \
    log/rl_games/open-tesol/left/grasp-v2/lstm_test6/nn/last_....pth
```

`GPU=N` → `CUDA_VISIBLE_DEVICES`. GPU 여러 장을 한 잡에 묶으려면 `NPROC=2`.

### 두 잡을 동시에 띄울 때 — `--standalone` 을 쓰지 말 것

같은 호스트에서 `--standalone` 잡을 둘 이상 띄우면 rendezvous 포트가 겹쳐 포트 충돌이
나거나, **더 나쁘게는 두 잡이 하나의 잡으로 병합된다**(torch 공식 문서 경고).
right/left를 GPU0/GPU1에서 동시에 돌리면 정확히 이 상황이고, 병합되면 서로의 gradient가
섞이는데 겉보기엔 그냥 도는 것처럼 보인다.

`distill.sh` 는 `--rdzv-backend=c10d --rdzv-endpoint=localhost:0` 으로 잡마다 빈 포트를
새로 잡아 이 사고를 막는다. 직접 torchrun을 부를 거면 반드시 같은 옵션을 쓸 것.

### 인자 (`distill.sh` 뒤에 그대로 전달됨)

| 인자 | 설명 |
|---|---|
| `--task` | `…-distill` 로 등록된 태스크 id (필수) |
| `--teacher` | teacher 체크포인트 `.pth` (필수) |
| `--student` | student 체크포인트 — 중단된 학습 재개용 |
| `--label` | 로그 하위 폴더명 (기본 `distill`) |
| `--num_envs` | 환경 수 (기본 256) |
| `--play_policy` | 학습 없이 student rollout만 (평가) |
| `--seed`, `--headless`, `--device` | IsaacLab 공통 |

`--task`가 `-distill`이 아니면(=`env_cfg.distillation=False`) 카메라도 student obs도
생성되지 않으므로 스크립트가 먼저 막는다.

### 중단된 학습 재개

```bash
GPU=0 ./distill.sh open-tesol_r_grasp_v2-distill test1 <teacher.pth> \
    --student log/distillation/open-tesol_r_grasp_v2-distill/test1/nn/grasp_student_20000_iters.pth
```

### 학습된 student 재생

```bash
GPU=0 ./distill.sh open-tesol_r_grasp_v2-distill test1 <teacher.pth> \
    --student log/distillation/.../nn/grasp_student_20000_iters.pth \
    --play_policy --num_envs 16
```

teacher 체크포인트는 `--play_policy` 에서도 필수다 — Dagger가 teacher 모델을 항상 빌드한다.

---

## DEXTRAH 원본 커맨드와 뭐가 다른가

DEXTRAH는 hydra로 `env.*` 를 잔뜩 오버라이드한다.

```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=N run_distillation.py \
  --headless --distributed --task=Dextrah-Kuka-Allegro \
  --num_envs 256 env.distillation=True \
  --enable_cameras env.simulate_stereo=True \
  --teacher <ckpt> env.img_aug_type="rgb" env.aux_coeff=10. ...
```

hdgp는 이렇게 바꿨다.

| DEXTRAH | hdgp | 이유 |
|---|---|---|
| `env.distillation=True`, `--num_envs 256`, `env.aux_coeff=10.` 을 CLI로 | `…-distill` gym id 의 `*EnvCfg_DISTILL` 이 내장 | 매번 손으로 넘기면 하나 빠뜨렸을 때 조용히 다른 실험이 된다 |
| `--enable_cameras`, `--distributed` 를 CLI로 | `run_distillation.py` 가 강제 | **빼먹으면 TiledCamera가 렌더되지 않고, student가 빈 이미지를 본 채 "정상적으로" 학습된다** — 실패가 드러나지 않는 종류라 옵션으로 두면 안 된다 |
| `env.simulate_stereo=True` | mono | D435i는 단일 시점 RGB-D라 stereo 경로가 없다 |
| `env.img_aug_type="rgb"` + 시각 DR | **동일하게 유지** | 아래 참조 |
| torchrun 직접 호출 | `distill.sh` | GPU 지정 + 잡별 포트 격리 |
| reward/ADR 오버라이드 (`max_pose_angle=45`, `lift_weight=[5→0]` …) | grasp_v2 cfg 기본값 | 이미 teacher 학습에서 같은 레시피를 쓰고 있다 |

### student는 RGB를 본다 — 그래서 textures.zip이 필요하다

DEXTRAH의 학생망은 **전부 `use_depth = False`** 로 하드코딩되어 있다
(`a2c_mono_transformer:454`, `a2c_stereo_transformer:317` …). 즉 인코더 입력은 depth가
아니라 **RGB 3채널**이다. depth는 `stereo_recon` 변종에서 aux 재구성 타깃으로 쓰일 뿐이다.

따라서 외형이 고정되면 student는 단 하나의 장면에만 맞는 정책이 된다. DEXTRAH가
`textures.zip`을 요구하는 이유가 이것이고, 우리도 같은 이유로 필요하다.

`textures.zip`은 4가지를 담고 있고, **앞의 3개는 증강이 아니라 env의 시각 DR** 이다
(`cfg.distillation=True`면 항상 돈다).

| 에셋 | 용도 | 코드 |
|---|---|---|
| `curated_table_textures/*.png` | 테이블 재질 | `visual_dr.py` |
| `dome_light_textures/*.exr` | 조명 HDRI | `visual_dr.py` |
| `object_textures/**/*.png` | 물체 텍스처 | `visual_dr.py` |
| `background_imgs/voc_resized/` | RGB 배경 교체 증강 | `rgb_augs.py` |

### 설치 (8.4GB — git 비추적, server에도 따로 받아야 한다)

```bash
cd hdgp
curl -L -o /tmp/textures.zip \
  https://huggingface.co/datasets/nvidia/dextrah_textures/resolve/main/textures.zip
unzip /tmp/textures.zip -d assets/dextrah_textures/
```

`assets/dextrah_textures/` 는 `.gitignore` 대상이다. 없으면 `TextureBank` 가 기동 시점에
에러를 던진다 — 조용히 no-op으로 넘어가면 학습은 멀쩡히 도는데 sim2real에서만 무너지기 때문이다.

`img_aug_type="depth"` 로 바꾸면 시각 DR과 RGB 증강이 꺼지고 depth 증강만 남는다
(구 `a2c_with_aux_depth` 처럼 depth를 직접 입력받는 네트워크를 쓸 때만 의미 있다).

---

## 로그

```
log/distillation/<task>/<label>/
├── nn/          grasp_student_<N>_iters.pth   (5,000 iter 마다)
└── summaries/   TFEvents
```

핵심 스칼라: `imitation_loss`(student가 teacher를 얼마나 따라잡았나),
`in_success_region`(student rollout의 실제 태스크 성공률), `aux_loss_object_pos`
(이미지에서 물체 위치를 얼마나 맞히나 — 인코더가 학습되고 있는지의 선행지표).

```bash
python3 scripts/tools/parse_tfevents.py log/distillation/<task>/<label>/summaries
```

---

## 반드시 먼저 할 일: 카메라 캘리브레이션

`grasp_right_preset.py`의 `CAMERA_POS` / `CAMERA_ROT`는 **PLACEHOLDER**다.
실물 D435i를 마운트한 적이 없어서, 작업공간을 내려다보는 look-at 값을 계산해 넣어둔 것뿐이다.

**hand-eye 캘리브레이션 값으로 교체하지 않으면 student가 배운 시점과 실기 시점이 어긋나
sim2real이 통째로 무너진다.** 증류를 실제로 돌리기 전에 교체할 것.

intrinsics(`CAMERA_*`)는 D435i depth 실측 사양(1280×720, HFOV 87°/VFOV 58°)에서 유도했으므로
그대로 두면 된다. 해상도를 바꾼다면 16:9를 유지해야 FOV가 맞는다.

---

## env 수

teacher는 4096 env로 돌지만 증류는 **256이 기본**이다. env마다 320×180 RGB-D 렌더 타깃이
붙어 teacher 규모를 GPU가 감당하지 못한다. 메모리가 남으면 `--num_envs`로 올린다.

---

## 테스트

```bash
python3 -m pytest source/openarm/openarm/distillation/tests -q
python3 -m pytest source/openarm/openarm/tesollo/right/grasp_v2/tests -q
```

후자에는 teacher obs(193/247)가 증류 이식으로 변하지 않았음을 고정하는 회귀 테스트가 있다.
이게 깨지면 기존 teacher 체크포인트가 죽는다.

---

## 다른 태스크에 이식하려면

`dagger.py` / `a2c_mono_transformer` / `mono_encoder` / `depth_augs`는 태스크를 모르므로
그대로 공유된다. 태스크마다 필요한 것:

1. env에 `distillation` 분기 + `compute_student_policy_observations()`
2. TiledCamera cfg (카메라 상수는 로봇별 preset — tesollo / rh56f1 별도)
3. student yaml + `…-distill` gym id 등록

**하드 블로커 하나:** Dagger는 로깅에서 `ov_env.in_success_region`을 읽는다. grasp 고유
지표라 pour 계열에는 없어서(=`success_flag` 사용) 그대로 붙이면 `AttributeError`로 죽는다.
태스크별 성공지표 훅으로 일반화가 먼저 필요하다.

pour 증류는 관측 51D 중 12D(`pour_point_to_opening`, 각 축들)가 두 컵의 pose 파생이라
추론 부담이 grasp보다 훨씬 크다. 게다가 teleop 스택에 FoundationPose가 이미 컵 pose를 뽑고
있어, 증류 없이 그 pose를 obs에 꽂는 선택지가 있다. 착수 전 명분을 정할 것.
