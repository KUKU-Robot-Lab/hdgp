# [2026-09-02 18:00] Pinocchio + Meshcat + Fabrics 기구학 스튜디오 구축 및 Isaac Sim 정합 가이드

> **문서 목적**: 무거운 시뮬레이터 없이 1초 만에 윈도우(Windows) 및 리눅스 환경에서 OpenArm-Tesollo 로봇의 기구학(FK), 3D 좌표계(TF), 충돌체, Geometric Fabrics 궤적을 시각화/검증하고 Isaac Lab(USD)과 100% 정합성을 유지하며 확장할 수 있는 표준 가이드입니다.

---

## 1. ⚙️ 가상환경 구축 및 패키지 설치 절차

Windows 환경에서 C++ 바이너리 충돌 없이 100% 완벽하게 설치하는 표준 절차입니다.

### Step 1-1. 전용 Python 3.10 Conda 가상환경 생성
```bash
conda create -n kuku_kinematics python=3.10 -y
conda activate kuku_kinematics
```

### Step 1-2. C++ Pinocchio, Meshcat, CasADi 바이너리 설치 (Conda-Forge)
```bash
# 초고속 설치 (C++ Micromamba 엔진 사용 권장)
curl.exe -Ls https://micro.mamba.pm/api/micromamba/win-64/latest -o %TEMP%\micromamba.tar.bz2
tar.exe -xf %TEMP%\micromamba.tar.bz2 -C %USERPROFILE%\anaconda3 Library/bin/micromamba.exe

%USERPROFILE%\anaconda3\Library\bin\micromamba.exe install -p %USERPROFILE%\anaconda3\envs\kuku_kinematics -c conda-forge pinocchio meshcat-python example-robot-data casadi trimesh scipy pyyaml -y
```

### Step 1-3. Geometric Fabrics 및 Pixar OpenUSD 설치 (Pip)
```bash
conda activate kuku_kinematics
pip install fabrics usd-core
```

### Step 1-4. 1초 종합 환경 검증
```bash
python verify_kuku_env.py
```
*결과 판정*: 모든 항목이 `[PASS]`로 출력되고 브라우저에 3D 로봇 및 초록색 타겟 컵이 뜨면 성공입니다.

---

## 2. 🗂️ 핵심 구성 스크립트 및 도구 목록

| 스크립트 파일 | 실행 배치 파일 | 주요 기능 및 역할 |
|---|---|---|
| [`scripts/kinematics/interactive_studio.py`](file:///c:/Users/User/RL/KUKU_hdgp/scripts/kinematics/interactive_studio.py) | [`run_studio.bat`](file:///c:/Users/User/RL/KUKU_hdgp/run_studio.bat) | **Isaac Lab 스타일 통합 웹 스튜디오**<br>• 27개 관절 실시간 슬라이더<br>• Palm EE 중심점 & 파지 정렬 점수 계측<br>• 반투명 충돌 구체(`_sphere`) 3D 렌더링 |
| [`scripts/kinematics/inspect_urdf.py`](file:///c:/Users/User/RL/KUKU_hdgp/scripts/kinematics/inspect_urdf.py) | [`run_inspect_openarm.bat`](file:///c:/Users/User/RL/KUKU_hdgp/run_inspect_openarm.bat) | **순수 Zero-Touch URDF 인스펙터**<br>• URDF 원본 XML 그대로 파싱<br>• 모든 링크/관절의 원본 TF 좌표축 표시 |
| [`scripts/kinematics/inspect_usd.py`](file:///c:/Users/User/RL/KUKU_hdgp/scripts/kinematics/inspect_usd.py) | [`run_inspect_usd.bat`](file:///c:/Users/User/RL/KUKU_hdgp/run_inspect_usd.bat) | **Pixar OpenUSD Ground-Truth 검증기**<br>• Isaac Lab 실제 학습용 `.usd` 파일의 231개 Prim 계층 구조 및 58개 Physics 관절 검증 |
| [`verify_kuku_env.py`](file:///c:/Users/User/RL/KUKU_hdgp/verify_kuku_env.py) | - | 전체 Python/Pinocchio/Meshcat/Fabrics 라이브러리 일괄 자가 진단 |

---

## 3. 🎯 Isaac Sim $\leftrightarrow$ Pinocchio/Fabrics 기구학 및 좌표계 정합 규격

로봇 학습(Isaac Lab)과 기구학 뷰어(Pinocchio) 간의 오차를 0%로 보장하는 Ground-Truth 좌표계 정의입니다.

### 3-1. 손바닥(Palm EE) 기준 3D 좌표계 정의
```text
  🔴 +X축 (Red Axis)   : 엄지손가락 (Thumb) 방향
  🟢 +Y축 (Green Axis) : 손바닥 피부 정면 (장풍 방향 / Palm Skin Normal)
  🔵 +Z축 (Blue Axis)  : 손목 (Wrist) 방향 (손등 상공)
  ⬇️ -Z축             : 4손가락이 아래로 뻗은 방향 (Finger Extension)
```

### 3-2. 가상 엔드이펙터 (`palm_ee`) 오프셋 정의
* **물리적 마운트 링크**: `palm_link` (손목 결합부)
* **진짜 손바닥 살 표면 중심 (`palm_ee`)**:
  $$\mathbf{p}_{\text{palm\_ee}} = \mathbf{p}_{\text{palm\_link}} + \mathbf{R}_{\text{palm\_link}} \begin{bmatrix} +0.028 \\ 0.000 \\ +0.040 \end{bmatrix} \text{ (단위: m)}$$
* **파지 정렬 지표 (Alignment Dot Product)**:
  $$\text{Score} = \vec{n}_{\text{palm}} \cdot \vec{v}_{\text{to\_cup}} \quad (\text{+1.0일 때 컵을 정면으로 완벽히 대면})$$

### 3-3. Isaac Lab USD $\leftrightarrow$ Fabrics URDF 1:1 매핑표
| 부위 / 역할 | **Isaac Lab USD Prim (`.usd`)** | **Fabrics URDF 프레임 (`.urdf`)** | 비고 |
|---|---|---|---|
| 베이스 바디 | `/World/Robot/body_link` | `body_link` | 로봇 베이스 기둥 |
| 팔 7관절 | `r_aj_1` ~ `r_aj_7` | `openarm_right_joint1` ~ `7` | Fabrics Task-space |
| 팔 7링크 | `r_al_1` ~ `r_al_7` | `openarm_right_link1` ~ `7` | 링크 바디 |
| 손바닥 마운트 | `r_hl_mount` / `r_hl_palm` | `palm_link` | 손목 결합 베이스 |
| 진짜 손바닥 중심 | `r_hl_palm_ee` | `palm_ee` | **RL 제어/보상 기준점** |
| 5개 손끝점 | `r_hl_thumb_tip` ~ `pinky_tip` | `rl_dg_1_tip` ~ `5_tip` | 파지 접촉점 |

---

## 4. 🚀 실행 가이드

### 방법 1. 원클릭 실행 (배치 파일 - 가장 권장)
* **통합 스튜디오 실행**: `c:\Users\User\RL\KUKU_hdgp\run_studio.bat` 더블 클릭
* **USD 계층구조 검증**: `c:\Users\User\RL\KUKU_hdgp\run_inspect_usd.bat` 더블 클릭
* **순수 URDF 검증**: `c:\Users\User\RL\KUKU_hdgp\run_inspect_openarm.bat` 더블 클릭

### 방법 2. 터미널 수동 실행
```cmd
cd /d c:\Users\User\RL\KUKU_hdgp
conda activate kuku_kinematics
python scripts/kinematics/interactive_studio.py
```

---

## 5. 🔮 향후 개발 및 확장 가이드 (Roadmap)

1. **계층적 하이브리드 액션 공간 (Hybrid Action Space)**
   - 팔 7-DoF는 6차원 손바닥 궤적($\Delta X, \Delta Y, \Delta Z, \Delta \text{Rot}$)으로 Fabrics에 전달
   - 손가락 20-DoF는 5차원 시너지($\text{PCA}_{1\sim5}$) 또는 직접 관절 명령으로 파지 제어
2. **7자유도 잉여성(Null-space) 활용**
   - 팔꿈치(Elbow)가 몸체에 닿지 않고 위/바깥쪽의 편안한 자세를 유지하도록 `Elbow Swivel Attractor` 부여
3. **스튜디오 프리셋 $\to$ RL 초기화 포즈 직결**
   - 스튜디오의 `Ready / Reach / Grasp` 27개 관절 배열을 `grasp_right_preset.py`의 `RESET_JOINT_POS`로 동기화하여 학습 탐색 효율을 극대화
