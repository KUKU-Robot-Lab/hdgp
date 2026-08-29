# Workspace Project Guidelines & Specifications (hdgp)

## 1. Git & Branch Management
- **Git LFS**: Ensure Git LFS is active (`git lfs install`) and all binary assets (.usd, model weights, datasets) are fully downloaded (`git lfs pull`).
- **Working Branch**: Always work on personal branch `dev/beomsu`. **NEVER** commit or push directly to `main`.
- **Ignore / Clean Repository**: Never commit checkpoints, large log files, datasets, or private tokens/credentials.

## 2. Environment & Simulator Requirements
- **Isaac Sim**: `>= 5.1.0`
- **IsaacLab**: `2.3.1`
- **Robomimic Module**: Must be installed via `./isaaclab.sh -i robomimic`.

## 3. ROS 2 & Network Configuration
- **ROS 2 Domain ID**: `export ROS_DOMAIN_ID=126`
- **Tesollo / Delto Hardware Communication**:
  - Local Host IP: `169.254.186.10/16`
  - Device Target IP / Port: `169.254.186.72` / Port `502`
  - **Wi-Fi Interface**: Disable conflicting Wi-Fi interfaces when running physical hardware tests (`nmcli radio wifi off`).

## 4. Controller & Execution Matching
- Match controller modes consistently between OpenArm execution and policy inference:
  - Differentiate clearly between Joint Trajectory Controller and Forward Position Controller using `--tesollo_mode` and `--arm_mode`.
