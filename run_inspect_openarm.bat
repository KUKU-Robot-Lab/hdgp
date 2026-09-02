@echo off
cd /d "%~dp0"
echo [*] Launching Pure URDF Kinematics Inspector for OpenArm-Tesollo...
"C:\Users\User\anaconda3\envs\kuku_kinematics\python.exe" scripts/kinematics/inspect_urdf.py source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo/openarm_tesollo.urdf
pause
