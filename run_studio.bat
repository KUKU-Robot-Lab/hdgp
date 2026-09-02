@echo off
cd /d "%~dp0"
echo [*] Launching Isaac Lab Style Interactive Kinematics Studio for OpenArm-Tesollo...
"C:\Users\User\anaconda3\envs\kuku_kinematics\python.exe" scripts/kinematics/interactive_studio.py source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo/openarm_tesollo.urdf
pause
