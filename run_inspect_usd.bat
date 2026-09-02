@echo off
cd /d "%~dp0"
echo [*] Inspecting Isaac Lab Ground-Truth USD Asset with Pixar OpenUSD...
"C:\Users\User\anaconda3\envs\kuku_kinematics\python.exe" scripts/kinematics/inspect_usd.py
pause
