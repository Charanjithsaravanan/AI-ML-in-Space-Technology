@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_geosat.ps1"
if errorlevel 1 pause
