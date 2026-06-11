@echo off
setlocal

set "ROOT_DIR=%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%
