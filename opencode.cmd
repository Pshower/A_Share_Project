@echo off
rem Launch opencode through the local Clash proxy, for this repository only.
rem
rem opencode is a Bun binary: it reads HTTP_PROXY / HTTPS_PROXY, not the Windows
rem system proxy that Clash toggles for browsers.  The logic lives in
rem tools\opencode_proxy.ps1; this shim exists because the default Windows
rem execution policy refuses to run unsigned .ps1 files.
rem
rem   opencode.cmd                      same as running opencode, but proxied
rem   opencode.cmd -Check               show the resolved proxy and test the port
rem   opencode.cmd -Direct ...          run opencode with no proxy variables
setlocal
set "PS=pwsh"
where pwsh >nul 2>nul || set "PS=powershell"
"%PS%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\opencode_proxy.ps1" %*
exit /b %ERRORLEVEL%
