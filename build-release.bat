@echo off
REM Release build helper. Sources vcvars64, sets env, configures + builds.
REM Usage: build-release.bat [configure|build|all]    (default: all)
setlocal
set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=all"

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo vcvars64.bat failed
    exit /b 1
)

set "VCPKG_ROOT=C:\vcpkg"
set "PATH=C:\VulkanSDK\1.4.341.1\Bin;%PATH%"

cd /d "%~dp0"

if /i "%ACTION%"=="configure" goto :configure
if /i "%ACTION%"=="build" goto :build
if /i "%ACTION%"=="all" goto :all
echo Unknown action: %ACTION%
exit /b 1

:configure
cmake --preset release
exit /b %errorlevel%

:build
cmake --build build-release
exit /b %errorlevel%

:all
cmake --preset release || exit /b 1
cmake --build build-release
exit /b %errorlevel%
