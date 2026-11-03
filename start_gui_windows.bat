@echo off
rem HF model downloader GUI - Windows launcher (double-click or run from cmd).
rem Creates .venv on first run, installs requirements, starts the server and opens the browser.
rem Keep this file ASCII only: cmd.exe reads .bat files as cp932.
rem Note: no variable is set and then used inside a ( ... ) block, because cmd expands %VAR% when it parses the whole block.
setlocal EnableExtensions
cd /d "%~dp0"
set "VENV=%~dp0.venv"
set "VPY=%VENV%\Scripts\python.exe"

echo == HF model downloader GUI (Windows) ==

rem A .venv copied from another OS or machine does not run here: rebuild it.
if not exist "%VPY%" goto :no_venv
"%VPY%" -c "import sys" >nul 2>&1
if not errorlevel 1 goto :have_venv
echo Existing .venv is not usable on this machine, recreating it ...
rmdir /s /q "%VENV%"

:no_venv
set "BASEPY="
if defined PYTHON_BIN call :trypy "%PYTHON_BIN%"
if not defined BASEPY call :trypy py -3.13
if not defined BASEPY call :trypy py -3.12
if not defined BASEPY call :trypy py -3.11
if not defined BASEPY call :trypy py -3.10
if not defined BASEPY call :trypy py -3
if not defined BASEPY call :trypy python
if not defined BASEPY goto :nopython
echo Creating virtualenv with: %BASEPY%
%BASEPY% -m venv "%VENV%"
if errorlevel 1 goto :venvfail

:have_venv
"%VPY%" -c "import huggingface_hub" >nul 2>&1
if not errorlevel 1 goto :run
echo Installing requirements - internet access to PyPI is needed ...
"%VPY%" -m pip install -q -r "%~dp0requirements.txt"
if errorlevel 1 goto :pipfail

:run
echo Starting the GUI. The browser opens automatically. Close this window or press Ctrl+C to stop.
echo.
"%VPY%" "%~dp0gui_server.py" %*
if errorlevel 1 goto :fail
endlocal
exit /b 0

:trypy
%* -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
set BASEPY=%*
exit /b 0

:nopython
echo.
echo ERROR: Python 3.10 or newer was not found.
echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then run this file again.
goto :fail

:venvfail
echo ERROR: could not create the virtualenv.
goto :fail

:pipfail
echo ERROR: pip install failed.
goto :fail

:fail
echo.
echo The launcher stopped because of an error ^(see the messages above^).
pause
endlocal
exit /b 1
