@echo off
REM ========================================================================
REM Scheduled Backup Task Creation Script
REM
REM This script creates a Windows Task Scheduler task to run backup.py daily
REM at 2:00 AM. It includes error handling and validation checks.
REM ========================================================================

echo Setting up daily backup task...

REM Save the current directory path
set "SCRIPT_DIR=%~dp0"
set "BACKUP_SCRIPT=%SCRIPT_DIR%backup.py"
set "TASK_NAME=BlockchainDailyBackup"

REM Verify that backup.py exists
if not exist "%BACKUP_SCRIPT%" (
    echo ERROR: Could not find backup.py at "%BACKUP_SCRIPT%"
    echo Please make sure the script exists and try again.
    exit /b 1
)

REM Verify Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in the PATH.
    echo Please install Python and try again.
    exit /b 1
)

REM Check if the task already exists
schtasks /query /tn %TASK_NAME% > nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Task "%TASK_NAME%" already exists.
    
    choice /C YN /M "Do you want to replace it?"
    if %ERRORLEVEL% equ 2 (
        echo Task creation canceled by user.
        exit /b 0
    )
    
    echo Removing existing task...
    schtasks /delete /tn %TASK_NAME% /f > nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to delete existing task.
        exit /b 1
    )
)

REM Create the scheduled task to run daily at 2:00 AM
echo Creating scheduled task "%TASK_NAME%" to run daily at 2:00 AM...
schtasks /create /tn %TASK_NAME% /tr "python \"%BACKUP_SCRIPT%\"" /sc DAILY /st 02:00 /ru SYSTEM
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create scheduled task.
    echo Check your administrator privileges and try again.
    exit /b 1
)

REM Verify the task was created successfully
schtasks /query /tn %TASK_NAME% > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Task verification failed. The task may not have been created correctly.
    exit /b 1
) else (
    echo Success! Scheduled task "%TASK_NAME%" has been created.
    echo The backup script will run daily at 2:00 AM.
    
    REM Display task details
    echo.
    echo Task details:
    schtasks /query /tn %TASK_NAME% /fo LIST
)

echo.
echo Note: You can manually run a backup at any time by executing:
echo python "%BACKUP_SCRIPT%"
echo.

exit /b 0

