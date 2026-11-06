@echo off
echo ======================================================================
echo Compiling MA4998P BTP Project Report locally via Tectonic...
echo ======================================================================
cd /d "%~dp0report"
"%~dp0bin\tectonic.exe" report.tex
if %ERRORLEVEL% EQU 0 (
    copy /y report.pdf "%~dp0MA4998P_BTP_Report_Dinkesh_Pal.pdf"
    echo.
    echo [SUCCESS] PDF compiled successfully!
    echo Output: %~dp0MA4998P_BTP_Report_Dinkesh_Pal.pdf
) else (
    echo.
    echo [ERROR] LaTeX compilation failed. Check the error log above.
)
pause
