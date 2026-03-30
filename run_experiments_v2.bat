@echo off
echo ============================================================
echo ASIMOV'S MIND - GOVERNED ML RESEARCH EXPERIMENT (v2)
echo ============================================================
echo.
echo This runs (RTX 4060 Laptop ~10 min/experiment):
echo   Phase 0: Baseline              (~10 min)
echo   Phase 1: 10 isolated           (~100 min)
echo   Phase 2: 3x cumulative replay  (~170 min)
echo   Phase 3: Optimal cumulative    (~60 min)
echo   Total:                         ~5.5 hours
echo.
echo Results: governed\results\all_results.tsv
echo Logs:    governed\results\logs\
echo.
echo Press Ctrl+C to abort at any time.
echo ============================================================
echo.

cd /d "C:\Users\swebs\Projects\asimovs-mind-research"

python governed/experiment_runner.py all

echo.
echo ============================================================
echo EXPERIMENT COMPLETE
echo.
echo Results: governed\results\all_results.tsv
echo Logs:    governed\results\logs\
echo ============================================================
pause
