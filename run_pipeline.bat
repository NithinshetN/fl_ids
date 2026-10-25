@echo off
cd /d "%~dp0"
python preprocessing\preprocess.py --label-scope project8 --sample-size 0
if errorlevel 1 exit /b 1
python run_experiments.py --suite
if errorlevel 1 exit /b 1
python evaluation\plot_results.py --results results\main
if errorlevel 1 exit /b 1
