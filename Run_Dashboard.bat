@echo off
title IPL Streamlit Dashboard

echo Starting IPL Dashboard...
echo.

python -m pip install -r requirements.txt

echo.
echo Opening dashboard...
python -m streamlit run app.py

pause
