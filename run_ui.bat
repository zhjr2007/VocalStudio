@echo off
REM Launch the NEW interface (vstudio_ui). The archived first-generation GUI
REM (gui\fluent_main.py, run.bat) is untouched and can still be started.
venv\Scripts\python -m vstudio_ui.app %*
