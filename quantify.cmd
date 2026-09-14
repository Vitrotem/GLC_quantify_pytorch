@echo off
cd /d "%~dp0"
poetry run python -m src.quantify %*
