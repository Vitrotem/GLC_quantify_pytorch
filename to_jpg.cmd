@echo off
cd /d "%~dp0"
poetry run python -m src.to_jpg %*
