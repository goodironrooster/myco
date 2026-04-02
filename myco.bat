@echo off
REM MYCO Launcher - Just type "myco" to start
REM
REM Usage:
REM   myco                    # Interactive mode
REM   myco "task"             # Run a single task
REM   myco --help             # Show all options

python -m cli.main myco %*
