#!/bin/bash
set -ex
printenv

if [[ "$TASK_COMMAND" = "serve" ]]; then
  python3 web_app_run.py
else
    eval "$@"
fi

# prevent docker exit
#tail -f /dev/null
