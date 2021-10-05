#!/bin/bash
set -ex
printenv

export USED_PORT=80

if [[ "$TASK_COMMAND" = "serve" ]]; then
  python3 web_app_run.py
else
    eval "$@"
fi

# prevent docker exit
#tail -f /dev/null
