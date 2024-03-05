#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running lint check...'

configure_script

python3 -m pylint --rcfile config/pylint_check/.pylintrc config

directories=$(get_project_directories)

for directory in $directories; do
  python3 -m pylint --rcfile config/pylint_check/.pylintrc "${directory}"
  check_if_failed
done

echo "Lint check passed."
