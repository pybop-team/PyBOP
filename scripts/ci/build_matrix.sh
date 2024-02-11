#!/bin/bash

# This helper script generates a matrix for further use in the
# scheduled/nightly builds for PyBOP, i.e., in scheduled_tests.yaml
# It generates a matrix of all combinations of the following variables:
# - python_version: 3.X
# - os: ubuntu-latest, windows-latest, macos-latest
# - pybamm_version: the last four versions of PyBaMM from PyPI, excluding release candidates

# To update the matrix, the variables below can be modified as needed.

python_version=("3.8" "3.9" "3.10" "3.11")
os=("ubuntu-latest" "windows-latest" "macos-latest")
# This command fetches the last four PyBaMM versions excluding release candidates from PyPI
pybamm_version=($(curl -s https://pypi.org/pypi/pybamm/json | jq -r '.releases | keys[]' | grep -v rc | tail -n 4 | awk '{print "\"" $1 "\"" }' | paste -sd " " -))

# open dict
json='{
  "include": [
'

# loop through each combination of variables to generate matrix components
for py_ver in "${python_version[@]}"; do
  for os_type in "${os[@]}"; do
    for pybamm_ver in "${pybamm_version[@]}"; do
      json+='{
        "os": "'$os_type'",
        "python_version": "'$py_ver'",
        "pybamm_version": '$pybamm_ver'
      },'
    done
  done
done

# fix structure, removing trailing comma
json=${json%,}

# close dict
json+='
  ]
}'

echo "$json" | jq -c .
