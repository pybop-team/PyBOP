#!/bin/bash

# This helper script generates a matrix for further use in the
# scheduled/nightly builds for PyBOP, i.e., in scheduled_tests.yaml
# It generates a matrix of all combinations of the following variables:
# - python_version: 3.X
# - os: ubuntu-latest, windows-latest, macos-latest
# - pybamm_version: the last X versions of PyBaMM from PyPI, excluding release candidates

# To update the matrix, the variables below can be modified as needed.

python_version=("3.8" "3.9" "3.10" "3.11" "3.12")
os=("ubuntu-latest" "windows-latest" "macos-latest")
# This command fetches the last three PyBaMM versions excluding release candidates from PyPI
pybamm_version=($(curl -s https://pypi.org/pypi/pybamm/json | jq -r '.releases | keys[]' | grep -v rc | tail -n 3 | paste -sd " " -))

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
          "pybamm_version": "'$pybamm_ver'"
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

# Filter out incompatible combinations
echo "$json" | jq -c 'del(.include[] | select(.pybamm_version == "23.5" and .python_version == "3.12"))'
