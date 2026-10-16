#!/bin/bash

# This helper script generates a matrix for further use in the scheduled
# continuous integration tests for PyBOP, i.e., in scheduled_tests.yaml
# It generates a matrix of all combinations of the following variables:
# - python_version: 3.X
# - os: ubuntu-latest, windows-latest, macos-15-intel (amd64), macos-latest (arm64)
# - pybamm_version: the last X versions of PyBaMM from PyPI, excluding release candidates

# To update the matrix, the variables below can be modified as needed.

python_version=("3.11" "3.12" "3.13" "3.14" "3.15")
os=("ubuntu-latest" "windows-latest" "macos-15-intel" "macos-latest")
# This command fetches the last PyBaMM version excluding release candidates from PyPI
pybamm_version=($(curl -s https://pypi.org/pypi/pybamm/json | jq -r '.releases | keys[]' | grep -v rc | sort -V | tail -n 1 | paste -sd " " -))

# This command fetches the last PyBaMM versions including release candidates from PyPI
#pybamm_version=($(curl -s https://pypi.org/pypi/pybamm/json | jq -r '.releases | keys[]' | sort -V | tail -n 1 | paste -sd " " -))

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

# Example for filtering out incompatible combinations
#json=$(echo "$json" | jq -c 'del(.include[] | select(.pybamm_version == "23.9" and .python_version == "3.12"))')

echo "$json" | jq -c .
