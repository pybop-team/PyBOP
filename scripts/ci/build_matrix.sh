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
# Function to check if a PyBaMM version is compatible with a Python version
is_compatible() {
  local pybamm_ver="$1"
  local py_ver="$2"

  # Compatibility check
  if [[ "$pybamm_ver" == "23.5" && "$py_ver" == "3.12" ]]; then
    return 1 # Incompatible
  elif [[ "$pybamm_ver" == "23.9" && "$py_ver" == "3.12" ]]; then
    return 1 # Incompatible
  fi

  return 0 # Compatible
}

# loop through each combination of variables to generate matrix components
for py_ver in "${python_version[@]}"; do
  for os_type in "${os[@]}"; do
    for pybamm_ver in "${pybamm_version[@]}"; do
      if is_compatible "$pybamm_ver" "$py_ver"; then
        json+='{
          "os": "'$os_type'",
          "python_version": "'$py_ver'",
          "pybamm_version": "'$pybamm_ver'"
        },'
      fi
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
