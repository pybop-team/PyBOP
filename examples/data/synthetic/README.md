# Data description
The data files in this directory have the following metadata

`{model}_charge_discharge_{soc}.csv`
- Time, current, terminal voltage, and open-circuit voltage data for a Chen2020 parameters at the corresponding SOC. 1mV of noise is applied to both voltage signals.
- See `discharge_charge_data_gen.py` for reference

`{model}_pulse_{soc}.csv`
- Time, current, terminal voltage, and open-circuit voltage data for a Chen2020 parameters at the corresponding SOC. 1mV of noise is applied to both voltage signals.
- See `pulse_data_gen.py` for reference
