# Synthetic Battery Data Generation

Generate synthetic battery cycling data using PyBaMM simulations, output as PyProBE-compatible parquet files.

## Quick Start

1. Create a JSON procedure file defining your experiments and store it in `procedures` (example files already available there).
2. Run the `generate_synthetic_data.py` script to generate synthetic data for the defined experiments. You may want to edit the `cell_info` dictionary in the script and the procedures being used.
3. Synthetic data will be written into `examples/data` in a PyProBE-compatible parquet format.

## Cell Info

Cell metadata should be added to the `cell_info` dictionary so that it is saved in the output.
The following minimum set of metadata is used to create the output path.

| Field         | Description                                                |
|---------------|------------------------------------------------------------|
| `Cell type`   | Cell brand name and type                                   |
| `Cell format` | "full cell", "half cell positive", or "half cell negative" |
| `Cell label`  | Cell or channel identifier                                 |

## Experiments

Each experiment is a list of PyBaMM experiment step strings. See [PyBaMM documentation](https://docs.pybamm.org/en/latest/source/api/experiment/experiment_steps.html) for step syntax.

To define an EIS experiment, add `"Type": "EIS"` and the EIS-specific fields to the
experiment object. The synthetic generator uses `pybop.pybamm.EISSimulator` for this
path and writes the parquet manually in a PyProBE-compatible format.

Notes on EIS:
- `Frequencies [Hz]` can be either a list of values or an object with `Min`, `Max`, `Count`, and optional `Spacing`.
- `Input amplitude [V]` is stored in the spec for completeness but is not used by the linear EIS solver.

## Output

Data is written to `examples/data/<Cell type>/<Cell format>/<Cell label>/<Experiment name>.parquet`

The parquet files are compatible with PyProBE and contain standard columns:
- `Time [s]`, `Current [A]`, `Voltage [V]`, `Capacity [Ah]`, `Step`, `Event`
- EIS procedures additionally include `Frequency [Hz]`, `Impedance (real) [Ohm]`, and `Impedance (imag) [Ohm]`
