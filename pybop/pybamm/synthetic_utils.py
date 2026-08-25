"""Utilities for synthetic data generation."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pybamm

import pybop

# List of parameters to swap for negative half cell
PARAMETERS_TO_SWAP: list[str] = [
    "Positive electrode conductivity [S.m-1]",
    "Maximum concentration in positive electrode [mol.m-3]",
    "Positive particle diffusivity [m2.s-1]",
    "Positive electrode OCP [V]",
    "Positive electrode porosity",
    "Positive electrode active material volume fraction",
    "Positive particle radius [m]",
    "Positive electrode Bruggeman coefficient (electrolyte)",
    "Positive electrode Bruggeman coefficient (electrode)",
    "Positive electrode charge transfer coefficient",
    "Positive electrode double-layer capacity [F.m-2]",
    "Positive electrode exchange-current density [A.m-2]",
    "Positive electrode density [kg.m-3]",
    "Positive electrode specific heat capacity [J.kg-1.K-1]",
    "Positive electrode thermal conductivity [W.m-1.K-1]",
    "Positive electrode OCP entropic change [V.K-1]",
    "Positive electrode thickness [m]",
    "Initial concentration in positive electrode [mol.m-3]",
]

DEFAULT_COLUMN_DEFINITIONS: dict[str, str] = {
    "Time": "The time passed from the start of the procedure.",
    "Step": "The step number.",
    "Event": "The event number. Counts the changes in cycles and steps.",
    "Current": "The current through the cell.",
    "Voltage": "The terminal voltage.",
    "Capacity": "The net charge passed since the start of the procedure.",
    "Temperature": "The temperature of the cell.",
}
EIS_COLUMN_DEFINITIONS: dict[str, str] = {
    "Time": "The time passed from the start of the procedure.",
    "Step": "The step number.",
    "Event": "The event number. Counts the changes in cycles and steps.",
    "Current": "The current through the cell.",
    "Voltage": "The terminal voltage.",
    "Capacity": "The net charge passed since the start of the procedure.",
    "Initial voltage": "The terminal voltage prior to the EIS measurement.",
    "Frequency": "The input perturbation frequency.",
    "Impedance (real)": "The real component of the complex impedance.",
    "Impedance (imag)": "The imaginary component of the complex impedance.",
    # "Temperature": "The temperature of the cell.",
}


def _canonical_cell_format(cell_type: str) -> str:
    """Normalize cell type string to archive format."""
    value = cell_type.strip().lower()
    if value in ["full cell", "full", "full-cell"]:
        return "Full cell"
    elif value in ["half cell positive", "half-cell positive", "positive half cell"]:
        return "Half cell positive"
    elif value in ["half cell negative", "half-cell negative", "negative half cell"]:
        return "Half cell negative"
    raise ValueError(f"Unsupported cell type: {cell_type}")


def convert_to_half_cell_parameters(
    parameter_values: pybamm.ParameterValues,
    electrode_type: str,
) -> pybamm.ParameterValues:
    """Function to adapt a full cell parameter set to a half cell.

    Positive electrode parameters are set from the working electrode,
    negative electrode parameters are set for a lithium counter, from Xu2019.
    """
    updated_parameter_values = parameter_values.copy()

    if electrode_type.lower() == "negative":
        for parameter_name in PARAMETERS_TO_SWAP:
            parameter_name_negative = parameter_name.replace(
                "Positive", "Negative"
            ).replace("positive", "negative")
            try:
                source_value = parameter_values[parameter_name_negative]
            except KeyError as error:
                raise KeyError(
                    f"Cannot swap '{parameter_name}' into '{parameter_name_negative}': "
                    f"source parameter was not found."
                ) from error
            updated_parameter_values.update({parameter_name: source_value})

        updated_parameter_values.update(
            {
                "Lower voltage cut-off [V]": 0.005,
                "Upper voltage cut-off [V]": 1.5,
                "Open-circuit voltage at 0% SOC [V]": 0.005,
                "Open-circuit voltage at 100% SOC [V]": 1.5,
            }
        )

    else:
        updated_parameter_values.update(
            {
                "Lower voltage cut-off [V]": 3.5,
                "Upper voltage cut-off [V]": 4.2,
                "Open-circuit voltage at 0% SOC [V]": 3.5,
                "Open-circuit voltage at 100% SOC [V]": 4.2,
            }
        )

    updated_parameter_values.update(
        {
            "Negative electrode OCP [V]": 0.0,
            "Negative electrode conductivity [S.m-1]": 10776000.0,
            "Negative electrode OCP entropic change [V.K-1]": 0.0,
            "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
            "Exchange-current density for lithium metal electrode [A.m-2]": (
                lambda c_e, c_Li, T: 3.5e-8 * pybamm.constants.F * c_Li**0.7 * c_e**0.3
            ),
            "Negative electrode charge transfer coefficient": 0.5,
            "Negative electrode double-layer capacity [F.m-2]": 0.2,
        }
    )
    return updated_parameter_values


def _safe_name(value: str) -> str:
    """Create a safe filename from a string."""
    return str(value).strip().replace(" ", "").replace("/", "-")


def _get_experiment_type(experiment_info: dict[str, Any]) -> str:
    """Return the synthetic experiment type."""
    return str(experiment_info.get("Type", "Time domain")).strip().lower()


def _get_experiment_sequences(
    experiment_info: dict[str, Any],
) -> tuple[list[tuple[int, ...]], list[tuple[str, ...]]]:
    """Expand a spec experiment into PyBaMM experiment step tuples."""
    step_numbers = experiment_info["Steps"]
    step_descriptions = experiment_info["Step Descriptions"]
    cycle_info = experiment_info.get("Cycles", [])
    step_index = {step: index for index, step in enumerate(step_numbers)}

    if len(cycle_info) == 0:
        return [tuple(step_numbers)], [tuple(step_descriptions)]

    n_loops = len(cycle_info)
    subset = [[] for _ in range(n_loops)]  # create distinct empty lists
    cycle_ended = [False] * n_loops
    step_sequence = []
    for step in step_numbers:
        step_included = False
        for i, cycle in enumerate(cycle_info[::-1]):
            # Iterate from inner to outer cycles
            if not cycle_ended[i]:
                if not step_included and cycle[0] <= step <= cycle[1]:
                    # This step is part of cycle i
                    subset[i].append(step)
                    step_included = True
                if step >= cycle[1]:
                    # The cycle has ended
                    if i < n_loops - 1:
                        subset[i + 1].extend(subset[i] * cycle[2])
                    else:
                        step_sequence.extend([tuple(subset[i])] * cycle[2])
                    cycle_ended[i] = True
        if not step_included:
            step_sequence.append(tuple([step]))
    for i, cycle in enumerate(cycle_info):
        # Add any cycles that did not yet end
        if not cycle_ended[i]:
            step_sequence.extend([tuple(subset[i])] * cycle[2])

    description_sequence = [
        tuple([step_descriptions[step_index[s]] for s in t]) for t in step_sequence
    ]

    return step_sequence, description_sequence


def _flatten_sequence(sequences: list[tuple[str, ...]]) -> list[str]:
    """Flatten a list of experiment tuples into a list of per-event descriptions."""
    return [step for sequence in sequences for step in sequence]


def _remap_cycle_info(
    dataframe: pl.LazyFrame, step_sequence: list[tuple[int, ...]], event_offset: int
) -> pl.LazyFrame:
    """Map PyBaMM step and cycle numbers back to match the cycle information."""
    step_column = pl.col("Step").cast(pl.Int64)
    cycle_column = pl.col("Cycle").cast(pl.Int64)
    event_column = (
        (
            (step_column - step_column.shift()).abs()
            + (cycle_column - cycle_column.shift()).abs()
            != 0
        )
        .fill_null(strategy="zero")
        .cum_sum()
        .add(event_offset)
        .cast(pl.Int64)
    )
    dataframe = dataframe.with_columns(Event=event_column)
    event_numbers = (
        dataframe.select(event_column.unique(maintain_order=True))
        .collect()
        .to_numpy()
        .flatten()
    )
    return dataframe.with_columns(
        Step=pl.col("Event").replace(event_numbers, _flatten_sequence(step_sequence))
    )


def _build_frequency_grid(frequency_spec: list[float] | dict[str, Any]) -> np.ndarray:
    """Build the frequency grid for an EIS experiment."""
    if isinstance(frequency_spec, list):
        return np.asarray(frequency_spec, dtype=float)

    minimum = float(frequency_spec["Min"])
    maximum = float(frequency_spec["Max"])
    count = int(frequency_spec["Count"])
    spacing = str(frequency_spec.get("Spacing", "log")).strip().lower()

    if spacing == "log":
        return np.geomspace(maximum, minimum, count)
    elif spacing == "linear":
        return np.linspace(maximum, minimum, count)
    raise ValueError(f"Unsupported EIS frequency spacing: {spacing}")


def _solution_to_dataframe(
    solution: pybamm.Solution,
    step_sequence: list[tuple[int, ...]],
    start_time: float,
    event_offset: int,
) -> pl.LazyFrame:
    """Convert a PyBaMM solution into the PyProBE-required dataframe layout."""
    pybamm_data = solution.get_data_dict(
        [
            "Time [s]",
            "Current [A]",
            "Voltage [V]",
            "Discharge capacity [A.h]",
            "Volume-averaged cell temperature [K]",
        ]
    )
    pybamm_data["Time [s]"] += start_time
    try:
        raw_dataframe = pl.LazyFrame(pybamm_data)
    except pl.exceptions.ShapeError:
        # Account for an error where the Step is one longer than the Solution
        pybamm_data["Step"] = pybamm_data["Step"][:-1]
        raw_dataframe = pl.LazyFrame(pybamm_data)
    raw_dataframe = _remap_cycle_info(raw_dataframe, step_sequence, event_offset)
    return raw_dataframe.select(
        [
            pl.col("Time [s]"),
            pl.col("Step"),
            pl.col("Event"),
            (pl.col("Current [A]") * -1).alias("Current [A]"),
            pl.col("Voltage [V]").alias("Voltage [V]"),
            (pl.col("Discharge capacity [A.h]") * -1).alias("Capacity [Ah]"),
            (pl.col("Volume-averaged cell temperature [K]") - 273.15).alias(
                "Temperature [degC]"
            ),
        ]
    )


def _solve_time_domain_experiment(
    experiment: pybamm.Experiment,
    model: pybamm.BaseModel,
    parameter_values: pybamm.ParameterValues,
    previous_solution: pybamm.Solution | None,
    solve_kwargs: dict[str, float] | None,
) -> pybamm.Solution:
    """Solve a standard time-domain experiment."""
    if previous_solution is not None:
        model.set_initial_conditions_from(previous_solution)

    sim = pybamm.Simulation(
        model, experiment=experiment, parameter_values=parameter_values
    )

    solve_kwargs = {} if solve_kwargs is None else solve_kwargs
    return sim.solve(**solve_kwargs)


def _build_eis_sweep_dataframe(
    step_number: int,
    event_number: int,
    frequencies: np.ndarray,
    impedance: np.ndarray,
    initial_voltage: float | None,
) -> pl.LazyFrame:
    """Create a PyProBE-compatible dataframe for one EIS sweep."""
    row_count = len(frequencies)
    return pl.LazyFrame(
        {
            "Time [s]": [None] * row_count,
            "Step": [step_number] * row_count,
            "Event": [event_number] * row_count,
            "Initial voltage [V]": [initial_voltage] * row_count,
            "Frequency [Hz]": frequencies,
            "Impedance (real) [Ohm]": np.real(impedance),
            "Impedance (imag) [Ohm]": np.imag(impedance),
            "Current [A]": [None] * row_count,
            "Voltage [V]": [None] * row_count,
            "Capacity [Ah]": [None] * row_count,
        }
    )


def _solve_eis_experiment(
    experiment_info: dict[str, Any],
    model: pybamm.BaseModel,
    parameter_values: pybamm.ParameterValues,
    previous_solution: pybamm.Solution | None,
    step_offset: int,
    solve_kwargs: dict[str, float] | None,
) -> tuple[pl.LazyFrame, dict[str, Any], pybamm.Solution | None, int, int]:
    """Solve an EIS experiment and return a PyProBE-compatible dataframe."""
    frequencies = _build_frequency_grid(experiment_info["Frequencies [Hz]"])
    step_sequence, description_sequence = _get_experiment_sequences(experiment_info)
    step_numbers = _flatten_sequence(step_sequence)
    sweep_descriptions = _flatten_sequence(description_sequence)

    sweep_frames: list[pl.LazyFrame] = []
    latest_solution: pybamm.Solution | None = previous_solution

    for i, (step, description) in enumerate(
        zip(step_numbers, sweep_descriptions, strict=True)
    ):
        if "EIS" not in description:
            solution = _solve_time_domain_experiment(
                experiment=pybamm.Experiment([description]),
                model=model,
                parameter_values=parameter_values,
                previous_solution=latest_solution,
                solve_kwargs=solve_kwargs,
            )
            sweep_frame = _solution_to_dataframe(
                solution=solution,
                step_sequence=[tuple([step])],
                start_time=0.0
                if latest_solution is None
                else latest_solution["Time [s]"].data[-1],
                event_offset=step_offset + i,
            )
            latest_solution = solution
        else:
            # This is the EIS experiment, needs a new copy due to the extra voltage variable
            eis_model = model.new_copy()
            if latest_solution is not None:
                eis_model.set_initial_conditions_from(latest_solution)
            eis_simulator = pybop.pybamm.EISSimulator(
                eis_model, f_eval=frequencies, parameter_values=parameter_values
            )
            impedance = np.asarray(eis_simulator.solve()["Impedance"].data)
            initial_voltage = (
                None
                if latest_solution is None
                else float(latest_solution["Terminal voltage [V]"].data[-1])
            )
            sweep_frame = _build_eis_sweep_dataframe(
                step_number=step,
                event_number=step_offset + i,
                frequencies=frequencies,
                impedance=impedance,
                initial_voltage=initial_voltage,
            )
        sweep_frames.append(sweep_frame)

    return pl.concat(sweep_frames, how="diagonal_relaxed"), len(sweep_descriptions)


def simulate_procedure(
    info: dict,
    model: pybamm.BaseModel,
    parameter_values: pybamm.ParameterValues,
    spec_path: Path,
    solve_kwargs: dict[str, float] | None = None,
) -> None:
    from pyprobe import Cell
    from pyprobe.filters import Procedure

    """Run synthetic data generation from a spec file."""
    if isinstance(spec_path, list):
        procedures = {}
        for sp in spec_path:
            procedures.update(_validate_spec(sp))
    else:
        procedures = _validate_spec(spec_path)

    cell_type = info.get("Cell type", "Unknown type")
    cell_format = _canonical_cell_format(info.get("Cell format", "Full cell"))
    cell_label = _safe_name(info.get("Cell label", "No label"))
    cell_name = info.get("Name", f"{cell_type} {cell_label}")
    cell_capacity = parameter_values["Nominal cell capacity [A.h]"]

    print("\n" + "=" * 80)
    print(f"SPEC: {spec_path}")
    print("=" * 80)
    print(f"Name: {cell_name}")
    print(f"Cell type: {cell_type}")
    print(f"Cell format: {cell_format}")
    print(f"Cell label: {cell_label}")
    print(f"Nominal cell capacity [A.h]: {cell_capacity}")
    print(f"Experiments: {list(procedures.keys())}")

    # Create PyProBE cell with metadata compatible with downstream scripts.
    cell_info = {
        "Name": cell_name,
        "Cell type": cell_type,
        "Cell format": cell_format,
        "Cell label": cell_label,
        "Synthetic": True,
        "Nominal cell capacity [A.h]": cell_capacity,
    }
    cell = Cell(info=cell_info)

    for procedure_name, procedure_info in procedures.items():
        print("\n" + "-" * 80)
        print(f"Processing procedure: {procedure_name}")
        print("-" * 80)

        readme_dict = {}
        model = model.new_copy()
        procedure_frames: list[pl.LazyFrame] = []
        latest_solution: pybamm.Solution | None = None
        next_step_offset = 0

        for experiment_name, experiment_info in procedure_info.items():
            experiment_type = _get_experiment_type(experiment_info)
            print(f"  Experiment: {experiment_name}")
            print(f"  Type: {experiment_type}")
            number_steps = experiment_info.get("Steps", [])
            print(f"  Steps: {len(number_steps)}")

            readme_dict[experiment_name] = {
                "Steps": experiment_info["Steps"],
                "Step Descriptions": experiment_info["Step Descriptions"],
                "Cycles": experiment_info.get("Cycles", []),
            }

            if experiment_type == "eis":
                column_definitions = EIS_COLUMN_DEFINITIONS.copy()
                frequencies = _build_frequency_grid(experiment_info["Frequencies [Hz]"])
                print(f"  Frequencies: {len(frequencies)}")
                procedure_frame, step_count = _solve_eis_experiment(
                    experiment_info=experiment_info,
                    model=model,
                    parameter_values=parameter_values,
                    previous_solution=latest_solution,
                    step_offset=next_step_offset,
                    solve_kwargs=solve_kwargs,
                )
                next_step_offset += step_count
            else:
                column_definitions = DEFAULT_COLUMN_DEFINITIONS.copy()
                step_sequence, description_sequence = _get_experiment_sequences(
                    experiment_info
                )
                solution = _solve_time_domain_experiment(
                    experiment=pybamm.Experiment(description_sequence),
                    model=model,
                    parameter_values=parameter_values,
                    previous_solution=latest_solution,
                    solve_kwargs=solve_kwargs,
                )
                procedure_frame = _solution_to_dataframe(
                    solution=solution,
                    step_sequence=step_sequence,
                    start_time=0.0
                    if latest_solution is None
                    else latest_solution["Time [s]"].data[-1],
                    event_offset=next_step_offset,
                )
                latest_solution = solution
                next_step_offset = (
                    procedure_frame.select("Event").last().collect().item() + 1
                )

            procedure_frames.append(procedure_frame)

        if not procedure_frames:
            print(f"  Skipping {procedure_name}: no experiments provided")
            continue

        combined_frame = (
            procedure_frames[0]
            if len(procedure_frames) == 1
            else pl.concat(procedure_frames, how="diagonal_relaxed")
        )
        indexed_frame = combined_frame.with_row_index("id")
        unique_rows = (
            indexed_frame.drop_nulls("Time [s]")
            .unique("Time [s]", maintain_order=True, keep="first")
            .collect()
        )
        lf = indexed_frame.filter(
            pl.col("Time [s]").is_null() | pl.col("id").is_in(unique_rows["id"])
        ).drop("id")
        cell.procedure[procedure_name] = Procedure(
            lf=lf,
            info=cell.info,
            readme_dict=readme_dict,
            column_definitions=column_definitions,
        )

        if latest_solution is not None:
            print(
                f"  Simulation completed: {latest_solution['Time [h]'].data[-1]:.2f} hours"
            )
        else:
            print("  Simulation completed: EIS-only procedure")

    return cell


def archive_data(cell, archive_root: Path):
    """Archive the data fron the cell object."""
    # Get info
    cell_label = cell.info["Cell label"]
    cell_format = cell.info["Cell format"]
    cell_type = cell.info["Cell type"]

    # Export archive
    archive_dir = archive_root / cell_type / cell_format / cell_label
    archive_dir.mkdir(parents=True, exist_ok=True)
    cell.archive(path=str(archive_dir))

    metadata_path = archive_dir / "metadata.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        with metadata_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(metadata, indent=4) + "\n")

    print(f"\nArchived cell to: {archive_dir}")


def _validate_steps(experiment_name: str, experiment_info: dict[str, Any]) -> None:
    steps = experiment_info.get("Steps")
    step_descriptions = experiment_info.get("Step Descriptions")
    if not isinstance(steps, list) or not steps:
        raise ValueError(
            f"Experiment '{experiment_name}' must define a non-empty 'Steps' list"
        )
    if not isinstance(step_descriptions, list) or len(step_descriptions) != len(steps):
        raise ValueError(
            f"Experiment '{experiment_name}' must define 'Step Descriptions' with the same length as 'Steps'"
        )


def _validate_frequency_spec(experiment_name: str, frequency_spec: Any) -> None:
    if isinstance(frequency_spec, list):
        if not frequency_spec:
            raise ValueError(
                f"EIS experiment '{experiment_name}' must define at least one frequency"
            )
        return

    if not isinstance(frequency_spec, dict):
        raise ValueError(
            f"EIS experiment '{experiment_name}' must define 'Frequencies [Hz]' as a list or object"
        )

    required_keys = {"Min", "Max", "Count"}
    missing = required_keys.difference(frequency_spec)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"EIS experiment '{experiment_name}' missing frequency keys: {missing_list}"
        )


def _validate_experiment(experiment_name: str, experiment_info: Any) -> None:
    if not isinstance(experiment_info, dict):
        raise ValueError(f"Experiment '{experiment_name}' must be an object")

    _validate_steps(experiment_name, experiment_info)

    experiment_type = str(experiment_info.get("Type", "Time domain")).strip().lower()
    if experiment_type != "eis":
        return

    _validate_frequency_spec(experiment_name, experiment_info.get("Frequencies [Hz]"))


def _load_spec(path: Path) -> dict[str, Any]:
    """Load a JSON spec file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_spec(spec_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    procedures = _load_spec(spec_path)

    for procedure_name, procedure_info in procedures.items():
        if not isinstance(procedure_info, dict):
            raise ValueError(
                f"Procedure '{procedure_name}' in {spec_path} must be an object"
            )
        for experiment_name, experiment_info in procedure_info.items():
            _validate_experiment(experiment_name, experiment_info)

    return procedures
