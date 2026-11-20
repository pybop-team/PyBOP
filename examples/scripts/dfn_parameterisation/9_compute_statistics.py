import argparse
import pickle
from pathlib import Path

import polars as pl

from pybop._utils import is_numeric

"""
or each cell type, compute the parameter statistics.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--draft", action="store_true", help="Draft mode")
args = parser.parse_args()

DIRECTIONS = ["charge", "discharge"]

# Iterate over the cell types, saving the results in each directory
results_path = Path(__file__).parent / "results"
for cell_type in results_path.iterdir():
    if cell_type.is_dir():
        # Create a dataframe with the values for each cell, param and direction combination
        schema = {
            "Parameter": pl.String,
            "cell_label": pl.String,
            "direction": pl.String,
            "value": pl.Float64,
        }
        df_float = pl.DataFrame(schema=schema)

        # Go through the cells, and add the values to the dataframe
        for cell_label in cell_type.iterdir():
            if cell_label.is_dir():
                for direction in DIRECTIONS:
                    # Load parameters
                    filename = cell_label / f"params_{direction}.pickle"
                    try:
                        with open(filename, "rb") as file:
                            params = pickle.load(file)
                    except FileNotFoundError:
                        continue

                    for key, value in params.items():
                        if is_numeric(value):
                            new_row = pl.DataFrame(
                                {
                                    "Parameter": key,
                                    "cell_label": f"{cell_label.stem}",
                                    "direction": direction,
                                    "value": float(value),
                                }
                            )
                            df_float.vstack(new_row, in_place=True)
                        else:  # str, another class
                            pass

        if df_float.is_empty():
            continue  # skip to next cell type

        # Create a dataframe with the mean and variance of the floats
        schema = {
            "Parameter": pl.String,
            "Mean (charge)": pl.String,
            "Std (charge)": pl.String,
            "Mean (discharge)": pl.String,
            "Std (discharge)": pl.String,
        }
        df_stats = pl.DataFrame(schema=schema)

        # Split the dataframe up by parameter and then direction
        for param_name, df_by_param in df_float.group_by("Parameter"):
            stats_dict = {
                "Parameter": param_name,
                "Mean (charge)": None,
                "Std (charge)": None,
                "Mean (discharge)": None,
                "Std (discharge)": None,
            }
            for direction, df_by_direction in df_by_param.group_by("direction"):
                direction = direction[0]
                std = df_by_direction.select("value").std(ddof=1).item()
                mean = df_by_direction.select("value").mean().item()
                stats_dict[f"Mean ({direction})"] = f"{mean:.4e}"
                stats_dict[f"Std ({direction})"] = (
                    None if std == 0 or std is None else f"{std:.4e}"
                )  # don't print zeros
            new_row = pl.DataFrame(stats_dict, schema=schema)
            df_stats.vstack(new_row, in_place=True)

        df_stats = df_stats.sort("Parameter")
        df_stats.write_csv(cell_type / "parameter_statistics.csv")
