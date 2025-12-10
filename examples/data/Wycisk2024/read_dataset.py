from pyarrow import parquet
from pybamm import citations

from pybop import Datasets


def read_parquet_cycling(filename):
    datafile = parquet.ParquetFile(filename)
    indices = []
    timepoints = []
    currents = []
    voltages = []
    for row_group_number in range(datafile.num_row_groups):
        row_group = datafile.read_row_group(row_group_number)
        indices.append(row_group.column("indices")[0].as_py())
        timepoints.append(
            row_group.column("timepoints [s]").combine_chunks().to_numpy().tolist()
        )
        currents.append(
            row_group.column("currents [A]").combine_chunks().to_numpy().tolist()
        )
        voltages.append(
            row_group.column("voltages [V]").combine_chunks().to_numpy().tolist()
        )
    return Datasets(
        [
            {
                "Time [s]": t,
                "Current function [A]": c,
                "Voltage change [V]": v,
                "Cycle indices": [i] * len(t),
            }
            for t, c, v, i in zip(timepoints, currents, voltages, indices)
        ],
        domain="Time [s]",
        control_variable="Current function [A]",
    )


citations.register("""@article{
    Wycisk2024,
    title={{Challenges of open-circuit voltage measurements for silicon-containing Li-Ion cells}},
    author={Wycisk, D and Mertin, G and Oldenburger, M and von Kessel, O and Latz, A},
    journal={Journal of Energy Storage},
    volume={89},
    pages={111617},
    year={2024},
    doi={10.1016/j.est.2024.111617}
}""")

gitt_on_graphite_with_5_percent_silicon = read_parquet_cycling(
    "../../data/Wycisk2024/gitt_on_graphite_with_5_percent_silicon.parquet"
)
