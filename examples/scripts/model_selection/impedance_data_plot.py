import json

import matplotlib.pyplot as plt
from ep_bolfi.utility.dataset_formatting import read_parquet_table
from ep_bolfi.utility.visualization import nyquist_plot

x_lims = {
    "monolayer_17_microns": (0, 300),
    "porous_42_microns": (0, 75),
    "porous_80_microns": (0, 75),
}
y_lims = {
    "monolayer_17_microns": (0, 200),
    "porous_42_microns": (0, 50),
    "porous_80_microns": (0, 50),
}

with open("../../data/Gunther2025/impedance_ocv_alignment.json") as f:
    alignment = json.load(f)

data = {}
"""
for filename, plotname in [
    ("monolayer_17_microns", "17 µm"),
    ("porous_42_microns", "42 µm"),
    ("porous_80_microns", "80 µm")
]:
    soc = alignment[filename]["Positive electrode SOC [-]"]
    ocv = alignment[filename]["OCV [V]"]
    data[filename] = read_parquet_table(filename + ".parquet", 'impedance')
    for (direction, index) in (("delithiation", 0), ("lithiation", 1)):
        fig, ax = plt.subplots(figsize=(5, 3))
        nyquist_plot(
            fig,
            ax,
            data[filename].frequencies[index::2],
            data[filename].complex_impedances[index::2],
            title_text="NMC " + plotname + " - " + direction + " direction",
            legend_text=['{:3.3g} V'.format(o) for o in ocv[index::2]]
        )
        ax.set_xlim(x_lims[filename])
        ax.set_ylim(y_lims[filename])
        plt.draw()
        legend_position = ax.get_legend().get_bbox_to_anchor().transformed(ax.transAxes.inverted())
        legend_position.x0 -= 1.25
        legend_position.x1 -= 1.25
        legend_position.y0 += 0.15
        legend_position.y1 += 0.15
        ax.get_legend().set_bbox_to_anchor(legend_position, transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(filename + "_" + direction + ".pdf", bbox_inches='tight', pad_inches=0.0)
"""
filename = "18650_LG_3500_MJ1_EIS_anode_discharge"
soc = alignment[filename]["Negative electrode SOC [-]"]
data[filename] = read_parquet_table(
    "../../data/Kuhn2026/" + filename + ".parquet", "impedance"
)
fig, ax = plt.subplots(figsize=(5, 3))
nyquist_plot(
    fig,
    ax,
    data[filename].frequencies,
    data[filename].complex_impedances,
    title_text="graphite - delithiation",
    legend_text=[f"{int(round(100 * s)):d} %" for s in soc],
)
handles = ax.get_legend().legend_handles
labels = [t._text for t in ax.get_legend().texts]
ax.legend(handles, labels, fontsize=6, ncols=2)
ax.set_xlim((0, 5))
ax.set_ylim((0, 5))
plt.draw()
legend_position = (
    ax.get_legend().get_bbox_to_anchor().transformed(ax.transAxes.inverted())
)
legend_position.x0 -= 1.25
legend_position.x1 -= 1.25
legend_position.y0 += 0.15
legend_position.y1 += 0.15
ax.get_legend().set_bbox_to_anchor(legend_position, transform=ax.transAxes)
fig.tight_layout()
fig.savefig(filename + "_delithiation.pdf", bbox_inches="tight", pad_inches=0.0)

plt.show()
