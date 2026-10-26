
import os
import matplotlib.pyplot as plt
import numpy as np

# 1. Data

conditions = [
    "Missing\nmild",
    "Missing\nsevere",
    "Outliers\nmild",
    "Outliers\nsevere",
    "Label noise\nmild",
    "Label noise\nsevere"
]

data = {
    "AutoML-based": [0.004, -0.006, 0.000, 0.000, -0.041, -0.185],
    "Hybrid": [-0.025, -0.008, -0.001, -0.010, -0.010, -0.076],
    "Manual": [0.002, -0.007, 0.002, 0.004, 0.008, -0.072],
    "LLM-based": [-0.003, -0.004, -0.005, -0.009, -0.008, -0.173]
}

colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


# 2. Create the bar chart

plt.rcParams["font.family"] = "serif"

fig, ax = plt.subplots(figsize=(9, 4.5))

x = np.arange(len(conditions))
bar_width = 0.2

for i, name in enumerate(data):

    values = data[name]

    # Position the four bars next to each other
    positions = x + (i - 1.5) * bar_width

    ax.bar(
        positions,
        values,
        width=bar_width,
        color=colors[i],
        label=name
    )

    # Show the value for severe label noise
    last_value = values[-1]

    ax.text(
        positions[-1],
        last_value - 0.012,
        f"{last_value:+.3f}",
        ha="center",
        va="top",
        fontsize=8
    )


# 3. Customize the plot

ax.axhline(0, color="gray", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(conditions)

ax.set_ylabel("Mean change in F1 vs. clean baseline")
ax.set_ylim(-0.21, 0.03)

ax.grid(axis="y", color="#dddddd", linewidth=0.6)
ax.set_axisbelow(True)

#
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add the legend
ax.legend(
    ncol=4,
    frameon=False,
    loc="lower left",
    fontsize=9
)


# 4. Save and show the plot

output_folder = "results/figures"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "figure_3_1_f1_change.png"
)

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")

print("Saved to:", os.path.abspath(output_file))

plt.show()
