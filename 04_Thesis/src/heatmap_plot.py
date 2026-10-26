
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data

pipelines = ["AutoML-based", "Hybrid", "Manual", "LLM-based"]

hourly_rate = 16.5
development_hours = [3.6, 17.5, 18.2, 4.7]

costs = []
for hours in development_hours:
    costs.append(hours * hourly_rate)

data = [
    ("Performance, ordinary data (T)", "higher",
     [0.566, 0.564, 0.571, 0.578]),

    ("Robustness to label noise (T)", "lower",
     [0.185, 0.076, 0.072, 0.173]),

    ("Computation time (O)", "lower",
     [304.4, 310.9, 32.0, 39.4]),

    ("Reproducibility (T)", "fixed",
     [3, 4, 5, 3]),

    ("Development time (O)", "lower",
     development_hours),

    ("Cost (O)", "lower",
     costs),

    ("Expertise required (O)", "lower",
     [1, 2, 3, 1]),

    ("Data privacy (E)", "fixed",
     [5, 5, 5, 2])
]


# 2. Calculate scores

row_names = []
all_scores = []
all_labels = []

for name, rule, values in data:

    row_names.append(name)
    row_scores = []
    row_labels = []

    if rule == "higher":
        best = max(values)
    elif rule == "lower":
        best = min(values)

    for value in values:

        if rule == "higher":
            score = 1 + 4 * (value / best)

        elif rule == "lower":
            score = 1 + 4 * (best / value)

        else:
            score = value

        row_scores.append(score)

        if rule == "fixed":
            row_labels.append(str(int(score)))
        else:
            row_labels.append(f"{score:.2f}")

    all_scores.append(row_scores)
    all_labels.append(row_labels)


# 3. Create heatmap

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 6))

ax = sns.heatmap(
    all_scores,
    annot=all_labels,
    fmt="",
    cmap="Blues",
    vmin=1,
    vmax=5,
    linewidths=3,
    linecolor="white",
    xticklabels=pipelines,
    yticklabels=row_names,
    annot_kws={"size": 11, "weight": "bold"},
    cbar_kws={"label": "Score", "ticks": [1, 2, 3, 4, 5]}
)


# 4. Adjust text and labels

# Use white text on dark cells and black on light cells
for text in ax.texts:
    score = float(text.get_text())

    if score >= 3:
        text.set_color("white")
    else:
        text.set_color("black")

# Put pipeline names at the top
ax.xaxis.tick_top()

# Keep labels horizontal
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# Remove small tick marks
ax.tick_params(length=0)


# 5. Save and show the plot

plt.tight_layout()
plt.savefig("toe_heatmap_seaborn.png", dpi=300)
plt.show()
