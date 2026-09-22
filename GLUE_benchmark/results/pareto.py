from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent
CSV_PATH = RESULTS_DIR / "master_results.csv"

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV_PATH)

df = df[
    (df["Task"] == "SST-2")
    & (df["Model"].isin([
        "TinyLlama-1.1B",
        "Qwen3-1.7B"
    ]))
].copy()


# ============================================================
# Normalize method names
# ============================================================

df["Method"] = (
    df["Method"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["Method"] = df["Method"].replace({
    "full-ft": "full_ft",
    "full ft": "full_ft",
    "fullft": "full_ft",
    "lora+": "loraplus",
    "lora_plus": "loraplus",
})


pretty_method = {
    "full_ft": "Full-FT",
    "lora": "LoRA",
    "loraplus": "LoRA+",
    "qlora": "QLoRA",
    "bitfit": "BitFit",
}


# ============================================================
# Pareto calculation
#
# Higher accuracy = better
# Lower energy = better
# ============================================================

def is_pareto(row, data):

    for _, other in data.iterrows():

        if (
            other["Performance"] >= row["Performance"]
            and
            other["FT_Energy_Wh"] <= row["FT_Energy_Wh"]
            and
            (
                other["Performance"] > row["Performance"]
                or
                other["FT_Energy_Wh"] < row["FT_Energy_Wh"]
            )
        ):
            return False

    return True


df["Pareto"] = df.apply(
    lambda row: is_pareto(row, df),
    axis=1
)


# ============================================================
# Short labels
# ============================================================

model_short = {
    "TinyLlama-1.1B": "Tiny",
    "Qwen3-1.7B": "Qwen",
}

df["Label"] = (
    df["Model"].map(model_short)
    + " "
    + df["Method"].map(pretty_method)
)


# ============================================================
# Split groups
# ============================================================

pareto = df[df["Pareto"]].copy()
dominated = df[~df["Pareto"]].copy()


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))


# Dominated configurations
ax.scatter(
    dominated["FT_Energy_Wh"],
    dominated["Performance"],
    s=100,
    alpha=0.65,
    label="Dominated"
)


# Pareto configurations
ax.scatter(
    pareto["FT_Energy_Wh"],
    pareto["Performance"],
    s=150,
    alpha=0.95,
    edgecolors="black",
    linewidths=1,
    label="Pareto-optimal"
)


# ============================================================
# Manual label offsets
# Prevent labels from overlapping
# ============================================================

offsets = {
    "Tiny Full-FT": (8, -16),
    "Tiny LoRA": (-42, 12),
    "Tiny LoRA+": (-40, -18),
    "Tiny QLoRA": (8, -12),
    "Tiny BitFit": (8, 8),

    "Qwen Full-FT": (-55, -18),
    "Qwen LoRA": (8, -15),
    "Qwen LoRA+": (8, 10),
    "Qwen QLoRA": (-60, -15),
    "Qwen BitFit": (8, -10),
}


for _, row in df.iterrows():

    offset = offsets.get(
        row["Label"],
        (6, 6)
    )

    ax.annotate(
        row["Label"],
        (
            row["FT_Energy_Wh"],
            row["Performance"]
        ),
        xytext=offset,
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(
            arrowstyle="-",
            linewidth=0.5,
            alpha=0.5
        )
    )


# ============================================================
# Highlight high-accuracy region
# ============================================================

ax.axhline(
    y=0.90,
    linestyle="--",
    linewidth=1,
    alpha=0.6
)

ax.text(
    df["FT_Energy_Wh"].max(),
    0.905,
    "90% accuracy threshold",
    fontsize=9,
    ha="right"
)


# ============================================================
# Axes
# ============================================================

ax.set_xlabel(
    "Fine-tuning energy (Wh)  ← lower is better",
    fontsize=12
)

ax.set_ylabel(
    "SST-2 accuracy  ↑ higher is better",
    fontsize=12
)

ax.set_ylim(
    0.48,
    0.985
)


ax.set_title(
    "Accuracy–Energy Trade-off on SST-2\n"
    "RTX 5090 Reproduction",
    fontsize=14
)


ax.grid(
    alpha=0.2
)

ax.legend(
    loc="lower right",
    frameon=True
)


# ============================================================
# Explanation
# ============================================================

fig.text(
    0.5,
    0.015,
    "Pareto-optimal = no other configuration achieves both "
    "higher accuracy and lower energy. "
    "BitFit is retained for completeness despite its large accuracy drop.",
    ha="center",
    fontsize=9
)


plt.tight_layout(
    rect=[0, 0.05, 1, 1]
)


# ============================================================
# Save
# ============================================================

png_path = (
    FIGURES_DIR
    / "sst2_accuracy_energy_pareto_fixed.png"
)

pdf_path = (
    FIGURES_DIR
    / "sst2_accuracy_energy_pareto_fixed.pdf"
)


plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Print useful configurations
# ============================================================

print("\n==========================================")
print("PARETO-OPTIMAL CONFIGURATIONS")
print("==========================================\n")

print(
    pareto[
        [
            "Model",
            "Method",
            "Performance",
            "FT_Energy_Wh"
        ]
    ]
    .sort_values("FT_Energy_Wh")
    .to_string(index=False)
)


print("\n==========================================")
print("HIGH-ACCURACY CONFIGURATIONS (>= 0.90)")
print("==========================================\n")

high_accuracy = df[
    df["Performance"] >= 0.90
].sort_values(
    ["FT_Energy_Wh", "Performance"],
    ascending=[True, False]
)

print(
    high_accuracy[
        [
            "Model",
            "Method",
            "Performance",
            "FT_Energy_Wh"
        ]
    ].to_string(index=False)
)


print("\nSaved:")
print(png_path)
print(pdf_path)
