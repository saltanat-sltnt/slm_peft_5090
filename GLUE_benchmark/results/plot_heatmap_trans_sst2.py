from pathlib import Path

import numpy as np
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
# Original paper values
# Table IV — SST-2
# ============================================================

paper_data = [
    # Model, Method, Performance, Time_min, VRAM_MB, Power_W, Energy_Wh

    ["TinyLlama-1.1B", "full_ft",  0.954, 34.01, 22850.4, 314.50, 178.27],
    ["TinyLlama-1.1B", "lora",     0.959, 19.18, 16479.8, 261.43, 83.57],
    ["TinyLlama-1.1B", "loraplus", 0.957, 19.33, 15957.0, 268.84, 86.61],
    ["TinyLlama-1.1B", "qlora",    0.952, 31.05, 4225.2,  284.56, 147.26],
    ["TinyLlama-1.1B", "bitfit",   0.920, 18.85, 17950.0, 288.41, 90.61],

    ["Qwen3-1.7B", "full_ft",  0.955, 51.56, 22586.7, 338.59, 290.96],
    ["Qwen3-1.7B", "lora",     0.955, 21.46, 23977.5, 360.31, 128.87],
    ["Qwen3-1.7B", "loraplus", 0.962, 22.30, 20047.6, 350.26, 130.18],
    ["Qwen3-1.7B", "qlora",    0.958, 34.17, 7808.1,  376.07, 214.17],
    ["Qwen3-1.7B", "bitfit",   0.900, 22.27, 18683.6, 333.06, 123.62],
]


paper_columns = [
    "Model",
    "Method",
    "Performance",
    "FT_Time_min",
    "FT_VRAM_MB",
    "FT_Power_W",
    "FT_Energy_Wh",
]

paper = pd.DataFrame(
    paper_data,
    columns=paper_columns
)


# ============================================================
# Load reproduction results
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Could not find master results CSV:\n{CSV_PATH}"
    )

repro = pd.read_csv(CSV_PATH)


# Keep only SST-2 + completed Transformer models
repro = repro[
    (repro["Task"] == "SST-2")
    & (
        repro["Model"].isin(
            ["TinyLlama-1.1B", "Qwen3-1.7B"]
        )
    )
].copy()


# ============================================================
# Normalize method names
# ============================================================

repro["Method"] = (
    repro["Method"]
    .astype(str)
    .str.strip()
    .str.lower()
)

repro["Method"] = repro["Method"].replace({
    "full-ft": "full_ft",
    "full ft": "full_ft",
    "fullft": "full_ft",
    "full_ft": "full_ft",

    "lora": "lora",

    "lora+": "loraplus",
    "lora_plus": "loraplus",
    "loraplus": "loraplus",

    "qlora": "qlora",

    "bitfit": "bitfit",
})


# ============================================================
# Metrics to compare
# ============================================================

metrics = [
    "Performance",
    "FT_Time_min",
    "FT_VRAM_MB",
    "FT_Power_W",
    "FT_Energy_Wh",
]


# Make sure required columns exist
required_columns = [
    "Model",
    "Method",
] + metrics

missing_columns = [
    col
    for col in required_columns
    if col not in repro.columns
]

if missing_columns:
    raise ValueError(
        "Missing columns in master_results.csv:\n"
        + ", ".join(missing_columns)
    )


# ============================================================
# Merge reproduction with paper values
# ============================================================

merged = repro[
    ["Model", "Method"] + metrics
].merge(
    paper,
    on=["Model", "Method"],
    suffixes=("_repro", "_paper"),
)


if merged.empty:
    print("\nNo matching rows found.")
    print("\nRows found in master_results.csv:")
    print(
        repro[["Model", "Method"]]
        .to_string(index=False)
    )
    raise SystemExit


# ============================================================
# Percentage difference
#
# Δ% = (RTX5090 - Paper) / Paper * 100
#
# Positive:
# reproduction value is higher
#
# Negative:
# reproduction value is lower
# ============================================================

for metric in metrics:

    repro_col = f"{metric}_repro"
    paper_col = f"{metric}_paper"

    merged[f"{metric}_pct"] = (
        (
            merged[repro_col]
            - merged[paper_col]
        )
        / merged[paper_col]
    ) * 100


# ============================================================
# Sort rows
# ============================================================

model_order = {
    "TinyLlama-1.1B": 0,
    "Qwen3-1.7B": 1,
}

method_order = {
    "full_ft": 0,
    "lora": 1,
    "loraplus": 2,
    "qlora": 3,
    "bitfit": 4,
}


merged["model_order"] = (
    merged["Model"]
    .map(model_order)
)

merged["method_order"] = (
    merged["Method"]
    .map(method_order)
)


merged = (
    merged
    .sort_values(
        ["model_order", "method_order"]
    )
    .reset_index(drop=True)
)


# ============================================================
# Pretty labels
# ============================================================

pretty_method = {
    "full_ft": "Full-FT",
    "lora": "LoRA",
    "loraplus": "LoRA+",
    "qlora": "QLoRA",
    "bitfit": "BitFit",
}


merged["Configuration"] = (
    merged["Model"]
    + " | "
    + merged["Method"].map(pretty_method)
)


# ============================================================
# Build heatmap matrix
# ============================================================

heatmap_columns = {
    "Performance_pct": "Performance",
    "FT_Time_min_pct": "Time",
    "FT_VRAM_MB_pct": "VRAM",
    "FT_Power_W_pct": "Power",
    "FT_Energy_Wh_pct": "Energy",
}


heatmap = merged[
    list(heatmap_columns.keys())
].copy()

heatmap.columns = (
    list(heatmap_columns.values())
)


values = heatmap.to_numpy(
    dtype=float
)


# Remove accidental inf values
values = np.where(
    np.isfinite(values),
    values,
    np.nan
)


if np.all(np.isnan(values)):
    raise ValueError(
        "All calculated values are NaN."
    )


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 7)
)


# Symmetric scale around zero
max_abs = np.nanmax(
    np.abs(values)
)

if max_abs == 0:
    max_abs = 1


image = ax.imshow(
    values,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-max_abs,
    vmax=max_abs,
)


# ============================================================
# X axis
# ============================================================

ax.set_xticks(
    np.arange(
        len(heatmap.columns)
    )
)

ax.set_xticklabels(
    heatmap.columns,
    fontsize=11
)


# ============================================================
# Y axis
# ============================================================

ax.set_yticks(
    np.arange(
        len(merged)
    )
)

ax.set_yticklabels(
    merged["Configuration"],
    fontsize=10
)


# ============================================================
# Percentage labels inside cells
# ============================================================

for i in range(
    values.shape[0]
):

    for j in range(
        values.shape[1]
    ):

        value = values[i, j]

        if np.isnan(value):
            text = "N/A"
            text_color = "black"

        else:
            text = f"{value:+.1f}%"

            if abs(value) > (
                0.55 * max_abs
            ):
                text_color = "white"
            else:
                text_color = "black"

        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
        )


# ============================================================
# Separator between models
# ============================================================

tiny_count = (
    merged["Model"]
    == "TinyLlama-1.1B"
).sum()

if (
    tiny_count > 0
    and tiny_count < len(merged)
):
    ax.axhline(
        y=tiny_count - 0.5,
        linewidth=1.5,
        color="black",
    )


# ============================================================
# Title
# ============================================================

ax.set_title(
    "RTX 5090 Reproduction vs. Original RTX 4090 Results\n"
    "SST-2 Fine-Tuning Percentage Difference",
    fontsize=13,
    pad=14,
)


# ============================================================
# Color bar
# ============================================================

cbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.035,
    pad=0.03,
)

cbar.set_label(
    "Difference from original paper (%)",
    rotation=270,
    labelpad=20,
)


# ============================================================
# Explanation
# ============================================================

fig.text(
    0.5,
    0.015,
    "Δ% = (RTX 5090 reproduction − original RTX 4090 result) "
    "/ original result × 100. "
    "For time, VRAM, power and energy, negative values mean lower cost.",
    ha="center",
    fontsize=9,
)


plt.tight_layout(
    rect=[0, 0.055, 1, 1]
)


# ============================================================
# Save image
# ============================================================

png_path = (
    FIGURES_DIR
    / "paper_vs_rtx5090_heatmap_trans_sst2.png"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)


# IMPORTANT:
# Do not use plt.show().
# On your Windows environment it caused the
# Matplotlib cursor normalization error.

plt.close(fig)


# ============================================================
# Print results to terminal
# ============================================================

print("\n==========================================")
print("HEATMAP CREATED SUCCESSFULLY")
print("==========================================")

print("\nSaved:")
print(png_path)


output_columns = [
    "Model",
    "Method",
    "Performance_pct",
    "FT_Time_min_pct",
    "FT_VRAM_MB_pct",
    "FT_Power_W_pct",
    "FT_Energy_Wh_pct",
]


print("\nPercentage differences:\n")

print(
    merged[
        output_columns
    ]
    .round(1)
    .to_string(
        index=False
    )
)
