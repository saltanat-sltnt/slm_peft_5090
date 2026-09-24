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

TASKS = ["SST-2", "QNLI"]
MODELS = ["TinyLlama-1.1B", "Qwen3-1.7B"]

df = df[
    (df["Task"].isin(TASKS))
    & (df["Model"].isin(MODELS))
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
    "full_ft": "full_ft",
    "lora": "lora",
    "lora+": "loraplus",
    "lora_plus": "loraplus",
    "loraplus": "loraplus",
    "qlora": "qlora",
    "bitfit": "bitfit",
})


pretty_method = {
    "full_ft": "Full-FT",
    "lora": "LoRA",
    "loraplus": "LoRA+",
    "qlora": "QLoRA",
    "bitfit": "BitFit",
}


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
# Pareto calculation
#
# Higher accuracy = better
# Lower energy = better
#
# IMPORTANT:
# Pareto optimality is calculated separately for each task.
# SST-2 and QNLI should not dominate each other because they
# are different benchmark tasks.
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


# ============================================================
# Manual label offsets
# ============================================================

TASK_OFFSETS = {
    "SST-2": {
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
    },

    "QNLI": {
        "Tiny Full-FT": (8, 8),
        "Tiny LoRA": (-48, 10),
        "Tiny LoRA+": (-48, -18),
        "Tiny QLoRA": (8, -12),
        "Tiny BitFit": (8, 8),

        "Qwen Full-FT": (-58, -16),
        "Qwen LoRA": (8, -16),
        "Qwen LoRA+": (8, 10),
        "Qwen QLoRA": (-62, -16),
        "Qwen BitFit": (8, -10),
    },
}


# ============================================================
# Plot one Pareto figure per task
# ============================================================

for task in TASKS:

    task_df = df[
        df["Task"] == task
    ].copy()

    if task_df.empty:
        print(f"\nNo rows found for {task}. Skipping.")
        continue

    task_df["Pareto"] = task_df.apply(
        lambda row: is_pareto(row, task_df),
        axis=1
    )

    pareto = task_df[
        task_df["Pareto"]
    ].copy()

    dominated = task_df[
        ~task_df["Pareto"]
    ].copy()


    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )


    # Dominated configurations
    ax.scatter(
        dominated["FT_Energy_Wh"],
        dominated["Performance"],
        s=100,
        alpha=0.65,
        label="Dominated"
    )


    # Pareto-optimal configurations
    ax.scatter(
        pareto["FT_Energy_Wh"],
        pareto["Performance"],
        s=150,
        alpha=0.95,
        edgecolors="black",
        linewidths=1,
        label="Pareto-optimal"
    )


    # ========================================================
    # Labels
    # ========================================================

    offsets = TASK_OFFSETS.get(
        task,
        {}
    )

    for _, row in task_df.iterrows():

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


    # ========================================================
    # Highlight high-accuracy region
    # ========================================================

    ax.axhline(
        y=0.90,
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )

    ax.text(
        task_df["FT_Energy_Wh"].max(),
        0.905,
        "90% accuracy threshold",
        fontsize=9,
        ha="right"
    )


    # ========================================================
    # Axes
    # ========================================================

    ax.set_xlabel(
        "Fine-tuning energy (Wh)  ← lower is better",
        fontsize=12
    )

    ax.set_ylabel(
        f"{task} accuracy  ↑ higher is better",
        fontsize=12
    )

    ax.set_ylim(
        0.48,
        0.985
    )

    ax.set_title(
        f"Accuracy–Energy Trade-off on {task}\n"
        "RTX 5090 Reproduction — Transformer Models",
        fontsize=14
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        loc="lower right",
        frameon=True
    )


    # ========================================================
    # Explanation
    # ========================================================

    fig.text(
        0.5,
        0.015,
        "Pareto-optimal = no other configuration on the same task achieves both "
        "higher accuracy and lower energy. "
        "BitFit is retained for completeness despite its accuracy drop.",
        ha="center",
        fontsize=9
    )

    plt.tight_layout(
        rect=[0, 0.05, 1, 1]
    )


    # ========================================================
    # Save
    # ========================================================

    task_slug = (
        task
        .lower()
        .replace("-", "")
    )

    png_path = (
        FIGURES_DIR
        / f"trans_{task_slug}_pareto.png"
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ========================================================
    # Print useful configurations
    # ========================================================

    print("\n==========================================")
    print(f"{task}: PARETO-OPTIMAL CONFIGURATIONS")
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
    print(f"{task}: HIGH-ACCURACY CONFIGURATIONS (>= 0.90)")
    print("==========================================\n")

    high_accuracy = task_df[
        task_df["Performance"] >= 0.90
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
