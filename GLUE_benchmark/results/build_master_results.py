import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
CSV_PATH = RESULTS_DIR / "master_results.csv"

FIELDNAMES = [
    "Task", "Model", "Method", "Params_M", "Performance",
    "FT_TFLOPs", "FT_Time_min", "FT_VRAM_MB", "FT_Power_W",
    "FT_Energy_Wh", "FT_NS", "FT_NS_E", "FT_NS_M", "FT_NS_hash",
    "INF_TFLOPs", "INF_Time_s", "INF_VRAM_MB", "INF_Power_W",
    "INF_Energy_mWh", "INF_NS", "INF_NS_E", "INF_NS_M", "INF_NS_hash"
]


def model_short_name(model_name):
    name = model_name.lower()

    if "tinyllama" in name:
        return "TinyLlama-1.1B"
    if "qwen" in name:
        return "Qwen3-1.7B"
    if "mamba2" in name:
        return "Mamba2-1.3B"
    if "mamba" in name:
        return "Mamba-1.4B"

    return model_name


rows = []

for task_dir in ["sst2", "qnli", "stsb"]:
    folder = RESULTS_DIR / task_dir

    if not folder.exists():
        continue

    for json_file in folder.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        inference = data.get("inference_stats", {})
        ft_ns = data.get("NetScore", {})
        inf_ns = data.get("NetScore_inference", {})
        flops = data.get("flops", {})

        batch_size = data.get("batch_size", 32)

        ft_tflops = (
            flops.get("training_flops_per_sequence", 0)
            * batch_size / 1e12
        )

        inf_tflops = (
            flops.get("forward_flops_per_sequence", 0)
            * batch_size / 1e12
        )

        task_name = {
            "sst2": "SST-2",
            "qnli": "QNLI",
            "stsb": "STS-B"
        }[task_dir]

        row = {
            "Task": task_name,
            "Model": model_short_name(data.get("model_name", "")),
            "Method": data.get("method", ""),

            "Params_M": data.get("trainable_parameters", 0) / 1e6,
            "Performance": data.get("fine_tuned_accuracy", ""),

            "FT_TFLOPs": ft_tflops,
            "FT_Time_min": data.get("training_time_minutes", ""),
            "FT_VRAM_MB": data.get("max_gpu_vram_used_mb", ""),
            "FT_Power_W": data.get("avg_gpu_power_watts", ""),
            "FT_Energy_Wh": data.get("estimated_energy_Wh", ""),

            "FT_NS": ft_ns.get("NS", ""),
            "FT_NS_E": ft_ns.get("NS-E", ""),
            "FT_NS_M": ft_ns.get("NS-M", ""),
            "FT_NS_hash": ft_ns.get("NS#", ""),

            "INF_TFLOPs": inf_tflops,
            "INF_Time_s": inference.get("total_time_s", ""),
            "INF_VRAM_MB": inference.get("peak_vram_mb", ""),
            "INF_Power_W": inference.get("avg_power_w", ""),
            "INF_Energy_mWh": inference.get("total_energy_wh", 0) * 1000,

            "INF_NS": inf_ns.get("NS", ""),
            "INF_NS_E": inf_ns.get("NS-E", ""),
            "INF_NS_M": inf_ns.get("NS-M", ""),
            "INF_NS_hash": inf_ns.get("NS#", ""),
        }

        rows.append(row)


with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {CSV_PATH}")