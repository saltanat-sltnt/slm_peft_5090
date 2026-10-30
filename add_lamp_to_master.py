#!/usr/bin/env python
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

if (HERE / "GLUE_benchmark").exists() and (HERE / "LaMP_benchmark").exists():
    ROOT = HERE
elif (HERE.parent / "GLUE_benchmark").exists() and (HERE.parent / "LaMP_benchmark").exists():
    ROOT = HERE.parent
else:
    raise FileNotFoundError("Put this script in the repo root or inside GLUE_benchmark/LaMP_benchmark.")

MASTER_CSV = ROOT / "GLUE_benchmark" / "results" / "master_results.csv"
LAMP_OUTPUTS = ROOT / "LaMP_benchmark" / "outputs"

TASK_NAMES = {"lamp1":"LaMP-1","lamp2":"LaMP-2","lamp3":"LaMP-3","lamp4":"LaMP-4"}
MODEL_NAMES = {
    "tinyllama":"TinyLlama-1.1B",
    "qwen3":"Qwen3-1.7B",
    "qwen":"Qwen3-1.7B",
    "mamba1":"Mamba-1.4B",
    "mamba2":"Mamba2-1.3B",
}
METHOD_NAMES = {
    "full_ft":"Full-FT","full-ft":"Full-FT",
    "lora":"LoRA","loraplus":"LoRA+","lora+":"LoRA+",
    "qlora":"QLoRA","bitfit":"BitFit",
}

ALIASES = {
    "Task":["Task"], "Model":["Model"], "Method":["Method"],
    "Params_M":["Params_M","Trainable_Params_M","Parameters_M"],
    "Performance":["Performance","Perf","Perf."],
    "FT_TFLOPs":["FT_TFLOPs","FT_FLOPs_TF","FT_TF"],
    "FT_Time_min":["FT_Time_min"],
    "FT_VRAM_MB":["FT_VRAM_MB"],
    "FT_Power_W":["FT_Power_W"],
    "FT_Energy_Wh":["FT_Energy_Wh"],
    "FT_NS":["FT_NS","NS"],
    "FT_NS_E":["FT_NS_E","FT_NS-E","NS-E"],
    "FT_NS_M":["FT_NS_M","FT_NS-M","NS-M"],
    "FT_NS_hash":["FT_NS_hash","FT_NS#","NS#"],
    "INF_TFLOPs":["INF_TFLOPs","INF_FLOPs_TF","INF_TF"],
    "INF_Time_s":["INF_Time_s","Inference_Time_s","INF_Time"],
    "INF_VRAM_MB":["INF_VRAM_MB"],
    "INF_Power_W":["INF_Power_W"],
    "INF_Energy_mWh":["INF_Energy_mWh"],
    "INF_NS":["INF_NS"],
    "INF_NS_E":["INF_NS_E","INF_NS-E"],
    "INF_NS_M":["INF_NS_M","INF_NS-M"],
    "INF_NS_hash":["INF_NS_hash","INF_NS#"],
}

def val(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def fmt(x, n=6):
    if x is None:
        return ""
    return round(float(x), n)

if not MASTER_CSV.exists():
    raise FileNotFoundError(f"Missing: {MASTER_CSV}")
if not LAMP_OUTPUTS.exists():
    raise FileNotFoundError(f"Missing: {LAMP_OUTPUTS}")

with MASTER_CSV.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    existing = list(reader)

resolved = {}
for canon, names in ALIASES.items():
    for name in names:
        if name in fieldnames:
            resolved[canon] = name
            break

for req in ("Task","Model","Method"):
    if req not in resolved:
        raise ValueError(f"Could not map required column {req}. Existing columns: {fieldnames}")

rows = []
seen = set()

for p in sorted(LAMP_OUTPUTS.rglob("benchmark_results_*.json")):
    low_parts = {x.lower() for x in p.parts}
    if "debug" in low_parts or "smoke" in low_parts:
        continue

    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)

    task = str(d.get("task","")).lower()
    model = str(d.get("model_type","")).lower()
    method = str(d.get("method","")).lower()

    if task not in TASK_NAMES:
        continue

    hp = d.get("hyperparameters", {}) or {}
    flops = d.get("flops", {}) or {}
    ns = d.get("NetScore", {}) or {}
    nsi = d.get("NetScore_inference", {}) or {}
    inf = d.get("inference_stats", {}) or {}

    bs = hp.get("batch_size", 4) or 4
    ft_seq = flops.get("training_flops_per_sequence")
    inf_seq = flops.get("inference_flops_per_sequence")
    trainable = d.get("trainable_parameters")

    c = {
        "Task": TASK_NAMES[task],
        "Model": MODEL_NAMES.get(model, d.get("model_name", model)),
        "Method": METHOD_NAMES.get(method, method),
        "Params_M": fmt(trainable / 1e6, 5) if trainable is not None else "",
        "Performance": fmt(d.get("primary_metric_value"), 6),
        "FT_TFLOPs": fmt(ft_seq * bs / 1e12, 4) if ft_seq is not None else "",
        "FT_Time_min": fmt(d.get("training_time_minutes"), 4),
        "FT_VRAM_MB": fmt(d.get("max_gpu_vram_used_mb"), 4),
        "FT_Power_W": fmt(d.get("avg_gpu_power_watts"), 4),
        "FT_Energy_Wh": fmt(d.get("estimated_energy_Wh"), 6),
        "FT_NS": fmt(ns.get("NS"), 4),
        "FT_NS_E": fmt(ns.get("NS-E"), 4),
        "FT_NS_M": fmt(ns.get("NS-M"), 4),
        "FT_NS_hash": fmt(ns.get("NS#"), 4),
        "INF_TFLOPs": fmt(inf_seq * bs / 1e12, 4) if inf_seq is not None else "",
        "INF_Time_s": fmt(inf.get("total_time_s"), 4),
        "INF_VRAM_MB": fmt(inf.get("peak_vram_mb"), 4),
        "INF_Power_W": fmt(inf.get("avg_power_w"), 4),
        "INF_Energy_mWh": fmt(inf.get("total_energy_wh") * 1000, 6) if inf.get("total_energy_wh") is not None else "",
        "INF_NS": fmt(nsi.get("NS"), 4),
        "INF_NS_E": fmt(nsi.get("NS-E"), 4),
        "INF_NS_M": fmt(nsi.get("NS-M"), 4),
        "INF_NS_hash": fmt(nsi.get("NS#"), 4),
    }

    out = {col:"" for col in fieldnames}
    for canon, value in c.items():
        actual = resolved.get(canon)
        if actual:
            out[actual] = value

    key = (out[resolved["Task"]], out[resolved["Model"]], out[resolved["Method"]])
    if key in seen:
        continue
    seen.add(key)
    rows.append(out)

keys = {(r[resolved["Task"]], r[resolved["Model"]], r[resolved["Method"]]) for r in rows}

kept = []
for r in existing:
    key = (r.get(resolved["Task"],""), r.get(resolved["Model"],""), r.get(resolved["Method"],""))
    if key not in keys:
        kept.append(r)

backup = MASTER_CSV.with_name("master_results_before_lamp.csv")
backup.write_bytes(MASTER_CSV.read_bytes())

with MASTER_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept + rows)

print(f"Added/updated {len(rows)} LaMP rows")
print(f"Updated: {MASTER_CSV}")
print(f"Backup:  {backup}")
