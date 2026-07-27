from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .catalog import ROOT, get_lab, list_labs


def _run_directories(root: Path, lab_id: str | None = None) -> list[Path]:
    base = root / "runs"
    lab_ids = [lab_id] if lab_id else list_labs()
    paths: list[Path] = []
    for current in lab_ids:
        lab_root = base / current
        if lab_root.exists():
            paths.extend(path for path in lab_root.iterdir() if path.is_dir() and (path / "metrics.json").is_file())
    return sorted(paths)


def collect_runs(root: Path = ROOT, lab_id: str | None = None) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for run_dir in _run_directories(root, lab_id):
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config = {}
        if (run_dir / "config.yaml").is_file():
            import yaml
            config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
        current_lab = run_dir.parent.name
        record: dict[str, Any] = {
            "lab": current_lab,
            "title": get_lab(current_lab)["title"],
            "run": run_dir.name,
            "run_dir": str(run_dir),
            "config": config.get("_config_name", config.get("config_name", "unknown")),
            "seed": config.get("seed"),
            "device": config.get("device"),
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)):
                record[key] = value
        records.append(record)
    return pd.DataFrame(records)


def build_leaderboard(root: Path = ROOT, output: Path | None = None) -> dict[str, Any]:
    frame = collect_runs(root)
    output = output or root / "reports" / "leaderboard"
    output.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame.to_csv(output / "leaderboard.csv", index=False)
        (output / "leaderboard.md").write_text("# Leaderboard\n\nNo hay ejecuciones registradas.\n", encoding="utf-8")
        return {"runs": 0, "output": str(output)}
    rows: list[pd.Series] = []
    for lab_id, group in frame.groupby("lab", sort=True):
        metric = str(get_lab(lab_id).get("selection_metric") or "")
        candidates = [metric, "macro_f1", "f1", "accuracy", "r2", "pr_auc", "mae", "rmse", "loss"]
        selected = next((name for name in candidates if name and name in group.columns and group[name].notna().any()), None)
        if selected is None:
            rows.append(group.iloc[-1])
            continue
        ascending = selected in {"mae", "rmse", "mape", "loss", "ece", "brier"}
        rows.append(group.sort_values(selected, ascending=ascending).iloc[0])
    leaderboard = pd.DataFrame(rows).sort_values("lab")
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    visible = [column for column in ["lab", "title", "run", "accuracy", "macro_f1", "f1", "pr_auc", "r2", "mae", "rmse", "wall_time_seconds", "parameters"] if column in leaderboard.columns]
    markdown = "# Leaderboard\n\n" + leaderboard[visible].to_markdown(index=False) + "\n"
    (output / "leaderboard.md").write_text(markdown, encoding="utf-8")
    (output / "leaderboard.json").write_text(leaderboard.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")
    return {"runs": len(frame), "labs": leaderboard["lab"].nunique(), "output": str(output)}


def compare_runs(run_paths: list[Path], output: Path) -> Path:
    records: list[dict[str, Any]] = []
    for path in run_paths:
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        records.append({"run": str(path), **{key: value for key, value in metrics.items() if isinstance(value, (int, float, str, bool))}})
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(frame.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")
    else:
        frame.to_csv(output, index=False)
    return output


def summarize_benchmark(records: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted({
        key
        for record in records
        for key, value in (record.get("metrics") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    })
    summary: dict[str, Any] = {"runs": len(records), "metrics": {}}
    for key in numeric_keys:
        values = [float(record["metrics"][key]) for record in records if isinstance(record.get("metrics", {}).get(key), (int, float))]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        std = variance ** 0.5 if len(values) > 1 else 0.0
        margin = 1.96 * std / max(1.0, len(values) ** 0.5)
        summary["metrics"][key] = {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "normal_95_lower": mean - margin,
            "normal_95_upper": mean + margin,
            "values": values,
        }
    return summary


def write_benchmark_report(lab_id: str, records: list[dict[str, Any]], root: Path = ROOT) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = root / "reports" / "benchmarks" / lab_id / stamp
    output.mkdir(parents=True, exist_ok=False)
    summary = summarize_benchmark(records)
    payload = {"lab": lab_id, "title": get_lab(lab_id)["title"], "created_at_utc": datetime.now(timezone.utc).isoformat(), "summary": summary, "runs": records}
    (output / "benchmark.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    lines = [f"# Benchmark — {get_lab(lab_id)['title']}", "", f"Ejecuciones: {len(records)}", "", "| Métrica | Media | Desv. estándar | Mínimo | Máximo |", "|---|---:|---:|---:|---:|"]
    for metric, values in summary["metrics"].items():
        lines.append(f"| {metric} | {values['mean']:.6g} | {values['std']:.6g} | {values['min']:.6g} | {values['max']:.6g} |")
    lines += ["", "La banda normal aproximada es orientativa; con pocas semillas no sustituye un análisis estadístico más amplio."]
    (output / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
