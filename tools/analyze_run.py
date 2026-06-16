#!/usr/bin/env python3
"""Auto-analyze a StepProgram run and write summary files.

Inputs: a run directory containing metrics.csv, analysis_epoch_*.json,
events_epoch_*.jsonl.

Outputs inside the run directory:
  auto_summary.json
  AUTO_SUMMARY.md

The summary is intentionally compact enough to push to GitHub reports.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_metrics(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out=[]
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rr={}
            for k,v in r.items():
                try: rr[k]=float(v)
                except Exception: rr[k]=v
            out.append(rr)
    return out


def top(rows: List[Dict[str, Any]], key: str, n: int = 20, pred=None) -> List[Dict[str, Any]]:
    xs=[r for r in rows if pred is None or pred(r)]
    xs.sort(key=lambda r: float(r.get(key, 0.0) or 0.0), reverse=True)
    return xs[:n]


def compact_row(r: Dict[str, Any]) -> Dict[str, Any]:
    keep=["type","address","prob","grad_x_gate","cost","selected","computed","top_margin","write_gate","class_read_mass","safe_mass","nontrivial_mass","expensive_mass","utility_proxy"]
    return {k:r[k] for k in keep if k in r}


def analyze(run_dir: Path, topn: int = 30) -> Dict[str, Any]:
    metrics=read_metrics(run_dir/"metrics.csv")
    analyses=sorted(run_dir.glob("analysis_epoch_*.json"))
    events=sorted(run_dir.glob("events_epoch_*.jsonl"))
    latest_analysis=read_json(analyses[-1]) if analyses else {}
    latest_events=read_jsonl(events[-1]) if events else []
    best=max(metrics, key=lambda r: float(r.get("val_acc", 0.0) or 0.0)) if metrics else {}
    last=metrics[-1] if metrics else {}

    by_type=Counter(str(r.get("type","unknown")) for r in latest_events)
    computed_prims=[r for r in latest_events if r.get("type")=="primitive" and r.get("computed") is True]
    skipped_prims=[r for r in latest_events if r.get("type")=="primitive" and r.get("computed") is False]
    selected_prims=[r for r in latest_events if r.get("type")=="primitive" and r.get("selected") is True]
    utility_rows=[r for r in latest_events if r.get("type")=="utility"]

    suspicious=[]
    for r in utility_rows:
        if float(r.get("class_read_mass",0) or 0)>0.025 and float(r.get("safe_mass",0) or 0)>0.55:
            suspicious.append({"kind":"class_reads_safe_slot", **compact_row(r)})
        if float(r.get("expensive_mass",0) or 0)>0.50 and float(r.get("utility_proxy",0) or 0)<0.002:
            suspicious.append({"kind":"expensive_low_utility", **compact_row(r)})
        if float(r.get("write_gate",0) or 0)>0.10 and float(r.get("class_read_mass",0) or 0)<0.005:
            suspicious.append({"kind":"writes_but_class_barely_reads", **compact_row(r)})

    competition=top([r for r in latest_events if r.get("type")=="primitive"], "prob", topn, pred=lambda r: float(r.get("top_margin",1) or 1)<0.06)
    expensive_low_grad=top(computed_prims, "cost", topn, pred=lambda r: float(r.get("cost",0) or 0)>0.7 and float(r.get("grad_x_gate",0) or 0)<1e-6)
    suppressed_high_prob=top(skipped_prims, "prob", topn, pred=lambda r: float(r.get("prob",0) or 0)>0.10)

    result={
        "run_dir": str(run_dir),
        "num_epochs": len(metrics),
        "best": best,
        "last": last,
        "latest_analysis_file": str(analyses[-1]) if analyses else None,
        "latest_events_file": str(events[-1]) if events else None,
        "event_counts_by_type": dict(by_type),
        "exec_stats": latest_analysis.get("program_report",{}).get("exec_stats",{}),
        "top_grad": [compact_row(r) for r in top(latest_events, "grad_x_gate", topn)],
        "top_utility": [compact_row(r) for r in top(utility_rows, "utility_proxy", topn)],
        "low_utility_writers": [compact_row(r) for r in top(utility_rows, "write_gate", topn, pred=lambda r: float(r.get("utility_proxy",0) or 0)<0.002)],
        "competing_primitives_low_margin": [compact_row(r) for r in competition],
        "expensive_low_grad_computed": [compact_row(r) for r in expensive_low_grad],
        "skipped_high_prob_primitives": [compact_row(r) for r in suppressed_high_prob],
        "suspicious": suspicious[:topn],
    }
    return result


def md_table(rows: List[Dict[str, Any]], cols: List[str], limit: int = 10) -> str:
    if not rows:
        return "_none_\n"
    out=["| " + " | ".join(cols) + " |", "|" + "|".join(["---"]*len(cols)) + "|"]
    for r in rows[:limit]:
        vals=[]
        for c in cols:
            v=r.get(c,"")
            if isinstance(v,float): v=f"{v:.6g}"
            vals.append(str(v).replace("|","/"))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)+"\n"


def write_markdown(run_dir: Path, summary: Dict[str, Any]) -> None:
    best=summary.get("best",{}) or {}; last=summary.get("last",{}) or {}
    lines=[]
    lines.append(f"# Auto summary: `{run_dir.name}`\n")
    lines.append("## Quality\n")
    lines.append(f"- epochs: `{summary.get('num_epochs',0)}`")
    lines.append(f"- best val acc: `{float(best.get('val_acc',0) or 0)*100:.2f}%` at epoch `{int(best.get('epoch',0) or 0)}`")
    lines.append(f"- last val acc: `{float(last.get('val_acc',0) or 0)*100:.2f}%`")
    lines.append(f"- last train acc: `{float(last.get('train_acc',0) or 0)*100:.2f}%`")
    lines.append(f"- exec stats: `{summary.get('exec_stats',{})}`\n")
    lines.append("## Top gradient addresses\n")
    lines.append(md_table(summary.get("top_grad",[]), ["type","address","grad_x_gate","prob","cost","computed","selected"], 15))
    lines.append("## Top utility slots\n")
    lines.append(md_table(summary.get("top_utility",[]), ["address","utility_proxy","write_gate","class_read_mass","safe_mass","expensive_mass"], 15))
    lines.append("## Low utility writers\n")
    lines.append(md_table(summary.get("low_utility_writers",[]), ["address","write_gate","utility_proxy","class_read_mass","safe_mass","expensive_mass"], 15))
    lines.append("## Competing primitives: low top margin\n")
    lines.append(md_table(summary.get("competing_primitives_low_margin",[]), ["address","prob","top_margin","computed","selected","cost","grad_x_gate"], 15))
    lines.append("## Expensive computed with low gradient\n")
    lines.append(md_table(summary.get("expensive_low_grad_computed",[]), ["address","cost","prob","grad_x_gate","computed","selected"], 15))
    lines.append("## Skipped but high probability\n")
    lines.append(md_table(summary.get("skipped_high_prob_primitives",[]), ["address","prob","computed","selected","cost","top_margin"], 15))
    lines.append("## Suspicious\n")
    lines.append(md_table(summary.get("suspicious",[]), ["kind","address","utility_proxy","write_gate","class_read_mass","safe_mass","expensive_mass"], 20))
    (run_dir/"AUTO_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--top", type=int, default=30)
    args=p.parse_args()
    run_dir=Path(args.run_dir)
    s=analyze(run_dir, args.top)
    (run_dir/"auto_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(run_dir, s)
    print(run_dir/"AUTO_SUMMARY.md")
    print(run_dir/"auto_summary.json")

if __name__ == "__main__":
    main()
