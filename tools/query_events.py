#!/usr/bin/env python3
"""Query StepProgram JSONL events.

Examples:
  python tools/query_events.py runs/x/events_epoch_001.jsonl --top 20 --sort grad_x_gate
  python tools/query_events.py runs/x/events_epoch_001.jsonl --type primitive --computed false
  python tools/query_events.py runs/x/events_epoch_001.jsonl --type utility --where expensive_mass'>0.5' --sort utility_proxy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_value(x: str):
    if x.lower() == "true": return True
    if x.lower() == "false": return False
    try: return int(x)
    except Exception: pass
    try: return float(x)
    except Exception: return x


def load_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows=[]
    for p in paths:
        with Path(p).open("r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def apply_where(rows: List[Dict[str, Any]], expr: str) -> List[Dict[str, Any]]:
    # Tiny safe-ish filter: field OP value, OP in >= <= > < == !=
    ops=[">=","<=","!=","==",">","<"]
    op=None
    for cand in ops:
        if cand in expr:
            op=cand; break
    if not op:
        return rows
    key,val=expr.split(op,1); key=key.strip(); val=parse_value(val.strip().strip('"').strip("'"))
    def ok(r):
        a=r.get(key, None)
        if a is None: return False
        try:
            if op==">=": return a>=val
            if op=="<=": return a<=val
            if op==">": return a>val
            if op=="<": return a<val
            if op=="==": return a==val
            if op=="!=": return a!=val
        except Exception:
            return False
        return False
    return [r for r in rows if ok(r)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--type", default="")
    p.add_argument("--address-contains", default="")
    p.add_argument("--where", action="append", default=[])
    p.add_argument("--sort", default="grad_x_gate")
    p.add_argument("--desc", action="store_true", default=True)
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--computed", choices=["true","false","any"], default="any")
    p.add_argument("--selected", choices=["true","false","any"], default="any")
    args=p.parse_args()

    rows=load_rows(args.paths)
    if args.type:
        rows=[r for r in rows if r.get("type")==args.type]
    if args.address_contains:
        rows=[r for r in rows if args.address_contains in str(r.get("address",""))]
    if args.computed != "any":
        want=args.computed == "true"
        rows=[r for r in rows if r.get("computed") is want]
    if args.selected != "any":
        want=args.selected == "true"
        rows=[r for r in rows if r.get("selected") is want]
    for w in args.where:
        rows=apply_where(rows,w)
    rows.sort(key=lambda r: float(r.get(args.sort, 0.0) or 0.0), reverse=True)
    for r in rows[:args.top]:
        print(json.dumps(r, ensure_ascii=False))

if __name__ == "__main__":
    main()
