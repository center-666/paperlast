"""Extract the locked four-cell primary table from the path-level analysis."""
import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "task4_confirm"
GROUP = "U_unstable_R35x2_primary"
CELLS = (
    ("iid", "A_q0_raw"),
    ("iid", "S_fixed_table_T"),
    ("no_repeat", "A_q0_raw"),
    ("no_repeat", "S_fixed_table_T"),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = []
    with (OUT / "absolute.csv").open(encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            if source["configuration"] != GROUP or source["fixture"] != "equal_weighted":
                continue
            if (source["watermark"], source["fusion"]) not in CELLS:
                continue
            rows.append({
                "configuration": source["configuration"],
                "watermark": source["watermark"],
                "fusion": source["fusion"],
                "metric": source["metric"],
                "mean": source["mean"],
                "lower95": source["lower95"],
                "upper95": source["upper95"],
                "n": source["n"],
            })
    expected = 4 * 8
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} primary rows, observed {len(rows)}.")
    pairs = {(row["watermark"], row["fusion"]) for row in rows}
    if pairs != set(CELLS):
        raise ValueError("The registered four-cell table is incomplete.")
    destination = OUT / "main_2x2.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "PASS_four_cell_main_table",
        "group": GROUP,
        "cells": [list(cell) for cell in CELLS],
        "rows": len(rows),
        "source_sha256": sha(OUT / "absolute.csv"),
        "table_sha256": sha(destination),
        "source_analysis_sha256": sha(OUT / "ANALYSIS.json"),
        "interpretation": "Point estimates and path-bootstrap intervals are read from the locked analysis output. This extraction does not rerun or select a strategy.",
    }
    (OUT / "MAIN_2X2_VERIFY.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
