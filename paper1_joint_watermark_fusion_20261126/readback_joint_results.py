"""Independently reconstruct primary cells and every scored M7 violation from archives."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "fusion_attribution"
PRIMARY = "U_unstable_R35x2"
POLICIES = ("A_q0_raw", "A_M7_min", "S_fixed_table_T", "q1_full")
WATERMARKS = ("iid", "no_repeat")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    folder = parser.parse_args().archives
    manifest = json.loads((HERE/"task4_confirm/ARCHIVE_MANIFEST.json").read_text())
    run = json.loads((HERE/"task4_confirm/RUN.json").read_text())
    with (HERE/"task4_confirm/scenes.csv").open() as stream:
        scenes = {(r["configuration"], r["fixture"], r["watermark"], r["fusion"]): r
                  for r in csv.DictReader(stream)}
    primary = {}
    violations = {}
    paths = 0
    for item in manifest["files"]:
        path = folder/Path(item["path"]).name
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"] == run["archives"][path.name]
        assert path.stat().st_size == item["bytes"]
        config, fixture, purpose = path.stem.rsplit("_", 2)
        topology, dynamics, _ = config.split("_")
        with np.load(HERE/"quality_models"/config/f"{topology}_{dynamics}_target35_qwords.npz") as graph_file:
            graph = graph_file["legal"]
        with np.load(path) as a:
            for wm in WATERMARKS:
                gates = a[wm+"_gates"][:, :, 6].astype(bool)
                for pol in POLICIES:
                    key = wm+"_"+pol
                    actions = a[key+"_actions"]
                    hist = np.full(len(actions), 7**6-1, dtype=np.int64)
                    bad = np.zeros(len(hist), dtype=np.int64)
                    for tick in range(1, actions.shape[1]):
                        q = actions[:,tick].astype(int)
                        bad += gates[:,tick] & ~graph[hist,q]
                        hist = (hist*7+q) % 7**6
                    batches = a[key+"_M7_violations"].reshape(-1,32)
                    np.testing.assert_array_equal(batches, np.broadcast_to(bad.reshape(-1,32).sum(1)[:,None], batches.shape))
                    if purpose == "attack":
                        assert int(bad.sum()) == int(scenes[config,fixture,wm,pol]["M7_violations"])
                        violations[key] = violations.get(key,0) + int(bad.sum())
                    if config == PRIMARY and (purpose == "attack" or fixture == "continuous"):
                        col = 0 if purpose == "attack" else 5
                        primary.setdefault((wm,pol,purpose),[]).append(a[key+"_losses"][:,col])
            paths += len(a["path_id"])
    OUT.mkdir(exist_ok=True)
    with (OUT/"primary_cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer=csv.writer(stream, lineterminator="\n")
        writer.writerow(["watermark","policy","attack_mean","attack_median","attack_p99","attack_max","healthy_mean","attack_paths","healthy_paths"])
        for wm in WATERMARKS:
            for pol in POLICIES:
                a = np.concatenate(primary[wm,pol,"attack"])
                h = np.concatenate(primary[wm,pol,"health"])
                writer.writerow([wm,pol,a.mean(),np.median(a),np.quantile(a,.99),a.max(),h.mean(),len(a),len(h)])
    result = dict(verified_archives=len(manifest["files"]), archive_path_rows=paths, attack_violations=violations,
                  published_scene_rows_checked=len(scenes), graph_and_batch_counts_agree=True)
    (OUT/"READBACK.json").write_text(json.dumps(result,indent=2)+"\n", encoding="utf-8",newline="\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
