"""Reconstruct only historical paths and validate their registered endpoints."""
import argparse
import hashlib
import json
import subprocess
import numpy as np
from audit_graph import enumerate_graph
from audit_replay import D1, OUT, CURVE, b2, healthy, modes, d0, matrices, materialize


def check_hash(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Frozen input hash mismatch: {path}")


def check_provenance():
    head = subprocess.check_output(["git","-C",str(D1),"rev-parse","HEAD"],text=True).strip()
    assert head == "7e32a41b9802438f6645d9048ee46386caa53418"
    for folder,manifest,inkey,outkey in [(CURVE,"response_manifest.json","input_sha256","output_sha256"),
            (b2.OUT,"run_manifest.json","inputs_sha256","outputs_sha256")]:
        j=json.loads((folder/manifest).read_text(encoding="utf-8"))
        for key,digest in j[inkey].items():check_hash(D1.parent/key.replace(chr(92),"/"),digest)
        for key,digest in j[outkey].items():check_hash(folder/key,digest)
    d0reg=json.loads((d0.OUTPUT/"analysis.json").read_text())
    for key,digest in d0reg["source_sha256"].items():check_hash(d0.HERE/key,digest)
    check_hash(d0.OUTPUT/"paths.npz",d0reg["path_sha256"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("structures",type=int,nargs="+",choices=range(6))
    args=parser.parse_args()
    check_provenance()
    with np.load(modes.OUTPUT/"mode_tables.npz") as gains:
        for pair in args.structures:
            spec=healthy.scenarios()[2*pair]
            model=healthy.setup(spec,gains)
            m=matrices(gains,spec,model)
            legal=enumerate_graph(model,gains,spec,OUT/"graphs")
            for scene in (2*pair,2*pair+1):
                spec=healthy.scenarios()[scene]
                checks={}
                sources=["B2","B1"]+(["D0"] if spec[1]=="U" else [])
                for source in sources:
                    for fixture in (["healthy"] if source=="B1" else ["continuous","burst"]):
                        checks[source+"_"+fixture]=materialize(scene,spec,model,m,legal,source,fixture)
                (OUT/f"replay_{scene:02}.json").write_text(json.dumps(checks,indent=2)+"\n")
                print("verified archived targets and materialized scene",scene,flush=True)


if __name__ == "__main__":
    main()
