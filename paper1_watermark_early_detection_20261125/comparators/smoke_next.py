"""Disposable shape, schema and summary checks for tail and partial-support runners."""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import experiment as exp
import tail_study
import analyze_tail
import partial_study
import analyze_partial


def main():
    with tempfile.TemporaryDirectory(prefix="wm_recovery_smoke_") as temp:
        root=Path(temp)
        tail_dir=root/"tail_dev";tail_dir.mkdir()
        tail_rows={}
        for mi,name in enumerate(exp.MODEL):
            for scene in range(6):
                row=tail_study.trial((1,mi,scene,0,2))
                tail_rows[(name,scene)]={k:v for k,v in row.items() if k!="path_id"}
                arrays={k:np.asarray(v) for k,v in row.items()}
                exp.old.base._atomic_npz(tail_dir/f"{name}_{scene}.npz",arrays)
        hashes={p.name:exp.old.base.digest(p) for p in tail_dir.glob("*.npz")}
        ident={"namespace":[20261127,52387,1],"paths_per_cell":2,
               "threshold_manifest_sha256":exp.old.base.digest(exp.HERE/"confirm/RUN.json")}
        exp.old.base._atomic_json(tail_dir/"RUN.json",{"identity":ident,"archives":hashes})
        # Tail analyzer's paths are rooted at the module package.
        old_here=tail_study.HERE
        tail_study.HERE=root
        analyze_tail.HERE=root
        analyze_tail.summarize("dev")

        partial_dir=root/"partial_dev";partial_dir.mkdir()
        hashes={}
        for mi,name in enumerate(exp.MODEL):
            for scene in range(6):
                row=partial_study.task((3,mi,scene,0,1))
                exp.old.base._atomic_npz(partial_dir/f"{name}_{scene}.npz",
                                         {k:np.asarray(v) for k,v in row.items()})
        hashes={p.name:exp.old.base.digest(p) for p in partial_dir.glob("*.npz")}
        ident={"source_sha256":{},"W1_threshold_manifest_sha256":exp.old.base.digest(exp.HERE/"confirm/RUN.json"),
               "paths_per_scene":1}
        exp.old.base._atomic_json(partial_dir/"RUN.json",{"identity":ident,"archives":hashes})
        old_partial_here=partial_study.HERE
        old_analyze_here=analyze_partial.HERE
        partial_study.HERE=root
        analyze_partial.HERE=root
        report=analyze_partial.summarize("dev")
        assert len(report["support_masks"])==8
        partial_study.HERE=old_partial_here
        analyze_partial.HERE=old_analyze_here
        tail_study.HERE=old_here
        analyze_tail.HERE=old_here
        print(json.dumps({"tail_schema":"pass","partial_schema":"pass",
                          "tail_scenes":12,"partial_supports":8},indent=2))


if __name__=="__main__":
    main()

\n