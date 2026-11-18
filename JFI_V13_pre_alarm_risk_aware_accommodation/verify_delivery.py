"""Recompute saved observed event counts and check frozen sources and test phases."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from inputs import DEFAULT_BASELINE, HERE, MODELS, digest, load_json, setup, write_json
from experiments import verify_design
from nominal_audit import bounded_mean_upper
from replay import first


def verify(base):
    design = load_json(HERE/"design.json")
    verify_design(design)
    _,settings,spec,_ = setup(base)
    certificate = load_json(HERE/"frozen_policy.json")
    assert certificate["development_sha256"] == digest(HERE/"results/development/summary.json")
    assert certificate["design_sha256"] == digest(HERE/"design.json")
    published = load_json(HERE/"results/validation/summary.json")
    assert published["frozen_policy_sha256"] == digest(HERE/"frozen_policy.json")
    assert published["design_sha256"] == digest(HERE/"design.json")
    health = load_json(HERE/"results/healthy_check/summary.json")
    methods = published["methods"]
    T = spec["horizon"]
    threshold = spec["danger_squared"]
    tail = {}
    for model in MODELS:
        cases = published["models"][model]["profiles"]
        path = HERE/"results"/"validation"/f"{model}_paths.npz"
        method_rows = {}
        with np.load(path,allow_pickle=False) as z:
            for case,row in cases.items():
                get = lambda k: z[case+"__"+k]
                loss, bias = get("mse"),get("offset_sq")
                assert len(loss)==design["paths"]["validation"]
                first_alarm = get("first_alarm")
                hit = bias>=threshold
                np.testing.assert_array_equal(first(hit),get("first_danger"))
                np.testing.assert_array_equal(get("first_danger")<first_alarm[:,None],get("risk"))
                start = get("start")
                prior = np.arange(T)[None,:]<start[:,None]
                assert np.max(bias*prior[:,None,:])<1e-23
                alpha = get("alpha")
                assert np.all((alpha>=0)&(alpha<=1)&np.isfinite(alpha))
                arm = np.arange(T)[None,:]>=first_alarm[:,None]
                j = methods.index("V1.3")
                np.testing.assert_array_equal(alpha[:,j][arm],1.)
                assert np.isnan(get("risk_score")[:,j][arm]).all()
                for m,k in enumerate(methods):
                    assert get("risk")[:,m].sum()==row["methods"][k]["risk_before_alarm"]
                    assert get("first_packet")[:,m].sum()==row["methods"][k]["first_packet_danger"]
                    if case!="healthy":
                        np.testing.assert_allclose(loss[:,m].mean(),row["methods"][k]["active_mse"][0],
                                                   atol=1e-14)
                if case=="healthy":
                    continue
                after = get("after_mask")
                j0,j1 = methods.index("V1a"),methods.index("V1.3")
                k = np.arange(len(start))
                stop = get("stop")
                m = {}
                for name,j in (("V1a",j0),("V1.3",j1)):
                    last = np.where(hit[:,j]&after,np.arange(T),stop[:,None]-1).max(axis=1)
                    m[name] = {"mean_stop_state_offset_squared":float(bias[k,j,stop].mean()),
                               "danger_at_or_after_stop":int(np.sum(last>=stop)),
                               "mean_updates_until_final_recovery":float(np.mean(last+1-stop)),
                               "horizon_censored_count":int(np.sum(last==T-1))}
                method_rows[case] = m
        tail[model] = method_rows
        with np.load(HERE/"results"/"healthy_check"/f"{model}_integrands.npz",allow_pickle=False) as z:
            assert len(z["conditional"])==design["paths"]["healthy_check"]
            # The conditional floor is shared, so the two computed bounds' difference is independently checkable.
            contrast = np.diff(bounded_mean_upper(z["conditional"],2.65,.05/4))[0]
            upper = [health[model]["policies"][k]["upper_one_sided_joint95"] for k in ("V1a","V1.3")]
            np.testing.assert_allclose(contrast,upper[1]-upper[0],atol=3e-15)
    evaluator = ("import pathlib, unittest, test_v13 as t; "
                 "t.DEFAULT_BASELINE=pathlib.Path(__import__('sys').argv.pop()); "
                 "unittest.main(module=t,verbosity=2)")
    test = subprocess.run([sys.executable,"-B","-X","utf8","-c",evaluator,str(base.resolve())],
                          cwd=HERE,capture_output=True,text=True,check=True)
    write_json(HERE/"results/recovery.json",tail)
    proof = {"V1_V11_V12_frozen_inputs_matched":True,"legacy_regressions":load_json(
              HERE/"results/legacy_pareto.json")["n_regression_method_groups"],
              "independent_paths_per_model_and_profile":design["paths"]["validation"],
              "development_policy_frozen":certificate["lambda"],"identity_checks":test.stderr,
              "verified_result_hashes":{str(f.relative_to(HERE)):digest(f) for f in (HERE/"results").rglob("*")
                if f.is_file() and f.name!="delivery_checks.json"}}
    write_json(HERE/"results/delivery_checks.json",proof)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    verify(parser.parse_args().baseline)
    print("Verified stored trajectories, conditional evidence, and two-world receiver checks.")
