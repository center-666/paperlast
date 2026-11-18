"""Read the frozen validation, regenerate only registered replay inputs, and check receiver bounds."""
import argparse
from pathlib import Path

import numpy as np

from energy_inputs import (BASE_COMMIT, DEFAULT_BASELINE, HERE, MODELS, PREVIOUS,
                           digest, load_json, load_prepared, setup, validate_spec, write_json)
from frozen_inputs import fresh_streams
from nonlinear_bounds import certificate
from receiver import collect, method_table
from replay import paired_risk


METHODS = ("V0","V1a","V1.2a","V1.4","anchor@2%","innovation-clip@2%","trusted-ledger@2%")


def verify_archived_codec(stream, healthy, limit):
    """Evaluate declared units and inverse consistency only on immutable replay records."""
    offset = stream.target-healthy.target
    np.testing.assert_array_equal(stream.decoder,healthy.decoder)
    np.testing.assert_array_equal(stream.trusted,healthy.trusted)
    np.testing.assert_array_equal(stream.state,healthy.state)
    np.testing.assert_allclose(offset,stream.offset,atol=2e-13,rtol=2e-13)
    nominal = np.linalg.solve(stream.decoder,healthy.target[...,None])
    coded_offset = np.linalg.solve(stream.decoder,offset[...,None])
    reconstructed = (stream.decoder @ (nominal+coded_offset))[...,0]
    np.testing.assert_allclose(reconstructed,stream.target,atol=2e-12,rtol=2e-13)
    energy = np.sum(coded_offset[...,0]**2,axis=2)
    count = np.arange(energy.shape[1])[None,:]
    mask = (count>=stream.start[:,None]) & (count<stream.stop[:,None])
    assert float(np.max(energy*~mask))<1e-20
    assert float(energy.sum(axis=1).max())<=limit+2e-10
    return {"mean_episode_coded_energy":float(energy.sum(axis=1).mean()),
            "maximum_episode_coded_energy":float(energy.sum(axis=1).max()),
            "mean_coded_energy_by_time":energy.mean(axis=0).tolist(),
            "reconstruction_max_abs":float(np.max(np.abs(reconstructed-stream.target)))}


def recompute(p, config, proof, parameters, profiles, arrays, cases):
    names = [item["name"] for item in parameters["method_table"]]
    clean = profiles["healthy"]
    radii = [x["radius"] for x in proof["minimum_radii"]]
    fits = [x["norm_threshold"] for x in config["clip_calibration"]]
    table = [e for e in method_table(p,config["radius"],radii,fits) if e.name in METHODS]
    tiny = clean.take(slice(0,2))
    nominal_pair = collect(p,tiny,table)
    rc = {}
    detailed = None
    for key,stream in profiles.items():
        summary = cases[key]
        def info(field):
            return arrays[key+"__"+field]
        n0 = names.index("V1a")
        e = verify_archived_codec(stream,clean,p.settings.coded_energy_reference)
        np.testing.assert_array_equal(stream.start,info("onset"))
        np.testing.assert_array_equal(stream.stop,info("stop"))
        e["first_BF_count"] = int(np.sum(info("first_alarm")<p.settings.horizon))
        e["methods"] = {}
        for method in METHODS:
            j = names.index(method)
            original = summary["methods"][method]
            active = None if key=="healthy" else float(info("active_mean_mse")[:,j].mean())
            post = None if key=="healthy" else float(info("tail_mean_mse")[:,j].mean())
            healthy = float(info("health_mean_mse")[:,j].mean())
            if active is not None:
                np.testing.assert_allclose(active,original["active_actual_mse"][0],atol=1e-13)
            np.testing.assert_allclose(healthy,original["health_actual_mse"][0],atol=1e-13)
            danger = int(info("pre_alarm")[:,j].sum())
            packet = int(info("first_packet")[:,j].sum())
            assert danger==original["pre_alarm_count"] and packet==original["first_packet_count"]
            e["methods"][method] = {"healthy_actual_MSE":healthy,"active_actual_MSE":active,
                                    "tail_actual_MSE":post,"tail_cross_count":original["tail_cross_count"],
                                    "first_packet_count":packet,
                                    "strictly_pre_alarm_count":danger,
                                    "strictly_pre_alarm_contrast_v1a":paired_risk(info("pre_alarm")[:,j],
                                                                                 info("pre_alarm")[:,n0])}
        if key in ("unknown_start","A1_C_replay"):
            view = stream.take(slice(0,2))
            tested = collect(p,view,table)
            observed = np.linalg.norm(tested.post-nominal_pair.post,axis=-1)
            for episode in range(2):
                bounds = certificate(p,view.trusted[episode],view.target[episode],view.decoder[episode],
                                     p.settings.coded_energy_reference,radii[-1],fits[-1])
                for j,method in enumerate(table):
                    assert np.all(observed[episode,j]<=bounds[method.name]+2e-11)
                    np.testing.assert_allclose(np.max(observed[episode,j]**2),
                                               info("peak_paired_squared")[episode,names.index(method.name)],
                                               atol=2e-10,rtol=3e-11)
                if key=="A1_C_replay" and episode==0:
                    detailed = {a.name:{"archived_episode_0_max_bound":float(bounds[a.name].max()),
                                         "archived_episode_0_max_displacement":float(observed[episode,j].max())}
                                for j,a in enumerate(table)}
        rc[key] = e
    return {"profiles":rc,"selected_pair_upper_envelope_examples":detailed}


def run(base):
    common,settings,spec,_ = setup(base)
    validate_spec(settings,spec)
    root = PREVIOUS/"results"
    upstream = load_json(root/"delivery_checks.json")
    for name,sha in upstream["results_sha256"].items():
        assert digest(root/name)==sha,name
    registration = load_json(PREVIOUS/"registration.json")
    assert registration["horizon"]==settings.horizon
    assert registration["D_squared"]==spec["danger_squared"]
    config,calib,proof = (load_json(root/name) for name in
                          ("comparison.json","calibration_frozen.json","frontier.json"))
    models = {}
    for name in MODELS:
        p = load_prepared(base,name,common,settings)
        rep = registration["holdout"]
        streams = fresh_streams(base,p,spec,rep["seed"][name],rep["paths_per_model_and_profile"])
        previous = config["models"][name]
        with np.load(root/(name+"_paired.npz"),allow_pickle=False) as arrays:
            models[name] = recompute(p,{"radius":registration["strict_ball_radius"],
                         "clip_calibration":calib["models"][name]},proof["models"][name],
                         previous,streams,arrays,previous["cases"])
    return {"baseline":BASE_COMMIT,"new_physical_seeds":0,"live_adaptive_trials":0,
            "existing_data_sha256_files":upstream["results_sha256"],
            "defense_vs_defense_reference":"paired V1a risk within each immutable replay",
            "models":models}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    opts = parser.parse_args()
    report = run(opts.baseline)
    (HERE/"results").mkdir(exist_ok=True)
    write_json(HERE/"results/archived_comparison.json",report)
    print("Frozen inputs, coded units, paired metrics, envelopes, and preserved results verified.")
