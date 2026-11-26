"""Disposable W5 endpoint and Markov transition smoke."""
import json
import numpy as np
import rotation_markov as study
import experiment as exp


def main():
    n=2;mi=0;scene=0;batch=0;split=5
    calibration=study.path_outputs(split,mi,scene,batch,n,kind="cal")
    assert all(calibration[a+"_max"].shape==(n,) for a in study.ARMS)
    assert not np.diag(calibration["transition_count"]).any()
    old=exp.read(exp.HERE/"confirm/RUN.json")
    thresholds={}
    for arm in study.ARMS:
        source=arm.replace("quarter_no_repeat","quarter").replace("quarter_iid","quarter")
        thresholds[arm]={"threshold":old["thresholds"]["unstable"][source][1]}
    health=study.path_outputs(split,mi,scene,batch,n,thresholds,kind="health")
    attack=study.path_outputs(split,mi,scene,batch,n,thresholds,kind="attack")
    for arm in study.ARMS:
        for key in ("_fwer","_post20_affected_mse"):
            assert np.isfinite(health[arm+key]).all()
        assert attack[arm+"_early_link"].shape==(n,6)
        assert np.isfinite(attack[arm+"_attack_mse"]).all()
        assert (attack[arm+"_attack_mse"]>=0).all()
    print(json.dumps({"calibration":"pass","health":"pass","attack":"pass",
                      "no_repeat_transition_diagonal":np.diag(calibration["transition_count"]).tolist()},indent=2))


HERE=study.HERE
if __name__=="__main__":main()
