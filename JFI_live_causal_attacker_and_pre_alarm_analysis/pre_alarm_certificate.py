"""Exact nominal-precision lower certificate. No packet or attack generation."""
import argparse
from pathlib import Path
import sys

from scipy.special import ndtr, ndtri
import sympy as sp

from archive_timing import BASE, ENERGY, OUTPUT, save

sys.dont_write_bytecode = True
sys.path.insert(0,str(ENERGY))

from affine_frontier import GAMMA, raw_covariance_certificate, rounded_bounds
from energy_inputs import (DEFAULT_BASELINE, MODELS, MODEL_PATH, digest, load_prepared,
                           rounded_exact, setup, validate_spec)


ALPHA = sp.Rational(1,20)


def blind_interval_survival_lower(delta, false_alarm_upper=.05):
    """Gaussian shift lower bound on no alarm, for a fixed independent prefix shift."""
    if delta<0 or not 0<false_alarm_upper<1:
        raise ValueError("Require nonnegative Gaussian shift length and false alarm level in (0,1).")
    return float(ndtr(ndtri(1-false_alarm_upper)-delta))


def health_lower(full,avg,penalty,eligible,horizon=32):
    exact = full+penalty*eligible/horizon
    return {"eligible_ticks":eligible,"MSE_lower_down":rounded_exact(exact),
            "percentage_vs_V0_lower_down":rounded_exact(100*(exact/avg-1)),
            "gap_over_two_pct_down":rounded_exact(exact-sp.Rational(51,50)*avg),
            "rejects_two_pct":bool(exact>sp.Rational(51,50)*avg)}


def calculate(base):
    common,settings,spec,_ = setup(base)
    validate_spec(settings,spec)
    horizon = settings.horizon
    assert horizon==32
    resource = base/MODEL_PATH
    raw_lower,maximum = rounded_bounds(GAMMA)
    penalty = raw_lower-ALPHA*maximum
    assert penalty>0
    record = {"baseline":BASE,"model_sha256":digest(resource),"nominal_finite_window_bf_alpha":"1/20",
        "required_increment_norm_per_root_coded_energy":str(GAMMA),
        "mean_stage_penalty_if_absolute_zero_lower_down":rounded_exact(raw_lower),
        "max_stage_penalty_exact":str(maximum),"healthy_false_alarm_allowance_exact":str(ALPHA*maximum),
        "mean_stage_penalty_if_strict_pre_alarm_zero_lower_down":rounded_exact(penalty),
        "zero_risk_scope":"Current-packet affine receiver, fixed admissible full-energy single-packet changes at eligible ticks.",
        "abstract_gaussian_shift_examples":[{"mahalanobis_length":x,
            "nominal_level_005_no_alarm_probability_lower":blind_interval_survival_lower(x)}
            for x in [0.,1.,2.,3.,4.]],
        "physical_episodes_generated":0,"models":{}}
    for name in MODELS:
        p = load_prepared(base,name,common,settings)
        full,nominal,_,stages = raw_covariance_certificate(resource,name,p)
        cap = sp.Rational(51,50)*nominal
        B = health_lower(full,nominal,penalty,32,horizon)
        assert B["rejects_two_pct"]
        m = min(n for n in range(1,horizon+1) if full+penalty*n/horizon>cap)
        K = {"B_all_32_ticks":B,"A_if_single_packet_required_at_each_of_10_ticks":
             health_lower(full,nominal,penalty,10,horizon),
             "A_if_single_packet_required_on_union_of_archived_supports":
             health_lower(full,nominal,penalty,18,horizon)}
        record["models"][name] = {"V0_nominal_MSE":float(nominal),"full_nominal_MMSE":float(full),
                "two_percent_nominal_cap_up":rounded_exact(cap,up=True),
                "minimum_eligible_ticks_for_this_certificate":m,"exact_covariance_stages_checked":len(stages),
                "windows":K}
        assert not K["A_if_single_packet_required_at_each_of_10_ticks"]["rejects_two_pct"]
        print(name,"Window B exact lower (%)",B["percentage_vs_V0_lower_down"],"minimum eligible ticks",m,
              flush=True)
    return record


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    args = parser.parse_args()
    save(OUTPUT/"pre_alarm_certificate.json",calculate(args.baseline))
