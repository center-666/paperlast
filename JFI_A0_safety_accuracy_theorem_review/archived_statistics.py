"""Confidence calculations for the already recorded health scores and A0 outcomes."""

from math import comb

from flint import arb
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import beta

from certified_cost import outward, real


def binomial_tail(n, k, p, increasing):
    p = real(p)
    q = 1-p
    term = arb(comb(n,k))*p**k*q**(n-k)
    total = term
    indices = range(k,n) if increasing else range(k,0,-1)
    for j in indices:
        term *= (n-j)*p/((j+1)*q) if increasing else j*q/((n-j+1)*p)
        total += term
    return total


def binomial_interval(k, n, alpha):
    low = 0. if k==0 else max(0.,float(beta.ppf(alpha/2,k,n-k+1))-1e-12)
    high = 1. if k==n else min(1.,float(beta.ppf(1-alpha/2,k+1,n-k))+1e-12)
    if low>0.:
        assert binomial_tail(n,k,low,True) < real(alpha)/2
    if high<1.:
        assert binomial_tail(n,k,high,False) < real(alpha)/2
    return low,high


def health_interval(scores, healthy_floor, clip, tail_upper, alpha):
    """Inversion of e-values with a deterministic allowance for the clipped tail."""
    y = np.asarray(scores)
    assert np.all(np.isfinite(y)) and np.all(y>=0.)
    log_bound = -np.log(alpha)
    fractions = (.015625,.03125,.0625,.125,.25)
    def upper_test(m):
        return np.log1p(.95/clip*(m-np.minimum(y,clip))).sum()-log_bound
    def lower_test(m):
        log_bets = [np.log1p(f*(y/m-1)).sum() for f in fractions]
        return logsumexp(log_bets)-np.log(len(fractions))-log_bound
    lower = max(0.,brentq(lower_test,1e-14,max(1.,2*y.mean()),xtol=1e-13)-1e-9)
    upper = brentq(upper_test,0.,clip,xtol=1e-13)+1e-9
    if lower>0.:
        paid = [real(x)/real(lower)-1 for x in y]
        logs = [sum(((1+real(f)*z).log() for z in paid),arb(0)) for f in fractions]
        peak = real(max(float(z) for z in logs))
        log_e = peak+sum(((z-peak).exp() for z in logs),arb(0)).log()-arb(len(logs)).log()
        assert log_e > -real(alpha).log()
    wager = real(.95/clip)
    log_e = sum(((1+wager*(real(upper)-real(min(x,clip)))).log() for x in y),arb(0))
    assert log_e > -real(alpha).log()
    low_mse = outward(healthy_floor+real(lower))[0]
    high_mse = outward(healthy_floor+real(upper)+real(tail_upper))[1]
    return {
        "n":len(y),
        "conditional_integrand_mean":float(y.mean()),
        "mean_square_estimate":float(healthy_floor)+float(y.mean()),
        "mean_square_interval":[low_mse,high_mse],
        "per_endpoint_error_probability":alpha,
        "clip":clip,"clip_tail_upper":tail_upper,
        "maximum_saved_score":float(y.max()),
        "observations_above_clip":int(np.sum(y>clip)),
    }


def archive_events(path, methods, horizon):
    with np.load(path,allow_pickle=False) as archive:
        start = archive["unknown_start__onset"]
        stop = archive["unknown_start__stop"]
        keep = (start==7)&(stop==17)
        alarm = archive["unknown_start__first_alarm"]
        danger = archive["unknown_start__first_danger"]
        onset = archive["unknown_start__first_packet"]
        before = danger < alarm[:,None]
        first = (danger==start[:,None])&before
        assert np.all((0<=alarm)&(alarm<=horizon))
        assert np.all((start[:,None]<=danger)&(danger<=horizon))
        np.testing.assert_array_equal(before,archive["unknown_start__pre_alarm"])
        np.testing.assert_array_equal(first,onset&(start[:,None]<alarm[:,None]))
        np.testing.assert_array_equal(first,archive["unknown_start__first_packet_pre_alarm"])
        v1a = archive["healthy__conditional_health_squared"][:,methods.index("V1a")]
        match = {
            label: {"first": first[keep,j], "third": before[keep,j]&(danger[keep,j]<=9),
                    "first_alarm_tie": int(np.sum((danger[keep,j]==alarm[keep])&(danger[keep,j]<horizon)))}
            for j,label in enumerate(methods)
            if label=="V1a" or label.startswith("anchor@")
        }
        return v1a, match, int(keep.sum())


def report_events(events, count, alpha):
    methods = {}
    for name, info in events.items():
        rows = {"same_tick_first_crossing_and_alarm_over_32":info["first_alarm_tie"]}
        for window in ("first","third"):
            observed = info[window]
            k = int(observed.sum())
            row = {"events":k,"paths":count,"point":k/count,
                   "clopper_pearson":binomial_interval(k,count,alpha)}
            if name!="V1a":
                base = events["V1a"][window]
                plus,minus = int((base&~observed).sum()),int((~base&observed).sum())
                pu, pd = binomial_interval(plus,count,alpha),binomial_interval(minus,count,alpha)
                row["V1a_minus_saturation"] = {"discordant":[plus,minus],
                                                "point":(plus-minus)/count,
                                                "ci":[pu[0]-pd[1],pu[1]-pd[0]]}
            rows[window] = row
        methods[name] = rows
    return methods


def matching_interval(health, frontier, confidence):
    """Map a health interval to an enclosure of the equal-cost radius."""
    tr = np.array([float(health.full)])
    eigenvalues = np.array([[float(u),float(v)] for u,v in health.eigenvalues])
    lo, hi = confidence["mean_square_interval"]
    ends = []
    for target,sign in [(hi,-1),(lo,1)]:
        if target<=outward(health.full)[1]:
            assert sign==1
            ends.append(None)
            continue
        center = frontier.radius_for_cost(target,tr,eigenvalues)
        radius = max(0.,center+sign*1e-7)
        if radius>0.:
            value = health.at(radius)
            assert (value>real(target) if sign==-1 else value<real(target))
        ends.append(radius)
    return ends
