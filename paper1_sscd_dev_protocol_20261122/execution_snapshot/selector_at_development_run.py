"""First-paper SSCD: development-set-only, locked candidate selector.

Fixes exact MI-boundary exclusion and guards against accidental reuse of old
128-path test data. No final-test data should be passed to this module.
Requires numpy, scipy, mpmath.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.stats import t as student_t

REFERENCE_COMMIT = 'b9fc33c324fd539c6e223d0da9950dfd00645895'
THETA_DEFAULT = {'neg2': -2.0, 'neg_boundary': -1.315236976827179,
                 'neg06': -0.6, 'iid': 0.0, 'pos06': 0.6,
                 'pos_boundary': 1.315236976827179, 'pos2': 2.0}


def information(theta: float, m: int = 2, precision: int = 70) -> mp.mpf:
    """Watson adjacent-direction mutual information, high precision."""
    if m < 2 or not np.isfinite(theta):
        raise ValueError('m>=2 and finite theta required')
    with mp.workdps(precision):
        z = mp.mpf(str(theta))
        aa, bb = mp.mpf('0.5'), mp.mpf(m)/2
        norm = mp.hyper([aa], [bb], z)
        mean = (aa/bb) * mp.hyper([aa+1], [bb+1], z) / norm
        return +(z*mean-mp.log(norm))


def project_theta(theta: float, budget: float, m: int = 2,
                  rounding_tolerance: float = 1e-11) -> dict:
    """Project *only* near-boundary roundoff inward; otherwise reject.

    Any projection is logged. By making the final magnitude strictly below the
    high-precision budget root, the implemented theta is genuinely feasible.
    """
    if budget < 0 or not math.isfinite(budget):
        raise ValueError('budget must be finite and nonnegative')
    with mp.workdps(80):
        mival = information(theta, m, 80)
        cap = mp.mpf(str(budget))
        if mival <= cap:
            return {'status':'feasible', 'theta_requested':theta,
                    'theta_used':theta, 'MI_nats':str(mival), 'adjusted':False}
        over = mival - cap
        if over > mp.mpf(str(rounding_tolerance)) or theta == 0 or budget == 0:
            return {'status':'over_budget', 'theta_requested':theta,
                    'MI_nats':str(mival), 'excess_nats':str(over), 'adjusted':False}
        # Monotonicity of I(theta) on either side of zero; solve in abs(theta).
        signed = 1 if theta > 0 else -1
        target = mp.mpf(str(budget))
        def ff(absval):
            q = signed*absval
            Z = mp.hyper([mp.mpf('.5')],[mp.mpf(m)/2],q)
            c = (mp.mpf(1)/m) * mp.hyper([mp.mpf('1.5')],[mp.mpf(m)/2+1],q)/Z
            return q*c-mp.log(Z)-target
        lo, hi = mp.mpf('0'), abs(mp.mpf(str(theta)))
        for _ in range(240):
            mid=(lo+hi)/2
            if ff(mid)>0:hi=mid
            else:lo=mid
        # A conservative interior margin (far bigger than float roundoff).
        used = signed*float(lo*(1-mp.mpf('1e-12')))
        mi_used = information(used,m,80)
        if not mi_used <= target:
            raise ArithmeticError('projected parameter still exceeds budget')
        return {'status':'feasible', 'theta_requested':theta,
                'theta_used':used, 'MI_nats':str(mi_used),
                'MI_requested_nats':str(mival), 'adjusted':True,
                'adjustment_reason':'rounding-only exceedance projected inward'}


def select_from_development(data: dict, *, budget: float=0.1,
                            m: int=2, alpha: float=0.05,
                            min_relative_gain: float=0.0,
                            expected_source: str=REFERENCE_COMMIT) -> dict:
    """Choose one global parameter, with Bonferroni one-sided paired t-UCB.

    Development JSON schema: {source_commit, split:'development', theta: {name:float},
       scenarios:{name:{path_ids:[str], losses:{policy:[float]}}}}.
    The samples must be independent *across paths*, paired *within a scenario*.
    """
    if data.get('split')!='development':
        raise ValueError('This selector rejects non-development data')
    if data.get('source_commit')!=expected_source:
        raise ValueError('Frozen source commit mismatch')
    theta=data['theta']; scenarios=data['scenarios']
    if not scenarios or 'iid' not in theta or theta['iid']!=0:
        raise ValueError('Nonempty scenarios and iid theta 0 required')
    names=set(theta)
    if not 0<alpha<1 or min_relative_gain<0:raise ValueError('invalid alpha/min gain')
    for scenario, rec in scenarios.items():
        if set(rec['losses']) != names:raise ValueError('every scenario must contain each candidate')
        ids=rec.get('path_ids')
        if not ids or len(ids)<3 or len(set(ids))!=len(ids):
            raise ValueError(f'{scenario}: invalid path IDs')
        for policy,values in rec['losses'].items():
            arr=np.asarray(values,float)
            if arr.shape!=(len(ids),) or not np.isfinite(arr).all() or np.any(arr<0):
                raise ValueError(f'{scenario}/{policy}: invalid loss vector')
    # Simultaneous bounds cover every alternative/scenario numerator and each
    # scenario's iid denominator. Student-t calibration is approximate.
    numerator_bounds=(len(names)-1)*len(scenarios)
    denominator_bounds=len(scenarios)
    tests=numerator_bounds+denominator_bounds
    baseline_lcb={}
    for scenario,rec in scenarios.items():
        ref=np.asarray(rec['losses']['iid'],float)
        n=len(ref);mean=float(ref.mean());se=float(ref.std(ddof=1)/np.sqrt(n))
        q=student_t.ppf(1-alpha/tests,n-1)
        lower=mean-q*se
        if lower<=0:
            raise ValueError(f'{scenario}: simultaneous iid baseline lower bound is not positive')
        baseline_lcb[scenario]=lower
    report={}; selected='iid'; best=float('inf')
    for policy in sorted(names):
        admissible=project_theta(float(theta[policy]),budget,m)
        if admissible['status']!='feasible':
            report[policy]={'feasible':False,**admissible};continue
        stats=[]
        for scenario,rec in scenarios.items():
            x=np.asarray(rec['losses'][policy],float)
            ref=np.asarray(rec['losses']['iid'],float)
            base=ref.mean()
            if base<=0:raise ValueError('iid mean loss must be positive')
            paired=x-ref
            n=len(paired);avg=paired.mean();se=paired.std(ddof=1)/math.sqrt(n)
            q=student_t.ppf(1-alpha/tests,n-1) if tests else 0.
            upper_abs=avg+q*se
            # A ratio UCB uses simultaneous numerator and positive denominator
            # bounds; dividing by the observed iid mean understates uncertainty.
            relative_upper=upper_abs/baseline_lcb[scenario]
            stats.append({'scenario':scenario,'n':n,'mean_abs_diff':float(avg),
                'se_abs':float(se),'numerator_upper_abs':float(upper_abs),
                'mean_relative':float(avg/base),
                'simultaneous_baseline_lower':float(baseline_lcb[scenario]),
                'simultaneous_relative_upper_t_approx':float(relative_upper),
                'baseline_mean':float(base)})
        worst=max(s['simultaneous_relative_upper_t_approx'] for s in stats)
        report[policy]={'feasible':True,**admissible,'worst_upper_relative':worst,'per_scenario':stats}
        if policy!='iid' and worst<-min_relative_gain and worst<best:
            selected=policy;best=worst
    return {'selected':selected, 'theta_locked':report[selected]['theta_used'],
      'MI_locked_nats':report[selected]['MI_nats'],'info_budget_nats_per_step':budget,
      'selector':'one-sided paired Bonferroni-UCB, development-only',
      'interval_method':'simultaneous one-sided paired-difference Student-t UCB and positive iid-baseline Student-t LCB; Bonferroni; approximate under skewed path losses',
      'alpha':alpha,'familywise_bounds':tests,
      'alternative_numerator_bounds':numerator_bounds,
      'iid_denominator_lower_bounds':denominator_bounds,
      'minimum_relative_gain':min_relative_gain,
      'source_commit':expected_source,'scenarios':list(scenarios),
      'candidate_audit':report,'status':'DEVELOPMENT_CHOICE_ONLY_NOT_FINAL_TEST',
      'source_sha256':hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()}


def self_test():
    bound=project_theta(-1.315236976827179,0.1,m=2)
    assert bound['status']=='feasible' and bound['adjusted']
    assert float(bound['MI_nats']) < 0.1
    assert project_theta(-2,0.1)['status']=='over_budget'
    rng=np.random.default_rng(227)
    n=256;ids=[f'fresh_{i}' for i in range(n)]
    base=.065+rng.normal(size=n)*.002
    def scenario(benefit):
        return {'path_ids':ids,'losses':{k: (base+shift+rng.normal(size=n)*.00002).tolist()
            for k,shift in [('iid',0),('negative',benefit),('positive',+.0004)]}}
    data={'source_commit':REFERENCE_COMMIT,'split':'development',
      'theta':{'iid':0,'negative':-1.315236976827179,'positive':1.315236976827179},
      'scenarios':{'s1':scenario(-.0007),'s2':scenario(-.0009)}}
    result=select_from_development(data)
    assert result['selected']=='negative',result
    data['scenarios']['s2']=scenario(.0004)
    result2=select_from_development(data)
    assert result2['selected']=='iid'
    print(json.dumps({'result':'PASS','boundary_projection':bound,
      'all_benefit_selection':'negative','conflict_selection':result2['selected']},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--development-json',type=Path)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--budget',type=float,default=.1)
    ap.add_argument('--minimum-relative-gain',type=float,default=0.)
    args=ap.parse_args()
    if args.development_json is None:self_test();return
    if args.output is None:ap.error('--output required')
    data=json.loads(args.development_json.read_text(encoding='utf8'))
    res=select_from_development(data,budget=args.budget,min_relative_gain=args.minimum_relative_gain)
    args.output.write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'selected':res['selected'],'theta_locked':res['theta_locked'],
                     'mi':res['MI_locked_nats'],'output':str(args.output)},indent=2))

if __name__=='__main__':main()
