"""First-paper SSCD development-only selection: constrained expected benefit.

Replaces the overly strict demand of 'strict positive gain in EVERY geometry'
with two separate, meaningful pre-declared engineering obligations:
  (A) simultaneous per-scenario noninferiority: rel MSE <= margin;
  (B) simultaneous across-candidate benefit for the EQUAL-WEIGHT average.

This is a statistical DESIGN rule; Student-t and delta-method bounds are
approximate, not finite-sample certificates. Margin cannot be chosen from
held-out final-test data. It must be justified in the next new registration.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t, norm
from watson_crn import theoretical_zeta_mi


def _bonf_ratio_upper(diff,baseline,alpha):
    """Paired mean diff divided by iid mean, with simultaneous upper bound.

    Allocations: one numerator upper, iid lower+upper endpoints. Caller splits
    global alpha by total number of planned bounds, not per scenario.
    """
    d=np.asarray(diff,float);b=np.asarray(baseline,float)
    n=len(d)
    if n<5 or b.shape!=d.shape or (b<0).any():raise ValueError('invalid paired paths')
    c=student_t.ppf(1-alpha,n-1)
    mub=float(b.mean()); mbse=b.std(ddof=1)/math.sqrt(n)
    L=mub-c*mbse;U=mub+c*mbse
    if L<=0:raise ValueError('baseline lower confidence endpoint not positive')
    md=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(n))
    u=md+c*se
    ratio=md/mub
    # Paired delta-method influence for ratio E[D]/E[B], including
    # cross-covariance of D and B, not just numerator variability.
    influence=(d-ratio*b)/mub
    relative_se=float(influence.std(ddof=1)/math.sqrt(n))
    return float(u/(L if u>=0 else U)),float(ratio),relative_se


def select(data,information_budget=.1,noninferiority_margin=.001,
           minimum_average_benefit=.001,alpha=.05):
    """Select one parameter based exclusively on independent development paths.

    The supplied loss distributions are per-path *final common-G* MSE values,
    paired by path_id across candidates, independent across scenario path IDs.
    The historic test results are NEVER read and the JSON split must be
    'development_crn'. MI budget uses each candidate's dimension explicitly.
    """
    if data.get('split')!='development_crn':
        raise ValueError('Only fresh development_crn split is admissible')
    if not 0<alpha<1 or min(information_budget,noninferiority_margin,minimum_average_benefit)<0:
        raise ValueError('invalid constraints')
    candidates=data['theta'];S=data['scenarios'];m=int(data['dimension'])
    if not S or not candidates or candidates.get('iid')!=0.:raise ValueError('iid=0 mandatory')
    if m<2:raise ValueError('dimension must be at least 2')
    names=set(candidates); seen=set();nsc=len(S)
    for label,sc in S.items():
        if set(sc['losses'])!=names:raise ValueError('candidate set mismatch')
        ids=sc['path_ids'];n=len(ids)
        if n<5 or len(set(ids))!=n or (set(ids)&seen):raise ValueError('duplicate, overlapping or insufficient path IDs')
        seen.update(ids)
        for p,v in sc['losses'].items():
            a=np.asarray(v,float)
            if a.shape!=(n,) or not np.isfinite(a).all() or (a<0).any():
                raise ValueError(f'{label}/{p}: invalid losses')
    alts=sorted(names-{'iid'})
    # Allocate alpha/2 to all per-scenario numerator/denominator confidence
    # statements, and alpha/2 to candidate aggregate benefit comparisons.
    n_safety=(len(alts)+2)*nsc
    n_mean=max(len(alts),1)
    safety_single=.5*alpha/n_safety
    mean_single=.5*alpha/n_mean
    report={};chosen='iid';best=math.inf
    for name in alts:
        theta=float(candidates[name]);zeta,mi=theoretical_zeta_mi(theta) if m==2 else (None,None)
        if m!=2:
            from watson_nd import zeta_mi
            zeta,mi=zeta_mi(theta,m)
        if mi>information_budget+1e-12:
            report[name]={'feasible':False,'reason':'information budget exceeded','MI_nats':mi};continue
        rows=[]
        for label,sc in S.items():
            x=np.asarray(sc['losses'][name],float)
            y=np.asarray(sc['losses']['iid'],float)
            d=x-y
            upper,estimate,relative_se=_bonf_ratio_upper(d,y,safety_single)
            rows.append({'scenario':label,'estimated_relative':estimate,
                         'upper_relative':upper,'relative_se':relative_se,
                         'n_paths':len(d)})
        # Equal fixed weight per scenario. Independent scenario paths.
        average=np.mean([r['estimated_relative'] for r in rows])
        se_average=float(np.sqrt(sum(r['relative_se']**2 for r in rows))/nsc)
        # Conservative normal critical; an approximate stratified delta t.
        scenario_vars=np.array([r["relative_se"]**2 for r in rows])/nsc**2
        df=float((scenario_vars.sum()**2)/sum(v*v/(r["n_paths"]-1) for v,r in zip(scenario_vars,rows))) if scenario_vars.sum()>0 else 1e9
        upper_mean=float(average+student_t.ppf(1-mean_single,df)*se_average)
        upper_worst=max(r['upper_relative'] for r in rows)
        good=upper_worst<=noninferiority_margin and upper_mean<-minimum_average_benefit
        report[name]={'feasible':True,'theta':theta,'zeta':zeta,'MI_nats':mi,
                      'average_relative':float(average),'upper_mean_relative':upper_mean,
                      'aggregate_satterthwaite_df':df,
                      'worst_upper_relative':upper_worst,'passes':bool(good),'scenarios':rows}
        if good and upper_mean<best:
            chosen=name;best=upper_mean
    result={'status':'FRESH_DEVELOPMENT_DESIGN_ONLY_NOT_A_TEST',
            'selection':chosen,'theta_locked':float(candidates[chosen]),
            'dimension':m,'information_budget_per_step_nats':information_budget,
            'noninferiority_margin_relative':noninferiority_margin,
            'minimum_average_benefit_relative':minimum_average_benefit,
            'familywise_nominal_alpha':alpha,'safety_bound_count':n_safety,
            'mean_bound_count':n_mean,'per_bound_alpha_safety':safety_single,
            'per_bound_alpha_mean':mean_single,'candidate_audit':report,
            'data_sha256':hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest(),
            'limitations':'Approximate Student-t/normal confidence bounds; only conditional on source scenarios; not an unknown-support security guarantee.'}
    return result


def test():
    rng=np.random.default_rng(20261122);n=512
    def scen(label,mu):
        iid=rng.uniform(.059,.061,size=n)
        return {'path_ids':[label+f'_{i}' for i in range(n)],
                'losses':{'iid':iid.tolist(),
                          'negative':(iid+mu+rng.normal(size=n)*.0001).tolist(),
                          'positive':(iid+.002+rng.normal(size=n)*.0001).tolist()}}
    data={'split':'development_crn','dimension':2,
          'theta':{'iid':0.,'negative':-1.3152369768257723,'positive':1.3152369768257723},
          'scenarios':{'near_zero':scen('zero',-.00001),'strong1':scen('one',-.0008),'strong2':scen('two',-.0007)}}
    r=select(data)
    assert r['selection']=='negative',r
    data['scenarios']['near_zero']=scen('zero',+.0003)
    assert select(data)['selection']=='iid'
    print(json.dumps({'status':'PASS','toy_only':True,
          'average_gain_with_noninferiority':'negative','harmful_case_fallback':'iid'},indent=2))

if __name__=='__main__':test()
