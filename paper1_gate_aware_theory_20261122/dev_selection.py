"""Paper 1: conservative *development-only* gate-aware SSCD candidate selector.

Never run this on test paths; freeze its output before independent evaluation.
Input: a JSON object with scenario -> policy -> list[per-path final common-G MSE].
Scenarios can encode dynamics, attack directions and onset variants. All
policies in one scenario must use the same paired physical paths.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as t_dist


def select_candidate(paths, *, info_nats, info_budget, alpha=0.05):
    """Minimize maximum multiplicity-adjusted upper CI of relative risk change.

    info_nats and info_budget must use identical units (per-step MI in
    this implementation, not window-total correlation).

    iid must exist in every scenario. Nonzero policies failing the MI budget
    are excluded. Reject all non-iid policies if no *positive certified
    development margin* appears. This is not a theorem about the test set.
    """
    scenarios = list(paths)
    if not scenarios: raise ValueError('No development scenarios provided')
    names=set(paths[scenarios[0]])
    if 'iid' not in names: raise ValueError('iid is mandatory')
    if any(set(paths[s]) != names for s in scenarios):
        raise ValueError('Policies must be present in every scenario')
    alt=sorted(names-{'iid'})
    ntests=len(alt)*len(scenarios)
    if not ntests: return {'selected':'iid','reason':'no alternatives'}
    # One-sided Bonferroni familywise upper bound across all candidate/scenario pairs.
    q=1.-alpha/ntests
    report={}
    for name in sorted(names):
        if name!='iid' and info_nats[name]>info_budget:
            report[name]={'feasible':False,'reason':'information budget exceeded'}
            continue
        risk=[]
        for scenario in scenarios:
            src=paths[scenario]
            x=np.asarray(src[name],float)
            ref=np.asarray(src['iid'],float)
            if x.shape != ref.shape or x.ndim != 1 or len(x)<2:
                raise ValueError('Need paired equal-length path arrays (n>=2)')
            if not np.isfinite(x).all() or not np.isfinite(ref).all():
                raise ValueError('Nonfinite path loss')
            baseline=float(ref.mean())
            if baseline<=0: raise ValueError('iid baseline risk must be >0')
            d=(x-ref)/baseline
            mean=float(d.mean())
            se=float(d.std(ddof=1)/np.sqrt(len(d)))
            upper=float(mean+t_dist.ppf(q,len(d)-1)*se)
            risk.append({'scenario':scenario,'mean_relative':mean,'se':se,
                         'adjusted_upper_relative':upper,'paths':len(d)})
        report[name]={'feasible':True, 'MI_nats':info_nats[name] if name!='iid' else 0.,
                      'scenario_metrics':risk,
                      'worst_upper_relative':max(r['adjusted_upper_relative'] for r in risk)}
    feasible=[(p,v['worst_upper_relative']) for p,v in report.items()
              if p!='iid' and v.get('feasible')]
    if feasible:
        best=min(feasible,key=lambda z:z[1])
        selected=best[0] if best[1]<0 else 'iid'
    else: selected='iid'
    return {'selected':selected, 'reason':('positive robust development margin' if selected!='iid' else 'no robust alternative certified on development paths'),
            'alpha':alpha,'familywise_tests':ntests,'info_budget':info_budget,
            'scenarios':scenarios,'candidates':report,
            'warning':'Development-only statistical screening, not global optimum or test-set guarantee.'}


def test():
    rng=np.random.default_rng(2026112221)
    n=800
    b1=.05+rng.normal(size=n)*.005
    b2=.07+rng.normal(size=n)*.006
    paths={'unstable':{'iid':b1.tolist(),'negative':(b1-.002+rng.normal(size=n)*.001).tolist(),
                       'positive':(b1+.002+rng.normal(size=n)*.001).tolist()},
           'stable':{'iid':b2.tolist(),'negative':(b2-.003+rng.normal(size=n)*.001).tolist(),
                     'positive':(b2+.002+rng.normal(size=n)*.001).tolist()}}
    got=select_candidate(paths,info_nats={'iid':0.,'negative':.1,'positive':.1},info_budget=.12)
    assert got['selected']=='negative'
    restricted=select_candidate(paths,info_nats={'iid':0.,'negative':.1,'positive':.1},info_budget=.01)
    assert restricted['selected']=='iid'
    conflict={'unstable':paths['unstable'], 'rotated':{'iid':b2.tolist(),
        'negative':(b2+.003+rng.normal(size=n)*.001).tolist(),
        'positive':(b2-.002+rng.normal(size=n)*.001).tolist()}}
    conflict_result=select_candidate(conflict,info_nats={'iid':0.,'negative':.1,'positive':.1},info_budget=.12)
    assert conflict_result['selected']=='iid'
    print(json.dumps({'status':'PASS','synthetic_only':True,
                      'positive_case':got['selected'],
                      'information_limited':restricted['selected'],
                      'geometric_conflict':conflict_result['selected']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development-json',type=Path)
    p.add_argument('--info-json',type=Path)
    p.add_argument('--information-budget',type=float,default=.1)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.development_json is None:
        test()
    else:
        if args.info_json is None or args.output is None:
            p.error('Need --info-json and --output with --development-json')
        dev=json.loads(args.development_json.read_text(encoding='utf-8'))
        info=json.loads(args.info_json.read_text(encoding='utf-8'))
        result=select_candidate(dev,info_nats=info,info_budget=args.information_budget)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'selected':result['selected'],'output':str(args.output)},indent=2))
