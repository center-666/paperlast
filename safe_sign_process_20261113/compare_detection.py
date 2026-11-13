"""Same fixed detectors and finite sign-word sums, no noise simulation.

GLRT retains the preceding known-onset diagnostic and is explicitly not an
unknown-onset scanner. Safety is separate and holds for arbitrary onset.
"""
from pathlib import Path
from itertools import product
import importlib.util,json
from fractions import Fraction
import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar
from scipy.linalg import eigh
from scipy.stats import ncx2

HERE=Path(__file__).resolve().parent
P=HERE.parent
spec=importlib.util.spec_from_file_location('old_glrt',P/'glrt_screening_20261112/minimal_screen.py')
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
graph=json.loads((HERE/'graph_results.json').read_text(encoding='utf-8'))
words=np.array(list(product([-1.,1.],repeat=10)))
pi=dict(zip(graph['states'],graph['stationary']))
weights=np.zeros(1024)
for row in graph['transitions']:
    weights[row['word_index']]+=pi[row['source_state']]*row['probability']
assert abs(weights.sum()-1)<1e-12
assert np.allclose(weights,weights[::-1],atol=1e-15)
assert np.count_nonzero(weights)==912
safe=json.loads((HERE/'safe_words.json').read_text(encoding='utf-8'))
assert all(i in safe['safe_indices'] for i in np.flatnonzero(weights))
model=old.build(words)
no=old.build(np.ones((1,10)))
# Explicit low-cost nonsecret legal periodic comparison, not a new mechanism.
alt=old.build(np.array([[-1.,1.]*5]))
assert 341 in safe['safe_indices'] and 682 in safe['safe_indices']
dyadic=graph['dyadic_16bit_chain']
dpi=dict(zip(graph['states'],dyadic['stationary']))
dweights=np.zeros(1024)
for row in dyadic['transitions']:
    wi=(row['source_state']<<1)+(row['target_state']&1)
    dweights[wi]+=dpi[row['source_state']]*row['numerator']/row['denominator']
assert abs(dweights.sum()-1)<1e-12

def metrics(M,w,xi,kind):
    q=old.qvalues(M,xi,kind)
    power=np.sum(np.einsum('wkti,i->wkt',M['B'],xi)**2,axis=-1)
    hit=power>=.1; anyhit=hit.any(axis=1); first=hit.argmax(axis=1); rows=np.arange(len(w))
    cum=q.cumsum(axis=1); mx=np.maximum.accumulate(q,axis=1)
    lo=np.where(anyhit,np.maximum(0,1-cum[rows,first]),0)
    hi=np.where(anyhit,1-mx[rows,first],0)
    return dict(PD_last=float(w@q[:,-1]),event_PD_lower=float(w@q.max(axis=1)),
                event_PD_upper=float(w@np.minimum(1,q.sum(axis=1))),
                risk_lower=float(w@lo),risk_upper=float(w@hi),
                danger_probability=float(w@anyhit),
                first_alarm_CDF_lower=(w@mx).tolist(),
                first_alarm_CDF_upper=(w@np.minimum(1,cum)).tolist())

def final_min(M,w,kind):
    def f(t):
        x=np.sqrt(10)*old.Winvhalf@np.array([np.cos(t),np.sin(t)])
        return float(w@old.qvalues(M,x,kind)[:,-1])
    ts=np.linspace(0,np.pi,129,endpoint=False); vals=np.array([f(t) for t in ts]); best=[]
    for i in vals.argsort()[:3]:
        step=np.pi/129; opt=minimize_scalar(f,bounds=(ts[i]-step,ts[i]+step),method='bounded')
        best.append((float(opt.fun),float(opt.x%np.pi)))
    v,t=min(best); x=np.sqrt(10)*old.Winvhalf@np.array([np.cos(t),np.sin(t)])
    if x@old.W@x>10:x*=1-1e-14
    return dict(xi=x.tolist(),energy=float(x@old.W@x),PD_last_min_float=v,
                metrics_at_this_direction=metrics(M,w,x,kind),
                global_direction_interval_certificate=False)

comparisons={}
for name,M,w,kind in [
 ('no_parity',no,np.ones(1),'sliding'),('no_glrt_diagnostic',no,np.ones(1),'glrt'),
 ('iid_parity',model,np.full(1024,1/1024),'sliding'),('iid_glrt',model,np.full(1024,1/1024),'glrt'),
 ('constrained_parity',model,weights,'sliding'),('constrained_glrt',model,weights,'glrt'),
 ('dyadic_constrained_parity',model,dweights,'sliding'),('dyadic_constrained_glrt',model,dweights,'glrt'),
 ('public_alternating_parity',alt,np.ones(1),'sliding'),('public_alternating_glrt',alt,np.ones(1),'glrt')]:
    comparisons[name]=final_min(M,w,kind)
    for label,xi in [('P1',old.old['harm']['no_encoding']['maximizing_xi']),
                     ('iid_risk_witness',[-.26350224221152657,2.263155874216922])]:
        comparisons[name][label]=metrics(M,w,np.array(xi),kind)

# Mean detection matrices for any fixed 4-packet injection, not only O xi.
Z=np.array(old.old['Z4'],float); theta=np.array([[float(Fraction(x)) for x in row] for row in old.old['Theta4']])
P4=Z.T@np.linalg.solve(theta,Z)
anyG=np.zeros((4,4)); marginal=np.zeros(16)
for i,s in enumerate(words):
    anyG+=weights[i]*np.diag(s[:4])@P4@np.diag(s[:4])
    b=sum(int(v>0)<<(3-j) for j,v in enumerate(s[:4])); marginal[b]+=weights[i]
assert np.all(marginal>0)
Jact=old.J[:,3:13]
Pfull=Jact.T@np.linalg.solve(old.Sigma,Jact)
assert np.linalg.matrix_rank(Jact)==10
# For any finite preselected a, diag(s) preserves norm, so each realization
# has full-event lambda >= lambda_min(Pfull)*||a||^2, including no watermark.
edge_states=graph['states']; adjacent={v:[] for v in edge_states}
for row in graph['transitions']:adjacent[row['source_state']].append(row)
conditioned=[]
for state in edge_states:
    paths=[(state,[],1.)]
    for _ in range(4):
        paths=[(e['target_state'],bits+[e['next_sign']],pr*e['probability'])
               for v,bits,pr in paths for e in adjacent[v]]
    GG=sum((pr*np.diag(bits)@P4@np.diag(bits) for _,bits,pr in paths),np.zeros((4,4)))
    Zint=sp.Matrix([[1,-2,1,0],[0,1,-2,1]])
    gram=sp.zeros(4)
    for _,bits,_ in paths:
        C=Zint*sp.diag(*[int(x) for x in bits]); gram+=C.T*C
    minors=[int(gram[:j,:j].det()) for j in range(1,5)]
    assert all(v>0 for v in minors)
    conditioned.append(dict(state=state,rank=int(np.linalg.matrix_rank(GG,tol=1e-10)),
                            minimum_eigenvalue=float(np.linalg.eigvalsh(GG)[0]),
                            support_gram_leading_minors_exact=minors))
out=dict(date='2026-11-13',weights=weights.tolist(),weight_sum=float(weights.sum()),
         stationary_word_support=int(np.count_nonzero(weights)),comparisons=comparisons,
         alpha_total_upper=.05,per_test_alpha=old.alpha,attack_energy_upper=10,danger=.1,
         detector_onset_knowledge='known-onset diagnostic only; unknown-onset GLRT NOT validated',
         exact_event_PD_computed=False,conditional_CDF_functions_float_not_interval=True,
         fixed_four_packet_expected_detection=dict(matrix=anyG.tolist(),
             eigenvalues=np.linalg.eigvalsh(anyG).tolist(),four_sign_pattern_probabilities=marginal.tolist(),
             all_16_patterns_positive=True),
         conditional_known_state_four_packet_gram=conditioned,
         minimum_conditional_rank=min(x['rank'] for x in conditioned),
         finite_event_arbitrary_attack=dict(dimension=10,rank=10,
             per_realization_lower_constant=float(np.linalg.eigvalsh(Pfull)[0]),
             property_also_holds_without_encoding=True),
         risk_constrained_uniform_zero='pathwise certificate, not detection power100%',
         physical_trajectories=0,noise_samples=0)
(HERE/'detection_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:dict(PD_last=v['PD_last_min_float'],event_bounds=v['metrics_at_this_direction'])
                  for k,v in comparisons.items()},indent=2))
