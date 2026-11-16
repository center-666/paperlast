"""Register the finite audit before evaluating the existing sample bank."""
from pathlib import Path
import sys,json,hashlib,datetime
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;P=H.parent;ROOT=P.parent.parent
REF=P/'common_g_reference_revision_20261115';OLD=P/'attack_gain_switch_audit_20261115'
sys.path.insert(0,str(REF))
import common_g_reference as ref
import numpy as np
import sympy as sp
from scipy.linalg import block_diag,solve_discrete_are,eigh
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf8')
def modes(m,redesign):
    out=[]
    for mask in range(4):
        ids=[j for h,j in enumerate((0,2)) if mask>>h&1]
        if not redesign or mask==3:K,G=m.K[1],m.G[1];res=None
        else:
            C=np.vstack([m.H[1],sum((m.H[j] for j in ids),np.zeros((2,2)))]) if ids else m.H[1]
            V=block_diag(m.R[1],sum((m.R[j] for j in ids),np.zeros((2,2)))) if ids else m.R[1]
            pp=solve_discrete_are(m.A.T,C.T,m.Q,V)
            gain=np.linalg.solve(C@pp@C.T+V,C@pp).T
            K=gain[:,:2];G=gain[:,2:] if ids else np.zeros((2,2))
            res=float(np.max(np.abs(pp-(m.A@(pp-gain@C@pp)@m.A.T+m.Q))))
        J=np.eye(2)-K@m.H[1]-G@sum((m.H[j] for j in ids),np.zeros((2,2)));F=J@m.A
        r=sp.Matrix([[sp.Rational(str(x)) for x in row] for row in F]);Q=sp.Rational(81,100)*sp.eye(2)-r.T*r
        assert Q[0,0]>0 and Q.det()>0
        out.append(dict(mask=mask,accepted=ids,K=K.tolist(),G=G.tolist(),J=J.tolist(),F=F.tolist(),
          spectral_radius=float(max(abs(np.linalg.eigvals(F)))),norm2=float(np.linalg.norm(F,2)),DARE_residual=res,
          rational_minors_081I_minus_FtF=[str(Q[0,0]),str(Q.det())]))
    return out
def attack_profiles(m,j,encoding):
    O=np.stack([m.H[j]@np.linalg.matrix_power(m.A,t) for t in range(10)]);W=np.einsum('tji,tjk->ik',O,O)
    q=0 if encoding=='identity' else .5;best=(-1,None,None)
    for k in range(10):
        C=[np.linalg.matrix_power(m.F(1),k-t)@m.G[1] for t in range(k+1)]
        B=sum(C[t]@O[t] for t in range(k+1));S=B.T@B
        for t,c in enumerate(C):S+=q*q*O[t].T@(c.T@c+np.trace(c.T@c)*np.eye(2))@O[t]
        ev,v=eigh(S,W)
        if 10*ev[-1]>best[0]:best=(float(10*ev[-1]),v[:,-1],k)
    def norm(u):
        x=u*np.sqrt(10/(u@W@u))*(1-1e-12)
        return dict(xi=x.tolist(),coding_energy=float(x@W@x))
    return dict(W=W.tolist(),ordinary=norm(np.array([-1.89,.42])),P1={**norm(best[1]),'max_expected_all_accept_damage':best[0],'peak_offset':best[2]})
def main():
    assert not (H/'registration.json').exists(),'do not overwrite frozen registration'
    accepted=json.loads((P/'minimum_contribution_gate_20261115/freeze_manifest.json').read_text('utf8'))
    for f in ('common_g_reference.py','inputs/five_node_aggregate_model.json'):assert sha(REF/f)==accepted['local_reference_artifact_hashes'][f]
    frozen={str(p):sha(p) for p in P.rglob('*') if p.is_file() and H not in p.parents and not set(p.relative_to(P).parts)&{'tmp','__pycache__','literature_work','_build','publication'}}
    for p in (ROOT/'AGENTS.md',ROOT/'参考文献/文献库.json',ROOT/'研究规划/第一篇小论文/secure distributed safety.tex'):frozen[str(p)]=sha(p)
    dump(H/'protected_inputs.json',dict(sha256=frozen))
    c=dict(baseline='f5460ed6932f648cfb1ae6715392417c3d2e9529',T=32,attack_start=7,L=10,E=10.,D=.1,
      receiver=1,neighbors=[0,2],models=['unstable','stable'],encodings=['identity','original_Gaussian'],lam=.5,q=.5,
      monitoring=[7,31],event_FA=.05,hold_including_alarm=5,
      bank='previous 128 paths per model, regenerated only from saved seeds; no new physical realizations',
      search_indices=list(range(64)),evaluation_indices=list(range(64,128)),
      split_limit='historically reused bank; disjoint for THIS selection, not untouched external validation',
      attack_grid=dict(angles=24,energy_fractions=[.25,.5,1.],both_signs=True),
      search_objectives=['danger_before_first_alarm','danger_on_attacked_edge_readmission_after_alarm'],
      search_scope='finite grid in xi before secrets/noise; not global P3; per policy and true attacked edge separately',
      known_target='a priori declared target; other neighbor trusted within this experiment only',
      unknown_target='no target label given; at most one of edges 0/2 attacked; both received and tested',
      B0='all accept frozen original common G; diagnostic single detector only',
      B1='raw local-vs-neighbor spatial tests; fixed original K/G after rejections',
      B2='same B1 alarm histories; nominal steady aggregate-MMSE gains for accepted set',
      B4='conditional Gaussian/leave-one GLRT, max over two hypotheses when unknown; reconfigured common G',
      B5='pairwise-consistency maximum-clique gate including trusted local; nominal accepted-set common G; standard principle baseline, not exact He reproduction',
      B5_tie='largest allowed clique; ties smallest sum pair score then lexicographic sensor IDs; known-case must keep the declared healthy neighbor',
      calibration='Bonferroni over 25 times and 1/2 local tests (known/unknown); B4 1/2 conditional tests; B5 3 pair tests, known also3; no independence assumption within time',
      B4H_aux='same B4 high alarm; release after 5 consecutive score<=chi2_2(.95) passes; supplementary mature hysteresis, not a new candidate',
      prefix_GLRT='detector-only: public episode aligned onset7, L10 template, 10 distinct prefixes, two link hypotheses if unknown, Bonferroni alpha/10 or20; not unknown-onset online scanner',
      sample_bank_sha256=sha(OLD/'screen_arrays.npz'),old_registration_sha256=sha(OLD/'registration.json'),
      original_counterexample_sha256=sha(OLD/'reentry_counterexample_exact.json'),
      normal_cost='target and network MSE; no tuning tolerance to manufacture gap',
      main_bias='actual minus own independently gated healthy shadow; true error and difference vs all-accept clean separately',
      reentry_events='0->1 of attacked edge after first alarm; same-packet alarm before fusion; no attack reading this label',
      evidence='finite empirical Wilson/paired SE diagnostics only; no formal all-attack risk bound',
      stop='standard Gaussian/robust gating explains gain and no new provable central property -> stop new method claim',
      registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),evaluations_started=False)
    dump(H/'registration.json',c)
    rows={}
    for name in c['models']:
        m=ref.load_model(name)
        rows[name]=dict(fixed_modes=modes(m,False),redesigned_modes=modes(m,True),
          attacks={str(j):{e:attack_profiles(m,j,e) for e in c['encodings']} for j in (0,2)})
    dump(H/'analytic_preflight.json',dict(registration_sha256=sha(H/'registration.json'),models=rows,
      certificate_scope='finite computed F decimals as exact rational; 0.9 common Euclidean contraction; no physical DARE interval certification'))
    print(json.dumps(dict(protected=len(frozen),registered=True,max_norm={n:max(z['norm2'] for typ in ('fixed_modes','redesigned_modes') for z in r[typ]) for n,r in rows.items()})))
if __name__=='__main__':main()
