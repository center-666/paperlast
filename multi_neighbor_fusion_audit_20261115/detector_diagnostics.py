"""Standard detector-only diagnostics, no additional fusion design or samples."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
import joint_screen as js
import numpy as np
from scipy.stats import chi2,ncx2,norm
def main():
    out={};first={};store={}
    for name in js.CFG['models']:
        m=js.ref.load_model(name);data,M=js.sample_inputs(m,name);idx=js.CFG['evaluation_indices']
        y0=data['y'][idx];M=M[idx];info=js.gaussian_info(m)
        for j in (0,2):
          h=0 if j==0 else 1
          for enc in js.CFG['encodings']:
            mm=np.broadcast_to(np.eye(2),M.shape).copy() if enc=='identity' else M
            row=js.PRE['models'][name]['attacks'][str(j)][enc];xi=np.array(row['P1']['xi'])
            a=js.signals(m,j,xi[None])[0];eta=np.einsum('ntij,tj->nti',mm[:,:,j],a)
            y=y0.copy();y[:,:,j]+=eta
            for scope in ('known','unknown'):
                tag=f'{name}/{scope}/j{j}/{enc}';record={}
                clean_b0,_,_=js.closed(m,js.PRE['models'][name],'B0',y0,j if scope=='known' else None)
                full_b0,_,_=js.closed(m,js.PRE['models'][name],'B0',y,j if scope=='known' else None)
                tauD=js.first(np.sum((full_b0['post']-clean_b0['post'])**2,axis=2)>=.1)
                for local in (True,False):
                    alarm,score=js.prefix_glrt(m,y,mm,j,scope,local)
                    clean,cscore=js.prefix_glrt(m,y0,mm,j,scope,local)
                    nm='local_prefix_GLRT' if local else 'joint_prefix_GLRT'
                    record[nm]=dict(Pd=js.prop(alarm.any(axis=1)),normal_event_FA=js.prop(clean.any(axis=1)),
                      all_accept_counterfactual_danger_before_diagnostic_alarm=js.prop(tauD<js.first(alarm)),
                      first_packet_detection=js.prop(alarm[:,7]),mean_first_time_alarms_only=js.summary(js.first(alarm)[alarm.any(axis=1)]),
                      not_unknown_onset=True,number_distinct_prefixes=10,threshold=float(chi2.isf(.05/(10*(1 if scope=='known' else 2)),2)))
                    store[tag.replace('/','__')+'__'+nm+'__alarms']=alarm
                # Whole four-dimensional per-packet omnibus, correct df and total event FA.
                sc=js.scores(m,y);csc=js.scores(m,y0);tau4=chi2.isf(.05/25,4)
                aa=sc['omnibus'][:,7:]>tau4;ca=csc['omnibus'][:,7:]>tau4
                record['omnibus4']=dict(Pd=js.prop(aa[:,:10].any(axis=1)),normal_event_FA=js.prop(ca.any(axis=1)),df=4,threshold=float(tau4))
                # Exact integration over nominal Gaussian measurement noise, fixed saved secrets.
                delta=eta@info['inv'][j].T
                l0=np.einsum('nti,ij,ntj->nt',delta,np.linalg.inv(info['pairV'][h]),delta)
                lj=np.einsum('nti,ij,ntj->nt',delta,np.linalg.inv(info['Gamma'][h]),delta)
                tau=chi2.isf(.05/(25*(1 if scope=='known' else 2)),2)
                # For unknown edge, each own focused test exceeds implies a global alarm,
                # but an argmax attribution need not select the correct edge.
                record['noise_integrated_own_link_power']=dict(single=float(ncx2.sf(tau,2,l0[:,7:17]).mean()),
                    joint_focused=float(ncx2.sf(tau,2,lj[:,7:17]).mean()),
                    meaning='mean per-packet marginal power over ten attack times and saved secret draws, not event power or unknown-link attribution',
                    mean_ncp_single=float(l0[:,7:17].mean()),mean_ncp_joint=float(lj[:,7:17].mean()))
                out[tag]=record
            # An oracle with true x and xi, an UPPER bound for legitimate first-packet tests.
            # Per-time .05 relaxed from .05/25: gives the oracle even more FA resources.
            a0=a[7];ncp0=np.einsum('ni,ij,nj->n',eta[:,7],np.linalg.inv(m.R[j]),eta[:,7])
            first[f'{name}/j{j}/{enc}']=dict(first_attack=a0.tolist(),xi=xi.tolist(),
              unprotected_first_bias_squared=[float(v) for v in np.sum((eta[:,7]@m.G[1].T)**2,axis=1)],
              conditional_oracle_power_relaxed_alpha05=[float(v) for v in norm.sf(norm.isf(.05)-np.sqrt(ncp0))],
              oracle_scope='conditional fixed current secret NP test; unconditional secret-dependent alpha allocation is not bounded by its unweighted mean')
        # Identity first-packet ordinary witness: deterministic harm, so no secret-allocation caveat.
        xi=np.array(js.PRE['models'][name]['attacks']['0']['identity']['ordinary']['xi']);a0=m.H[0]@xi
        damage=float(np.sum((m.G[1]@a0)**2));lam=float(a0@np.linalg.inv(m.R[0])@a0)
        first[name+'/identity_ordinary_first_packet_impossibility']=dict(xi=xi.tolist(),coding_energy=float(xi@np.array(js.PRE['models'][name]['attacks']['0']['identity']['W'])@xi),
          damage=damage,oracle_NP_miss_lower_bound=float(norm.cdf(norm.isf(.05)-np.sqrt(lam))),
          danger_is_deterministic=damage>=.1,
          legal_scope='no watermark, all-accept before first alarm, clean prior; offline oracle knows x and xi; at most .05 total event false alarm',
          not_a_lower_bound_for_preemptive_clipping_or_permanent_rejection=True)
    js.dump(H/'detector_diagnostics.json',dict(results=out,first_packet_oracle=first,new_physical_samples=0,
       finite_time_marginal_power_is_not_danger_risk=True,overlap_prefixes_not_independent=True))
    np.savez_compressed(H/'detector_diagnostic_arrays.npz',**store)
    print(json.dumps({k:v for k,v in first.items() if 'impossibility' in k}))
if __name__=='__main__':main()
