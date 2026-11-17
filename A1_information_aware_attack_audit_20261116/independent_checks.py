"""Independent affine trajectories, causal-prefix tests, exact budget checks."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import numpy as np
import attack_models as am
from screen_attack_classes import detectors,evaluate,policies
from attacker_observer_bounds import primitive_maps
H=Path(__file__).resolve().parent;P=H.parent
def main():
    source=json.loads((H/'validation_results.json').read_text('utf8'))
    errors=dict(affine_trajectory=0.,bias_recursion=0.,mse_identity=0.,causality=0.,A0_embedding=0.,watermark_cancellation=0.,nominal_covariance=0.)
    count=0;largest_energy=0.;overlaps={}
    for name in ('unstable','stable'):
        m=am.ref.load_model(name);bank=am.old_sample_bank(m,[16,33,63]);dets=detectors(m,bank)
        for pid,row in source['models'][name]['validation_rows'].items():
            p=row['policy'];metrics,z,alarms,_=evaluate(m,bank,p,dets,True)
            largest_energy=max(largest_energy,float(z['energy'].max()))
            for i in range(3):
                clean=np.zeros(2);att=np.zeros(2);direct=[];delta=[]
                for k in range(32):
                    y=bank['y'][i,k]
                    # Expand the update independently of ref.common_step and
                    # independently of the saved bias recursion.
                    D=m.K[1]@y[1]+m.G[1]@(y[0]+y[2])
                    clean=m.J(1)@(m.A@clean)+D
                    attacked_source=bank['M'][i,k]@(bank['q'][i,k]+z['a'][i,k])
                    da=attacked_source-y[0]
                    att=m.J(1)@(m.A@att)+D+m.G[1]@da
                    direct.append(att);delta.append(att-clean)
                E=np.array(direct)-bank['x'][i];d=np.array(delta)
                errors['bias_recursion']=max(errors['bias_recursion'],float(np.max(np.abs(d-z['b'][i]))))
                errors['affine_trajectory']=max(errors['affine_trajectory'],float(np.max(np.abs(E-(bank['e'][i]+z['b'][i])))))
                errors['mse_identity']=max(errors['mse_identity'],float(np.max(np.abs(np.sum(E*E,1)-z['mse'][i]))))
                count+=32
            # Future packets cannot change present/past actions.
            q=bank['q'].copy();cut=9 if p['family']=='A1-H' else 10
            q[:,cut:]+=100.
            altered,_=am.causal_policy(q,m.A,m.H[0],p)
            errors['causality']=max(errors['causality'],float(np.max(np.abs(altered[:,:10]-z['a'][:,:10]))))
            # All theta0 policies strictly embed A0.
            if p.get('kind')=='projected_feedback':
                zero={**p,'theta':0.};a0={**zero,'family':'A0'}
                ah,_=am.causal_policy(bank['q'],m.A,m.H[0],zero)
                ab,_=am.causal_policy(bank['q'],m.A,m.H[0],a0)
                errors['A0_embedding']=max(errors['A0_embedding'],float(np.max(np.abs(ah-ab))))
            # Pure current feedback exposes M q=y irrespective of hidden M.
            dp=np.einsum('ntij,ntj->nti',bank['M'],bank['q'])
            errors['watermark_cancellation']=max(errors['watermark_cancellation'],float(np.max(np.abs(dp-bank['y'][:,:,0]))))
        X,Y,_,_=primitive_maps(m);noise=[]
        for end in (8,9):
            d=dets['joint_GLRT'];rows=np.concatenate([Y[t,j] for t in range(end-4,end+1) for j in (1,0,2)],axis=0)
            L=d['V']@rows;noise.append(L)
            errors['nominal_covariance']=max(errors['nominal_covariance'],float(np.max(np.abs(L@L.T-np.eye(10)))))
        overlaps[name]=float(np.linalg.norm(noise[0]@noise[1].T))
        assert overlaps[name]>1e-6
    assert largest_energy<=10
    for k,v in errors.items():assert v<1e-8,(k,v)
    old_path=H/'before_roundoff_guard/selection_before_validation.json'
    now=json.loads((H/'selection_before_validation.json').read_text('utf8'))
    if old_path.exists():assert json.loads(old_path.read_text('utf8'))==now
    assert hashlib.sha256((H/'selection_before_validation.json').read_bytes()).hexdigest()==json.loads((H/'roundoff_guard_record.json').read_text('utf8'))['prior_selection_sha256']
    extra=json.loads((H/'strong_baseline_checks.json').read_text('utf8'))
    for name,z in extra['models'].items():
        assert z['public_first_packet']['validation']['energy_max']<=10
        assert z['analytic_A0_P1']['validation']['energy_max']<=10
        for r in z['population_P3_validation'].values():
            assert r['metrics']['energy_max']<=10 and r['policy']['population_peak_damage']>=.1
    out=dict(date='2026-11-17',errors=errors,independent_target_updates=count,maximum_executed_energy=largest_energy,
      joint_overlap_cross_covariance_norms=overlaps,no_future_packet_dependence=True,A0_exact_embedding=True,
      selection_unchanged_after_roundoff_guard=True,new_physical_paths=0,
      actual_MSE_includes_cross_term=True,global_attack_optimality=False)
    (H/'independent_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
