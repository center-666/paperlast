"""Small registered policy screening on OLD paired physical samples."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import numpy as np
from scipy.stats import chi2
from scipy.linalg import svd
import attack_models as am
H=Path(__file__).resolve().parent;P=H.parent
sys.path.insert(0,str(P/'active_trusted_measurement_G0_G1'))
import measurement_model as mm
CFG=am.CFG;ENDS=np.arange(4,32)
def dump(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf8')
def detectors(m,bank):
    out={}
    for label,kind in [('temporal','B0'),('joint_GLRT','B1')]:
        r=mm.raw_matrices(m,mm.rows_for_window(4,kind),None)
        ids=(0,) if kind=='B0' else (1,0,2)
        if kind=='B0':V=r['white'];target=V;df=len(V)
        else:
            choose=[t*6+2+c for t in range(5) for c in (0,1)]
            W=r['white'][:,choose];U,sv,_=svd(W,full_matrices=False);rank=int((sv>1e-10).sum())
            assert rank==10
            V=U[:,:rank].T@r['white'];target=V[:,choose];df=rank
        Y=np.stack([bank['y'][:,k-4:k+1][:,:,ids,:].reshape(len(bank['x']),-1) for k in ENDS],axis=1)
        clean=np.einsum('dw,ntw->ntd',V,Y)
        assert np.max(np.abs(V@r['cov']@V.T-np.eye(df)))<1e-9
        out[label]=dict(clean=clean,target=target,V=V,df=df,threshold=float(chi2.isf(.05/28,df)))
    return out
def evaluate(m,bank,pol,dets,details=False):
    a,meta=am.causal_policy(bank['q'],m.A,m.H[0],pol)
    delta=np.einsum('ntij,ntj->nti',bank['M'],a)
    b=np.zeros_like(a);prev=np.zeros((len(a),2))
    for k in range(32):
        prev=prev@m.F(1).T+delta[:,k]@m.G[1].T;b[:,k]=prev
    bias=np.einsum('nti,nti->nt',b,b)
    cross=2*np.einsum('nti,nti->nt',bank['e'],b)
    mse=np.einsum('nti,nti->nt',bank['e']+b,bank['e']+b)
    normal=np.einsum('nti,nti->nt',bank['e'],bank['e'])
    assert np.max(np.abs(mse-normal-bias-cross))<1e-10
    bad=bias>=.1;tauD=np.where(bad.any(1),bad.argmax(1),1000)
    out=dict(energy_mean=float(meta['energy'].mean()),energy_max=float(meta['energy'].max()),
       peak_mean_bias=float(bias.mean(0).max()),mean_actual_MSE=float(mse[:,7:].mean()),
       mean_MSE_increment=float((mse-normal)[:,7:].mean()),mean_bias_energy=float(bias[:,7:].mean()),
       mean_cross=float(cross[:,7:].mean()),danger_rate=float((tauD<1000).mean()),
       clipping_fraction=float(meta['clipped'][:,7:17].mean()),detectors={})
    alarms={};residuals={}
    windows=np.stack([delta[:,k-4:k+1].reshape(len(a),-1) for k in ENDS],axis=1)
    for lab,d in dets.items():
        r=d['clean']+np.einsum('dw,ntw->ntd',d['target'],windows)
        score=np.einsum('ntd,ntd->nt',r,r);alarm=score>d['threshold']
        first=np.where(alarm.any(1),ENDS[alarm.argmax(1)],1000)
        pre=(first<7);alive=~pre
        risk=tauD<first
        nominal=np.einsum('ntd,ntd->nt',d['clean'],d['clean'])
        rr=dict(event_alarm_rate=float(alarm.any(1).mean()),event_miss_rate=float((~alarm.any(1)).mean()),
          pre_alarm_danger_rate=float(risk.mean()),pre_attack_alarm_count=int(pre.sum()),
          alive_at_start_count=int(alive.sum()),risk_given_no_pre_alarm=float(risk[alive].mean()) if alive.any() else None,
          score_decrease_internal_fraction=float((score[:,7:13]<nominal[:,7:13]-1e-10).mean()),
          mean_first_alarm_capped_at32=float(np.minimum(first,32).mean()),
          mean_nominal_score=float(nominal.mean()),mean_attacked_score=float(score.mean()))
        out['detectors'][lab]=rr;alarms[lab]=dict(first=first,score=score,alarm=alarm);residuals[lab]=r
    if details:return out,dict(a=a,delta=delta,b=b,bias=bias,cross=cross,mse=mse,normal=normal,tauD=tauD,**meta),alarms,residuals
    return out
def policies(m):
    G0=json.loads((P/'active_trusted_measurement_G0_G1/batch_power_results.json').read_text('utf8'))['models'][m.name]['schemes']['B2']
    W=np.load(P/'active_trusted_measurement_G0_G1/batch_matrices.npz')[m.name+'__W']
    xis=[np.zeros(2)]
    for p in (.5,1.):
        for angle in np.linspace(0,2*np.pi,12,endpoint=False):
            u=np.array([np.cos(angle),np.sin(angle)]);xis.append(u*np.sqrt(10*p/(u@W@u))*(1-1e-12))
    for key in ('P1','P3'):
        u=np.array(G0[key]['xi']);xis.extend([u,-u])
    out=[]
    for family in ('A0','A1-H','A1-C'):
        for j,xi in enumerate(xis):
            for theta in ([0.] if family=='A0' else [-2.,-1.,-.25,0.,.25,1.]):
                out.append(dict(id=f'{family}/x{j}/t{theta}',family=family,xi=xi.tolist(),theta=theta,kind='projected_feedback'))
    out.append(dict(id='A1-C/feasible_reflection',family='A1-C',xi=[0.,0.],theta=-2.,kind='feasible_reflection'))
    return out
def select(rows):
    chosen={}
    for family in ('A0','A1-H','A1-C'):
        rr=[r for r in rows if r['policy']['family']==family]
        def keep(key,row):chosen[family+'/'+key]=row['policy']['id']
        keep('P1_peak_mean_bias',max(rr,key=lambda r:r['metrics']['peak_mean_bias']))
        keep('actual_MSE',max(rr,key=lambda r:r['metrics']['mean_MSE_increment']))
        for d in ('temporal','joint_GLRT'):
            keep('danger_'+d,max(rr,key=lambda r:r['metrics']['detectors'][d]['pre_alarm_danger_rate']))
            feasible=[r for r in rr if r['metrics']['peak_mean_bias']>=.1]
            if feasible:keep('P3_proxy_'+d,min(feasible,key=lambda r:(r['metrics']['detectors'][d]['event_alarm_rate'],-r['metrics']['peak_mean_bias'])))
    return chosen
def main():
    allrows={};selection={}
    for name in ('unstable','stable'):
        m=am.ref.load_model(name);bank=am.old_sample_bank(m,range(16));dets=detectors(m,bank);rows=[]
        for p in policies(m):rows.append(dict(policy=p,metrics=evaluate(m,bank,p,dets)))
        allrows[name]=rows;selection[name]=select(rows)
        print(json.dumps({'training_completed':name,'policies':len(rows),'chosen':selection[name]}),flush=True)
    dump('training_results.json',allrows)
    selection_record=dict(date='2026-11-16',selection=selection,training_only=True,
      registration_sha256=hashlib.sha256((H/'registration.json').read_bytes()).hexdigest(),validation_started=False)
    selection_path=H/'selection_before_validation.json'
    if selection_path.exists():assert json.loads(selection_path.read_text('utf8'))==selection_record
    else:dump(selection_path.name,selection_record)
    results={};arrays={};examples=[]
    for name in ('unstable','stable'):
        m=am.ref.load_model(name);bank=am.old_sample_bank(m,range(16,64));dets=detectors(m,bank)
        pool={r['policy']['id']:r['policy'] for r in allrows[name]}
        nominal={lab:dict(df=d['df'],threshold=d['threshold'],
          alarm_count=int((np.einsum('ntd,ntd->nt',d['clean'],d['clean'])>d['threshold']).any(1).sum()),
          samples=48) for lab,d in dets.items()}
        rows={}
        for pid in sorted(set(selection[name].values())|{'A1-C/feasible_reflection'}):
            p=pool[pid];metrics,paths,alarms,residuals=evaluate(m,bank,p,dets,True)
            rows[pid]=dict(policy=p,metrics=metrics)
            tag=name+'__'+pid.replace('/','_')
            for k in ('a','b','bias','cross','mse','tauD','energy'):arrays[tag+'__'+k]=paths[k]
            for lab,z in alarms.items():
                for k in ('first','score','alarm'):arrays[tag+'__'+lab+'__'+k]=z[k]
            if p['family']=='A1-C':
                for i in range(48):
                    for k in range(7,17):
                        ti=k-4
                        sc=alarms['temporal']['score'][i,ti];clean=dets['temporal']['clean'][i,ti]@dets['temporal']['clean'][i,ti]
                        if sc<clean-1e-6 and paths['bias'][i,k]>.1:
                            examples.append(dict(model=name,policy=pid,validation_index=i,old_index=i+16,k=k,
                              physical_seed=bank['physical_seeds'][i],watermark_seed=bank['watermark_seeds'][i],
                              temporal_clean=float(clean),temporal_attacked=float(sc),bias=float(paths['bias'][i,k]),
                              actual_MSE_increment=float(paths['mse'][i,k]-paths['normal'][i,k]),
                              cross=float(paths['cross'][i,k]),energy=float(paths['energy'][i])))
                            break
                    if examples and examples[-1]['policy']==pid:break
        results[name]=dict(nominal=nominal,selected=selection[name],validation_rows=rows,
          unique_old_validation_paths=48,new_physical_paths=0,
          population_P3_not_certified=True,global_attack_optimality_not_claimed=True)
        print(json.dumps({'validation_completed':name,'policies':len(rows)}),flush=True)
    dump('validation_results.json',dict(date='2026-11-16',models=results,new_physical_paths=0,
      selection_unchanged_after_validation=True,sample_rates_not_global_risk_certificates=True))
    dump('mechanism_witnesses.json',dict(witnesses=examples,meaning='saved old-path observations, not a theorem or global attack optimum'))
    np.savez_compressed(H/'validation_arrays.npz',**arrays)
if __name__=='__main__':main()
