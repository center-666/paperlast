from pathlib import Path
from dataclasses import replace
import json,sys,hashlib,importlib.metadata
sys.dont_write_bytecode=True
import numpy as np
from protocol import ref,run,bank_path,Packet,IdealVerifier,CausalActor,INF
H=Path(__file__).resolve().parent;P=H.parent
def main():
    checks=[];rows=[];saved={};refpath=P/'common_g_reference_revision_20261115/common_g_reference.py'
    before=hashlib.sha256(refpath.read_bytes()).hexdigest()
    def ck(n,z):
        assert z,n
        checks.append(dict(test=n,passed=True))
    old=json.loads((P/'A1_G2_attack_information_defense_20261117/training_metrics.json').read_text('utf8'))
    schedules=[('none',np.zeros(32,bool),0,False),('all',np.ones(32,bool),0,False),
      ('public_even',np.arange(32)%2==0,0,False),
      ('private_five_in_support',np.array([k in (7,9,11,13,15) for k in range(32)]),0,True),
      ('late1_all',np.ones(32,bool),1,False),('late2_all',np.ones(32,bool),2,False)]
    for name in ('unstable','stable'):
        m=ref.load_model(name)
        r=old[name]['Niu']['rows']
        word=lambda n:r[n]['policy']['plan']
        pp=r['H_signed1']['policy'];pc=r['C_signed1']['policy']
        plans=[
          dict(id='none',kind='none',family='OL'),
          dict(id='A0_original',kind='plan',family='A0',word=word('A0_peak1')),
          dict(id='H',kind='signed',family='H',word=pp['plan'],ell=pp['ell'],obs=6),
          dict(id='C',kind='signed',family='C',word=pc['plan'],ell=pc['ell'],obs=7),
          dict(id='OL_pulse7',kind='pulse',family='OL',time=7,packet=[np.sqrt(10)*(1-1e-12),0.]),
          dict(id='tiny_edit',kind='pulse',family='OL',time=7,packet=[.001,0.]),
          dict(id='protected_timing_C',kind='signed',family='C',word=pc['plan'],ell=pc['ell'],obs=7,avoid_public_protected=True)]
        for idx in (0,1):
            path,M=bank_path(m,idx);q,rec=ref.transmit_decode(m,path['y'],M)
            reference=ref.run_common(m,path['y'],rec)['posterior']
            for sn,gamma,delay,private in schedules:
                healthy=run(m,path,M,plans[0],gamma,delay,private)
                ck(name+str(idx)+sn+'.healthy_ideal_auth_equals_original',
                   np.max(abs(healthy['posterior']-reference))<1e-10)
                ck(name+str(idx)+sn+'.all_healthy_packets_accepted',healthy['available'].all())
                ck(name+str(idx)+sn+'.no_auth_false_reject',healthy['resource']['failures']==0)
                for p in plans[1:]:
                    if p.get('avoid_public_protected') and private:continue
                    a=run(m,path,M,p,gamma,delay,private)
                    b=a['posterior'][:,1]-healthy['posterior'][:,1]
                    norm=np.sum(b*b,axis=1);bad=norm>=.1;tD=int(np.flatnonzero(bad)[0]) if bad.any() else INF
                    gates=[e['accepted'] for e in a['logs'] if e['phase']=='decode_admission']
                    prior_b=np.zeros(2);error=0.;terms=[];bh=np.zeros(2)
                    # Independent expanded affine update, with two actual masks.
                    for k in range(32):
                        z=int(gates[k]);p0=m.A@healthy['posterior'][k-1,1] if k else np.zeros(2)
                        u=path['y'][k,0]-m.H[0]@p0
                        F=(np.eye(2)-m.K[1]@m.H[1]-m.G[1]@(m.H[2]+z*m.H[0]))@m.A
                        bh=F@bh+m.G[1]@(z*(M[k,0]@a['actions'][k])+(z-1)*u)
                        error=max(error,float(np.max(abs(bh-b[k]))))
                    ck(name+str(idx)+sn+p['id']+'.switching_identity',error<1e-10)
                    ck(name+str(idx)+sn+p['id']+'.hard_energy',a['energy']<=10)
                    ck(name+str(idx)+sn+p['id']+'.no_state_pollution_at_other_nodes',
                       np.max(abs(a['posterior'][:,[0,2,3,4]]-healthy['posterior'][:,[0,2,3,4]]))<1e-10)
                    ck(name+str(idx)+sn+p['id']+'.one_output_per_deadline',
                       sum(e['phase']=='update' for e in a['logs'])==32)
                    ck(name+str(idx)+sn+p['id']+'.natural_stopping_tail',
                       max(np.max(abs(b[k]-m.F(1)@b[k-1])) for k in range(17,32))<1e-10)
                    if sn=='all':
                        ck(name+str(idx)+p['id']+'.all_changed_packets_rejected',
                           all(not gates[k] for k in range(32) if np.any(a['actions'][k])))
                        ck(name+str(idx)+p['id']+'.no_danger_strictly_before_combined_alarm',
                           not tD<min(a['tauth'],a['tstat']['joint_GLRT']))
                    if p['id']=='OL_pulse7' and sn.startswith('late'):
                        raw=run(m,path,M,p,np.zeros(32,bool))
                        ck(name+str(idx)+sn+'.cannot_undo_published_outputs',
                           np.max(abs(a['posterior']-raw['posterior']))<1e-10)
                        ck(name+str(idx)+sn+'.auth_verdict_late',a['tauth']==7+delay)
                        ck(name+str(idx)+sn+'.first_pulse_danger',tD==7)
                    if p.get('avoid_public_protected'):
                        ck(name+str(idx)+sn+'.masked_waveform_explicitly_distinct_from_A0',
                           not np.any(a['actions'][gamma]))
                    rows.append(dict(model=name,index=idx,schedule=sn,plan=p['id'],energy=a['energy'],
                      tau_D=tD,tau_auth=a['tauth'],tau_stat=a['tstat'],
                      risk_event_stat={d:bool(tD<t) for d,t in a['tstat'].items()},
                      risk_event_combined={d:bool(tD<min(t,a['tauth'])) for d,t in a['tstat'].items()},
                      maximum_bias_squared=float(norm.max()),actual_MSE_mean=float(np.mean(np.sum((a['posterior'][:,1]-path['x'])**2,axis=1))),
                      healthy_MSE_mean=float(np.mean(np.sum((healthy['posterior'][:,1]-path['x'])**2,axis=1))),
                      recurrence_error=error,resource=a['resource']))
                    if idx==0 and p['id'] in ('tiny_edit','OL_pulse7') and sn in ('all','late1_all','none'):
                        key=name+'__'+sn+'__'+p['id'];saved[key+'__bias']=b;saved[key+'__post']=a['posterior']
                        if name=='unstable' and sn=='all' and p['id']=='tiny_edit':
                            (H/'protocol_trace_tiny_edit.json').write_text(json.dumps(a['logs'],ensure_ascii=False,indent=2)+'\n','utf8')
            # Causal information tests: changing q7 cannot affect H7.
            h=CausalActor(plans[2]);h2=CausalActor(plans[2])
            for k in range(8):
                x,v=h.act(k,None);x2,v2=h2.act(k,None)
                ck(name+str(idx)+f'.H_nonanticipative_{k}',np.array_equal(x,x2))
                h.observe_after_decision(q[k,0]);h2.observe_after_decision(q[k,0]+(1000 if k==7 else 0))
            ck(name+str(idx)+'.private_schedule_not_in_attack_view',v.public_gamma is None)
            altered={k:v.copy() for k,v in path.items()}
            altered['y'][17:]+=100
            M2=M.copy();M2[17:]*=3
            p=plans[4]
            first=run(m,path,M,p,np.ones(32,bool),2)
            second=run(m,altered,M2,p,np.ones(32,bool),2)
            ck(name+str(idx)+'.future_data_and_secrets_do_not_change_past_outputs',
               np.max(abs(first['posterior'][:17]-second['posterior'][:17]))<1e-10)
    # Binding, replay, timestamp and domain checks are protocol tests, not new
    # energy-budget FDI attack models.
    v=IdealVerifier();p=v.issue(Packet(0,1,1,7,np.zeros(2,dtype='<f8').tobytes()))
    ck('modified_payload_rejected',not v.verify(replace(p,payload=np.ones(2,dtype='<f8').tobytes()),7))
    ck('modified_time_rejected',not v.verify(replace(p,time=8),8))
    ck('modified_receiver_rejected',not v.verify(replace(p,receiver=2),7))
    ck('modified_epoch_rejected',not v.verify(replace(p,epoch=2),7))
    ck('untampered_valid',v.verify(p,7));ck('replay_rejected',not v.verify(p,7))
    ck('reference_bytes_unchanged',hashlib.sha256(refpath.read_bytes()).hexdigest()==before)
    result=dict(executed='2026-11-18',checks=checks,passed=len(checks),cases=rows,
      reused_physical_fixtures=4,new_physical_samples=0,probability_estimation=False,
      sampled_rates_are_not_reported=True,statistical_monitors_separate_FA_budgets=True,
      authentication_interface='ideal token Boolean; not a MAC implementation or timing benchmark',
      dependency_versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','sympy')},
      reference_sha256=before)
    (H/'protocol_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8')
    np.savez_compressed(H/'protocol_regression.npz',**saved)
    print(json.dumps(dict(passed=len(checks),cases=len(rows),new_physical_samples=0)))
if __name__=='__main__':main()
