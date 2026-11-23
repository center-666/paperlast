"""Exploratory n=4,m=2 extension of the frozen 50-node common-G system.

Each sensor sees only one invariant 2-D block. The original graph, six attack
waveforms, message-space parity detector and signed Watson law are retained.
This is explicitly a new physical model, never a relabeling of old results.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import platform
import sys
import numpy as np
import scipy
from scipy.linalg import block_diag, solve_discrete_are
from scipy.stats import beta as beta_dist, t as student_t

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
for folder in ('paper1_50node_tv_gate_theory_20261121','paper1_sscd_commonrandom_20261122'):
    sys.path.insert(0,str(ROOT/folder))
from source_model import load_model
from replay import scores
from watson_crn import make_common_draws,beta_from_common,watermark_attack_from_beta


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(x):
    x=np.asarray(x,float);se=float(x.std(ddof=1)/np.sqrt(len(x)))
    c=float(student_t.ppf(.975,len(x)-1))
    return {'mean':float(x.mean()),'path_se':se,'n':len(x),
            'conditional_unadjusted_ci95':[float(x.mean()-c*se),float(x.mean()+c*se)]}


def obs_rank(A,C):
    O=np.vstack([C@np.linalg.matrix_power(A,j) for j in range(A.shape[0])])
    return int(np.linalg.matrix_rank(O,tol=1e-10))


def design(label,cfg):
    old=load_model(label)
    A=block_diag(old.A,old.A);Q=block_diag(old.Q,old.Q)
    H=np.zeros((50,2,4));types=np.arange(50)%2
    for i in range(50):H[i,:,2*types[i]:2*types[i]+2]=old.H[i]
    adjacency=np.asarray(cfg['graph']['adjacency'],float)
    rows=[];lookup={};min_complement=50;parity_residual=0.
    for i in range(50):
        incident=[e for e,(r,s) in enumerate(old.links) if r==i]
        for bits in range(1<<len(incident)):
            dropped=[e for j,e in enumerate(incident) if bits&(1<<j)]
            removed={int(old.links[e,1]) for e in dropped}
            active=[j for j in cfg['graph']['neighbors'][i] if j not in removed]
            opposite=sum(types[j]!=types[i] for j in active)
            min_complement=min(min_complement,int(opposite))
            if not opposite:
                raise ArithmeticError(f'Undetectable registered mode receiver={i}, dropped={dropped}; no fallback gain')
            C=np.vstack([H[i],H[active].sum(axis=0)])
            V=block_diag(old.R[i],old.R[active].sum(axis=0))
            P=solve_discrete_are(A.T,C.T,Q,V)
            L=np.linalg.solve(C@P@C.T+V,C@P).T
            F=(np.eye(4)-L@C)@A
            post=P-P@C.T@np.linalg.solve(C@P@C.T+V,C@P)
            residual=float(np.linalg.norm(A@post@A.T+Q-P,'fro'))
            row={'receiver':i,'dropped_edges':dropped,'active_neighbors':active,
                 'complementary_neighbor_count':int(opposite),'observability_rank':obs_rank(A,C),
                 'K':L[:,:2].tolist(),'G':L[:,2:].tolist(),
                 'F':F.tolist(),'norm_F_2':float(np.linalg.norm(F,2)),
                 'rho_F':float(max(abs(np.linalg.eigvals(F)))),'dare_residual_fro':residual}
            lookup[i,tuple(dropped)]=len(rows);rows.append(row)
    mapping=[[lookup[i,tuple(e for e,(r,s) in enumerate(old.links) if r==i and mask&(1<<e))]
              for i in range(50)] for mask in range(64)]
    idx=np.asarray(mapping)
    old.A2=old.A;old.H2=old.H;old.A=A;old.Q=Q;old.H=H;old.n=4
    old.types=types;old.adjacency=adjacency;old.Hsum=np.einsum('ij,jmn->imn',adjacency,H)
    old.K=np.asarray([r['K'] for r in rows])[idx]
    old.G=np.asarray([r['G'] for r in rows])[idx]
    old.F=np.asarray([r['F'] for r in rows])[idx]
    local_ranks=[obs_rank(A,H[i]) for i in range(50)]
    for e,(i,j) in enumerate(old.links):
        O=np.vstack([H[j]@np.linalg.matrix_power(A,k) for k in range(5)])
        parity_residual=max(parity_residual,float(np.linalg.norm(old.Z[e]@O)))
    eigenvalues=np.linalg.eigvals(A)
    pbh_ranks=[int(np.linalg.matrix_rank(np.vstack([lam*np.eye(4)-A,H[0]]),tol=1e-10))
               for lam in eigenvalues if abs(lam)>=1]
    report={'state_dimension':4,'message_dimension':2,'local_observability_ranks':local_ranks,
            'network_observability_rank':obs_rank(A,H.reshape(-1,4)),
            'minimum_complementary_neighbors_in_registered_modes':min_complement,
            'maximum_sender_parity_annihilation_residual':parity_residual,
            'all_neighbor_deletion_node0_PBH_ranks_on_unstable_eigenvalues':pbh_ranks,
            'all_neighbor_deletion_node0_detectable':all(r==4 for r in pbh_ranks),
            'local_mode_count':len(rows),'global_mask_to_local_row':mapping,'rows':rows,
            'max_norm_F_2':max(r['norm_F_2'] for r in rows),
            'max_rho_F':max(r['rho_F'] for r in rows),
            'max_dare_residual_fro':max(r['dare_residual_fro'] for r in rows),
            'eigenvalues_A':[[float(v.real),float(v.imag)] for v in eigenvalues]}
    assert len(rows)==57 and set(local_ranks)=={2}
    assert report['network_observability_rank']==4
    assert max(r['dare_residual_fro'] for r in rows)<1e-8
    assert max(r['observability_rank'] for r in rows)==min(r['observability_rank'] for r in rows)==4
    assert parity_residual<1e-10
    return old,report


def physical(model,count,seed):
    rng=np.random.default_rng(seed);x=np.empty((count,150,4));x[:,0]=rng.normal(size=(count,4))
    chol=np.linalg.cholesky(model.Q)
    for k in range(1,150):x[:,k]=x[:,k-1]@model.A.T+rng.normal(size=(count,4))@chol.T
    y=np.einsum('jmn,btn->btjm',model.H,x)+rng.normal(size=(count,150,50,2))*np.sqrt(.5)
    return SimpleNamespace(x=x,y=y)


def matvec(A,x):return np.einsum('...ij,...j->...i',A,x)


def run_filter(model,phy,delta,threshold,override=None,literal=False):
    B=len(phy.x);sc=scores(model,phy,delta);gates=sc>threshold
    if override is not None:gates=np.asarray(override,bool).copy()
    masks=np.sum(gates*(1<<np.arange(6)),axis=-1).astype(int)
    post=np.zeros((B,150,50,4));forcing=np.zeros((B,150))
    edge_lookup={tuple(map(int,edge)):e for e,edge in enumerate(model.links)}
    for k in range(1,150):
        pred=post[:,k-1]@model.A.T;K=model.K[masks[:,k]];G=model.G[masks[:,k]]
        local=phy.y[:,k]-matvec(model.H[None],pred)
        injection=np.zeros((B,50,2))
        for e,(i,j) in enumerate(model.links):injection[:,i]+=(~gates[:,k,e,None])*delta[:,k,e]
        if literal:
            innovation=np.zeros((B,50,2))
            for i,neighbors in enumerate(model.cfg['graph']['neighbors']):
                for j in neighbors:
                    e=edge_lookup.get((i,j));packet=phy.y[:,k,j];keep=np.ones(B,bool)
                    if e is not None:packet=packet+delta[:,k,e];keep=~gates[:,k,e]
                    innovation[:,i]+=keep[:,None]*(packet-matvec(model.H[j],pred[:,i]))
        else:
            yy=np.einsum('ij,bjm->bim',model.adjacency,phy.y[:,k])
            HS=np.broadcast_to(model.Hsum,(B,50,2,4)).copy()
            for e,(i,j) in enumerate(model.links):
                yy[:,i]+=(~gates[:,k,e,None])*delta[:,k,e]-gates[:,k,e,None]*phy.y[:,k,j]
                HS[:,i]-=gates[:,k,e,None,None]*model.H[j]
            innovation=yy-matvec(HS,pred)
        post[:,k]=pred+matvec(K,local)+matvec(G,innovation)
        forcing[:,k]=np.sum(matvec(G,injection)**2,axis=(1,2))/50
    return SimpleNamespace(post=post,gates=gates,scores=sc,masks=masks,forcing=forcing)


def measure(model,phy,world):
    square=(world.post-phy.x[:,:,None,:])**2;err=square.sum(axis=-1)
    missing=np.stack([square[:,:,i,2*(1-model.types[i]):2*(1-model.types[i])+2].sum(axis=-1)
                      for i in range(50)],axis=-1)
    return {'network_post20':err[:,20:].mean(axis=(1,2)),
            'affected_post20':err[:,20:,model.receiver_ids].mean(axis=(1,2)),
            'missing_block_post20':missing[:,20:].mean(axis=(1,2)),
            'locally_observed_block_post20':(err-missing)[:,20:].mean(axis=(1,2)),
            'network_attack':err[:,50:110].mean(axis=(1,2)),
            'accepted_attack_forcing_post20':world.forcing[:,20:].mean(axis=1),
            'eligible_link_time_alarm_fraction':world.gates[:,57:110].mean(axis=(1,2))}


def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--execute',action='store_true');args.add_argument('--design-only',action='store_true')
    a=args.parse_args();reg=json.loads((HERE/'registration.json').read_text(encoding='utf-8'))
    config=ROOT/'paper1_common_g50_gain_acceptance_20261121/physical_configuration.json'
    assert sha(config)==reg['input_sha256'];cfg=json.loads(config.read_text(encoding='utf-8'))
    if not a.execute:
        print(json.dumps({'status':'DRY_RUN','registration':reg},indent=2));return
    output=HERE/'partial50_results.json';archive_path=HERE/'partial50_paths.npz'
    if output.exists() or archive_path.exists():raise FileExistsError('Refusing to overwrite partial50 outputs')
    result={'kind':'NEW_N4_M2_LOCALLY_UNOBSERVABLE_50_NODE_EXPLORATORY',
            'registration_sha256':sha(HERE/'registration.json'),'script_sha256':sha(__file__),
            'source_configuration_sha256':sha(config),'source_root_commit_reference':'fbeab8d',
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
            'limitations':reg['limitations'],'models':{}};archive={}
    for mi,label in enumerate(('unstable','stable')):
        model,design_report=design(label,cfg)
        print(label,'designed',design_report['local_mode_count'],'modes; max norm',design_report['max_norm_F_2'],flush=True)
        (HERE/f'{label}_gain_design.json').write_text(json.dumps(design_report,indent=2)+'\n',encoding='utf-8',newline='\n')
        if a.design_only:continue
        base=reg['seed']+mi*100_000
        seeds={'calibration':base+1,'healthy_holdout':base+2,'attack_physical':base+3,
               'watermark_radial':base+4,'watermark_initial':base+5,'watermark_transitions':base+6}
        cal=physical(model,reg['calibration_paths_per_model'],seeds['calibration'])
        maxima=np.max(scores(model,cal,np.zeros((len(cal.x),150,6,2)))[:,4:],axis=(1,2))
        rank=int(np.ceil((len(maxima)+1)*.95));threshold=float(np.partition(maxima,rank-1)[rank-1])
        archive[f'{label}__calibration_maxima']=maxima
        healthy=physical(model,reg['healthy_holdout_paths_per_model'],seeds['healthy_holdout'])
        zero=np.zeros((len(healthy.x),150,6,2));healthy_gated=run_filter(model,healthy,zero,threshold)
        hits=healthy_gated.gates[:,4:].any(axis=(1,2));k=int(hits.sum());n=len(hits)
        record={'design_summary':{key:value for key,value in design_report.items() if key not in ['rows','global_mask_to_local_row']},
                'seeds':seeds,'threshold':threshold,'calibration_rank':rank,
                'healthy_alarms':k,'healthy_paths':n,'healthy_path_fwer':float(hits.mean()),
                'healthy_fwer_exact_ci95':[float(beta_dist.ppf(.025,k,n-k+1)) if k else 0.,
                                           float(beta_dist.ppf(.975,k+1,n-k)) if k<n else 1.],
                'healthy':{},'attacked':{}}
        archive[f'{label}__healthy_any_alarm']=hits
        health_metrics=measure(model,healthy,healthy_gated)
        record['healthy']['gated']={metric:summary(vals) for metric,vals in health_metrics.items()}
        for metric,vals in health_metrics.items():archive[f'{label}__healthy_gated__{metric}']=vals
        static=np.zeros((n,150,6),bool);static[:,1:]=True
        for name,override in [('all_registered_retained',np.zeros_like(static)),('static_six_drop',static)]:
            w=run_filter(model,healthy,zero,threshold,override)
            vals=measure(model,healthy,w);record['healthy'][name]={m:summary(v) for m,v in vals.items()}
            for metric,v in vals.items():archive[f'{label}__healthy_{name}__{metric}']=v
        phy=physical(model,reg['attacked_paths_per_model'],seeds['attack_physical']);B=len(phy.x)
        draws=make_common_draws(B,60,6,seeds['watermark_radial'],seeds['watermark_initial'],seeds['watermark_transitions'])
        metrics={}
        for name,theta in reg['watermarks_fixed_before_run'].items():
            delta=watermark_attack_from_beta(model.attack[50:110],beta_from_common(draws,theta),150,50)
            w=run_filter(model,phy,delta,threshold);metrics[name]=measure(model,phy,w)
            for metric,v in metrics[name].items():archive[f'{label}__{name}__{metric}']=v
            record['attacked'][name]={metric:summary(v) for metric,v in metrics[name].items()}
            if name=='iid':
                small=SimpleNamespace(x=phy.x[:2],y=phy.y[:2]);dsmall=delta[:2]
                pattern=np.broadcast_to((np.arange(150)%64)[None,:],(2,150))
                schedule=(pattern[:,:,None]&(1<<np.arange(6)))!=0
                literal=run_filter(model,small,dsmall,threshold,schedule,literal=True)
                aggregated=run_filter(model,small,dsmall,threshold,schedule)
                difference=float(np.max(np.abs(literal.post-aggregated.post)))
                assert difference<1e-8
                record['independent_literal_aggregate_max_abs_difference']=difference
                # Exact endogenous-mode paired error identity for n=4.
                clean=run_filter(model,small,np.zeros_like(dsmall),threshold)
                attacked=run_filter(model,small,dsmall,threshold)
                residual=0.
                for t in range(1,150):
                    bprev=attacked.post[:,t-1]-clean.post[:,t-1]
                    p0=clean.post[:,t-1]@model.A.T
                    Ka=model.K[attacked.masks[:,t]];K0=model.K[clean.masks[:,t]]
                    Ga=model.G[attacked.masks[:,t]];G0=model.G[clean.masks[:,t]]
                    nu=small.y[:,t]-matvec(model.H[None],p0)
                    raw=np.einsum('ij,bjm->bim',model.adjacency,small.y[:,t])-matvec(model.Hsum[None],p0)
                    ia=raw.copy();i0=raw.copy();inj=np.zeros_like(raw)
                    for e,(i,j) in enumerate(model.links):
                        nn=small.y[:,t,j]-matvec(model.H[j],p0[:,i])
                        ia[:,i]-=attacked.gates[:,t,e,None]*nn;i0[:,i]-=clean.gates[:,t,e,None]*nn
                        inj[:,i]+=(~attacked.gates[:,t,e,None])*dsmall[:,t,e]
                    forcing=matvec(Ka-K0,nu)+matvec(Ga,ia)-matvec(G0,i0)+matvec(Ga,inj)
                    rr=attacked.post[:,t]-clean.post[:,t]-matvec(model.F[attacked.masks[:,t]],bprev)-forcing
                    residual=max(residual,float(np.max(np.abs(rr))))
                assert residual<1e-8
                record['endogenous_gate_paired_identity_max_abs_residual']=residual
            print(label,name,record['attacked'][name]['network_post20']['mean'],flush=True)
        metrics['static_six_drop']=measure(model,phy,run_filter(model,phy,np.zeros((B,150,6,2)),threshold,static))
        record['attacked']['static_six_drop']={m:summary(v) for m,v in metrics['static_six_drop'].items()}
        for metric,v in metrics['static_six_drop'].items():archive[f'{label}__static_six_drop__{metric}']=v
        record['paired_vs_iid']={name:{metric:summary(v-metrics['iid'][metric]) for metric,v in vals.items()}
                                 for name,vals in metrics.items() if name!='iid'}
        result['models'][label]=record
    if not a.design_only:
        np.savez_compressed(archive_path,**archive);result['path_archive_sha256']=sha(archive_path)
        output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps({'status':'COMPLETE','output':str(output),'path_arrays':len(archive)},indent=2))


if __name__=='__main__':main()
