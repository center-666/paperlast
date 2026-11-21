"""Hash-pinned source-function loading and exact 50-node model metadata.

No plotting, global source import, old Monte Carlo wrapper or attack optimizer
is executed. Original function validation optionally instruments only return.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import hashlib
import json
import numpy as np
from scipy.linalg import block_diag, null_space, pinv

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'paper1_original50_source_20261121/source/simulation_revised_detection_metrics_complete_v2.py'
EXPECTED='2f4694e6180680ccd83ecc4ae60cb587b76276625c9d34c5946cfeca2d3e8a4a'


def load_model(label):
    cfg=json.loads((ROOT/'paper1_original50_source_20261121/configuration_snapshot.json').read_text())
    defaults=cfg['defaults'].copy();defaults['output_dir']=Path(defaults['output_dir'])
    defaults['attack_links']=[tuple(link) for link in defaults['attack_links']]
    raw=SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise ValueError('source hash mismatch')
    H=np.asarray(cfg['H'],float);R=np.asarray(cfg['R'],float)
    if not np.allclose(H,np.eye(2)[None]*np.diagonal(H,axis1=1,axis2=2)[:,None,:]):
        raise ValueError('fast replay requires diagonal recovered H matrices')
    if not np.allclose(R,.5*np.eye(2)):raise ValueError('fast replay requires recovered R=.5I')
    A=np.asarray(cfg['A_'+label],float);Q=np.asarray(cfg['Q'],float)
    links=np.asarray([[x['receiver'],x['sender']] for x in cfg['attack_links']],int)
    scope={'np':np,'Params':SimpleNamespace(**defaults),'H':np.moveaxis(H,0,2),
           'R':np.moveaxis(R,0,2),'Q':Q,'neighbors_in':cfg['graph']['neighbors'],
           'block_diag':block_diag,'null_space':null_space,'pinv':pinv}
    tree=ast.parse(raw.decode('utf-8-sig'));nodes=[]
    fnames={'build_parity_space','generate_zda_sequence','get_zda_biases','compute_KL_divergence',
            'metric_intervals','run_single_simulation'}
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name=='DynamicWatermark':nodes.append(copy.deepcopy(node))
        elif isinstance(node,ast.FunctionDef) and node.name in fnames:
            n=copy.deepcopy(node)
            if n.name=='run_single_simulation':
                last=n.body[-1]
                if not isinstance(last,ast.Return):raise ValueError('unexpected source return structure')
                n.body[-1]=ast.parse("return dict(x_true=x_true, y_nominal=y_nominal, x_hat=x_hat, P_cov=P_cov, scores=kld_stats, gates=isolated_links, watermark=watermark, attacks=zda_bias)").body[0]
            nodes.append(n)
    module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(module,str(SOURCE),'exec'),scope)
    Zs,Thetas=scope['build_parity_space'](A)
    evals,evecs=np.linalg.eig(A);dom=np.real(evecs[:,np.argmax(abs(evals))]);dom/=np.linalg.norm(dom)
    magnitude=6. if label=='unstable' else 10.
    xi=magnitude*dom
    d=np.zeros((150,len(links),2))
    biases=scope['get_zda_biases'](A,magnitude,60)
    for e,(i,j) in enumerate(links):d[50:110,e]=biases[int(j)]
    closed=np.asarray(cfg['graph']['adjacency'],float)+np.eye(50)
    hdiag=np.diagonal(H,axis1=1,axis2=2)
    info_packet=2*hdiag*hdiag
    Jfull=closed@info_packet
    receiver_ids=np.unique(links[:,0]);receiver_index={int(v):i for i,v in enumerate(receiver_ids)}
    rbar=.5*np.linalg.norm(A,'fro')
    return SimpleNamespace(label=label,cfg=cfg,A=A,Q=Q,H=H,R=R,hdiag=hdiag,closed=closed,
        links=links,source_functions=scope,Z=np.stack([Zs[int(j)] for i,j in links]),
        Theta=np.stack([Thetas[int(j)] for i,j in links]),info_packet=info_packet,Jfull=Jfull,
        receiver_ids=receiver_ids,receiver_index=receiver_index,xi=xi,attack=d,
        coded_energy=float(np.sum(d*d)),rbar=float(rbar),n=2,m=2,T=150,N=50)


def inv2(A):
    A=np.asarray(A,float);det=A[...,0,0]*A[...,1,1]-A[...,0,1]*A[...,1,0]
    if np.any(det<=0) or not np.all(np.isfinite(det)):raise ArithmeticError('nonpositive 2x2 determinant')
    out=np.empty_like(A);out[...,0,0]=A[...,1,1];out[...,1,1]=A[...,0,0]
    out[...,0,1]=-A[...,0,1];out[...,1,0]=-A[...,1,0]
    return out/det[...,None,None]


def matvec(A,x):return np.einsum('...ij,...j->...i',A,x)


def covariance_schedule(model,static_drop=False):
    """Exact recovered no-gate Riccati schedule, initial algorithmic P=5I."""
    P=np.broadcast_to(5*np.eye(2),(50,2,2)).copy()
    Jfull=model.Jfull.copy()
    if static_drop:
        for i,j in model.links:Jfull[int(i)]-=model.info_packet[int(j)]
    Fs=np.zeros((150,50,2,2));stars=np.zeros_like(Fs);Ps=np.zeros_like(Fs);Ps[0]=P
    Fs[0]=np.eye(2)
    for k in range(1,150):
        pred=model.A@P@model.A.T+model.Q;precision=inv2(pred)
        info=precision.copy();info[:,0,0]+=Jfull[:,0];info[:,1,1]+=Jfull[:,1]
        star=inv2(info);Fs[k]=star@precision@model.A;stars[k]=star
        P=star+1e-9*np.eye(2);Ps[k]=P
    return SimpleNamespace(F=Fs,star=stars,P=Ps,information=Jfull)


def physical_paths(model,count,seed):
    rng=np.random.default_rng(seed);x=np.empty((count,150,2));x[:,0]=rng.normal(size=(count,2))
    for k in range(1,150):x[:,k]=x[:,k-1]@model.A.T+rng.normal(size=(count,2))
    y=x[:,:,None,:]*model.hdiag[None,None]
    y=y+rng.normal(size=(count,150,50,2))*np.sqrt(.5)
    return prepare_physical(model,x,y)


def prepare_physical(model,x,y):
    weighted=2*np.asarray(y)*model.hdiag[None,None]
    suff=(weighted.swapaxes(-1,-2)@model.closed.T).swapaxes(-1,-2)
    return SimpleNamespace(x=np.asarray(x),y=np.asarray(y),weighted=weighted,suff=suff)
