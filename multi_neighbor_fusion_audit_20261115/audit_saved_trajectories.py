"""Reconstruct saved P1 paths by affine observation maps, not innovation code."""
from pathlib import Path
import json,sys,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;P=H.parent
sys.path.insert(0,str(H));import joint_screen as js
import numpy as np
import sympy as s
def main():
    saved=np.load(H/'screen_arrays.npz');res=json.loads((H/'screen_results.json').read_text('utf8'));cert=[];rows={};worst=0.
    source=json.loads((P/'common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json').read_text('utf8'))
    Wexact={}
    for name in js.CFG['models']:
        A=s.Matrix([[s.Rational(str(x)) for x in r] for r in source['models'][name]['A']])
        for j in (0,2):
            Hj=s.Matrix([[s.Rational(str(x)) for x in r] for r in source['H'][j]])
            Wexact[name,j]=sum(((Hj*A**t).T*(Hj*A**t) for t in range(10)),s.zeros(2))
    grid=json.loads((H/'finite_attack_grid.json').read_text('utf8'))
    for case,r in grid.items():
        name,scope,jj,enc=case.split('/');j=int(jj[1:]);W=Wexact[name,j]
        candidates=r['xi_grid']+[js.PRE['models'][name]['attacks'][str(j)][enc][p]['xi'] for p in ('ordinary','P1')]
        for x in candidates:
            xi=s.Matrix([s.Rational(str(v)) for v in x]);en=(xi.T*W*xi)[0];assert en<=10
            cert.append(float(10-en))
    for case in res['cases']:
        name,scope,jj,enc=case.split('/');j=int(jj[1:]);tag=case.replace('/','__');m=js.ref.load_model(name)
        data,M=js.sample_inputs(m,name);data={k:v[64:] for k,v in data.items()};M=M[64:]
        if enc=='identity':M=np.broadcast_to(np.eye(2),M.shape).copy()
        xi=np.array(js.PRE['models'][name]['attacks'][str(j)][enc]['P1']['xi']);a=js.signals(m,j,xi[None])[0]
        y=data['y'].copy();y[:,:,j]+=np.einsum('ntij,tj->nti',M[:,:,j],a)
        # Unaffected nodes run the exact frozen raw-sharing model. This is a
        # network metric supplement, not additional independent physical data.
        other=np.zeros((64,32));yp=data['y']
        for i in (0,2,3,4):
            z=np.zeros((64,2))
            for k in range(32):
                z=z@m.F(i).T+yp[:,k,i]@m.K[i].T+sum(yp[:,k,l]@m.G[i].T for l in m.neighbors[i])
                other[:,k]+=np.sum((z-data['x'][:,k])**2,axis=1)
        ccase={}
        for b in js.SCHEMES:
            z=np.zeros((64,2));post=saved[tag+'__'+b+'__post'];ph=saved[tag+'__'+b+'__phi'];er=0.
            modes=js.PRE['models'][name]['fixed_modes' if b in ('B0','B1') else 'redesigned_modes']
            for k in range(32):
                code=ph[:,k,0]+2*ph[:,k,1];new=np.zeros_like(z)
                for mask in range(4):
                    take=code==mask;mode=modes[mask];K=np.array(mode['K']);G=np.array(mode['G']);F=np.array(mode['F'])
                    new[take]=z[take]@F.T+y[take,k,1]@K.T
                    for h,l in enumerate((0,2)):
                        if mask>>h&1:new[take]+=y[take,k,l]@G.T
                z=new;er=max(er,float(np.max(np.abs(z-post[:,k]))))
            assert er<1e-11;worst=max(worst,er)
            clean,_,_=js.closed(m,js.PRE['models'][name],b,data['y'],j if scope=='known' else None)
            normal_loss=np.sum((clean['post']-data['x'])**2,axis=2)
            actual_loss=np.sum((post-data['x'])**2,axis=2)
            norm_network=js.summary(((normal_loss+other)/5)[:,7:].mean(axis=1))
            att_network=js.summary(((actual_loss+other)/5)[:,7:].mean(axis=1))
            sync=bool(np.array_equal(ph[:,21:],clean['phi'][:,21:]))
            # hold5 is synchronized from k21; hysteresis has no deterministic flush time.
            if b!='B4H':assert sync
            ccase[b]=dict(affine_reconstruction_max_error=er,normal_network_MSE=norm_network,
              P1_network_MSE=att_network,modes_equal_from21_in_saved_bank=sync,
              deterministic_hold5_tail_claim=(b!='B4H'))
        rows[case]=ccase
    # Independent verification of raw Gaussian conditioning as an alternative
    # GLS fit for ONE saved packet on each of the two physical models.
    for name in js.CFG['models']:
        m=js.ref.load_model(name);data,_=js.sample_inputs(m,name);info=js.gaussian_info(m);y=data['y'][0,7]
        z=[np.linalg.solve(m.H[j],y[j]) for j in (0,1,2)];S=info['S']
        for h,(j,o) in enumerate(((0,2),(2,0))):
            fit=np.linalg.solve(np.linalg.inv(S[1])+np.linalg.inv(S[o]),np.linalg.solve(S[1],z[1])+np.linalg.solve(S[o],z[o]))
            u=z[j]-fit;r=np.r_[z[0]-z[1],z[2]-z[1]];E=np.zeros((4,2));E[2*h:2*h+2]=np.eye(2)
            beta=np.linalg.solve(E.T@info['Vi']@E,E.T@info['Vi']@r)
            assert np.max(np.abs(beta-u))<1e-12
    out=dict(date='2026-11-16',finite_attack_directions_exactly_checked=len(cert),all_exact_coding_energies_le10=True,
      minimum_energy_budget_slack=min(cert),affine_reconstruction_global_max=worst,cases=rows,
      standard_GLS_crosscheck=True,new_physical_realizations=0)
    js.dump(H/'independent_trajectory_checks.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='cases'}))
if __name__=='__main__':main()
