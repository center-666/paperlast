"""Alternate GLS, primitive-noise and finite-optimization checks."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
import measurement_model as mm
from batch_power_and_attacks import secrets
from online_bounds import global_noise_map
import numpy as np
from scipy.stats import ncx2,chi2
def main():
    source=json.loads((H/'batch_power_results.json').read_text('utf8'));errors=dict(GLS_info=0.,shared_noise=0.,normal_covariance=0.,template_cover=0.)
    tests=0;opt=[]
    for name in ('unstable','stable'):
        m=mm.ref.load_model(name);M=secrets(m)[0]
        for c in (0,1):
          for var in (.25,.5,1.):
           for delay in (0,1):
            for rho in (0.,.5):
             aux=mm.schedule(coordinate=c,variance=var,delay=delay,rho=rho)
             for end in (10,11):
                G,_,r=mm.one_batch_info(m,7,'B1',aux)
                # Use this actual window's map, not the helper batch's end11.
                r=mm.raw_matrices(m,mm.rows_for_window(end,'B1',aux),aux);D=mm.attack_map(m,r['rows'],7)
                Vi=np.linalg.inv(r['cov']);O=r['O'];P=Vi-Vi@O@np.linalg.solve(O.T@Vi@O,O.T@Vi)
                G1=D.T@P@D;G2=(r['white']@D).T@(r['white']@D)
                errors['GLS_info']=max(errors['GLS_info'],float(np.max(np.abs(G1-G2))))
                L=global_noise_map(m,r['rows'],aux);err=np.max(np.abs(r['white']@L@L.T@r['white'].T-np.eye(len(r['white']))))
                errors['shared_noise']=max(errors['shared_noise'],float(err));tests+=1
        # Independent normal error covariance: concatenate coefficients of x0,
        # all process noise and measurement noises, rather than a Riccati call.
        A=m.A;K,G=m.K[1],m.G[1];J=m.J(1);F=m.F(1);L=np.zeros((2,256));Pc=[]
        for k in range(32):
            if k==0:L[:,:2]=-J
            else:L=F@L;L[:,2+2*(k-1):4+2*(k-1)]=-J@np.linalg.cholesky(m.Q)
            for j in (0,1,2):L[:,64+(k*3+j)*2:64+(k*3+j)*2+2]=(K if j==1 else G)@np.linalg.cholesky(m.R[j])
            Pc.append(L@L.T)
        errors['normal_covariance']=max(errors['normal_covariance'],float(np.max(np.abs(np.array(Pc)-mm.normal_covariances(m)))))
        # Unknown start is NOT passed into the detector. Check that nine public
        # templates cover every nonempty length10 truncation in these windows.
        for end in range(4,32):
            r=mm.raw_matrices(m,mm.rows_for_window(end,'B1'),None)
            bases=[np.linalg.qr(r['white']@D)[0] for D in mm.template_maps(m,r['rows'],M)]
            for Ta in range(23):
                U=r['white']@mm.attack_map(m,r['rows'],Ta,M)
                if np.linalg.norm(U)<1e-12:continue
                e=min(np.linalg.norm(U-B@B.T@U)/np.linalg.norm(U) for B in bases)
                errors['template_cover']=max(errors['template_cover'],float(e));tests+=1
        # Independent dense-ray check of the analytically enumerated P3. These
        # are algebraic verification directions, not new deployment attacks.
        S=np.load(H/'batch_matrices.npz')[name+'__S_identity'];W=np.load(H/'batch_matrices.npz')[name+'__W']
        for label in ('B0','B1','B3_coord0','B3_coord1'):
            row=source['models'][name]['schemes'][label];data=np.load(H/'batch_matrices.npz')[name+'__'+label+'__G'].reshape(-1,2,2)[0]
            minimum=2.
            for t in np.linspace(0,np.pi,721,endpoint=False):
                u=np.array([np.cos(t),np.sin(t)]);h=max(u@v@u for v in S);x=u*np.sqrt(.1/h)
                if x@W@x<=10:minimum=min(minimum,float(ncx2.sf(chi2.isf(.05,row['df']),row['df'],max(0.,x@data@x))))
            assert row['P3']['Pd']<=minimum+1e-8,(name,label,row['P3']['Pd'],minimum)
            assert row['P3']['energy']<=10+1e-10 and row['P3']['expected_peak_damage']>=.1
            opt.append(dict(model=name,label=label,analytic_candidate_Pd=row['P3']['Pd'],independent721_ray_min=minimum))
    for k,v in errors.items():assert v<1e-8,(k,v)
    result=dict(date='2026-11-16',errors=errors,independent_checks=tests,P3_crosschecks=opt,
      full_window_blindness_and_counterexamples_exact=True,unknown_onset_template_cover_verified=True,
      new_physical_samples=0,probability_values_not_interval_certified=True,new_G2_algorithm=False)
    (H/'independent_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='P3_crosschecks'}))
if __name__=='__main__':main()
