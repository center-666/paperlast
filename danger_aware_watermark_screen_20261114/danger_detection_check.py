"""Second-formula finite moment and feasibility checks; no old script is run."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import eigh

H=Path(__file__).resolve().parent
INPUT=H/'danger_detection_comparison.json'
d=json.loads(INPUT.read_text(encoding='utf-8'))

def exact_marginal_atoms(scheme):
    I=np.eye(2)
    if scheme=='whole_sign':return [(-I,.5),(I,.5)]
    if scheme=='coordinate_sign':return [(np.diag(x),.25) for x in itertools.product([-1.,1.],repeat=2)]
    if scheme=='Haar_SO2':
        return [(np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]),.25) for t in [0.,np.pi/2,np.pi,3*np.pi/2]]
    if scheme in ['Gaussian','normalized_Gaussian']:
        z,w=hermgauss(3);atoms=[]
        nu=.6346361709395337
        for i,j in itertools.product(range(3),repeat=2):
            beta=np.sqrt(2)*np.array([z[i],z[j]])
            M=I+np.outer(beta,beta)
            if scheme=='normalized_Gaussian':M*=np.sqrt(nu)
            atoms.append((M,float(w[i]*w[j]/np.pi)))
        return atoms
    return None

def deterministic_M(scheme,t):
    if scheme=='identity':return np.eye(2)
    if scheme=='public_scalar_alternation':return (-1.)**t*np.eye(2)
    if scheme=='public_coordinate_alternation':return np.diag([1.,(-1.)**t])
    if scheme=='public_R90_supplement':return np.array([[0.,-1.],[1.,0.]])
    raise ValueError(scheme)

def second_moment_recursion(A,F,G,scheme):
    # Column-major v=vec(B): v'= (I kron F) v + vec(G M A^t).
    K=np.kron(np.eye(2),F);m=np.zeros(4);P=np.zeros((4,4));moments=[]
    atoms=exact_marginal_atoms(scheme)
    for t in range(15):
        if t<10:
            current=atoms if atoms is not None else [(deterministic_M(scheme,t),1.)]
            uv=[((G@M@np.linalg.matrix_power(A,t)).reshape(-1,order='F'),p) for M,p in current]
            mu=sum((p*u for u,p in uv),start=np.zeros(4))
            U=sum((p*np.outer(u,u) for u,p in uv),start=np.zeros((4,4)))
        else:mu=np.zeros(4);U=np.zeros((4,4))
        km=K@m;P=K@P@K.T+np.outer(km,mu)+np.outer(mu,km)+U;m=km+mu
        S=np.empty((2,2))
        for i,j in itertools.product(range(2),repeat=2):S[i,j]=sum(P[2*i+r,2*j+r] for r in range(2))
        moments.append(S)
    return np.array(moments)

def main():
    out=dict(input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        method='independent vec(B) second-moment recursion; 3-node Hermite exact through fourth degree; 4-angle rotation quadrature exact second moment',
        physical_noise_samples=0,models={})
    for name,md in d['models'].items():
        A=np.array(md['A']);F=np.array(md['F']);G=np.array(md['G']);W=np.array(md['W']);S0=np.array(md['no_watermark_prefix_moment_matrices'])
        assert np.max(abs(W-sum((np.linalg.matrix_power(A,t).T@np.linalg.matrix_power(A,t) for t in range(10)),start=np.zeros((2,2)))))<1e-12
        B=np.zeros((2,2));direct=[]
        for t in range(15):
            B=F@B+(G@np.linalg.matrix_power(A,t) if t<10 else 0)
            direct.append(B.T@B)
        geometry_error=float(np.max(abs(np.array(direct)-S0)))
        assert geometry_error<1e-12
        results={}
        for scheme,sd in md['schemes'].items():
            check=second_moment_recursion(A,F,G,scheme);S=np.array(sd['prefix_expected_moment_matrices'])
            error=float(np.max(abs(check-S)));assert error<1e-11,(name,scheme,error)
            sol=sd['probes'][-1]['danger_set_minimum'];xi=np.array(sol['initial_state'])
            energy=float(xi@W@xi);peak=max(np.einsum('i,kij,j->k',xi,S0,xi))
            assert energy<=10*(1+1e-9)
            assert abs(peak-.1)<1e-12
            results[scheme]=dict(moment_max_abs_error=error,dangerous_direction_energy=energy,
                dangerous_direction_no_watermark_peak=float(peak))
        # Independent exact full-10 whole-sign path enumeration for all15 times.
        signs=np.array(list(itertools.product([-1.,1.],repeat=10)))
        B=np.zeros((len(signs),2,2));enumeration=[]
        for t in range(15):
            B=np.einsum('ij,njk->nik',F,B)
            if t<10:B+=signs[:,t,None,None]*(G@np.linalg.matrix_power(A,t))[None,:,:]
            enumeration.append(np.einsum('nji,njl->il',B,B)/len(B))
        enum_error=float(np.max(abs(np.array(enumeration)-np.array(md['schemes']['whole_sign']['prefix_expected_moment_matrices']))))
        assert enum_error<1e-12
        # Check angle-envelope segments against an independent dense angular grid.
        ev,U=eigh(W);V=U@np.diag(1/np.sqrt(ev))@U.T;bad=0
        for theta in np.linspace(0,np.pi,8193)[1:-1]:
            u=V@np.array([np.cos(theta),np.sin(theta)]);qs=np.einsum('i,kij,j->k',u,S0,u)
            for interval in md['geometry']['intervals']:
                if interval['lo']<theta<interval['hi']:
                    bad+=int((max(qs)>=.01)!=interval['dangerous'])
                    if abs(theta-interval['lo'])>1e-10 and abs(theta-interval['hi'])>1e-10:
                        bad+=int(int(np.argmax(qs))+1!=interval['active_bias_index'])
                    break
        assert bad==0
        out['models'][name]=dict(no_watermark_prefix_recursion_error=geometry_error,
            full_1024_sign_prefix_enumeration_error=enum_error,
            dense_geometry_grid=8193,dense_geometry_mismatches=bad,schemes=results)
    (H/'danger_detection_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
