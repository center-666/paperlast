"""Deterministic matrix diagnostics on the NEW original-aggregate instance.

No old four-dimensional F/g, simulation, attacker oracle, or codebook is used.
Analytic claims are stated in report two; this script is floating diagnostics.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.linalg import block_diag,eigh
from scipy.stats import chi2,ncx2
from numpy.polynomial.hermite import hermgauss

H=Path(__file__).resolve().parent
source=H/'five_node_aggregate_model.json'
data=json.loads(source.read_text(encoding='utf-8'))
E=10.;D=.1;LMAX=11;s=4;alpha_event=.05;tests=LMAX+s

def blocks(F,G,k,L):
    return [np.linalg.matrix_power(F,k-1-j)@G if j<k else np.zeros_like(G) for j in range(L)]

def template(A,L,cut=None):
    if cut is None:return np.vstack([np.linalg.matrix_power(A,j) for j in range(L)])
    return block_diag(template(A,cut),template(A,L-cut))

def attack_envelopes(F,G,A,L,cut=None):
    T=template(A,L,cut);W=T.T@T
    # H_attack=I in the separately declared compatible original-model instance.
    rows=[]
    for k in range(1,L+1):
        cb=blocks(F,G,k,L);C=np.hstack(cb)
        S=np.diag(np.repeat([(-1.)**j for j in range(L)],2))
        iid=block_diag(*[b.T@b for b in cb])
        gm=4*C.T@C+block_diag(*[b.T@b+np.trace(b.T@b)*np.eye(2) for b in cb])
        mats={'none':C.T@C,'public_alternation':S@C.T@C@S,
              'iid_sign_expected':iid,'Gaussian_lambda1_q1_expected':gm}
        row={'state':k}
        for name,Q in mats.items():row[name]=float(E*eigh(T.T@Q@T,W,eigvals_only=True)[-1])
        row['arbitrary_FDI_pathwise_sign_or_none']=float(E*np.linalg.norm(C,2)**2)
        row['arbitrary_FDI_iid_sign_expected']=float(E*max(np.linalg.norm(b,2)**2 for b in cb))
        row['arbitrary_FDI_Gaussian_expected']=float(E*np.linalg.eigvalsh(gm)[-1])
        rows.append(row)
    keys=[k for k in rows[0] if k!='state']
    return {'L':L,'cut':cut,'energy':'sum norm of actual encoded injections squared <= 10',
            'template_dimension':T.shape[1],
            'maxima':{key:max(rows,key=lambda x:x[key])[key] for key in keys},
            'maximizing_states':{key:max(rows,key=lambda x:x[key])['state'] for key in keys},
            'rows':rows}

def first_packet_test(F,G,A):
    # Equivalent complete parity basis for H_attack=I: y_l-A*y_(l-1).
    Z=np.zeros((2*s,2*(s+1)))
    for j in range(s):Z[2*j:2*j+2,2*j:2*j+2]=-A;Z[2*j:2*j+2,2*j+2:2*j+4]=np.eye(2)
    Theta=np.zeros((2*s,2*s));R=.5*np.eye(2);Q=np.eye(2)
    for j in range(s):
        Theta[2*j:2*j+2,2*j:2*j+2]=Q+R+A@R@A.T
        if j+1<s:
            Theta[2*j:2*j+2,2*j+2:2*j+4]=-R@A.T
            Theta[2*j+2:2*j+4,2*j:2*j+2]=-A@R
    M=np.linalg.inv(Theta)[-2:,-2:]
    _,_,vh=np.linalg.svd(G);a=np.sqrt(E)*vh[0]
    lam=float(a@M@a);at=alpha_event/tests
    tau8=float(chi2.isf(at,2*s));tau2=float(chi2.isf(at,2))
    p8=float(ncx2.sf(tau8,2*s,lam));p2=float(ncx2.sf(tau2,2,lam))
    gaussian=[]
    for order in [20,32,48,80,128]:
        x,w=hermgauss(order);xp,yp=np.meshgrid(np.sqrt(2)*x,np.sqrt(2)*x,indexing='ij')
        beta=np.column_stack([xp.ravel(),yp.ravel()]);weights=np.outer(w,w).ravel()/np.pi
        eff=a[None,:]+beta*(beta@a)[:,None]
        lams=np.einsum('bi,ij,bj->b',eff,M,eff)
        gaussian.append({'order_per_coordinate':order,'points':order*order,
            'ordinary_parity_Pd':float(weights@ncx2.sf(tau8,2*s,lams)),
            'standard_linear_GLRT_Pd':float(weights@ncx2.sf(tau2,2,lams))})
    # Joint covariance of adjacent parity residual windows: shared primitive
    # process/noise blocks, not an independence approximation.
    V=2*(s+2);J=2*(s+1)
    def noise_map(end):
        N=np.zeros((2*s,J+V))
        for row,t in enumerate(range(end-s+1,end+1)):
            N[2*row:2*row+2,2*(t-1):2*t]=np.eye(2)
            N[2*row:2*row+2,J+2*t:J+2*t+2]=np.eye(2)
            N[2*row:2*row+2,J+2*(t-1):J+2*t]=-A
        return N
    N0=noise_map(s);N1=noise_map(s+1);Vraw=block_diag(np.eye(J),.5*np.eye(V))
    joint=np.block([[N0@Vraw@N0.T,N0@Vraw@N1.T],
                    [N1@Vraw@N0.T,N1@Vraw@N1.T]])
    assert np.max(abs(joint[:2*s,:2*s]-Theta))<1e-12
    return {'parity_order':s,'single_test_alpha':at,'finite_test_count':tests,
        'finite_event_union_bound':alpha_event,'attack':a.tolist(),'attack_energy':float(a@a),
        'first_posterior_damage':float(np.linalg.norm(G@a)**2),'ordinary_df':2*s,'GLRT_rank':2,
        'noncentrality':lam,'ordinary_parity_Pd':p8,'standard_linear_GLRT_Pd':p2,
        'no_watermark_iid_sign_and_public_sign_same_for_this_single_impulse':True,
        'Gaussian_quadrature_diagnostics':gaussian,
        'overlapping_windows_joint_covariance':joint.tolist(),
        'overlap_cross_covariance_Frobenius_norm':float(np.linalg.norm(joint[:2*s,2*s:],'fro')),
        'unknown_start_event_detection_probability':None,
        'danger_before_alarm_probability':None,
        'note':'Single first-packet score using four healthy preceding samples, monitor begins at attack packet. No repeated-window independence assumption.'}

result={'date':'2026-11-14','type':'floating deterministic diagnostics; not rigorous interval certificate',
        'input_sha256':hashlib.sha256(source.read_text(encoding='utf-8-sig').replace('\r\n','\n').encode('utf-8')).hexdigest(),
        'input_hash_format':'UTF-8/LF normalized text; original source provenance retains byte hashes',
        'E':E,'D':D,'Lmax':LMAX,'Lmax_role':'small declared check covers previous L9/10/11; not original experimental parameter',
        'state_quantity':'posterior estimate displacement; predictive state needs original Phi/Psi separately',
        'models':{},'physical_samples':0,'Monte_Carlo':0,'old_four_dimensional_model_used':False}
for name,model in data['models'].items():
    node=model['nodes'][1];F=np.array(node['posterior_F']);G=np.array(node['posterior_attack_G']);A=np.array(model['A'])
    norm=float(np.linalg.norm(F,2));assert norm<1
    one=[attack_envelopes(F,G,A,L) for L in range(1,LMAX+1)]
    two=[attack_envelopes(F,G,A,L,c) for L in range(2,LMAX+1) for c in range(1,L)]
    result['models'][name]={'F_operator_norm':norm,
        'tail_statement':'For this numerical instance norm(F)<1, after final injection every posterior norm decreases. General theory still requires a tail proof.',
        'one_segment':one,'two_segment':two,'first_packet_detection':first_packet_test(F,G,A)}
(H/'variable_attack_envelopes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name,m in result['models'].items():
    print(name,'first damage',m['first_packet_detection']['first_posterior_damage'],
          'first Pd ordinary/GLRT',m['first_packet_detection']['ordinary_parity_Pd'],m['first_packet_detection']['standard_linear_GLRT_Pd'])
    print('single maxima', {k:max(v['maxima'][k] for v in m['one_segment']) for k in m['one_segment'][0]['maxima']})
    print('two maxima', {k:max(v['maxima'][k] for v in m['two_segment']) for k in m['two_segment'][0]['maxima']})
