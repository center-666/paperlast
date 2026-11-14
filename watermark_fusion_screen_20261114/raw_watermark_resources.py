"""m=2 Gaussian normalization on the ORIGINAL aggregate-model instance.

Normalization is a B-task resource diagnostic, not a change to D-task candidate.
Conditional powers are single first-window values, not event detection risks.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy.stats import ncx2,chi2
from numpy.polynomial.hermite import hermgauss
H=Path(__file__).resolve().parent
model=json.loads((H.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json').read_text())
expected=json.loads((H/'rho_expected_damage.json').read_text())
v,err=quad(lambda u:np.exp(-u)/(1+2*u)**2,0,np.inf,epsabs=1e-12)
nu=.5+.5*v;c=np.sqrt(nu)
out=dict(date='2026-11-14',m=2,lambda_=1,q=1,
    original_Gaussian_normal_packet_energy_ratio=nu,scalar_integral_error_estimate=err,
    normalized_decode_multiplier='sqrt(nu)*M',normalization_scale=c,
    normalized_Gaussian_packet_energy_ratio=1,sign_packet_energy_ratio=1,
    original_encoder_peak_operator_norm=1,normalized_encoder_peak_operator_norm=1/c,
    sign_encoder_peak_operator_norm=1,normalization_changed_task_D=False,
    same_packet_count=True,extra_encode_wait=0,
    Gnorm_expected_damage_minus_sign_PSD_analytic=True,
    proof='nu>1/2; covariance term nu(Ct^T Ct+||Ct||F^2 I)-Ct^T Ct is PSD, as ||Ct||F^2 I>=Ct^T Ct; add 4nu B0^T B0',
    schemes_not_equal_in_peak_cost_or_secret_generation=True,rows=[])
for name,m in model['models'].items():
 A=np.array(m['A']);G=np.array(m['nodes'][1]['G']);F=np.array(m['nodes'][1]['posterior_F'])
 b=expected['models'][name]['baselines'];W=np.array(b['original_Gaussian']['W'])
 Sg=np.array(b['original_Gaussian']['S']);Ss=np.array(b['original_sign']['S'])
 B=sum((np.linalg.matrix_power(F,9-j)@G@np.linalg.matrix_power(A,j)for j in range(10)),start=np.zeros((2,2)))
 S0=B.T@B
 vals={label:float(10*eigh(S,W,eigvals_only=True)[-1])for label,S in
       [('none',S0),('sign',Ss),('Gaussian',Sg),('normalized_Gaussian',nu*Sg)]}
 theta=np.zeros((8,8));R=.5*np.eye(2)
 for j in range(4):
  theta[2*j:2*j+2,2*j:2*j+2]=np.eye(2)+R+A@R@A.T
  if j<3:
   theta[2*j:2*j+2,2*j+2:2*j+4]=-R@A.T;theta[2*j+2:2*j+4,2*j:2*j+2]=-A@R
 Q=np.linalg.inv(theta)[-2:,-2:];_,_,vh=np.linalg.svd(G);a=np.sqrt(10)*vh[0]
 threshold=chi2.isf(.05/15,8);powers=[]
 for n in [80,128]:
  x,w=hermgauss(n);xx,yy=np.meshgrid(np.sqrt(2)*x,np.sqrt(2)*x,indexing='ij')
  z=np.column_stack([xx.ravel(),yy.ravel()]);weights=np.outer(w,w).ravel()/np.pi
  eff=a[None,:]+z*(z@a)[:,None]
  lam=np.einsum('ni,ij,nj->n',eff,Q,eff)
  powers.append(dict(order=n,Gaussian=float(weights@ncx2.sf(threshold,8,lam)),
        normalized_Gaussian=float(weights@ncx2.sf(threshold,8,nu*lam)),
        sign=float(ncx2.sf(threshold,8,float(a@Q@a)))))
 old=b['original_Gaussian']['P3_phys_first_window'];new_p3={}
 if old['feasible']:
  xi=np.array(old['initial_state'])/c
  new_p3=dict(transformed_saved_direction=xi.tolist(),energy=float(xi@W@xi),
       expected_terminal_damage=float(xi@(nu*Sg)@xi),
       same_conditional_ncp_pathwise_as_original_saved_direction=True,
       saved_direction_Pd=old['Pd'],global_P3_optimum_certified=False)
 out['rows'].append(dict(model=name,P1_expected_terminal_damage_L10=vals,
       P3_phys_feasible={k:val>=.1 for k,val in vals.items()},
       normalized_P3_saved_direction_check=new_p3,first_packet_fixed_attack_power=powers,
       normalization_damage_difference_min_eigenvalue_diagnostic=float(np.linalg.eigvalsh(nu*Sg-Ss).min())))
(H/'raw_watermark_resources.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'nu':nu,'normalization_scale':c,'P1':[r['P1_expected_terminal_damage_L10']for r in out['rows']]},indent=2))
