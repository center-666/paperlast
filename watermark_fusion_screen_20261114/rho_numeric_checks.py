"""Independent finite-distribution checks of random transition/damage moments."""
import json
import numpy as np
from scipy.special import roots_laguerre
from rho_numeric import H,data,model_params,normGM,stationary
from rho_expected_damage import damage,deterministic_moments,gaussian_moments
out={'date':'2026-11-14','method':'direct deterministic finite distribution, no physical/noise samples','rows':[]}
for name in data['models']:
    p=model_params(name);A,G=p['A'],p['G'];gamma=.55
    x,w=roots_laguerre(4);angles=np.pi*(np.arange(8)+.5)/8
    vals=[]
    for t in angles:
        v=np.array([np.cos(t),np.sin(t)])
        for u,ww in zip(x,w):
            M=np.eye(2)+2*u*np.outer(v,v);r=min(1.,gamma/float(normGM(G,u,t)))
            vals.append((float(ww/8),r,p['F0']-r*p['X'],r*G@M))
    means=dict(m1=sum(w*r for w,r,F,B in vals),m2=sum(w*r*r for w,r,F,B in vals),
        d=sum((w*B.reshape(4)for w,r,F,B in vals),start=np.zeros(4)),
        rd=sum((w*r*B.reshape(4)for w,r,F,B in vals),start=np.zeros(4)),
        dd=sum((w*np.outer(B.reshape(4),B.reshape(4))for w,r,F,B in vals),start=np.zeros((4,4))))
    S=np.zeros((2,2))
    for w1,r1,F1,D1 in vals:
        for w2,r2,F2,D2 in vals:
            B2=F2@D1+D2@A
            S+=w1*w2*(B2.T@B2)
    formula=np.array(damage(p,means,L=2)['S'])
    err=float(np.max(abs(S-formula)));assert err<1e-12
    # Constant/no-watermark exact convolution under the declared matrices.
    rr=.73;Fc=p['F0']-rr*p['X'];B=np.zeros((2,2))
    for j in range(10):B=Fc@B+rr*G@np.linalg.matrix_power(A,j)
    normal=np.array(damage(p,deterministic_moments(p,rr),L=10)['S'])
    err0=float(np.max(abs(normal-B.T@B)));assert err0<1e-12
    # Gaussian + fixed rho has constant F and known fourth-moment formula.
    S0=B.T@B;extra=np.zeros((2,2))
    for j in range(10):
        T=np.linalg.matrix_power(A,j);K=np.linalg.matrix_power(Fc,9-j)@(rr*G)
        extra+=T.T@(K.T@K+np.linalg.norm(K,'fro')**2*np.eye(2))@T
    SG=4*S0+extra
    frommom=np.array(damage(p,gaussian_moments(p,'fixed',rr,32,32),L=10)['S'])
    errg=float(np.max(abs(SG-frommom)));assert errg<1e-10
    _,P=stationary(p,means['m1'],means['m2'])
    J0,Y=p['J0'],p['Y'];Q,R,K=p['Q'],p['R'],p['K']
    RHS=np.zeros((2,2))
    for ww,r,F,B in vals:
        J=J0-r*Y
        RHS+=ww*(F@P@F.T+J@Q@J.T+K@R@K.T+G@R@G.T+r*r*G@R@G.T)
    errp=float(np.max(abs(RHS-P)));assert errp<1e-12
    out['rows'].append(dict(model=name,finite_nodes=32,two_step_pairs=1024,
        aware_B2_second_moment_error=err,constant_no_water_convolution_error=err0,
        fixed_Gaussian_fourth_moment_error=errg,stationary_covariance_residual=errp))
(H/'rho_numeric_checks.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
