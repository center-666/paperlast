"""Task A only: independently rebuilt original-compatible aggregate estimator.

No old script is imported or executed. No watermark is designed. All writes
are beside this script. Angles/quadrature are floating diagnostics; the main
link all-direction and peak-time conclusions have rational PSD certificates
and the existing exact-DARE error bridge.
"""
from pathlib import Path
from decimal import Decimal
import hashlib, json, math
import numpy as np
from scipy.linalg import solve_discrete_are, eigh
from scipy.integrate import quad
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
P=HERE.parent
E=10.;D=.1;L=10;N=15
RR=sp.Rational
HVALUES=[('1','1'),('1.1','1.2'),('1.2','1.4'),('1.3','1.1'),('1.4','1.3')]
H=[np.diag([float(v)for v in x])for x in HVALUES]
NEIGH=[[1,4],[0,2],[1,3],[2,4],[3,0]]
ASPEC={'unstable':[['1.08','.05'],['.01','1.05']],
       'stable':[['.8','.05'],['.01','.9']]}
oldpath=P/'original_estimator_gate_20261114/five_node_aggregate_model.json'
bridgepath=P/'original_estimator_gate_20261114/aggregate_gap_certificate.json'
old=json.loads(oldpath.read_text(encoding='utf-8-sig'))
oldexact=json.loads(oldpath.read_text(encoding='utf-8-sig'),parse_float=Decimal)
bridge=json.loads(bridgepath.read_text(encoding='utf-8-sig'))

def canonical_theta(v):
    return float(math.atan2(float(v[1]),float(v[0]))%math.pi)

def roots(T,level=0.):
    """All roots of v(theta)'T v(theta)=level on [0,pi)."""
    c=float(np.trace(T)/2-level);a=float((T[0,0]-T[1,1])/2);b=float(T[0,1])
    z=math.hypot(a,b)
    if z<1e-15:return []
    t=-c/z
    if t < -1-1e-12 or t > 1+1e-12:return []
    t=min(1.,max(-1.,t));p=math.atan2(b,a);h=math.acos(t)
    return sorted(set([float(((p-h)/2)%math.pi),float(((p+h)/2)%math.pi)]))

def qval(T,theta):
    v=np.array([math.cos(theta),math.sin(theta)])
    return float(v@T@v)

def unique_sorted(x,tol=1e-10):
    ans=[]
    for y in sorted(x):
        if not ans or y-ans[-1]>tol:ans.append(float(y))
    return ans

def merge_intervals(ints,tol=1e-10):
    out=[]
    for a,b in sorted(ints):
        if out and a<=out[-1][1]+tol:out[-1][1]=max(b,out[-1][1])
        else:out.append([float(a),float(b)])
    return out

def geometry(Ts,W,Wi):
    # Pairwise equality and threshold roots partition the exact trigonometric
    # envelope. Floating endpoint values are never called certified intervals.
    cuts=[0.,math.pi]
    for T in Ts:cuts+=roots(T,D/E)
    for k,T in enumerate(Ts):
        for U in Ts[:k]:cuts+=roots(T-U)
    cuts=unique_sorted(cuts)
    segments=[];danger=[];amin=(float('inf'),None,None);amax=(-1.,None,None)
    area=0.;quaderr=0.
    for a,b in zip(cuts[:-1],cuts[1:]):
        if b-a<1e-12:continue
        mid=(a+b)/2;k=int(np.argmax([qval(T,mid)for T in Ts]));T=Ts[k]
        active=qval(T,mid)>=D/E
        if active:
            danger.append([a,b]);val,err=quad(lambda t:1-D/(E*qval(T,t)),a,b,epsabs=1e-12,epsrel=1e-12)
            area+=val;quaderr+=err
        cand=[a,b]
        ev,V=eigh(T)
        for z in range(2):
            th=canonical_theta(V[:,z])
            if a-1e-10<=th<=b+1e-10:cand.append(th)
        for th in cand:
            q=max(qval(U,th)for U in Ts)
            if q<amin[0]:amin=(q,th,int(np.argmax([qval(U,th)for U in Ts]))+1)
            if q>amax[0]:amax=(q,th,int(np.argmax([qval(U,th)for U in Ts]))+1)
        if segments and segments[-1]['active_bias_state']==k+1 and segments[-1]['danger_direction']==active:
            segments[-1]['theta_end_rad']=b
        else:segments.append(dict(theta_start_rad=a,theta_end_rad=b,active_bias_state=k+1,danger_direction=bool(active)))
    ints=merge_intervals(danger)
    temporal=[]
    for n,T in enumerate(Ts,1):
        e,V=eigh(T);directions=[]
        for z in range(2):
            xi=math.sqrt(E)*Wi@V[:,z]
            if xi[0]<0:xi=-xi
            directions.append(dict(whitened_theta_rad=canonical_theta(V[:,z]),xi_energy10=xi.tolist(),
                raw_xi_angle_rad=canonical_theta(xi),full_energy_damage=float(E*e[z])))
        # Loewner domination is sufficient and easily inspectable; absence
        # does not by itself mean a different peak time is active.
        dom=min(float(eigh(T-U,eigvals_only=True)[0])for m,U in enumerate(Ts)if m!=n-1)
        temporal.append(dict(bias_state=n,eigenvalues=e.tolist(),weakest=directions[0],strongest=directions[1],
            dominates_all_other_times_min_eigenvalue=dom))
    return dict(danger_theta_intervals_rad=ints,danger_theta_intervals_degrees=[[a*180/math.pi,b*180/math.pi]for a,b in ints],
        direction_coverage=sum(b-a for a,b in ints)/math.pi,energy_ellipsoid_area_fraction=area/math.pi,
        numerical_quadrature_error_estimate=quaderr/math.pi,
        min_full_energy_peak_damage=E*amin[0],min_peak_whitened_theta_rad=amin[1],min_peak_active_bias_state=amin[2],
        max_full_energy_peak_damage=E*amax[0],max_peak_whitened_theta_rad=amax[1],max_peak_active_bias_state=amax[2],
        finite_envelope_segments=segments,per_time_generalized_eigendirections=temporal,
        terminal_b10_min_damage=E*float(eigh(Ts[9],eigvals_only=True)[0]),
        terminal_b10_max_damage=E*float(eigh(Ts[9],eigvals_only=True)[1]),
        endpoint_b15_max_damage=E*float(eigh(Ts[14],eigvals_only=True)[1]))

def exact_pd(M):
    assert M==M.T
    mins=[M[0,0],M.det()]
    assert all(v>0 for v in mins)
    return [str(v)for v in mins]

def main_certificate(name):
    # Use only saved decimal center F/G as exact rational certificate input.
    # The old DARE enclosure bridges this center to the declared physical
    # aggregate MMSE gain. It is not a device-uncertainty claim.
    A=sp.Matrix([[RR(v)for v in r]for r in ASPEC[name]])
    nd=oldexact['models'][name]['nodes'][1]
    F=sp.Matrix([[RR(str(v))for v in r]for r in nd['posterior_F']])
    G=sp.Matrix([[RR(str(v))for v in r]for r in nd['G']])
    I=sp.eye(2);W=sum(((A**t).T*A**t for t in range(L)),start=sp.zeros(2));B=sp.zeros(2);Bs=[];Ss=[]
    fcert=exact_pd(RR(1,25)*I-F.T*F)
    for t in range(L):assert sum(v*v for v in A**t)<16
    for k in range(N):
        B=F*B+(G*A**k if k<L else sp.zeros(2));assert sum(v*v for v in B)<16
        Bs.append(B);Ss.append(B.T*B)
    err=RR(next(r for r in bridge['rows']if r['model']==name)['saved_F_G_operator_error_upper'])
    assert err<RR(1,10**10)
    # e_B <= 1e-8 by induction: (1/5+1e-10)e_B+8e-10 < e_B.
    eB=RR(1,10**8);delta=RR(1,10**10)
    assert (RR(1,5)+delta)*eB+8*delta<eB
    # ||S_exact-S_center|| <= 8eB+eB^2 <1e-7.
    eS=RR(1,10**7);assert 8*eB+eB**2<eS
    n=9 if name=='unstable'else 0
    # All energy-boundary rays are dangerous at one shared time.
    Q=10*Ss[n]-RR(1,10)*W-10*eS*I
    all_dirs=exact_pd(Q)
    # Same single time attains the whole b1..b15 envelope in every direction.
    dom=[]
    for k,S in enumerate(Ss):
        if k!=n:dom.append(dict(other_bias_state=k+1,leading_minors=exact_pd(Ss[n]-S-2*eS*I)))
    return dict(method='exact rational PSD at saved F/G plus prior exact-DARE norm enclosure',
        evaluated_bias_states=list(range(1,16)),shared_peak_state=n+1,
        W10=[[str(v)for v in r]for r in W.tolist()],F_center=[[str(v)for v in r]for r in F.tolist()],
        G_center=[[str(v)for v in r]for r in G.tolist()],F_norm_below_one_fifth_leading_minors=fcert,
        old_DARE_error_bound=str(err),uniform_B_error_upper=str(eB),uniform_S_error_upper=str(eS),
        all_energy_boundary_directions_strictly_dangerous_leading_minors=all_dirs,
        common_peak_time_domination_certificates=dom,
        exact_DARE_original_five_node_aggregate_only=True,original_50_node_model_certified=False)

def main():
    allmodels={};plotdata={}
    for name,aspec in ASPEC.items():
        A=np.array(aspec,dtype=float);nodes=[];links={}
        for i in range(5):
            C=np.vstack([H[i],sum((H[j]for j in NEIGH[i]),start=np.zeros((2,2)))]);V=np.diag([.5,.5,1,1])
            prior=solve_discrete_are(A.T,C.T,np.eye(2),V)
            gain=prior@C.T@np.linalg.inv(C@prior@C.T+V);K=gain[:,:2];G=gain[:,2:]
            F=(np.eye(2)-gain@C)@A
            dare=np.max(np.abs(prior-(A@(prior-prior@C.T@np.linalg.solve(C@prior@C.T+V,C@prior))@A.T+np.eye(2))))
            saved=old['models'][name]['nodes'][i]
            diff=max(float(np.max(np.abs(F-np.array(saved['posterior_F'])))),float(np.max(np.abs(G-np.array(saved['G'])))))
            assert diff<1e-12
            nodes.append(dict(node=i,neighbors=NEIGH[i],C=C.tolist(),R_aggregate=V.tolist(),
                prior_covariance=prior.tolist(),K=K.tolist(),G=G.tolist(),F=F.tolist(),DARE_max_residual=float(dare),
                max_saved_F_G_difference=diff))
            for j in NEIGH[i]:
                W=sum(((H[j]@np.linalg.matrix_power(A,t)).T@(H[j]@np.linalg.matrix_power(A,t))for t in range(L)),start=np.zeros((2,2)))
                e,U=eigh(W);Wi=(U/np.sqrt(e))@U.T;B=np.zeros((2,2));Bs=[];Ts=[];Ss=[]
                for k in range(N):
                    B=F@B+(G@H[j]@np.linalg.matrix_power(A,k)if k<L else np.zeros((2,2)))
                    Bs.append(B.copy());Ss.append(B.T@B);Ts.append(Wi@Ss[-1]@Wi)
                key=f'{j}->{i}';geo=geometry(Ts,W,Wi)
                links[key]=dict(source=j,target=i,H_source=H[j].tolist(),W10=W.tolist(),W10_inverse_sqrt=Wi.tolist(),
                    B_by_bias_state=[x.tolist()for x in Bs],S_by_bias_state=[x.tolist()for x in Ss],
                    energy_whitened_S_by_bias_state=[x.tolist()for x in Ts],geometry=geo)
                if j==0 and i==1:plotdata[name]=(Ts,Wi)
        mainlink=links['0->1'];peakidx=9 if name=='unstable'else 0
        Tmain=np.array(mainlink['energy_whitened_S_by_bias_state'][peakidx])
        closed_area=1-D/(E*math.sqrt(np.linalg.det(Tmain)))
        assert abs(closed_area-mainlink['geometry']['energy_ellipsoid_area_fraction'])<1e-12
        mainlink['geometry']['main_single_peak_closed_form_area_fraction']=closed_area
        allmodels[name]=dict(A=A.tolist(),nodes=nodes,links=links,main_link_rational_certificate=main_certificate(name))
    output=dict(date='2026-11-14',task='A only; no watermark design',attack='a_t=H_source A^t xi, t0..9; zero outside',
        estimator='independently reconstructed original common-G neighbor-sum aggregate MMSE; supplementary five-node instance',
        energy=10,danger_threshold=.1,duration=10,bias_states=list(range(1,16)),packet_indices=list(range(15)),
        danger_metric='attack-induced bias squared; not full estimation error, normal MSE or an attack probability',
        coordinate_interface='xi=r W10^(-1/2) (cos(theta),sin(theta)); theta in [0,pi), r^2<=10',
        danger_set='r^2 max_n v(theta)^T T_n v(theta) >= .1, T_n=W^-1/2 B_n^T B_n W^-1/2',
        input_hashes={str(q.relative_to(P)):hashlib.sha256(q.read_bytes()).hexdigest()for q in [oldpath,bridgepath,Path(__file__)]},
        angle_values_status='analytic trigonometric formulas evaluated in float; not rigorous angular interval enclosures',
        area_status='Lebesgue area ratio in the fixed energy ellipsoid, not a stochastic attacker probability',
        physical_noise_samples=0,old_scripts_run=False,models=allmodels)
    (HERE/'danger_geometry.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    fig,axs=plt.subplots(2,2,figsize=(11.4,7.4),layout='constrained')
    theta=np.linspace(0,math.pi,1501);vec=np.array([np.cos(theta),np.sin(theta)])
    for row,(name,(Ts,Wi))in enumerate(plotdata.items()):
        qs=np.array([np.einsum('in,ij,jn->n',vec,T,vec)for T in Ts]);qp=qs.max(axis=0)
        ax=axs[row,0];ax.plot(theta*180/math.pi,E*qp,label='peak b1..b15',lw=2.2)
        ax.plot(theta*180/math.pi,E*qs[9],label='terminal b10',ls='--');ax.axhline(D,c='firebrick',ls=':',label='D=0.1')
        ax.set(xlim=(0,180),xlabel='energy-whitened projective angle (degrees)',ylabel='bias squared at energy E=10',title=f'{name}: link 0 -> 1')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax=axs[row,1];tt=np.linspace(0,2*math.pi,1501);vv=np.array([np.cos(tt),np.sin(tt)])
        qq=np.max([np.einsum('in,ij,jn->n',vv,T,vv)for T in Ts],axis=0);rad=np.sqrt(D/qq)
        outer=math.sqrt(E)*vv;inner=rad*vv
        ax.fill(outer[0],outer[1],color='#d95f5f',alpha=.3,label='danger region')
        ax.fill(inner[0],inner[1],color='white',zorder=2,label='below D')
        ax.plot(outer[0],outer[1],c='black',lw=1);ax.plot(inner[0],inner[1],c='firebrick',lw=1.3)
        ax.set(aspect='equal',xlabel='whitened xi coordinate 1',ylabel='whitened xi coordinate 2',title='fixed energy ball and danger boundary')
        ax.grid(alpha=.15);ax.legend(fontsize=8,loc='upper right')
    fig.savefig(HERE/'danger_geometry_main.png',dpi=180)
    fig.savefig(HERE/'danger_geometry_main.pdf')
    print(json.dumps({name:{k:v['geometry'][k]for k in ['direction_coverage','energy_ellipsoid_area_fraction','min_full_energy_peak_damage','max_full_energy_peak_damage','terminal_b10_max_damage']}for name,d in allmodels.items()for link,v in d['links'].items()if link=='0->1'},indent=2))

if __name__=='__main__':main()
