"""Exact frozen-matrix bounds plus Arb Gaussian/probability enclosures."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import sympy as s
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
H=Path(__file__).resolve().parent;P=H.parent
sys.path.insert(0,str(P/'A1_G2_attack_information_defense_20261117'))
from risk_bounds import ba,arb,ctx,central_tail,nc_tail_upper,residual_info,enclosure,sm
from protocol import ref
ctx.prec=100
def encode(M):return [[str(v) for v in row] for row in M.tolist()]
def pd2(A):return A[0,0]>0 and A.det()>0
def covariance(m,mask):
    A=m.A;K=m.K[1];G=m.G[1];Ht=m.H[1]+0 # no changed K/G
    hs=sum((m.H[j] for j,on in zip([0,2],mask) if on),np.zeros((2,2)))
    J=np.eye(2)-K@Ht-G@hs;F=J@A
    V=K@m.R[1]@K.T+G@sum((m.R[j] for j,on in zip([0,2],mask) if on),np.zeros((2,2)))@G.T
    X=J@np.eye(2)@J.T+V;cost=np.trace(X)
    for k in range(1,32):
        X=F@X@F.T+J@m.Q@J.T+V;cost+=np.trace(X)
    steady=solve_discrete_lyapunov(F,J@m.Q@J.T+V)
    return dict(average_healthy_MSE_0_31=float(cost/32),steady_MSE=float(np.trace(steady)),
      spectral_radius=float(max(abs(np.linalg.eigvals(F)))))
def pulse_bound(g,df,tau,rho,pre_count=12):
    # Both original gains have e*.25*g^2>D. This condition is exact.
    e=s.Rational(10)*(1-s.Rational(1,10**12))**2
    assert e*g*g/4>s.Rational(1,10)
    pfa=central_tail(df,tau);v=arb(0);bins=64
    for i in range(bins):
        lo=s.Rational(4*i,bins);hi=s.Rational(4*(i+1),bins)
        upper=e*rho*(s.Rational(1,2)+hi)**2
        miss=1-nc_tail_upper(df,tau,upper)-pre_count*pfa
        if miss>0:v+=((-ba(lo)).exp()-(-ba(hi)).exp())*miss
    assert v>0
    return v
def main():
    cou=json.loads((P/'A1_G2_attack_information_defense_20261117/coupling_certificates.json').read_text('utf8'))['models']
    cfg=json.loads((H/'registration.json').read_text('utf8'));models={}
    for name in cfg['models']:
        m=ref.load_model(name);A=sm(m.A);G=sm(m.G[1]);K=sm(m.K[1]);I=s.eye(2);Hi=sm(m.H[1]);H0=sm(m.H[0]);H2=sm(m.H[2])
        J=I-K*Hi-G*(H0+H2);F=J*A;JR=I-K*Hi-G*H2;FR=JR*A
        # Rejection and full modes use the SAME original gains.
        assert pd2(s.Rational(16,25)*I-F.T*F)
        assert pd2(s.Rational(16,25)*I-FR.T*FR)
        g=s.Rational(cou[name]['Niu']['G7_sigma_min_lower_exact'])
        assert g>s.Rational(1,5)
        assert pd2(G.T*G-g*g*I)
        R=sm(m.R[1]);Rn=sm(m.R[0]+m.R[2]);Vnoise=K*R*K.T+G*Rn*G.T
        Pe=J*J.T+Vnoise
        for k in range(1,7):Pe=F*Pe*F.T+J*sm(m.Q)*J.T+Vnoise
        V=A*Pe*A.T+sm(m.Q);dropcov=G*(H0*V*H0.T+sm(m.R[0]))*G.T
        # A tiny predetermined nonzero edit is rejected. No valid payload can
        # be reconstructed from authentication alone. b7=-G*innovation0.
        sigma=ba(dropcov[0,0]);z=(ba(s.Rational(1,10))/(2*sigma)).sqrt()
        marginal=1-z.erf()
        rb={}
        for det,df,key in [('temporal',8,'temporal_threshold'),('joint_GLRT',10,'joint_threshold')]:
            tau=cfg[key];pfa=central_tail(df,tau);assert 28*pfa<ba(s.Rational(1,20))
            D,rho=residual_info(m,det)
            assert pd2(rho*I-D)
            # Uniform over10 fixed possible first-pulse times k7..16;
            # at most12 earlier monitored window ends. No independence used.
            low=pulse_bound(g,df,tau,rho,12)
            drop_lower=marginal-3*pfa;assert drop_lower>0
            rb[det]=dict(unprotected_pulse_uniform_risk_lower=enclosure(low),
              no_protection_k7_risk_lower=enclosure(pulse_bound(g,df,tau,rho,3)),
              full_timely_auth_statistical_only_rejection_risk_lower=enclosure(drop_lower),
              total_nominal_FA_upper=enclosure(28*pfa),pre_window_count_uniform=12,
              rho_exact=str(rho),private_B_out_of_10_factor='max(0,1-B/10)',
              full_timely_auth_combined_alarm_risk_exact='0',
              one_or_two_step_late_full_auth_risk_lower=enclosure(pulse_bound(g,df,tau,rho,3)))
        witness=-2*G*s.Matrix([1,0]);assert (witness.T*witness)[0]>s.Rational(1,10)
        small_xi=s.Matrix([s.Rational(1,1000),0])
        a0energy=sum(((H0*A**k*small_xi).dot(H0*A**k*small_xi) for k in range(10)),s.Integer(0))
        assert a0energy<10
        phealthy=J*V*J.T+Vnoise
        preject=JR*V*JR.T+K*sm(m.R[1])*K.T+G*sm(m.R[2])*G.T
        cross=-J*V*H0.T+G*sm(m.R[0])
        assert s.trace(preject-phealthy)==s.trace(dropcov)-2*s.trace(cross*G.T)
        models[name]=dict(full_F=encode(F),rejection_F=encode(FR),both_modes_common_norm_less_than='.8',
          G_sigma_min_lower_exact=str(g),all_secret_first_pulse_damage_lower_exact=str(10*(1-s.Rational(1,10**12))**2*g*g/4),
          rejection_covariance_at7_exact=encode(dropcov),drop_coordinate_danger_probability=enclosure(marginal),
          deterministic_rejection_witness=dict(healthy_target_innovation=['2','0'],tiny_coded_edit=['1/1000','0'],
            original_A0_extension_xi=['1/1000','0'],original_A0_L10_energy_exact=str(a0energy),
            bias_exact=encode(witness),damage_exact=str((witness.T*witness)[0]),
            interpretation='point plus an open continuous-support neighborhood; positive probability separately certified'),
          first_rejection_actual_MSE=dict(healthy_exact=str(s.trace(phealthy)),rejected_exact=str(s.trace(preject)),
            bias_squared_expectation_exact=str(s.trace(dropcov)),cross_term_exact=str(-2*s.trace(cross*G.T)),
            healthy_display=float(s.trace(phealthy)),rejected_display=float(s.trace(preject))),
          deterministic_availability_costs=dict(accept=covariance(m,[1,1]),permanent_target_reject=covariance(m,[0,1])),
          risk_bounds=rb)
        print(json.dumps(dict(model=name,drop_probability_lower=float(marginal.lower()),joint_pulse_risk_lower=float(ba(rb['joint_GLRT']['unprotected_pulse_uniform_risk_lower']['lower'])))))
    # Exact private-schedule minimax for the blind single-pulse surrogate:
    # sum p_k<=B => min p_k<=B/N; uniform B-subsets attain B/N.
    import itertools
    enum=[]
    for B in (0,1,5,9,10):
        cs=list(itertools.combinations(range(10),B));counts=[sum(k in x for x in cs) for k in range(10)]
        assert all(s.Rational(c,len(cs))==s.Rational(B,10) for c in counts)
        enum.append(dict(B=B,schedules=len(cs),protected_marginal_exact=str(s.Rational(B,10)),blind_pulse_unprotected_exact=str(1-s.Rational(B,10))))
    out=dict(executed='2026-11-18',models=models,private_uniform_subset_finite_check=enum,
      guarantees='frozen decimals as exact rational inputs; no DARE/device interval certification',
      public_B_less_than10='uniform positive single-pulse lower; B=10 combined risk0 only if every attacked packet timely checked',
      small_tamper_energy_exact='1/1000000',authentication_does_not_guarantee_delivery=True,
      resource_budget_physical_source_confirmed=False,new_physical_samples=0,Monte_Carlo=False)
    (H/'boundary_certificates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
if __name__=='__main__':main()
