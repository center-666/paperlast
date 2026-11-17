"""Exact algebra and finite-energy witnesses; independent of screen rankings."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
import sympy as s
import numpy as np
from scipy.stats import chi2
import attack_models as am
H=Path(__file__).resolve().parent;P=H.parent
def sm(x):return s.Matrix([[s.Rational(str(v)) for v in row] for row in x])
def enc(x):return [[str(v) for v in row] for row in x.tolist()]
def main():
    z=s.Matrix(s.symbols('y0:2'));u,v,w,t=s.symbols('u v w t')
    M=s.Matrix([[u,v],[w,t]]);c=s.symbols('c')
    assert s.simplify(M*((c-1)*M.inv()*z)-(c-1)*z)==s.zeros(2,1)
    source=json.loads((P/'common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json').read_text('utf8'))
    seedmap=json.loads((P/'attack_gain_switch_audit_20261115/sampling_seed_map.json').read_text('utf8'))
    rows=[]
    for name in ('unstable','stable'):
        model=am.ref.load_model(name);seeds=seedmap[name]
        # No parameter optimization: first saved validation input satisfying the
        # predeclared quarter-scale one-packet finite-resource witness.
        for i in range(16,64):
            path=am.ref.draw_physical_path(model,32,seeds['physical'][i])
            _,beta=am.ref.original_watermarks(model,32,seeds['watermark'][i],.5,.5)
            B=s.Matrix([s.Rational(str(v)) for v in beta[7,0]])
            M=s.eye(2)/2+B*B.T;y=s.Matrix([s.Rational(str(v)) for v in path['y'][7,0]])
            q=M.inv()*y;a=-q/4;energy=(a.T*a)[0]
            if energy>=10:continue
            assert M*a==-y/4
            node=source['models'][name]['nodes'][1];G=sm(node['G']);K=sm(node['K'])
            A=sm(source['models'][name]['A']);R=sm(source['R'][1]);R2=sm(source['R'][2])
            U=A*(K*R*K.T+G*R2*G.T)*A.T
            assert U[0,0]>0 and U.det()>0
            F=(s.eye(2)-K*sm(source['H'][1])-G*(sm(source['H'][0])+sm(source['H'][2])))*A
            assert s.trace(F.T*F)<s.Rational(1,25)
            # Exact healthy Gaussian regression at the FIRST attacked packet.
            J=F*A.inv();Px=s.eye(2);C=s.zeros(2);H0=sm(source['H'][0])
            for k in range(8):
                if k:Px=A*Px*A.T+sm(source['Q'])
                Cp=A*C*A.T
                Ey=J*(Cp-Px)*H0.T+G*sm(source['R'][0])
                C=J*Cp+(s.eye(2)-J)*Px
            regression=Ey*(H0*Px*H0.T+sm(source['R'][0])).inv()
            V=(G.T*regression+regression.T*G)/2
            assert V[0,0]>0 and V.det()>0
            rows.append(dict(model=name,old_sample_index=i,Ta=7,support='only7, within declared7..16',
              beta=enc(B),M=enc(M),y=enc(y),observed_q=enc(q),attack=enc(a),
              coded_energy=str(energy),energy_display=float(energy),strict_energy_margin=str(10-energy),
              exact_decoded_effect=enc(M*a),receiver_first_bias=enc(G*M*a),
              one_step_unseen_noise_covariance=enc(U),unseen_noise_PD=True,
              first_packet_cross_regression=enc(regression),
              first_packet_cross_quadratic=enc(V),cross_quadratic_PD=True,
              cross_term_consequence='for any negative current scale c with a=c*alpha(q)*q at first attack packet, alpha>=0 and positive with positive probability, E[2 e_healthy.T b] is strictly negative; valid also for causal energy clipping or feasible reflection',
              fixed_common_F_frobenius_squared=str(s.trace(F.T*F)),
              attack_API_uses='q only, coefficient-1/4; beta/y used solely to verify legitimate encoding',
              interpretation='rationalized beta and physical observation from one saved path; exact algebra, not a new stochastic draw'))
            break
    assert len(rows)==2
    df=8;alpha=.05/28;threshold=chi2.isf(alpha,df)
    out=dict(date='2026-11-17',invertible_linear_coding_malleability_exact=True,witnesses=rows,
      covariance_only_KL_counterexample=[dict(c=c,Pd=float(chi2.sf(threshold/(c*c),df)) if c else 0.,
         KL=float(df/2*(c*c-1-np.log(c*c))) if c else 'infinite: degenerate against full-support Gaussian',
         scope='full internal-window ideal constant scaling without hard-budget saturation; not a fair capped performance result')
        for c in (.5,0.,-1.)],
      first_packet_energy_feasible_reflection_law='For centrally symmetric y independent of M, mapping y to -y inside an even q-norm region is a measure-preserving involution conditional on M. Only first attacked packet marginal; not temporal/joint law.',
      no_global_or_full_event_strict_stealth_claim=True,new_physical_paths=0)
    (H/'exact_math_certificates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps({'exact_malleability':True,'strict_energy_witnesses':len(rows),'energies':[r['energy_display'] for r in rows],'unseen_covariance_PD':True}))
if __name__=='__main__':main()
