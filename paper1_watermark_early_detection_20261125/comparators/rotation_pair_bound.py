"""Two-report lower bounds and exact finite-word blind-window probabilities."""
import itertools
import json
import numpy as np
from scipy.linalg import eigh
import experiment as exp


def run_probability(states,designated,length=60,run=8):
    # Exact recursion in long-double arithmetic over (last designated state, run length).
    p=np.zeros((designated,run),dtype=np.longdouble)
    neutral=np.longdouble(1)
    for _ in range(length):
        new=np.zeros_like(p)
        for symbol in range(designated):
            new[symbol,1]=neutral/states+(p.sum()-p[symbol].sum())/states
            new[symbol,2:]=p[symbol,1:-1]/states
        new_neutral=(neutral+p.sum())*(states-designated)/states
        p,neutral=new,new_neutral
    return float(1-neutral-p.sum())


def audit():
    rows=[];checks=[]
    for model in exp.MODEL:
        _,m,_,_,ops=exp.old.backend(model)
        for e,(_,sender) in enumerate(m.links):
            H=m.H[sender];R=m.R[sender]
            D=H@m.A@np.linalg.inv(H)
            covariance=H@m.Q@H.T+R+D@R@D.T
            energy=np.eye(2)+D.T@D
            local=[]
            for j,k in itertools.permutations(range(4),2):
                N=exp.ROTATION[k]@D-D@exp.ROTATION[j]
                info=N.T@np.linalg.solve(covariance,N)
                gamma=float(eigh(info,energy,eigvals_only=True)[0])
                assert gamma>0
                rows.append({"model":model,"edge":e,"sender":int(sender),
                             "previous_rotation":j,"next_rotation":k,
                             "det_mismatch":float(np.linalg.det(N)),
                             "gamma_relative_pair_energy":gamma})
                local.append(gamma)
            # A pair residual is in the row space of the full eight-report parity residual.
            blocks,_=ops[8,False]
            full=blocks[e].transpose(1,0,2).reshape(14,16)
            pair=np.zeros((2,16));pair[:,-4:-2]=-D;pair[:,-2:]=np.eye(2)
            np.testing.assert_allclose(pair@np.linalg.pinv(full)@full,pair,atol=2e-12)
            covraw=exp.old.base.build_sender_geometry(m,int(sender)).covariance_raw
            np.testing.assert_allclose(pair@covraw@pair.T,covariance,atol=2e-12)
            rng=np.random.default_rng(52387+e)
            errors=[]
            for j,k in itertools.permutations(range(4),2):
                v=rng.normal(size=(10,2))
                signal=(exp.ROTATION[k]@D-D@exp.ROTATION[j])@v.T
                psi=np.sum(signal*np.linalg.solve(covariance,signal),axis=0)
                denom=np.sum(v*v+(v@D.T)**2,axis=1)
                assert np.all(psi/denom>=min(local)*(1-1e-12))
            checks.append({"model":model,"edge":e,"D":D.tolist(),
                           "trace_D":float(np.trace(D)),"det_D":float(np.linalg.det(D)),
                           "pair_covariance":covariance.tolist(),"gamma":min(local)})
    probabilities={name:{"per_sender_60_reports":run_probability(s,d),
                         "any_of_six_senders":1-(1-run_probability(s,d))**6}
                   for name,s,d in (("one_bit_scalar_sign",2,2),
                                    ("quarter_turn_sufficient_plus_minus_identity_run",4,2))}
    result={"scope":"fixed two-dimensional disturbance dynamics and frozen sensor models",
            "bound":"psi_eight >= psi_last_pair >= gamma * (||v_t||^2+||D v_t||^2) on every no-repeat transition",
            "all_144_transition_checks":rows,"sender_minima":checks,
            "minimum_gamma":min(row["gamma_relative_pair_energy"] for row in rows),
            "constant_plus_minus_identity_run_probabilities":probabilities,
            "run_probability_meaning":"sufficient blind windows due to >=8 consecutive +I or >=8 consecutive -I, among 60 reports per independent sender",
            "numerical_bounds_status":"double precision eigenvalue evaluations accompany the algebraic full-rank proof"}
    exp.old.base._atomic_json(exp.HERE/"ROTATION_PAIR_BOUND.json",result)
    print(json.dumps({"minimum_gamma":result["minimum_gamma"],"probabilities":probabilities},indent=2))


if __name__=="__main__":
    audit()
