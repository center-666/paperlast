"""Deterministic two-report mismatch-margin calculation from frozen 2-D models."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigvalsh

HERE=Path(__file__).resolve().parent.parent
RESEARCH=HERE/"watermark_research"
WATERMARK=Path(r"C:\Users\zhongxin\shixidoc\paperlast-watermark-w1\paper1_watermark_early_detection_20261125")
COMPARATOR=WATERMARK/"comparators"
sys.path[:0]=[str(COMPARATOR),str(WATERMARK),
              str(WATERMARK.parent/"paper1_sscd_commonrandom_20261122"),
              str(WATERMARK.parent/"paper1_50node_tv_gate_theory_20261121"),
              str(WATERMARK.parent/"paper1_common_g50_gain_acceptance_20261121")]
import experiment as exp

ROT=np.asarray([[[1.,0.],[0.,1.]],[[0.,-1.],[1.,0.]],
                [[-1.,0.],[0.,-1.]],[[0.,1.],[-1.,0.]]])
SENDERS=(1,3,16,31,49,34)
FRACTIONS=(0.,.5,.9,1.,1.1,2.)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    details=[];sender_rows=[];model_rows=[]
    phi=np.linspace(0,2*np.pi,512,endpoint=False)
    directions=np.stack((np.cos(phi),np.sin(phi)),axis=1)
    for label in ("unstable","stable"):
        model=exp.old.backend(label)[1]
        for sender in SENDERS:
            H=model.H[sender];R=model.R[sender];A=model.A;Q=model.Q
            D=H@A@np.linalg.inv(H)
            V2=H@Q@H.T+R+D@R@D.T
            vals,vec=np.linalg.eigh(.5*(V2+V2.T))
            W=(vec/np.sqrt(vals))@vec.T
            transitions=[]
            generalized=[]
            for a in range(4):
                for b in range(4):
                    if a==b:continue
                    N=ROT[b]@D-D@ROT[a]
                    WRb=W@ROT[b]
                    sigma=float(np.linalg.svd(W@N,compute_uv=False)[-1])
                    transitions.append((a,b,N,WRb,sigma))
                    generalized.append(float(eigvalsh(
                        N.T@np.linalg.solve(V2,N),np.eye(2)+D.T@D)[0]))
            margin=min(x[4] for x in transitions)
            worst=next((x[0],x[1]) for x in transitions if x[4]==margin)
            bnorm=max(float(np.linalg.norm(x[3],2)) for x in transitions)
            rho=margin/(bnorm*np.linalg.norm(D,2))
            gamma=min(generalized)
            bins=[]
            for fraction in FRACTIONS:
                eps=fraction*margin/bnorm
                mesh_min=np.inf;min_pair=None
                for a,b,N,WRb,_ in transitions:
                    full_base=(W@N)@directions.T
                    error=eps*WRb@directions.T
                    values=np.linalg.norm(full_base.T[:,None,:]+error.T[None,:,:],axis=2)
                    index=np.unravel_index(np.argmin(values),values.shape)
                    if values[index]<mesh_min:
                        mesh_min=float(values[index]);min_pair=(a,b)
                bins.append({"fraction_of_triangle_margin":fraction,
                             "error_over_v_bound":eps,
                             "relative_operator_error_bound":eps/np.linalg.norm(D,2),
                             "positive_uniform_lower_bound":max(0.,margin-bnorm*eps),
                             "direction_mesh_min_whitened_signal":mesh_min,
                             "mesh_worst_transition":min_pair})
                details.append({"model":label,"sender":sender,**bins[-1]})
            sender_rows.append({"model":label,"sender":sender,"D":D.tolist(),"V2":V2.tolist(),
                                "D_spectral_norm":float(np.linalg.norm(D,2)),
                                "m_min_sigma_WN":margin,"b_max_norm_WRb":bnorm,
                                "rho_relative_D_operator_bound":rho,"gamma_generalized":gamma,
                                "worst_transition":worst,"transition_count":len(transitions),
                                "V2_eigenvalues":vals.tolist(),"bins":bins})
        selected=[r for r in sender_rows if r["model"]==label]
        model_rows.append({"model":label,
                           "min_rho_relative_D_operator_bound":min(r["rho_relative_D_operator_bound"] for r in selected),
                           "max_rho_relative_D_operator_bound":max(r["rho_relative_D_operator_bound"] for r in selected),
                           "min_gamma_over_senders":min(r["gamma_generalized"] for r in selected),
                           "monitored_senders":list(SENDERS)})
    with (RESEARCH/"MISMATCH_SENDER_MARGIN.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]),lineterminator="\n");writer.writeheader();writer.writerows(details)
    result={"status":"PASS_deterministic_mismatch_margin",
            "formula":"For v+=Dv+e, mu=(RbD-DRa)v+Rb e. ||Wmu|| >= sigma_min(WN)||v|| - ||WRb|| ||e||.",
            "covariance":"V2=HQH.T+R+DRD.T; W is its symmetric inverse square root.",
            "direction_count":len(directions),"legal_transition_count_per_sender":12,
            "relative_operator_condition":"||E||2/||D||2 < min_sigma(WN)/(max_norm(WRb)*||D||2).",
            "error_bins":list(FRACTIONS),"sender_results":sender_rows,"model_summary":model_rows,
            "source_hashes":{"source_model.py":sha(WATERMARK.parent/"paper1_50node_tv_gate_theory_20261121/source_model.py"),
                             "MECHANISM_AND_THEORY_ZH.md":sha(WATERMARK/"comparators/MECHANISM_AND_THEORY_ZH.md"),
                             "script":sha(Path(__file__))},
            "scope":"This is a sufficient full-whitened-signal margin. It is not a lower bound on the receiver's fixed matched-template projection outside the correctly modeled attack family."}
    (RESEARCH/"MISMATCH_SENDER_SUMMARY.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"status":result["status"],"models":model_rows},indent=2))


if __name__=="__main__":main()
