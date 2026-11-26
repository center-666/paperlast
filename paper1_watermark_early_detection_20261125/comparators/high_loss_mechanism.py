"""Counterfactual gate/source decomposition of the W1 catastrophic path."""
import json
import numpy as np
import experiment as exp


def main():
    model_index, scene, path_id = 0,0,184
    batch,row=divmod(path_id,32)
    ns=tuple(exp.namespace(2,model_index,scene,2,batch))
    manifest=exp.read(exp.HERE/"confirm/RUN.json")
    _,model,_,estimator,ops=exp.old.backend("unstable")
    phy,draws,bits=exp.physical_and_keys(32,ns,model_index)
    raw=phy.y[:,:,model.links[:,1]]
    fixture=exp.old.backend("unstable")[0].rotated_raw_attack(model,0.)[0][50:110]
    onset,stop=50,110
    thresholds=manifest["thresholds"]["unstable"]
    schemes=("quarter","diagonal","sign","iid","negative")
    keys={s:exp.schedule(s,draws,bits,1.) for s in schemes}
    deltas={}
    scores={}
    gates={}
    for s in schemes:
        delta=np.zeros_like(raw);delta[:,onset:stop]=(keys[s][:,onset:stop]@fixture[None,...,None])[...,0]
        deltas[s]=delta
        scores[s]=exp.receiver.matched(raw+delta,keys[s],ops)
        gates[s]=scores[s]>thresholds[s+"_exact"][1]
    policies={}
    for s in schemes:
        g=gates[s][row].copy()
        policies[s+"_actual"]=g
        no=g.copy();no[:]=False;policies[s+"_no_gate"]=no
        oracle=g.copy();oracle[onset:stop]=True;policies[s+"_oracle"]=oracle
        latch=g.copy()
        for e in range(6):
            hit=np.flatnonzero(g[onset:stop,e])
            if hit.size:latch[onset+hit[0]:stop,e]=True
        policies[s+"_latch_after_first_alarm"]=latch
    # Cross-watermark gate policies isolate gate schedules from codec schedules.
    policies["quarter_gate"]=gates["quarter"][row].copy()
    policies["negative_gate"]=gates["negative"][row].copy()
    result={"path_id":path_id,"seed_namespace":list(ns),"attack_fixture":fixture.tolist(),
            "spectral_radius_A":float(max(abs(np.linalg.eigvals(model.A)))),
            "model_A":model.A.tolist(),"edge_order":model.links.astype(int).tolist(),
            "counterfactuals":{},"trace":[]}
    for codec in schemes:
        for policy,gate in policies.items():
            full_gate=gates[codec].copy()
            full_gate[row]=gate
            world=estimator(model,phy,deltas[codec],sender_packets=phy.y,gate_override=full_gate)
            err=np.sum((world.post-phy.x[:,:,None])**2,axis=-1)[row]
            result["counterfactuals"][codec+"__"+policy]={
                "affected5_attack_mse":float(err[onset:stop][:,model.receiver_ids].mean()),
                "network_attack_mse":float(err[onset:stop].mean()),
                "affected5_recovery_mse":float(err[stop:][:,model.receiver_ids].mean()),
                "receiver_trajectory_peak_attack":np.max(err[onset:stop][:,model.receiver_ids],axis=0).tolist(),
                "receiver_peak_time":(np.argmax(err[onset:stop][:,model.receiver_ids],axis=0)+onset).tolist(),
                "node0_peak":float(err[onset:stop,0].max()),
                "node0_peak_time":int(np.argmax(err[onset:stop,0])+onset),
            }
    for tick in range(onset,stop):
        entry={"tick":tick}
        for e in range(6):
            entry[f"edge{e}"]={
                "gate":int(gates["quarter"][row,tick,e]),
                "score_over_cutoff":float(scores["quarter"][row,tick,e]/thresholds["quarter_exact"][1]),
                "delta_norm":float(np.linalg.norm(deltas["quarter"][row,tick,e])),
                "accepted_delta_norm":float(np.linalg.norm(deltas["quarter"][row,tick,e])*(not gates["quarter"][row,tick,e])),
            }
        result["trace"].append(entry)
    gate = gates["quarter"].copy()
    latch_edge0 = gate[row].copy()
    hits=np.flatnonzero(latch_edge0[onset:stop,0])
    if hits.size:
        latch_edge0[onset+hits[0]:stop,0]=True
    interventions={}
    for name,mode in (("remove_e0_t109","remove"),("gate_e0_t109","gate"),
                      ("latch_e0_after_first_alarm","latch")):
        intervention_gate=gate.copy()
        intervention_gate[row]=latch_edge0 if mode=="latch" else gate[row].copy()
        intervention_delta=deltas["quarter"].copy()
        if mode=="remove":
            intervention_delta[row,109,0]=0.
        elif mode=="gate":
            intervention_gate[row,109,0]=True
        world=estimator(model,phy,intervention_delta,sender_packets=phy.y,
                        gate_override=intervention_gate)
        error=np.sum((world.post-phy.x[:,:,None,:])**2,axis=-1)[row]
        interventions[name]={
            "affected5_attack_mse":float(error[onset:stop][:,model.receiver_ids].mean()),
            "node0_error_t108_t110":error[108:111,0].tolist(),
            "receiver_error_t108_t110":error[108:111][:,model.receiver_ids].tolist(),
        }
    result["single_packet_interventions"]=interventions
    sender0=int(model.links[0,1])
    key0=keys["quarter"][row,:,0]
    identity=(np.max(abs(key0-np.eye(2)),axis=(1,2))<1e-12)
    runs=[];start=None
    for t in range(onset,stop+1):
        is_identity=t<stop and identity[t]
        if is_identity and start is None:start=t
        if not is_identity and start is not None:
            runs.append([start,t-1,t-start]);start=None
    result["identity_matrix_runs_edge0"]={
        "sender":sender0,"runs":runs,
        "key_indices_102_109":key0[102:110].tolist(),
        "attack_signal_norm_102_109":np.linalg.norm(deltas["quarter"][row,102:110,0],axis=1).tolist(),
        "edge0_gate_102_109":gates["quarter"][row,102:110,0].astype(int).tolist(),
        "edge0_score_ratio_102_109":(scores["quarter"][row,102:110,0]/thresholds["quarter_exact"][1]).tolist(),
    }
    w0=ops[8,False][0][0]
    residual=np.einsum("wrm,wm->r",w0,deltas["quarter"][row,102:110,0])
    raw_energy=float(np.sum(deltas["quarter"][row,102:110,0]**2))
    result["identity_matrix_runs_edge0"]["eight_report_white_signal_norm_at_109"]=float(np.linalg.norm(residual))
    result["identity_matrix_runs_edge0"]["eight_report_white_energy_at_109"]=float(residual@residual)
    result["identity_matrix_runs_edge0"]["relative_white_energy_at_109"]=float((residual@residual)/raw_energy)
    result["identity_matrix_runs_edge0"]["raw_perturbation_norm_at_109"]=float(np.linalg.norm(deltas["quarter"][row,109,0]))
    target=exp.HERE/"HIGH_LOSS_COUNTERFACTUAL.json"
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "rho":result["spectral_radius_A"],
        "receiver_mse":{k:v["affected5_attack_mse"] for k,v in result["counterfactuals"].items()
                        if k in ("quarter__quarter_actual","diagonal__diagonal_actual","iid__iid_actual","negative__negative_actual",
                                 "quarter__negative_gate","negative__quarter_gate","quarter__quarter_latch_after_first_alarm",
                                 "quarter__quarter_no_gate","quarter__quarter_oracle","sign__sign_actual")},
        "single_packet_interventions":interventions,
        "identity_matrix_runs_edge0":result["identity_matrix_runs_edge0"],
        "first_accepted":{str(e):[t for t in range(onset,stop) if gates["quarter"][row,t,e]==0][:5] for e in range(6)},
        "peak_times":{k:v["receiver_peak_time"] for k,v in result["counterfactuals"].items()
                      if k in ("quarter__quarter_actual","diagonal__diagonal_actual","iid__iid_actual","negative__negative_actual")},
        "file":str(target)},indent=2))


if __name__=="__main__":main()
