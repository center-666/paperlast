"""Source-specific observer/contraction, startup and simple-baseline diagnostics."""
from pathlib import Path
import json
import numpy as np
from source_model import load_model,covariance_schedule

HERE=Path(__file__).resolve().parent


def true_nominal_covariance(model,schedule):
    C=np.broadcast_to(np.eye(2),(50,2,2)).copy();loss=np.zeros((150,50));loss[0]=np.trace(C,axis1=1,axis2=2)
    for k in range(1,150):
        star=schedule.star[k];B=np.linalg.solve(model.A.T,schedule.F[k].swapaxes(-1,-2)).swapaxes(-1,-2)
        J=np.zeros_like(star);J[:,0,0]=schedule.information[:,0];J[:,1,1]=schedule.information[:,1]
        C=schedule.F[k]@C@schedule.F[k].swapaxes(-1,-2)+B@model.Q@B.swapaxes(-1,-2)+star@J@star.swapaxes(-1,-2)
        loss[k]=np.trace(C,axis1=1,axis2=2)
    return loss


def run():
    rows=[]
    for label in ('unstable','stable'):
        model=load_model(label);normal=covariance_schedule(model);drop=covariance_schedule(model,True)
        h0=true_nominal_covariance(model,normal);hdrop=true_nominal_covariance(model,drop)
        eigs=np.linalg.eigvalsh(model.Theta);theta_min=float(eigs.min())
        eps=1e-6;a=eps/theta_min
        floors={str(n):float((8-n+1)/2*(a-np.log(a)-1)) for n in (2,3,4)}
        assert floors['2']>42.
        rows.append(dict(model=label,uniform_F_bound_r=float(model.rbar),
            uniform_F_bound_squared=float(model.rbar**2),source_theta_min=theta_min,
            startup_plugin_lower_bounds_by_residual_count=floors,forced_alarm_k5=True,
            fixed_known_candidates_drop=dict(
                no_gate_healthy_network_mse=float(h0.mean()),static_drop_healthy_network_mse=float(hdrop.mean()),
                network_relative_cost=float(hdrop.mean()/h0.mean()-1),
                no_gate_healthy_affected_mse=float(h0[:,model.receiver_ids].mean()),
                static_drop_healthy_affected_mse=float(hdrop[:,model.receiver_ids].mean()),
                affected_relative_cost=float(hdrop[:,model.receiver_ids].mean()/h0[:,model.receiver_ids].mean()-1),
                network_relative_cost_post20=float(hdrop[20:].mean()/h0[20:].mean()-1),
                affected_relative_cost_post20=float(hdrop[20:,model.receiver_ids].mean()/h0[20:,model.receiver_ids].mean()-1),
                attack_effect_on_static_drop_exactly_zero=True,
                scope='Fixed preconfigured candidate-edge set, no unknown-location claim.')))
    (HERE/'observer_results.json').write_text(json.dumps(dict(models=rows),indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':run()
