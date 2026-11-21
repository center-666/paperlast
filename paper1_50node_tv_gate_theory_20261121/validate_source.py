"""Validate algebraic replay against original source body (return instrumented)."""
from pathlib import Path
import json
import time
import numpy as np
from source_model import load_model,covariance_schedule,prepare_physical
from replay import base_filter,scores,affected_filter,paired_forcing

HERE=Path(__file__).resolve().parent


def run():
    rows=[];started=time.perf_counter()
    for mi,label in enumerate(('unstable','stable')):
        model=load_model(label);schedule=covariance_schedule(model)
        for attack in (False,True):
            seed=202611310000+100*mi+int(attack)
            np.random.seed(seed % (2**32))
            start=time.perf_counter()
            original=model.source_functions['run_single_simulation'](True,attack,.5,model.A,
                                      6. if label=='unstable' else 10.,60)
            elapsed=time.perf_counter()-start
            x=original['x_true'].T[None];y=original['y_nominal'].transpose(2,0,1)[None]
            physical=prepare_physical(model,x,y);delta=np.zeros((1,150,6,2))
            if attack:
                for e,(i,j) in enumerate(model.links):
                    for k in range(50,110):
                        delta[0,k,e]=original['watermark'].M[(int(j),k)]@original['attacks'][int(j)][k-50]
            old_scores=np.stack([original['scores'][tuple(link)] for link in model.links],axis=-1)[None]
            old_gates=np.stack([original['gates'][tuple(link)] for link in model.links],axis=-1)[None]
            new_scores=scores(model,physical,delta)
            fast=affected_filter(model,physical,delta,new_scores,42.,True)
            base=base_filter(model,physical,schedule);base[:,:,model.receiver_ids]=fast.post
            old_est=original['x_hat'].transpose(1,2,0)[None]
            oldP=original['P_cov'].transpose(3,2,0,1)[None]
            max_est=float(np.max(abs(base-old_est)));maxP=float(np.max(abs(fast.P-oldP[:,:,model.receiver_ids])))
            margin=abs(old_scores-42.)<5.
            near=float(np.max(abs(old_scores-new_scores)[margin])) if np.any(margin) else 0.
            gate_equal=bool(np.array_equal(fast.gates,old_gates))
            assert gate_equal,(label,attack,'gate mismatch',np.argwhere(fast.gates!=old_gates)[:5])
            np.testing.assert_allclose(base,old_est,rtol=2e-10,atol=1e-6)
            assert maxP<1e-9
            # Exact paired-gate recursion, using this same physical stream.
            healthy_scores=scores(model,physical,np.zeros_like(delta))
            healthy=affected_filter(model,physical,np.zeros_like(delta),healthy_scores,42.,True)
            pair=paired_forcing(model,physical,healthy,fast,delta)
            identity=float(np.max(abs(pair.residual)))
            assert identity<1e-6
            rows.append(dict(model=label,attack=attack,source_seed=seed,source_seconds=elapsed,
                             estimator_max_abs_error=max_est,covariance_max_abs_error=maxP,
                             score_max_abs_error=float(np.max(abs(old_scores-new_scores))),
                             score_error_within5_of_threshold=near,all_gate_decisions_equal=gate_equal,
                             paired_bias_identity_max_error=identity))
    result=dict(scope='Original 50-node run_single_simulation body, only returned diagnostics extended; no source file modification, no original main or MC wrapper.',
                rows=rows,elapsed_seconds=time.perf_counter()-started)
    (HERE/'source_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
