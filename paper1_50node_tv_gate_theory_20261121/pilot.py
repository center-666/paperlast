"""Small runtime pilot; not used to tune source attacks, kernels or thresholds."""
from pathlib import Path
import json
import time
import numpy as np
from source_model import load_model,covariance_schedule,physical_paths
from replay import base_filter,scores,affected_filter,attack_watermarks,paired_forcing

HERE=Path(__file__).resolve().parent


def run():
    rows=[]
    for label in ('unstable','stable'):
        start=time.perf_counter();model=load_model(label);schedule=covariance_schedule(model)
        phys=physical_paths(model,16,202612040000+(label=='stable'))
        base=base_filter(model,phys,schedule)
        zero=np.zeros((16,150,6,2));healthscores=scores(model,phys,zero)
        healthy=affected_filter(model,phys,zero,healthscores,42.,True)
        delta,beta=attack_watermarks(model,16,-1.315236976827179,202612050000,202612060000,202612070000)
        attackscores=scores(model,phys,delta);attacked=affected_filter(model,phys,delta,attackscores,42.,True)
        pair=paired_forcing(model,phys,healthy,attacked,delta)
        err=np.max(abs(pair.residual));assert err<1e-6
        assert np.array_equal(attacked.gates[:,117:],healthy.gates[:,117:])
        rows.append(dict(model=label,paths=16,seconds=time.perf_counter()-start,identity_error=float(err)))
    (HERE/'runtime_pilot.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':run()
