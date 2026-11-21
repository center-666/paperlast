"""Prepare no-gate 50-node moments and buffered-residual support diagnostics."""
from pathlib import Path
import json
import numpy as np
from source_model import load_model,covariance_schedule
from moments import state_polynomials,buffer_support,design,polynomial

HERE=Path(__file__).resolve().parent


def run():
    result=[];arrays={}
    for label in ('unstable','stable'):
        model=load_model(label);schedule=covariance_schedule(model)
        state=state_polynomials(model,schedule);buffers=[buffer_support(model,e) for e in range(6)]
        selected=design(model,state,buffers)
        for i,bf in enumerate(buffers):
            for name in ('P','U','Z','theta','trace_coeff'):arrays[f'{label}_link{i}_{name}']=bf[name]
        for name in ('coeff','E','meanQ','radial_floor'):arrays[f'{label}_state_{name}']=state[name]
        eig_universal=np.linalg.norm(model.A,'fro')**2/4
        actualF=max(float(np.linalg.norm(F,2)) for F in schedule.F[1:].reshape(-1,2,2))
        result.append(dict(model=label,estimator='recovered no-gate time-varying stacked KF',
                           design=selected,observer=dict(r_frobenius_bound=float(np.sqrt(eig_universal)),
                                 r_squared_exact_decimal=eig_universal,max_actual_nogate_F_norm=actualF),
                           buffers=[dict(sender=b['sender'],raw_packet_dimension=16,ambient_residual_dimension=32,
                                effective_rank=b['rank'],effective_nominal_min_eigenvalue=float(np.linalg.eigvalsh(b['theta'])[0])) for b in buffers]))
        print(label,'energy',selected['energy_original_waveform'],'bound theta',selected['bound_optimum']['theta'],
              'witness',selected['strict_worst_dg_gain_witness'],flush=True)
    np.savez_compressed(HERE/'moment_arrays.npz',**arrays)
    (HERE/'design_results.json').write_text(json.dumps(dict(models=result),indent=2)+'\n',encoding='utf-8',newline='\n')
    print('design_results.json and moment_arrays.npz written')


if __name__=='__main__':run()
