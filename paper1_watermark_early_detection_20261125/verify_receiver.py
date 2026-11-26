"""Independent SVD regression on the compiled scorer and disposable end-to-end run."""
import tempfile
from pathlib import Path
import numpy as np

import study
import analyze_stage1
from matched_glrt import build_sender_geometry
from online_scores import matched
from watermark_keys import phase_for
from watson_crn import von_mises_cdf_signed


def main():
    errors=[]
    for mi,name in enumerate(study.base.MODELS):
        _,model,physical_paths,_,ops=study.backend(name)
        order=model.links[:,1]
        p=physical_paths(model,16,67201+mi)
        draws=study.short_draws(16,(20261125,8317,99),mi,0,40,0,order)
        key=study.matrices(draws,"negative_fixed")
        y=p.y[:,:,order].copy()
        actual=matched(y,key,ops)
        for t in (5,6,7,8,9,65,100,149):
            width=min(t+1,8); start=t-width+1
            for e,j in enumerate(order):
                g=build_sender_geometry(model,int(j),start==0,width)
                for b in range(2):
                    reports=y[b,start:t+1,e].copy()
                    if start==0: reports[0]=0
                    z=g.whitening_map@reports.ravel()
                    energy=[]
                    for offset in range(width):
                        design=key[b,start:t+1,e]@g.attack_state_maps[offset]
                        if start==0:design[0]=0
                        x=g.whitening_map@design.reshape(-1,2)
                        u=x@np.linalg.lstsq(x,z,rcond=1e-10)[0]
                        energy.append(u@u)
                    errors.append(abs(actual[b,t,e]-max(energy)))
                    np.testing.assert_allclose(actual[b,t,e],max(energy),atol=2e-7,rtol=1e-8)
        assert not actual[:,:5].any()
        changed=y.copy();changed[:,66:]+=100
        changed_key=key.copy();changed_key[:,66:]*=9
        np.testing.assert_array_equal(matched(changed,changed_key,ops)[:,:66],actual[:,:66])
    q=np.linspace(0,1,120002)[1:-1]
    table={}
    for theta in (-1.3152369768257723,-1.0149016255359247,-2.0298032510718493):
        error=np.max(np.abs(von_mises_cdf_signed(phase_for(q,theta),theta/2)-q))
        assert error<1e-12
        table[str(theta)]=float(error)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        study.run(99,path,smoke=True,workers=4)
        summary=analyze_stage1.summarize(path,reps=100)
        assert len(summary["scenes"])==12 and len(summary["geometry"])==12
    study.base._atomic_json(study.HERE/"NUMERICAL_AUDIT.json",{
        "startup_k":[5,6,7],"direct_lstsq_max_abs_error":max(errors),"inverse_cdf_residual":table,
        "matched_cases":len(errors),"future_invariance_through_65":True,"smoke_archive_analysis":"pass"})


if __name__=="__main__":
    main()
