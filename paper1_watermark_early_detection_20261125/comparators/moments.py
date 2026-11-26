"""Expected whitened energy of the *fixed offline fixture*, and packet costs."""
import csv
import json
import subprocess
import numpy as np
from scipy.special import exp1,i0,i1
import experiment as ex

J=np.array([[[1,0],[0,-1]],[[0,1],[1,0]]])
z=i1(-1.3152369768257723/2)/i0(-1.3152369768257723/2)


def expected(W,a):
    # W[edge,time,rank,2], a[edge,time,2]
    linear=np.einsum("etdm,etm->etd",W,a)
    signal=(linear.sum(1)**2).sum(1)
    var=(linear**2).sum((1,2))
    angular=np.einsum("etdm,imn,etn->etid",W,J,a)
    iid=signal+.25*var+.25*(angular**2).sum((1,2,3))
    negative=iid.copy()
    for lag in range(1,a.shape[1]):
        negative+=.25*z**lag*(angular[:,lag:]*angular[:,:-lag]).sum((1,2,3))
    return {"none":signal,"sign":var,"diagonal":((W*a[:,:,None,:])**2).sum((1,2,3)),
            "quarter":.5*((a*a).sum(2)*(W*W).sum((2,3))).sum(1),"iid":iid,"negative":negative}


if __name__=="__main__":
    arch=json.loads(subprocess.check_output(["git","show","cb3ad287:paper1_watermark_matching_20261125/watermark_moments.json"],
                                             cwd=ex.HERE,text=True,encoding="utf-8"))
    scale=ex.read(ex.HERE/"ENERGY_LOCK.json")
    rows=[]
    for mi,m in enumerate(ex.MODEL):
        source,model,_,_,ops=ex.old.backend(m)
        for age in (1,2,3,5,8):
            W=ops[8,False][0][:,-age:]
            sums={k:[] for k in ("iid","negative")}
            for angle in (0,-45,45):
                a=source.rotated_raw_attack(model,angle)[0][50:50+age].swapaxes(0,1)
                e=expected(W,a)
                for k in sums:sums[k].append(e[k].mean())
                c=scale[m]["encode_multiplier"]
                for k in ("iid","negative"):e[k+"_eq"]=e[k]/c**2
                for code,data in e.items():
                    rows.append({"model":m,"angle":angle,"reports":age,"scheme":code,"mean_psi":data.mean()})
            if age in (1,2,8):
                for k in sums:
                    np.testing.assert_allclose(np.mean(sums[k]),arch["models"][m][str(age)][k+"_mean"],rtol=1e-10)
    with (ex.HERE/"THEORY.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Integration by parts for S~Exp(1): E[(a+S)^-2]=1/a-exp(a)E1(a).
    print("Analytic encoded-to-raw energy ratio:",.5*(4+2-np.exp(.5)*exp1(.5)))
