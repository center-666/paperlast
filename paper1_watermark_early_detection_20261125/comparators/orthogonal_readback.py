"""Post-hoc descriptive contrasts for the two additional discrete comparators."""
import csv
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    folder=HERE/"confirm"
    rows=[]
    for mi,model in enumerate(("unstable","stable")):
        for s in range(6):
            with np.load(folder/f"{model}_2_{s}.npz") as data:
                cell=[]
                for alt in ("diagonal","quarter"):
                    for ref in ("sign","negative","negative_eq"):
                        for mode in ("plugin","exact"):
                            a=alt+"_"+mode;b=ref+"_"+mode
                            x=(data[a+"_delay"]<3).mean(1)-(data[b+"_delay"]<3).mean(1)
                            y=data[a+"_loss"]-data[b+"_loss"]
                            cell.append(np.column_stack([x,y[:,2],y[:,4]]))
                rows.append(np.stack(cell,axis=1))
    x=np.stack(rows);rng=np.random.default_rng([20261126,17291,2,90191])
    boot=[]
    for scene in x:
        n=len(scene)
        weights=rng.multinomial(n,np.full(n,1/n),size=10000).astype(float)/n
        boot.append((weights@scene.reshape(n,-1)).reshape(10000,*scene.shape[1:]))
    output=[]
    for group,idx in (("all",slice(None)),("unstable",slice(0,6)),("stable",slice(6,12))):
        point=x[idx].mean((0,1))
        ci=np.quantile(np.stack(boot[idx]).mean(0),[.025,.975],axis=0)
        from itertools import product
        for i,(alt,ref,mode) in enumerate(product(("diagonal","quarter"),("sign","negative","negative_eq"),("plugin","exact"))):
            for j,metric in enumerate(("early","disturbance_affected","recovery_affected")):
                output.append([group,alt,ref,mode,metric,point[i,j],ci[0,i,j],ci[1,i,j]])
    with (folder/"orthogonal_descriptive.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["group","left","right","scorer","metric","difference","lower","upper"])
        writer.writerows(output)


if __name__=="__main__":
    main()
