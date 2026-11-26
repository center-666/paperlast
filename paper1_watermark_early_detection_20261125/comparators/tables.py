"""Collect the registered absolute and paired results without recomputing experiments."""
import csv
from pathlib import Path

D=Path(__file__).resolve().parent/"confirm"


def read(name):
    with (D/(name+".csv")).open() as f:return list(csv.DictReader(f))


if __name__=="__main__":
    health={(r["model"],r["arm"]):r for r in read("health")}
    cost={(r["model"],r["scheme"]):r for r in read("resources")}
    delta={(r["group"],r["comparison"],r["metric"]):r for r in read("contrasts")}
    out=[]
    for r in read("absolute"):
        a=r["arm"];scheme,kind=a.rsplit("_",1)
        item=dict(r)
        for m in ("stable","unstable"):
            h=health[m,a];c=cost[m,scheme]
            item.update({m+"_"+k:h[k] for k in ("alarm_paths","n","FWER","lower","upper","health5","health50")})
            item[m+"_encoded_energy_ratio"]=c["pooled_tx_over_raw"]
        baseline="sign_"+kind
        if a==baseline:
            item.update({k:0 for k in ("early_minus_sign","paired95_lower","paired95_upper")})
        elif (r["group"],a+"-"+baseline,"early") in delta:
            p=delta[r["group"],a+"-"+baseline,"early"]
            item.update(dict(zip(("early_minus_sign","paired95_lower","paired95_upper"),(p["difference"],p["lower"],p["upper"]))))
        else:
            item.update({k:"" for k in ("early_minus_sign","paired95_lower","paired95_upper")})
        out.append(item)
    with (D/"main_table.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
