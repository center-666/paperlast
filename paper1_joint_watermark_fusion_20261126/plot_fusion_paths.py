"""Plot actual paired paths; cumulative improvement uses the 60-tick window."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
with (HERE/"task4_confirm/M7_EXAMPLE_TRACES.csv").open() as stream:
    rows=list(csv.DictReader(stream))
for selection in ("median","largest"):
    fig,axes=plt.subplots(3,2,figsize=(10,6.5),sharex=True,layout="constrained")
    for col,fixture in enumerate(("continuous","burst")):
        data=[r for r in rows if r["fixture"]==fixture and r["selection"]==selection]
        def series(name):
            return np.array([float(r[name]) for r in data])
        t=series("tick")
        axes[0,col].set_title(f"{fixture.capitalize()} input  ·  path {data[0]['path_id']}",loc="left")
        for row,(suffix,ylabel) in enumerate((("q","Executed weight"),("error","Squared error (2 states)"))):
            ax=axes[row,col]
            for prefix,color,label in (("min","#ad4c13","M7 minimum"),("table","#18598d","Frozen T")):
                ax.plot(t,series(prefix+"_"+suffix),color=color,linewidth=1.2,label=label,
                        drawstyle="steps-post" if suffix=="q" else "default")
            if not col:
                ax.set_ylabel(ylabel)
        axes[2,col].plot(t,series("cumulative_gain"),color="#24715d",linewidth=1.7)
        axes[2,col].set_xlabel("Tick")
        axes[2,0].set_ylabel("Accumulated MSE gain")
        for ax in axes[:,col]:
            ax.set_xlim(42,149)
            for start in (range(50,110,10) if fixture=="burst" else [50]):
                ax.axvspan(start-.5, start+(6 if fixture=="burst" else 60)-.5,color="#e6d7af",alpha=.35,zorder=-1)
            ax.axvline(109.5,color="#777",lw=.8,ls="--")
            ax.grid(axis="y",alpha=.2)
    axes[0,0].legend(fontsize=8,loc="lower right",ncol=2)
    fig.savefig(HERE/"fusion_attribution"/f"path_{selection}.png",dpi=180)
    plt.close(fig)
