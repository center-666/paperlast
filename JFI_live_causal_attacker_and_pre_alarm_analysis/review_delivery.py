"""Check release materials and visualize archived time ordering."""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from archive_timing import BASE, HERE, METHODS, MODELS, OUTPUT, PROFILES, digest, load, save, verify_frozen


TIMES = (
    ("strictly_before_alarm","Strictly before BF","#187d74"),
    ("same_tick_as_alarm","Same tick as BF","#e5ac64"),
    ("strictly_after_alarm","Strictly after BF","#756692"),
    ("no_danger_within_horizon","No danger in window","#eaeef0"),
)
HEADINGS = ("A0 archive","H archive","C archive")
METHOD_LABELS = ("V0","V1a","V1.2a","V1.4","State sat. 2%")


def plot_archive(doc):
    fig,axes = plt.subplots(2,3,figsize=(12.6,5.5),sharex=True)
    y = np.arange(len(METHODS))
    for model,row in zip(MODELS,axes,strict=True):
        for profile,title,ax in zip(PROFILES,HEADINGS,row,strict=True):
            cases = doc["models"][model]["profiles"][profile]["onset_7"]
            count = cases[METHODS[0]]["sample_size"]
            left = np.zeros(len(METHODS))
            for key,_,color in TIMES:
                fraction = np.array([100*cases[x][key]/count for x in METHODS])
                ax.barh(y,fraction,left=left,height=.56,color=color)
                left += fraction
            np.testing.assert_allclose(left,100,atol=2e-13)
            ax.set(title=f"{model.capitalize()} / {title} ({count:,})",
                   yticks=y,yticklabels=METHOD_LABELS,xlim=(0,100),ylim=(4.55,-.55))
            ax.grid(axis="x",alpha=.16)
    fig.suptitle("Ten-step receiver outcome archives with onset 7",y=.98,fontsize=13)
    fig.supxlabel("Share of all paired outcomes (%)",y=.093,fontsize=10)
    fig.legend(handles=[Patch(color=c,label=l) for _,l,c in TIMES],loc="lower center",
               bbox_to_anchor=(.5,.012),ncol=4,frameon=False)
    fig.subplots_adjust(left=.087,right=.984,top=.867,bottom=.176,wspace=.32,hspace=.30)
    fig.savefig(OUTPUT/"archived_timing.png",dpi=172,facecolor="white")
    plt.close(fig)


def main():
    old_verified = verify_frozen()
    archive,cert = (load(OUTPUT/name) for name in ("archived_timing.json","pre_alarm_certificate.json"))
    assert archive["verified_previous_results_sha256"]==old_verified
    assert archive["baseline"]==cert["baseline"]==BASE
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":9})
    plot_archive(archive)
    results = {f.name:digest(f) for f in sorted(OUTPUT.iterdir()) if f.name!="delivery_checks.json"}
    scripts = {f.name:digest(f) for f in sorted(HERE.glob("*.py"))}
    save(OUTPUT/"delivery_checks.json",{"baseline":BASE,"previous_result_files_verified":
            sum(len(items) for items in old_verified.values()),"new_trajectories":0,
            "live_attacker_runs":0,"results_sha256":results,"source_sha256":scripts})
    for md in HERE.glob("*.md"):
        for href in re.findall(r"\]\(([^)]+)\)",md.read_text(encoding="utf-8")):
            if not href.startswith(("https:","http:")):
                assert (HERE/href).exists(),(md.name,href)
    print("Verified both prior artifact groups, report links, and scoped delivery.")


if __name__=="__main__":
    main()
