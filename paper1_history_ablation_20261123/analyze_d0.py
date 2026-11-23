"""Summarize independent D0 outputs from paired arrays, including the watermark interaction."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import t

from locked_weight_check import HERE, OUTPUT


def interval(values):
    se = values.std(ddof=1)/np.sqrt(len(values))
    center = values.mean()
    margin = t.ppf(.975,len(values)-1)*se
    return center, center-margin, center+margin


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results/D0_audit/locked_summary.csv')
    target=parser.parse_args().output
    with np.load(OUTPUT/'paths.npz') as data, target.open('x',newline='',encoding='utf-8') as stream:
        out=csv.writer(stream)
        out.writerow(('model','fixture','watermark','policy_or_contrast','metric','n','mean','low95','high95'))
        for key in sorted(data.files):
            parts=key.split('__')
            if len(parts)==5:
                values=data[key]
                out.writerow((*parts,len(values),*interval(values)))
        for model in ('unstable','stable'):
            for fixture in ('continuous','burst'):
                prefix=model+'__'+fixture+'__'
                risk={wm:{policy:data[prefix+wm+'__'+policy+'__missing_fault'] for policy in
                           ('hard','locked','accept','fixed08','fixed09')} for wm in ('negative','iid')}
                for wm,values in risk.items():
                    for base in ('hard','accept','fixed08','fixed09'):
                        diff=values['locked']-values[base]
                        out.writerow((model,fixture,wm,'locked-minus-'+base,'missing_fault',len(diff),*interval(diff)))
                diff=risk['negative']['locked']-risk['iid']['locked']
                inter=diff-(risk['negative']['accept']-risk['iid']['accept'])
                for name,v in [('locked',diff),('interaction_vs_accept',inter)]:
                    out.writerow((model,fixture,'negative-minus-iid',name,'missing_fault',len(v),*interval(v)))
    print(target)


if __name__=='__main__':
    main()
