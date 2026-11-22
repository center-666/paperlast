"""Audit existing 99ee54f 128-path paired closed-loop screening results.

Running on the full source repo reads the original JSON. This script may also be
used against the explicitly extracted paper1_screening_summary_99ee.json; no
physical simulations or calibration are performed here.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.stats import t as student_t

BASE=Path(__file__).resolve().parent


def paired_ci(mean, se, n=128):
    crit=student_t.ppf(0.975,n-1)
    tstat=mean/se if se else float('inf')
    return {'difference':mean,'path_se':se,'t_stat':tstat,
            'conditional_95ci':[float(mean-crit*se),float(mean+crit*se)],
            'two_sided_pathwise_p_value':float(2*student_t.sf(abs(tstat),n-1))}


def audit_compact(summary):
    output={}
    for key in ['unstable','stable']:
        v=summary['models'][key]
        iid=v['iid']['network_mse_post20']
        neg=v['negative']['network_mse_post20']
        oracle=v['oracle']['network_mse_post20']
        healthy=v['healthy']['network_mse_post20']
        paired=v['negative_minus_iid_paired']
        assert np.isclose(neg-iid,paired['network_mse_post20']['mean'],atol=1e-12)
        output[key]={
            'closed_loop_iid_mse':iid,
            'closed_loop_negative_mse':neg,
            'relative_negative_reduction_percent':100*(iid-neg)/iid,
            'paired_network_mse':paired_ci(**paired['network_mse_post20']),
            'paired_affected_mse':paired_ci(**paired['affected_mse_post20']),
            'paired_eligible_TPR':paired_ci(**paired['eligible_tpr']),
            'static_known_six_oracle':oracle,
            'negative_above_oracle_abs':neg-oracle,
            'negative_above_oracle_percent':100*(neg/oracle-1),
            'oracle_healthy_cost_percent':100*(oracle/healthy-1),
            'no_gate_paired':paired_ci(**v['negative_minus_iid_no_gate']),
        }
    return {'source':'99ee54f, verified extracted aggregate metrics',
            'interpretation':'conditional-on-realized-calibration, paired path-level intervals; no full calibration uncertainty or fresh attack shapes',
            'models':output}


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--input',type=Path,default=BASE/'paper1_screening_summary_99ee.json')
    args=a.parse_args()
    summary=json.loads(args.input.read_text())
    result=audit_compact(summary)
    (BASE/'screening_audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
    for k,v in result['models'].items():
        print(k,'iid',round(v['closed_loop_iid_mse'],9),'negative',round(v['closed_loop_negative_mse'],9),
              'reduction%',round(v['relative_negative_reduction_percent'],4),
              'oracle_gap%',round(v['negative_above_oracle_percent'],4),
              'paired_CI',v['paired_network_mse']['conditional_95ci'])

if __name__=='__main__':main()
