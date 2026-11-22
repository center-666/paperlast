"""Small algebraic unit tests, no source-model data and no physical claims."""
import numpy as np
from causal_gate_ablation import shapley_attribution,score_metrics,metric_paths
from types import SimpleNamespace
r=np.random.default_rng(20261122)
x0=r.normal(size=40);x1=r.normal(size=40);x2=r.normal(size=40);x3=r.normal(size=40)
d=shapley_attribution(x0,x1,x2,x3)
np.testing.assert_allclose(d['total'],d['direct_watermark_at_fixed_gate']+d['gate_schedule_effect'])
S=np.zeros((4,150,6));S[0,57,2]=1.0;S[2,65,0]=1.0
scores=score_metrics(S,.5)
assert scores['eligible_tpr'].shape==(4,)
assert scores['path_any_alarm'].tolist()==[1,0,1,0]
assert scores['restricted_mean_first_alarm_lag'][1]==53
print('PASS: shapley identity and alarm-censor handling')
