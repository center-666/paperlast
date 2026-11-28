import sys,json
from pathlib import Path
sys.path[:0]=[r"C:\Users\zhongxin\shixidoc\paperlast-watermark-w1\paper1_watermark_early_detection_20261125",r"C:\Users\zhongxin\shixidoc\paperlast-watermark-w1\paper1_sscd_commonrandom_20261122",r"C:\Users\zhongxin\shixidoc\paperlast-watermark-w1\paper1_50node_tv_gate_theory_20261121",r"C:\Users\zhongxin\shixidoc\paperlast-watermark-w1\paper1_common_g50_gain_acceptance_20261121"]
import numpy as np
from run_common_g_50 import load_common_g
for label in ("unstable","stable"):
 m=load_common_g(label)
 vals,vec=np.linalg.eig(m.A)
 print(label,"A",m.A,"evals",vals)
 print("H diag ranges",np.min(m.hdiag,axis=0),np.max(m.hdiag,axis=0),"zero",np.count_nonzero(m.hdiag==0))
