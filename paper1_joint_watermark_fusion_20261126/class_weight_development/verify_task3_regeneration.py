import os
import sys
from pathlib import Path
os.environ["D2_D1_DIR"]=r"C:\Users\zhongxin\shixidoc\.d1_registration_worktree\paper1_D1_link_value_20261123"
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import task3, frontend
idx=2
model=task3.audit_frontend.load_model(idx)
ops=frontend.operators(model)
thr=task3.frontend_thresholds(idx)
with np.load(r"C:\Users\zhongxin\shixidoc\paperlast-joint-audit\paper1_joint_watermark_fusion_20261126\task3_dev\config02.npz") as a:
 for fi,fixture in enumerate(("continuous","burst")):
  data=task3.batch_data(model,ops,idx,fi,5,256,thr)
  print(fixture,"y",np.max(abs(data["paths"].y-a[fixture+"_y"])))
  print(fixture,"delta",np.max(abs(data["delta"]-a[fixture+"_delta"])))
  print(fixture,"matched",np.max(abs(data["matched"]-a[fixture+"_matched"])))
  print(fixture,"gates",np.array_equal(data["gates"],a[fixture+"_gates"]))
 health=task3.batch_data(model,ops,idx,0,6,256,thr,first_batch=100)
 health["delta"].fill(0.)
 for key in ("x","y"): print("health",key,np.max(abs(getattr(health["paths"],key)-a["health_"+key])))
 print("health gate",np.array_equal(health["gates"],a["health_gates"]))
