import numpy as np
p=r"C:\Users\zhongxin\shixidoc\paperlast-watermark-w1\paper1_watermark_early_detection_20261125\comparators\markov_confirm\unstable_attack_0.npz"
z=np.load(p)
print("\n".join(f"{k} {z[k].shape}" for k in z.files))
