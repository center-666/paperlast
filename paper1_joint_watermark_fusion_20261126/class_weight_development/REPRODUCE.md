# 统一与分类权重开发比较复现

此目录的归档路径是任务3 purpose 4/5/6 配置结果。它们作为开发数据发布，未标记为独立确认。

## 数据与依赖

- 分支 `joint-watermark-fusion-results-20261127`。
- 12个任务3配置NPZ及 RUN 清单来自 Release `joint-task3-development-ns83-data`，Release URL见`DEVELOPMENT_RELEASE_VERIFY.json)。依照`INPUT_MANIFEST.json`校验12个归档的大小与SHA-256。
- D1依赖使用冻结提交 `7e32a41b9802438f6645d9048ee46386caa53418`，目录内包含 `b1_health.py` 和 `b1_mode_models.py`。
- Python 3.11.18、NumPy 2.4.6、SciPy 1.17.1，`OPENBLAS_NUM_THREADS=1`。
- 归档和路径树身份见 `task3_dev/RUN.json`。D1数值与统计数据在解封前的参数锁见 `FIT_PARAMETERS.json`。

## Windows PowerShell

在项目根目录从冻结D1检出设置依赖路径，然后切换到联合实验目录。

~~~powershell
$env:D2_D1_DIR = 'C:\work\paperlast-d1\paper1_D1_link_value_20261123'
$env:D2_TASK3_DEV_DIR = 'C:\work\paperlast\paper1_joint_watermark_fusion_20261126\task3_dev'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location 'C:\work\paperlast\paper1_joint_watermark_fusion_20261126'
~~~

恢复任务3输入归档，校验再执行已冻结的评价重放与配对bootstrap。

~~~powershell
gh release download joint-task3-development-ns83-data --repo center-666/paperlast --pattern 'config*.npz' --dir task3_dev
python -B class_weight_development/build_input_manifest.py
python -B class_weight_development/verify_task3_regeneration_full.py
python -B class_weight_development/select_weights.py smoke
python -B class_weight_development/select_weights.py evaluate
python -B class_weight_development/select_weights.py finalize
python -B class_weight_development/verify_actions.py
python -B class_weight_development/configuration_comparison.py
python -B class_weight_development/build_fusion_development_summary.py
~~~

`evaluate`读取已提交的 `FIT_PARAMETERS.json`，只回放 purpose-5/6 归档路径。解封后不要再运行 `fit` 并覆盖参数锁。重新拟合开发参数需在另一个输出目录执行，先比较 SHA-256 再评估。健康路径和攻击评价路径均不得用于拟合。

## 输出口径

- `FIT_PARAMETERS.json`的权重和饱和阈值只由purpose-4训练路径选择。
- `evaluation.csv`记录配置×输入×策略端点、实际请求/执行权重和M7投影计数。
- `paired.csv`按完整物理路径成组重抽样，连续/突发配置等权，独立健康层按配置重采样。
- `pathwise_actions.npz`保存开发评价的逐拍请求与实际权重，可复核报警时序和M7约束。
- `INPUT_REGENERATION.json`确认12,288条攻击与3,072条健康归档输入、密钥、匹配分数及门控与冻结种子重建逐数组一致。
- `ACTION_VERIFY.json`重放全部72个配置×策略动作流并检查M7。结果没有违规动作或额外投影。
- purpose-5/6在历史任务3筛查中已用过，因而这些端点不构成新的独立确认。
