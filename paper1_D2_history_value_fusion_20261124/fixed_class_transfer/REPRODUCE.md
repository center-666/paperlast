# 迁移审计复现

注册提交为e33339a，父提交为d6d958348ec3bbc73fe42df0a9e9cf8a337a8d78。运行目录为本目录fixed_class_transfer，D1读取7e32a41b9802438f6645d9048ee46386caa53418的独立checkout。

~~~powershell
$env:D2_D1_DIR = 'C:\path\to\frozen-D1\paper1_D1_link_value_20261123'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B transfer.py --audit
python -B transfer.py
python -B analyze_verify.py
~~~

transfer.py先检查完成文件的身份和哈希，完成路径会直接回读。在独立checkout中重新生成时，可将results目录移出该checkout后按同一顺序运行。保留交付副本供逐数组比较。

新物理与攻击生成完整使用冻结b_run.py的make_seed接口，新namespace为[20261126,88403,1]，stage1用于健康，stage2用于攻击。每批64条。各场景分别使用冻结B2风险表、H1阈值与预算。Class-model和Class-topology的固定权重在PROTOCOL.md中逐项列明。

results/RUN.json记录新输入身份、旧输入哈希、执行提交和每场景NPZ的SHA-256。NPZ保留每条路径四策略动作、七边报警、物理探针及五列损失。损失列为目标缺失块受扰期、恢复期、20拍后，及全网受扰期、20拍后。12场景顺序沿用B2，连续与突发物理流独立。相同世界内四策略共享物理噪声、报文和检测掩码。

analyze_verify.py从每个场景的三种样本世界各重新生成首批64条路径，用独立决策循环重建动作，再以冻结逐邻居量测实现验证五列损失。统计对场景—输入条件分层重采样，四策略共享路径权重。健康样本单独按12场景分层。

全边FWER在本轮使用全部七条监测边。旧B2摘要中的masks[:,:,4:]切片在RESULTS_ZH.md中记录，旧代码、旧结果和原分支均保留其冻结身份。

DETERMINISTIC_AUDIT.json记录全部64历史和13风险单元的穷举，适用冻结模型与风险表。等价性归纳证明见RESULTS_ZH.md。新路径评价是冻结策略迁移审计，统计区间为条件于旧阈值与旧风险表的逐项95%区间。
