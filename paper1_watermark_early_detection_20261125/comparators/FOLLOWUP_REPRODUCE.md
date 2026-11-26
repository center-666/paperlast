# W3、W4、W5复现与审计

## 文件与冻结身份

所有路径相对本目录comparators。原比较基准为f97748fcba7c45981878352aa03bc1b15f78dc65，物理入口由study.py固定从8953e1dd9a5e2535ead2698acb6f345a85d4f532读取，水印接收器基准为da7a6c7d92b4bbf3514e86dbe51cca15e84660a5。

| 阶段 | 注册与锁 | 执行与独立核验 | 逐路径归档 |
|---|---|---|---|
| 原异常路径184 | NEXT_PROTOCOL.md中的案例段 | high_loss_trace.py、high_loss_mechanism.py | HIGH_LOSS_PATH.npz、HIGH_LOSS_TRACE.csv |
| W3尾部频率 | NEXT_PROTOCOL.md | tail_study.py、analyze_tail.py、verify_tail.py | tail_dev、tail_confirm |
| W4部分边定位 | NEXT_PROTOCOL.md、PARTIAL_LOCK.json | partial_study.py、analyze_partial.py、verify_partial.py、paired_localization_audit.py | partial_dev、partial_confirm |
| W5转移约束 | W5_MARKOV_PROTOCOL.md、W5_CORRECTION.md、MARKOV_LOCK.json | rotation_markov.py、analyze_markov.py、verify_markov_paths.py | markov_dev、markov_confirm |
| W5补充配对读数 | 原5%主比较及预登记2.5/7.5%灵敏度 | read_markov_results.py | markov_confirm/supplement_*.csv、sensitivity_health.csv |
| 理论核验 | MECHANISM_AND_THEORY_ZH.md | rotation_pair_bound.py、audit_markov_receiver.py | ROTATION_PAIR_BOUND.json、MARKOV_RECEIVER_AUDIT.json |
| 全包核验 | 各阶段运行清单 | verify_followup_bundle.py | FOLLOWUP_VERIFY.json |

各运行RUN.json记录执行提交、Python/NumPy/SciPy版本、源文件SHA-256、全部NPZ哈希、阈值与路径命名空间。W4确认之前的锁提交为7f65ea1，修正后W5确认之前的锁提交为b29145d。随机种子中的20261127是固定命名空间整数。

W3清单还绑定了两个案例JSON的原始Windows CRLF字节。交付时对HIGH_LOSS_REPLAY.json与HIGH_LOSS_COUNTERFACTUAL.json设置精确字节保留，Git归档中的SHA-256与原运行输入一致。新补充表及读数清单明确以LF写出。全包核验同时检查工作副本和Git索引中的冻结输入。

需要Python 3.11、NumPy、SciPy及C编译器。Windows使用MSVC Build Tools，POSIX使用cc。原投影内核在系统临时目录编译。固定版本运行记录采用Python 3.11.18、NumPy 2.4.6、SciPy 1.17.1。Git对象库需包含冻结物理提交。

PowerShell环境设置为

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
~~~

## 回读已交付结果

进入本目录运行以下命令。所有统计从逐路径数组复算。

~~~text
python -B verify_tail.py dev
python -B verify_tail.py confirm
python -B verify_partial.py dev
python -B verify_partial.py confirm
python -B paired_localization_audit.py dev
python -B paired_localization_audit.py confirm
python -B verify_markov_paths.py dev
python -B verify_markov_paths.py confirm
python -B read_markov_results.py
python -B audit_markov_receiver.py
python -B rotation_pair_bound.py
python -B verify_followup_bundle.py
~~~

原异常路径可用以下命令重放。源归档path_id为184，批次5、批内索引24。程序记录六条边的顺序、密钥、逐报文分数、门控和接收节点损失。

~~~text
python -B high_loss_trace.py
python -B high_loss_mechanism.py
~~~

## 全量重新生成

各执行器检查完成文件及其运行身份，已存在且核验一致时直接回读。独立复算可在另一个checkout中仅移出该次实验的输出目录和对应开发锁，再依顺序生成。保留当前交付checkout可供逐文件比较。若使用本轮之后修改过的代码，保存新的版本与运行身份。

~~~text
python -B tail_study.py dev --workers 8
python -B analyze_tail.py dev
python -B tail_study.py confirm --workers 8
python -B analyze_tail.py confirm
python -B partial_study.py dev --workers 8
python -B analyze_partial.py dev
python -B partial_study.py confirm --workers 8
python -B analyze_partial.py confirm
python -B paired_localization_audit.py dev
python -B paired_localization_audit.py confirm
python -B verify_markov.py
python -B rotation_markov.py dev --workers 8
python -B analyze_markov.py dev
python -B rotation_markov.py confirm --workers 8
python -B analyze_markov.py confirm
python -B read_markov_results.py
~~~

不同机器重算的墙钟时间、执行提交字段和NPZ文件容器元数据可能改变，优先逐数组比较数值、种子、阈值和端点。交付副本的精确文件哈希以RUN.json为准。

## 独立单位与统计范围

场景顺序为不稳定模型六格、稳定模型六格，各模型中角度依次为0°、−45°、+45°，每角度起点依次为50、65。受扰持续60拍，恢复窗分别为110至149、125至149。五节点集合为{0,2,14,15,17}。路径MSE按状态误差平方的两维和，再对节点和窗口平均。

W3开发和确认的随机namespace末项为1、2。W4为3、4，同场景每个物理路径共享全部八支持及全部编码—评分组。6144条独立物理路径在八支持下产生49152个支持条件，配对bootstrap以物理路径为单位。

W5修正后的namespace末项为15、16。种子树为

~~~text
[20261127, 52387, split, model, scene, population, batch, component]
population = 0 calibration, 1 healthy holdout, 2 attack evaluation
~~~

组件0生成物理样本，1生成四态编码原始均匀随机量，2及其子索引生成负耦合矩阵。每批32条路径。每个发送端的同拍随机量供其发送的报文共享，监测边发送端依次为1、3、16、31、49、34。

W5采用10000次场景内配对路径bootstrap。主检出和攻击MSE区间在ANALYSIS.json，新增恢复与健康配对区间在supplement_paired.csv和supplement_health_paired.csv。补充读数使用独立重采样种子，因此相同端点区间存在有限bootstrap误差，点估计一致。所有区间为条件于阈值的逐项区间。FWER以健康路径的全部六边、5至149拍和候选起点扫描的最大分校准，各编码的匹配评分单独校准，健康插件分数相同时共享阈值。

## 冻结公式的显示补充

NEXT_PROTOCOL.md原冻结版本中的反斜杠转义损坏影响部分公式显示。该文件字节保留以维持W3/W4运行哈希。执行程序中的事件定义以及注册英文说明均采用同一阈值，现补充为

$$
E_{\rm tail}
=\{\operatorname{MSE}_{Q,\rm match}>1\}
\cap\{\operatorname{MSE}_{Q,\rm match}
       \ge10\operatorname{MSE}_{N,\rm match}\}.
$$

原案例的恒定矩阵为 $I_2$。部分支持为 $\{0\},\ldots,\{5\}$、$\{3,5\}$、$\{0,1\}$。状态相容窗的判据为白化奇偶无噪信号能量除以当前扰动平方范数与1的较大值，小于 $10^{-20}$。首报后大幅误接纳以解码偏移范数至少100定义。原案例报告中的八拍总输入能量比另有字段，两个分母分别记录。

## 失败与撤回记录

markov_dev_failed_attempt_scene_index保留场景索引错误引起的中断运行，其清单为MARKOV_FAILED_ATTEMPT.json。

markov_superseded_health_reuse保留W5最初开发、确认及锁。原种子树遗漏用途索引，校准、健康留出、场景零攻击评价共享随机流，故这批结果全部撤回。MARKOV_RESTART_AUDIT.json逐文件登记原哈希。修正后的候选及转移概率沿用原值，重新开发并于b29145d冻结后启动namespace16的确认。旧数据已被查看，这一点随新确认结论披露。

W4原contrasts.csv中的差值区间遗漏了方法间路径配对，正式解释采用paired_contrasts.csv。各个支持共享物理路径的重采样关系也由同一套权重保留。原点估计和数组逐路径端点核验通过。补充审计文件记录更正后的种子、代码及表格哈希。

W3原VERIFY_INDEPENDENT.json的场景哈希字段错误引用最后一个文件，verify_tail.py已改为按场景文件名绑定。原始SHA验证循环和数值端点独立复核均通过，报告已重新生成。

W5实测零事件的经验bootstrap零宽区间与零失败的样本检出率均需结合有限样本上界。补充文件给出等权场景平均概率的95%单侧上界，并按固定场景列出精确二项区间。2.5/7.5%仅离线改变检测端点与健康FWER阈值，闭环MSE保持原注册5%门控。
