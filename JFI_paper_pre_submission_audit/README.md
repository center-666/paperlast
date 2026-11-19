# JFI 核心定理冻结与投稿前审查

冻结基线为 [b1509be](https://github.com/center-666/paperlast/commit/b1509be0265ecd0512befe999847fa5908087872)。本目录包含独立审稿、可编译的英文整合稿及离线复核。投稿决定为 **CONDITIONAL GO**，适合作者终审，正式投稿待阻断项关闭。

- [英文整合稿 PDF](manuscript/manuscript.pdf) · [LaTeX 入口](manuscript/manuscript.tex)
- [独立数学审稿](01_独立数学审稿.md) · [近邻排重表](02_近邻排重表.md)
- [实验与认证审计](03_实验与认证审计.md) · [附录规划](04_附录规划.md)
- [最终投稿判断](05_最终投稿判断.md) · [输入哈希](dependency_lock.json)
- [独立结果](results/independent_checks.json) · [冻结复算日志](results/frozen_replay.txt) · [版面与构建检查](results/build_check.txt)

## 独立检查和编译

以下命令在本目录执行。独立检查读取冻结研究树。依赖为 Python 3.10、NumPy 和 SciPy，版本见同级冻结目录的 requirements.txt。默认以本目录的父目录为冻结研究根，也支持通过 --baseline 指定审核镜像。输出 JSON 与执行日志为本次交付记录。

~~~sh
python -B audit_frozen.py
cd manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
~~~

冻结证书使用同级 JFI_A0_safety_accuracy_theorem_review 的原脚本和依赖，含 Arb。可通过程序接口复算到临时输出目录，避免写入旧结果。以下命令从本目录执行。

~~~sh
python -B -c "import pathlib,sys,tempfile; d=pathlib.Path.cwd().parent; sys.path.insert(0,str(d/'JFI_A0_safety_accuracy_theorem_review')); from health_cost_analysis import evaluate; out=pathlib.Path(tempfile.mkdtemp(prefix='jfi_frozen_')); evaluate(d,out); print(out)"
~~~

独立实现使用批量协方差、Schur 补、正尾积分和已有事件记录。确定性首包证书、统计置信区间和历史 Sobol 数值具有不同证据口径。正文专注已有接收器和指定合成模板。
