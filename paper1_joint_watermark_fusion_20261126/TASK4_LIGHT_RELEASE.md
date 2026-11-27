# 任务4轻量发布与归档取回

本分支以远端 `joint-watermark-fusion-audit-20261126` 的 `599a2501c5a882d67497d76954c408be222ed1b3` 为父提交，发布任务4复核需要的代码、小型数据和结果表。原始注册文件 `TASK4_CONFIRM_PROTOCOL.md` 与确认锁 `TASK4_CONFIRM_LOCK.json` 按本机原始字节复制，SHA-256分别记录在 `task4_confirm/ARCHIVE_MANIFEST.json`。它们对应本地历史提交 `2f5a631d4a2343e76b13c8e70d34f47bed71ba89`。正式运行归档记录的执行提交也是 `2f5a631d4a2343e76b13c8e70d34f47bed71ba89`，分析交付的本地提交为 `68c6d4f5cc1abc6c4ded5374cd7b9e6e18e4a26f`。

## 原始路径归档

48个NPZ共3,574,447,997字节，另存于研究工作站的 `D:\paperlast_task4_archive_20261127\task4_confirm`。副本SHA-256核验结果在 `D:\paperlast_task4_archive_20261127\ARCHIVE_BACKUP_VERIFY.json`，逐文件清单在 `task4_confirm/ARCHIVE_SHA256SUMS.csv`。`RUN.json` 中的原归档哈希与副本清单一致。

原始归档已按授权公开为 48 个 GitHub Release 资产，共 3,574,447,997 字节。C: 与 D: 位于同一块物理磁盘，Release 提供独立的异机副本。每个文件均可用清单中的字节数和 SHA-256 验收。本地固定标签 `preserve-task4-local-68c6d4f` 指向原始结果提交。[公开下载](https://github.com/center-666/paperlast/releases/tag/joint-task4-namespace85-data) · [匿名回读记录](fusion_attribution/PUBLIC_RELEASE_VERIFY.json)

可使用 `gh release download joint-task4-namespace85-data --repo center-666/paperlast --pattern '*.npz' --dir task4_confirm` 批量下载。运行 `python -B verify_public_archives.py` 可核对 GitHub 报告的全部摘要。加 `--download 文件名` 会额外下载该资产并逐字节计算 SHA-256。

## 干净检出复核

从GitHub检出本轻量分支后，先取回上述归档文件。复核依赖冻结D1分支提交 `7e32a41b9802438f6645d9048ee46386caa53418`。可按下列方式从同一远端检出D1并挂接路径。

~~~powershell
$repo = 'C:\work\paperlast-joint-results-20261127'
$d1 = 'C:\work\paperlast-d1-frozen'
git clone --branch joint-watermark-fusion-results-20261127 https://github.com/center-666/paperlast.git $repo
git -C $repo fetch origin d1-research-registration-20261123
git -C $repo worktree add --detach $d1 7e32a41b9802438f6645d9048ee46386caa53418
$env:D2_D1_DIR = Join-Path $d1 'paper1_D1_link_value_20261123'
$archive = 'D:\paperlast_task4_archive_20261127\task4_confirm'
Copy-Item -Path (Join-Path $archive '*.npz') -Destination (Join-Path $repo 'paper1_joint_watermark_fusion_20261126\task4_confirm')
Set-Location (Join-Path $repo 'paper1_joint_watermark_fusion_20261126')
~~~

运行环境沿用 `REPRODUCE.md` 中记录的 Python 3.11.18、NumPy 2.4.6、SciPy 1.17.1。

在本目录运行下列命令。

~~~powershell
$env:D2_D1_DIR = 'C:\path\to\frozen-D1\paper1_D1_link_value_20261123'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B analyze_task4_postrun.py
python -B correct_task4_contrast_directions.py
python -B build_task4_main_table.py
python -B analyze_task4_m7_interaction.py
python -B verify_task4_postrun.py
~~~

分析器复算任务4全量统计、四格主表的上游表格和M7批次计数。方向校正程序生成与标签方向一致的配对表。四格主表抽取程序读回分析表，并核对四个单元和32行指标。独立核验器重新检查归档SHA、锁定源文件、匹配评分、路径配对、逐邻居状态递推和M7违规数。由于轻量分支从远端基线提交开始，未携带原本地执行提交的祖先链，发布版核验器用清单中的源历史引用核对运行SHA和确认锁身份。原本地历史版核验器以`verify_task4_postrun_local_history.py`保留，其SHA-256见`ARCHIVE_MANIFEST.json`。

## 发布内容

- `TASK4_CONFIRM_PROTOCOL.md`、`TASK4_CONFIRM_LOCK.json`及其原始SHA-256。
- `task4_confirm/RUN.json`、小型统计表、路径级配对表、方向校正表和独立核验结果。
- 任务3、任务4实际运行代码及确认所需的小型模型、健康校准和分析归档。
- 48 个归档的大小、SHA-256 与公开下载位置。[归档清单](task4_confirm/ARCHIVE_MANIFEST.json)
- 主格中位与尾部路径的逐拍归因、两项消融、全健康留出结果及复算命令。[研究报告](FUSION_ATTRIBUTION_ZH.md)
