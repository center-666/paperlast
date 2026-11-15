# 定理、数值与文献证据分级

日期：2026-11-15。所有文件相对本目录；原模型只读，未使用旧数值结果作新脚本分类输入。

|ID|结论|证据等级|复算/正文|不可外推之处|
|---|---|---|---|---|
|G01|固定同源时间残差及其重叠过程law不受同期cross-R改变|严格证明|01节1；g0_independent_review；independent_g0_checks|条件于别源或数据依赖选窗另论|
|G02|外生水印、预定攻击下，偏移/单源报警联合law不变|严格推论|01节3；独立审查节4|无门控、无攻击噪声适应；总真实误差不包括|
|G03|真实聚合R与对角简化的精确条件|严格协方差代数|02节1；两套符号代码|不是必须逐对独立|
|G04|固定F、实际协方差/MSE及均值递推|严格代数及Schur标准推论|02节2|不代表任意K/G稳定|
|G05|五节点ρ=.5正常MSE约增90%|冻结有理输入的精确Lyapunov解/小数显示|analytic_results、exact_rational_witnesses|不是原50节点或工程区间认证|
|G06|空间协方差、方向改善/恶化二维例|严格公式及精确有限矩阵例|02节4；analytic_checks|不能笼统说所有正相关均改善|
|G07|独立空间阈值实际FA变化|标准χ²尾函数数值|analytic_results.scalar_checks|单次，不是全事件/无限运行|
|G08|边缘已知下空间协方差上界及Gaussianχ²支配|严格证明/经典事实|02节5|有限单次误报充分界，非最优功效|
|G09|时间噪声帮助空间去噪及已知均值联合似然等价|严格经典Gaussian恒等式；SymPy核验|02节6；analytic_checks|真实协方差oracle；V奇异须支持分支|
|G10|仅聚合的一般不可识别及实际模型模板交集|精确有限反例/有理秩|02节7；independent_g0_results|本五节点L≥2交集0，不能硬称都无法定位|
|L01|Niu共G来源与MMSE/攻击定理接口|实际指定全文页核读|literature_one、sources|不等于所有证明独立认证|
|L02|Greiff相关量测共享/信息界|实际6页正文核读、关键页视觉|literature_one|完整R已知，非本稿固定K/G|
|L03|Olfat协方差集合水印及渐近声明|官方预印本正文1—7；关键页视觉|literature_one、formula_checks|残差有色/归一化/df适用链待核；不认证有限FA|
|L04|Zhang两篇2026|摘要/出版社预览，全文未得|literature_two、openalex过程记录|不得补造式号、不得称定理排重|
|L05|Ma CI重放检测与界|正式PDF指定页；局部代数独立核|literature_two、check_ma_trace_step|局部trace反例不直接否定完整动态定理|
|L06|Juditsky等未知协方差Gaussian/二次鲁棒统计|arXiv v4指定页核读、视觉23|03节6|不是任意非凸危险集合/全事件最优性|
|D01|方向A当前C，不进入G2|研究判断|05|非全领域否定或所有未来相关检测不可能|

`input_hashes.json`为570个旧输入冻结，`protection_check.json`为最终复核。唯一库183个旧记录内容保持、只新增CNA01—04，详见`library_append_audit.json`；原始第三方PDF及全文提取仅本地保留，不上传公共仓库再分发。
