"""Build exact source-line mapping for the nine review-only corrections."""
from pathlib import Path
import hashlib,json,re
H=Path(__file__).resolve().parent;P=H.parent;R=P.parent.parent
O=R/'研究规划/第一篇小论文/secure distributed safety.tex'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def find(text,needle):
 i=text.find(needle);assert i>=0,needle;return text[:i].count('\n')+1
def main():
 original=O.read_text('utf-8-sig');tex=(H/'theory_review.tex').read_text('utf8')
 paths=[O]+[P/'sections'/n for n in ['02_model_detector.tex','03_attack.tex','04_watermark.tex']]
 expected=['0d59143f19ee62cdbb95f8207e47ec9cc481bedcda10b3a24398cd5bfa36b609','1d7c357efe2fa592fe87a6db050cbab9b1fe15b35b9faafd0edfc8c826456bee','71aa152fc1058b291c88fa232c6b94e27462beba6f74aa17c567ffdd502b4d9c','686a8aa9ec53edd1a3b1af294e359f65696330c4eed36fdaaa0df883d970ac42']
 assert all(sha(p)==v for p,v in zip(paths,expected)), 'protected source version mismatch'
 stages={p.name:p.read_text('utf8') for p in paths[1:]}
 details=[
  ('R1.3',[39,40,41,42,45,46],'04_watermark.tex',['wm:structural','wm:entropy','wm:fixed'],['rev:c1','rev:degenerate','rev:entropy'],'从D非零PSD及Z满行秩推出c1>0，并对随机水印边缘law套Gaussian KL等式。','改为逐时间块iff条件；给A=0局部可观反例；只以最大熵作边缘KL下界。','λ>0,q>0；水印时域独立且与全部物理噪声独立；整窗攻击独立于joint(r,β)；固定攻击定理还要求Δ不依赖q、ZΔ=0、c1>0。','逐块迹展开＋Gaussian第四矩＋最大熵；TP1/TP3/TP4。'),
  ('R2.1',[10,11,12],'02_model_detector.tex',['eq:klbudget','eq:score','mod:nominal'],['rev:calibration','rev:KLdefinition'],'直接按真实KL与θ逐拍比较，未给可观测统计量。','使用名义Θ的r^TΘ^-1r和χ²阈值；θ只定义攻击law预算。','预定完整窗口、准确名义协方差、Gaussian原噪声、理想同步；不读取受攻law或标签。','白化残差证明单次χ²；有限监测族union bound不需重叠独立。'),
  ('R3.1',[1,2,3,4],'02_model_detector.tex',['mod:system','mod:estimator'],['rev:model','rev:aggregate'],'共m与每个(A,H_i)可观的范围动机不清楚，易误写邻域可观与自由堆叠KF。','保留共m及原本地可观，说明求和/共G接口；Niu聚合MMSE只是显式补充设计。','图原连通无向无自环；m相同才可原样求和；稳定性另检，不由连通性自动推出。','等价聚合块公式直接展开；不把新增异构堆叠模型写成原稿推广。'),
  ('R3.2',[6,7,8,11],'02_model_detector.tex',['mod:noiseblocks','mod:nominalresidual','mod:nominal'],['rev:window','rev:Theta','rev:calibration'],'在线KL/时窗机制及跨窗统计相关未分清。','精确s个有效过程噪声块；保留零尾块也等价；预定单次/有限族误报与联合Σ分开。','名义残差不受估计器增益影响；缓冲原decoded数据不按报警删除；适应选窗/无限监测另校准。','从共同primitive-noise线性映射构造Θ及跨窗联合协方差；不把重叠统计当独立。'),
  ('R3.3',[13,14,15,16,17],'03_attack.tex',['att:window','att:sequence','att:completewindow','att:harmful'],['rev:autonomous','rev:finitesupport','rev:harm'],'单窗构造容易被解释为有限启动停止全过程隐蔽，非零注入与有害性未分。','主族固定为有限自主a_t=HA^tξ；选定全部受影响窗构造R；分别给有效/有害kernel存在条件。','攻击前后为0，起止/监测窗集合显式，零初始偏移；固定增益全接纳；无水印精确隐蔽与水印随机残差分开。','Rξ=0、kerR不包含kerT/kerB的标准线性判据；TP5显示起止非零。'),
  ('R3.4',[18,19,20,21,22,23,24,25,26,27,28,29],'03_attack.tex',['att:bounded','att:randomwalk','att:unstable','att:jordanmodel'],['rev:bounded','rev:convolution','rev:stop'],'只分Schur稳定与不稳定，单位圆/非平凡Jordan和有限停止未完整处理。','补幂有界、A=I非零极限、Jordan激励例；不稳定结论限定严格分离实模态；有限停止按F^r衰减。','固定增益；Φ/F Schur；持续输入与有限输入分开，Jordan存在不等于所有ξ增长。','稳定卷积与几何级数；Jordan两种ξ反例；TP6。'),
  ('R4.2',[10,40,48,50],'02_model_detector.tex',['eq:klbudget','mod:deterministicKL'],['rev:KLdefinition','rev:jointKL','rev:entropybound'],'定义先给Gaussian公式，虽已有历史补充一般释义，后续仍误用受攻Gaussian等式。','一般measure KL先定义；deterministic shift给等式；秘密条件/边缘/联合law分别写。','P≪P0，否则∞；Gaussian闭式需真实Gaussian；固定Δ时条件给秘密后Gaussian，随机Δ未必。','联合KL链式法则；最大熵只能给下界，不把joint与marginal KL混用。'),
  ('R4.4',[47,48,49,50,51,52,53],'04_watermark.tex',['wm:energy','wm:decodedenergy','wm:scoremean'],['rev:budget','rev:energybound','rev:decodedenergy'],'θ隐蔽约束与实用报警混淆；编码攻击能量下降易被等同实际MSE下降。','保留现有整窗能量界并完整列joint独立/c_Z>0；实际decoded能量系数及P3重选量词另写。','每个q均满足整窗攻击与(r,β)joint独立且有限二阶矩；允许响应公开参数但不读当前payload/secret。','迹下界≥q²c_Z E||Δ||²再反演f；TP3/TP7；q^-2=σβ^-4是一速率。'),
  ('R4.5',[18,19,30,39,42,47,53],'04_watermark.tex',['wm:moments','wm:entropy','wm:energy'],['rev:Fphi','rev:biasstep','rev:biasrec','rev:budget'],'传感器q与水印方差q、预测Φ与后验F、重复矩/熵公式可能混用。','本副本N传感器、q=σβ²；b+/-同物理时刻，F=JA、Φ=AJ；共享lemma一次证明后引用。','传输η=M a与原始量测域注入分名；同阈值不能跨预测/后验直接套用。','两递推相减及AB/BA特征多项式相同但矩阵不同，TP2；全文最终符号审计仍需采用后做。')]
 rows=[]
 for id,tags,stage,labels,targets,before,after,assumptions,proof in details:
  lines={str(tag):find(original,r'\tag{'+str(tag)+'}') for tag in tags if r'\tag{'+str(tag)+'}' in original}
  slines={label:find(stages[stage],r'\label{'+label+'}') for label in labels}
  targetlines={label:find(tex,r'\label{'+label+'}') for label in targets}
  rows.append({'reviewer':id,'original_tags_and_lines':lines,'stage_file':'sections/'+stage,'stage_labels_and_lines':slines,'review_tex_labels_and_lines':targetlines,'before':before,'after':after,'adopted_assumptions':assumptions,'proof_or_counterexample':proof,'status':'已有可采用证据；本轮形成独立审阅副本，未回写原稿，未宣称审稿意见已获接受'})
 log=(H/'tmp/theory_compile/theory_review.log').read_text('utf8',errors='replace')
 result={'date':'2026-11-15','baseline':'72a972ccd4c3dc5fb024ef1b178769073b10c422','source_hashes':{str(p.relative_to(R)):sha(p) for p in paths},'review_tex_sha256':sha(H/'theory_review.tex'),'rows':rows,'originality_unresolved':['R1.1','R1.2','R2.2','R3.5','R5.3'],'compile_check':{'engine':'pdfTeX/TeX Live 2025','draftmode':True,'pdf_generated':False,'fatal_error':bool(re.search(r'^! ',log,re.M)),'undefined_references':'There were undefined references' in log,'overfull_boxes':len(re.findall('Overfull',log)),'visual_QA':False},'old_sources_modified':False,'new_defense_method':False}
 (H/'theory_revision_mapping.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8')
 out=['# 九条数学修订的独立审阅副本与逐项映射','','日期：2026-11-15；审查基线72a972c。该副本不改原稿/阶段稿、不写摘要贡献，也不宣布重投就绪。两项防御候选及独立分析稿定位仍为C；最终采用由研究负责人决定。','','[可独立编译TeX](theory_review.tex)使用原共G架构，独立编号及`rev:`标签，方便逐段审查。它不是整篇新稿，也不是已自动应用到原稿的patch。已用pdflatex `-draftmode`双遍校验语法与交叉引用；不生成PDF，没有宣称完成PDF视觉检查。','','## 1. 输入与版本','','原依据是保留`研究规划/第一篇小论文/secure distributed safety.tex`；其中有历史补字，不等同已取得期刊实际送审PDF。11月2日stage只作已修证明来源；特别是不将其更宽因果攻击族全部采用。输入哈希见`theory_revision_mapping.json`。','','## 2. 必须先统一的接口','','同物理时刻采用 `b_k^- = A b_{k-1}^+`、`b_k^+ = J b_k^- + G η_k`，其中 `η_k=M_k a_k`、`F=JA`、`Φ=AJ`。F和Φ等谱但一般不是同一矩阵/范数；预测阈值与后验阈值不能仅换记号。代码与公式均需以此对齐。','','原共G更新保持；Niu聚合MMSE只是新增参考实现的显式补充设计。原作者原50节点K/G来源没有因此恢复。','','## 3. 九项before/after与证明','']
 for row in rows:
  out += [f"### {row['reviewer']}",'',f"**原位置：** O式/源行 {row['original_tags_and_lines']}。",f"**阶段修复：** `{row['stage_file']}`；labels/源行 `{row['stage_labels_and_lines']}`。",f"**本副本：** `{row['review_tex_labels_and_lines']}`。",'',f"**修改前：** {row['before']}",'',f"**修改后：** {row['after']}",'',f"**采用假设：** {row['adopted_assumptions']}",'',f"**证明/反例：** {row['proof_or_counterexample']}",'',f"**状态：** {row['status']}。",'']
 out += ['## 4. 最小严格反例与不能混用的结论','','1. **c1反例属于原模型类：** n=m=1，A=0、H=1、Q/R>0、s=1，局部可观；O=(1,0)ᵀ，Z=(0,1)，Δ=(ξ,0)ᵀ，非零D却ZDZᵀ=0。这里Z的第一时间块为0；不声称此前三个特定非退化模型有盲区。','2. **边界窗反例：** A=H=1，s=1，L=2，ξ=1，Z=(-1,1)，起点/内部/停止窗shift分别1/0/-1。内部奇偶隐蔽不等于全事件隐蔽。','3. **单位圆反例：** A为二阶单位Jordan、K=0、G=H=I，ξ=(0,1)给线性增长，ξ=(1,0)却有界；停止后F=0使下一后验偏移归零。持续与有限支持不能混说。','4. **编解码后效果：** m维Gaussian水印的E||Ma||²系数为λ²+2λq+(m+2)q²；a=b/q的encoded能量消失并不强制decoded能量消失，亦不直接证明任意KL预算可行。','','这些属于正确性修复与标准数学结果，不作新水印贡献。首包oracle、GLRT、公开符号、已停止候选及第二篇冻结成果不移入贡献段。','','## 5. 原创性待解决标识','','R1.1、R1.2、R2.2、R3.5、R5.3维持未关闭，不编造正面回应。九条中“有可采用证据”只代表数学修订材料就绪，不等于已经回写、原实验已重做或审稿人接受。最终完整论文仍需符号/交叉引用审计与真实门控闭环验证。','','## 6. 复算与交付','','`theory_patch_checks.py`独立以SymPy计算7项小代数证书，结果见JSON；不调用旧脚本，不是五节点性能实验、无物理样本或MC。`theory_revision_mapping.json`保存原行号、阶段label、修订label与输入哈希。TeX编译仅draftmode，实际PDF排版核验留给主流程需要时执行。']
 (H/'theory_patch_draft.md').write_text('\n'.join(out)+'\n','utf8');print(json.dumps({'review_items':len(rows),'originality_open':result['originality_unresolved'],'compile':result['compile_check']},ensure_ascii=False))
if __name__=='__main__':main()
