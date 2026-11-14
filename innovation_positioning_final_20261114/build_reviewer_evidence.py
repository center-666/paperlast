"""Read-only reviewer/source audit. Outputs remain in this independent folder.

Does not execute old scripts, alter manuscript/library, or claim external review.
"""
from pathlib import Path
import hashlib,json,re

H=Path(__file__).resolve().parent
ROOT=H.parents[2]
P=H.parent
O=ROOT/'研究规划/第一篇小论文/secure distributed safety.tex'
RAW=Path('E:/Apps/Codex/data/attachments/b64223d7-dfc7-4003-b7bf-761f7c412ced/已粘贴的文本.txt')
aliases={'O':O,'M':P/'sections/02_model_detector.tex','A':P/'sections/03_attack.tex',
 'W':P/'sections/04_watermark.tex','I':P/'sections/01_introduction.tex',
 'X':P/'sections/05_analytical_checks.tex','C':P/'sections/06_conclusion.tex',
 'F':P/'manuscript.tex'}
lines={key:path.read_text(encoding='utf-8-sig').splitlines() for key,path in aliases.items()}
def find(key,needle):
 matches=[i for i,line in enumerate(lines[key],1) if not line.lstrip().startswith('%') and needle in line]
 assert matches,(key,needle)
 return matches[0]
def loc(key,needle,description):
 i=find(key,needle)
 return dict(alias=key,path=str(aliases[key].relative_to(ROOT)),line=i,
   matched=lines[key][i-1],description=description)
def oeq(number):return loc('O','\\tag{'+str(number)+'}',f'原式({number})，无独立label时以原tag定位')
def lab(key,label,desc=''):
 return loc(key,'\\label{'+label+'}',desc or label)
def heading(key,title):return loc(key,title,title)
def rloc(relative,needle):
 path=P/relative
 ls=path.read_text(encoding='utf-8-sig').splitlines()
 ix=next(i for i,line in enumerate(ls,1) if needle in line)
 return dict(path=str(path.relative_to(ROOT)),line=ix,matched=ls[ix-1],status='C研究记录，未回写阶段TeX/原稿')

# Parse the actual attached decision text rather than relying on the old table.
raw_lines=RAW.read_text(encoding='utf-8-sig').splitlines()
raw_items={};reviewer=None
for i,line in enumerate(raw_lines,1):
 m=re.search(r'Reviewer #(\d):',line)
 if m:reviewer=int(m.group(1))
 m=re.match(r'\s*(\d)\.\s+(.+)',line)
 if reviewer and m:
  rid=f'R{reviewer}.{m.group(1)}'
  assert rid not in raw_items
  raw_items[rid]=dict(line=i,text=m.group(2))
assert len(raw_items)==29

lib_path=ROOT/'参考文献/文献库.json'
records={r['id']:r for r in json.loads(lib_path.read_text(encoding='utf-8-sig'))['records']}
literature={
 'Zhou':dict(id='INTRO06',theorem='Theorems 2—3、Corollary1；式(25)、Information Leakage',pages='PDF2、4—6/印刷629、631—633',
   scope='交换估计的共识结构；λI+ββᵀ乘性编解码、固定攻击KL与统计参数知情重新攻击已有。不是本稿原量测求和；不等于当前秘密实现知情。'),
 'Ferrari':dict(id='INTRO05',theorem='Theorems1、3、4；Lemma7、Theorem5、Corollaries2—3',pages='PDF6—7、11—14/2561—2562、2566—2569；既核附录17—18/2572—2573',
   scope='带状态互逆滤波、秘密切换、正常恢复及检测充分条件；失配稳定条件含末次切换后不再切换。不是任意切换Gaussian有限误报定理。'),
 'Miao':dict(id='FPN04',theorem='Theorem1、Lemma2；Definition1/Givens、Algorithm1、Lemma3；Definition2/Algorithms2—3',pages='PDF5—7/110—112；PDF8—9/113—114',
   scope='集中式Kalman/创新χ²；编码方向变化、旋转矩阵与刷新、辨识/有限损害/隐蔽持续时间已有。不是本稿有限自主族完整事件统计风险定理。'),
 'Gallo':dict(id='FPN16',theorem='Problem1、Lemma3.1、Theorem3.2、Algorithm1、Theorem3.3/Corollary3.3.1',pages='本机arXiv2502.18948v1 PDF3—6；未冒称出版社最终版',
   scope='残差能量/攻击能量约束下最坏输出损害及切换乘性参数设计。固定参数Schur结果不等于任意切换；所引Anand—Teixeira等价链未全审证。'),
 'Huang':dict(id='FPN06',theorem='Theorems4—5、Remark4、AppendixD',pages='PDF4、9、12—13/文章内同页',
   scope='部分通道编码、观测一致的数据拟合参数及重新攻击；不能保证严格隐蔽不等于所有方向高概率报警。是Automatica112701，非C&S104925。'),
 'He25':dict(id='FPN01',theorem='Theorem1及式(14)—(28)',pages='PDF3—5/18661—18663；题录PDF1/18659',
   scope='读当前编码包并作比例篡改可消去单水印，后续双水印全证明未审证。该攻击不满足本稿全窗独立注入假设。'),
 'He22':dict(id='FPN12',theorem='Algorithm1；Theorems1—3，Theorem4检测隔离',pages='PDF3—7/2845—2849',
   scope='有界噪声、稀疏受攻传感器、新息饱和和多轮估计共识。不是原一跳邻居原量测求和机制的完整同协议复现。'),
 'Niu':dict(id='FPN22',theorem='聚合MMSE设计式(9)—(13)、Assumption2；本条不假造定理号',pages='PDF4/4；已核2—5、7—8模型/算例页',
   scope='原模型来源：邻居量测求和、区域可观。原稿逐节点局部可观更强，异构分别堆叠KF不能直接替代。'),
 'Parity19':dict(id='FPN07',theorem='核心全文未得，不能填写具体定理或判定等价',pages='无实际核心PDF页',scope='仅题录/摘要线索；标记直接近邻缺口，不作为创新通过理由。'),
 'Parity24':dict(id='FPN08',theorem='核心全文未得，不能填写具体定理或判定等价',pages='无实际核心PDF页',scope='作者摘要表明奇偶核/充要条件/最坏攻击已有，不能由摘要决定具体健康历史和支持等价。'),
 'Parity23':dict(id='FPN11',theorem='核心全文未得，不能填写具体定理或判定等价',pages='无实际核心PDF页',scope='部分知识、奇偶残差不变/有害性主题已存在；具体窗口初始化权限未完成排重。'),
}
for v in literature.values():
 rec=records[v['id']]
 v.update(title=rec['title'],citation_key=rec['citation_key'],doi=rec.get('doi',''),
    actual_prior_read_scope=rec.get('read_scope',''),new_pdf_read_this_round=False)

MATH='数学核验可采用（保留明确条件）'
IMPL='实验实现可补（含编辑/引用/物理依据子项）'
NOVEL='防御新颖性未解决'
common=['Zhou','Ferrari','Miao','Gallo','Huang','He25']
rows=[]
def row(rid,issue,original,stage,research,near,correct,remaining,categories):
 assert rid in raw_items
 rows.append(dict(id=rid,raw_review=raw_items[rid],issue=issue,original=original,
   stage_tex=stage,research=research,nearest_literature=near,
   correct_result=correct,remaining_increment=remaining,classifications=categories,
   stage_tex_is_submission_ready=False,research_is_merged_into_tex=False))

row('R1.1','攻击由奇偶核直接导出，实质新条件不足',
 [oeq(13),heading('O','Proposition 1'),oeq(14),oeq(16)],
 [lab('A','att:window'),lab('A','att:causal'),lab('A','att:oneextra'),lab('A','att:harmful'),heading('I','neither membership')],
 [rloc('充分必要条件与旧水印盲区_精确核验_20261104.md','## 3.'),rloc('original_estimator_gate_20261114/报告二_变持续与分段FDI的检测损害边界.md','#')],
 ['Parity19','Parity24','Parity23','Miao'],
 '固定窗kerZ=imO属标准代数；阶段已补健康历史/重叠窗因果实现与独立有害性检验，后续补有限自主支持、起止及P1/P3。',
 '更宽因果自由输入攻击不是当前固定原自主主攻击；直接奇偶近邻具体定理仍未取得，不能认定新攻击原理或首次性。',[MATH])
row('R1.2','随机乘性/切换水印已有，缺区别与直接比较',
 [oeq(30),oeq(31),oeq(32),heading('O','Theorem 2')],
 [heading('I','their matrix form is not a contribution'),heading('W','We adopt'),lab('W','wm:matrix'),lab('W','wm:fixed'),lab('W','wm:energy')],
 [rloc('watermark_boundary_final_20261114/近邻定理对照与论文定位_C.md','## 2.'),rloc('danger_aware_watermark_screen_20261114/报告5_审稿对应与本轮决定_C.md','## 2.')],common,
 '阶段明确采用既有矩阵/互逆机制；后续完成符号/旋转/归一化Gaussian及资源边界，若干新防御候选已C停止。',
 '未有具体新机制＋简单强基线不可替代性质＋真实收益；动态水印同协议算法未复现，不能以KL修好或参数依赖秘密宣布创新。',[MATH,IMPL,NOVEL])
row('R1.3','式(42)非零PSD与Z满行秩不能推出正迹',
 [oeq(40),oeq(42),heading('O','Theorem 2')],
 [lab('W','wm:structural'),lab('W','wm:entropy'),lab('W','wm:fixed'),heading('X','A nonzero attack in a degenerate time block')],
 [rloc('充分必要条件与旧水印盲区_精确核验_20261104.md','## 5.')],['Zhou'],
 'c1>0 iff存在δ_l≠0且Z_l≠0；非Gaussian受攻律用最大熵下界，不再使用原Gaussian KL等式。阶段退化时间块反例支持必要性。',
 '采用时必须保留块条件和固定攻击量词；修复有效性不自动产生新防御，也不证明当前三个模型存在旧水印盲区。',[MATH])
row('R1.4','只测自构攻击，缺已有水印/抗攻击估计及其他攻击比较',
 [heading('O','\\section{Simulation}'),lab('O','tab:detection_metrics'),lab('O','fig:rmse_defense')],
 [heading('X','The following examples'),heading('F','Comparative simulation results are not included')],
 [rloc('original_estimator_gate_20261114/报告一_原稿估计器与后续KF的严格对照.md','## 1.'),rloc('watermark_boundary_final_20261114/统一功效与P1P3比较.md','#')],['Zhou','Ferrari','Gallo','He22','He25'],
 '已有P3逐方案重选ξ、信息边界和原共G补充五节点强基线；与研究模型不同的旧数值已隔离。',
 '原50节点最终H/图/增益/评分/门控与图表来源未闭合；补充模型诊断不替代原实验，也未完成动态方法同成本端到端比较。',[IMPL,NOVEL])
row('R2.1','系统端无法实时取得受攻分布，KL不是在线观测量',
 [heading('O','Definition 1'),oeq(11)],
 [lab('M','eq:klbudget'),lab('M','eq:score'),lab('M','mod:nominal'),lab('M','mod:algorithm')],
 [rloc('danger_aware_watermark_screen_20261114/报告3_水印与空间检测的统一强基线.md','#')],['Zhou','Ferrari'],
 '已区分KL攻击约束与名义Θ白化能量评分；预定完整窗Gaussian单次χ²校准是精确命题。',
 '有限监测并集校准和补充空间评分可采用；不等于未知起点自适应扫描、无限运行、模型误差/丢包和隔离闭环已验证。',[MATH,IMPL])
row('R2.2','相对既有水印的技术困难和优势不明确',
 [heading('O','The main contributions'),oeq(30),heading('O','Theorem 3')],
 [heading('I','The finite-window setting'),lab('W','wm:fixed'),lab('W','wm:energy')],
 [rloc('watermark_fusion_screen_20261114/研究决定_C_当前候选停止.md','## 2.'),rloc('danger_aware_watermark_screen_20261114/报告5_审稿对应与本轮决定_C.md','## 1.')],common+['He22'],
 '具体困难已收敛为时间块非退化、非Gaussian KL、独立性权限、编码/解码损害和有限证据时延；它们可作正确分析。',
 '没有证据表明采用水印超越已有方法提供新的安全能力；新损失函数名称、正熵、满秩、矩优化或NP界不等于新机制。',[MATH,NOVEL])
row('R2.3','需要已有方法的比较实验',
 [heading('O','\\section{Simulation}')],
 [heading('F','Comparative simulation results are not included'),heading('X','Reproducible calculation scope')],
 [rloc('watermark_boundary_final_20261114/统一功效与P1P3比较.md','#'),rloc('danger_aware_watermark_screen_20261114/证据分级与阶段验收.md','#')],['Zhou','Ferrari','Gallo','He22'],
 '单拍机制同协议补充比较和QMC/有限码词/代数复核已有，证据类型已区分。',
 '动态互逆水印和抗攻击滤波器没有同协议全面复现；功效浮点/QMC稳定性不冒充全局概率区间证书，原50节点比较尚未闭合。',[IMPL])
row('R3.1','共同m与逐节点局部可观过强',
 [oeq(2),heading('O','Assumption 2'),oeq(3)],
 [heading('M','Equal measurement dimensions'),heading('M','We assume that each pair')],
 [rloc('original_estimator_gate_20261114/报告一_原稿估计器与后续KF的严格对照.md','## 3.'),rloc('Niu原模型复核与局部修复候选评估_20261103.md','### 2.2')],['Niu'],
 '共m保证邻居求和有定义；原本地可观比Niu聚合区域可观强，后续分别堆叠KF不是同一估计器，原共G实例已重建。',
 '保留强假设可以范围说明；异构/网络可观推广需另证，不能以旧堆叠F/g替代。原实例可逆H还提供普通空间一致性强基线。',[MATH,IMPL])
row('R3.2','在线KL怎样评估，是否用滑窗',
 [oeq(6),oeq(10),oeq(11)],
 [lab('M','mod:window'),lab('M','eq:score'),lab('M','mod:nominal')],
 [rloc('danger_aware_watermark_screen_20261114/first_alarm_bounds.md','#')],['Zhou'],
 '量测奇偶窗不是受攻分布估计窗；在线评分只用名义Θ。重叠窗相关已显式保留，有限时域可用并集预算。',
 '不把滑窗评分相加当独立χ²，不把单次或15拍5%无限重复；长期误报标准和模型不确定性仍未完成。',[MATH,IMPL])
row('R3.3','命题1需存在充要条件而非单构造',
 [heading('O','Proposition 1'),oeq(13),oeq(16)],
 [lab('A','att:window'),lab('A','att:causal'),lab('A','att:oneextra'),lab('A','att:harmful')],
 [rloc('充分必要条件与旧水印盲区_精确核验_20261104.md','## 2.'),rloc('充分必要条件与旧水印盲区_精确核验_20261104.md','## 3.')],['Parity19','Parity24','Parity23'],
 '单窗、受限坐标、全支持/健康历史、有效非零与有害性条件已分开；阶段更宽因果族定理与原有限自主族判据不能混用。',
 '数学必要充分性可在各自假设下采用；直接攻击近邻核心定理未排重，不能把更长证明认定新攻击原理。',[MATH])
row('R3.4','遗漏单位圆模态',
 [heading('O','Proposition 2'),heading('O','Theorem 1'),oeq(29)],
 [lab('A','att:bounded'),lab('A','att:randomwalk'),heading('A','The unit-circle boundary'),lab('A','att:jordanmodel'),lab('A','att:jordanbias')],
 [rloc('有限资源自主轨迹攻击_启动结束与基准核验_20261104.md','#')],[],
 '阶段区分幂有界和非平凡Jordan，A=I可有非零稳态偏移、Jordan可多项式增长；有限停止后由Schur接收F传播。',
 '属标准LTI谱/卷积分析，不作为新防御；临界模态完整性能实验未补，不能把单位圆特征值一概说成幂有界。',[MATH,IMPL])
row('R3.5','乘性水印新颖性差异不够',
 [oeq(30),oeq(34),heading('O','Theorem 2'),heading('O','Theorem 3')],
 [heading('I','their matrix form is not a contribution'),lab('W','wm:moments'),lab('W','wm:fixed'),lab('W','wm:energy')],
 [rloc('adversarial_audit_20261113/投稿独立性评估.md','**投稿轴'),rloc('watermark_boundary_final_20261114/近邻定理对照与论文定位_C.md','## 3.')],common+['He22'],
 '已明确机制归属；新写出的有限窗分析/正确KL和参数权限可以保留，不能列为新水印设计。',
 '公开交替、普通空间检测和既有符号/正交具有强反证；当前无新机制的不可替代保证，指定近邻局部差异不自动提供投稿增量。',[MATH,NOVEL])
row('R3.6','二维对角例太简单，需高维及敏感性',
 [heading('O','The unstable and Schur-stable'),heading('O','\\{1+0.5\\nu_{i,1}')],
 [heading('X','A finite attack with no start-up'),heading('X','The following examples')],
 [rloc('safe_sign_process_20261113/尾部与鲁棒性_证明及结构敏感性.md','#'),rloc('adversarial_audit_20261113/物理来源审查.md','#'),rloc('danger_aware_watermark_screen_20261114/证据分级与阶段验收.md','#')],[],
 '已有三/六状态结构、旧四维与预设敏感性解析检查；原共G五节点为补充条件实例。',
 '结构检查不等于高维物理性能验证；旧堆叠KF鲁棒小邻域不等于原估计器工程鲁棒，不能换估计器来解决本条。',[IMPL])
row('R4.1','贡献叙述过长',
 [heading('O','The main contributions')],
 [heading('I','The analysis is organized'),heading('F','This paper studies')],[],[],
 '阶段已压缩并区分采用机制、正确分析和范围。',
 '编辑项，无可对应外部定理；贡献未成立时不能只改摘要措辞使其显得原创。',[IMPL])
row('R4.2','KL需一般定义而非只Gaussian公式',
 [heading('O','Definition 1'),oeq(10),heading('O','More generally')],
 [heading('M','For probability laws'),lab('M','eq:klbudget'),lab('W','wm:entropy')],[],[],
 'O:221—223已有历史补充的一般KL文字说明，不把当前O说成毫无修复；B给完整一般相对熵/绝对连续边界，确定均值平移才用Gaussian等式，随机水印受攻律用最大熵下界。',
 '信息论标准定义/界，无新的水印机制；并稿时继续区分边缘/给定秘密条件/完整联合记录KL。',[MATH])
row('R4.3','实际如何同步时变水印',
 [heading('O','synchronized pseudo-random'),oeq(31),oeq(32)],
 [heading('W','The transmitter and receiver must agree'),lab('M','mod:algorithm')],
 [rloc('watermark_boundary_final_20261114/近邻定理对照与论文定位_C.md','## 2.')],['Miao','Ferrari','Zhou'],
 '阶段给共享生成器/secret/源身份/packet epoch，按发送索引解码而非接收数；理想iid与实际PRNG已区分。',
 '这是声明的理想同步假设，不是失步、丢包重同步、有限精度或泄露抗性的部署证明；刷新/同步本身已有。',[IMPL])
row('R4.4','θ隐蔽是攻击约束还是检测量，如何取得受攻Θ*',
 [oeq(11),heading('O','Scenario 2'),oeq(47)],
 [lab('M','eq:klbudget'),lab('M','eq:score'),lab('W','wm:energy'),lab('W','wm:scoremean')],
 [rloc('danger_aware_watermark_screen_20261114/报告2_危险方向与检测弱方向的错配.md','#')],['Zhou'],
 'θ为分布级攻击预算、τα为名义评分阈值，接收端不用受攻协方差；E评分不是Pd，固定攻击极限不是参数知情家族一致极限。',
 '所有估计/在线校准/门控性能仍需匹配真实实现；只证明编码注入能量缩小不能推出解码损害消失。',[MATH,IMPL])
row('R4.5','定理2/3证明重复与符号混用',
 [heading('O','Theorem 2'),heading('O','Theorem 3')],
 [heading('I','The number of nodes is'),lab('W','wm:moments'),lab('W','wm:tracebound'),lab('W','wm:energy')],[],[],
 '阶段节点数N/水印方差q区分，复用矩引理及最大熵界，能量速率q^{-2}=σβ^{-4}为同一量。',
 '编辑/符号项，不适用外部新定理；后续研究G同时指融合增益和响应矩阵时仍须并稿统一，不能算独立贡献。',[IMPL])
row('R4.6','早期RMSE差异和σβ²的关系',
 [lab('O','fig:rmse_defense'),oeq(30),oeq(12)],
 [lab('W','wm:decodedenergy'),heading('X','Encoded energy, decoded disturbance')],
 [rloc('danger_aware_watermark_screen_20261114/报告4_有界水印的可行性与停止边界.md','## 3.'),rloc('watermark_fusion_screen_20261114/研究决定_C_当前候选停止.md','## 2.')],['Zhou','Gallo'],
 '解码Ma可同步放大检测与首包GM a；固定融合下偏移与噪声交叉项在明确独立/零均值时才化成MSE增量。',
 '首包偏移/NP瓶颈不等于原图RMSE来源已核；需原σβ/门控/启动/置信误差曲线与数据配置闭合，未补实验不能关闭。',[MATH,IMPL])
row('R5.1','补差分进化动态/隐蔽FDIA研究',
 [heading('O','\\section{Introduction}')],
 [heading('I','Parity-space monitoring has already')],[],[],
 '已增直接奇偶相关主题定位，未冒认差分进化具体论文。',
 '原意见没有完整题录；必须先定位真实来源及与本问题的关系。引用编辑项，无已核具体定理可填；不得机械猜引。',[IMPL])
row('R5.2','单一FDI能否扩为混合攻击',
 [oeq(5),oeq(34)],
 [heading('W','Two attack classes are considered'),heading('W','Finally, consider an attacker able')],
 [rloc('original_estimator_gate_20261114/报告二_变持续与分段FDI的检测损害边界.md','#'),rloc('adversarial_audit_20261113/反例与失效条件报告.md','#')],['He25','Huang'],
 '当前主线仍单链编码域模型一致性FDI；短时/分段、载荷读取和当前参数预补偿用于明确不同权限与失效边界。',
 '这些不等于FDI+DoS/欺骗+丢包等混合攻击理论。范围回应可用；未授权更换主攻击，不能把边界反例包装成混合防御保证。',[MATH,IMPL])
row('R5.3','公式相似，实质推导困难及贡献在哪里',
 [heading('O','The main contributions'),oeq(13),oeq(30),oeq(42),oeq(53)],
 [heading('I','neither membership'),heading('I','their matrix form is not a contribution'),lab('A','att:causal'),lab('W','wm:energy')],
 [rloc('watermark_boundary_final_20261114/近邻定理对照与论文定位_C.md','## 3.'),rloc('danger_aware_watermark_screen_20261114/报告5_审稿对应与本轮决定_C.md','## 3.')],common+['He22','Parity19','Parity24','Parity23'],
 '核、可达性、Gaussian四阶矩、PF、GLRT/NP、半定/特征值、标准范数缩放都承认为工具；正确量词与反例提高可信度。',
 '未建立新的不可替代防御机制。有限窗编码能量界等分析须相对具体定理单独评估；增加证明长度、改指标或命名都不能关闭本项。',[MATH,NOVEL])
row('R5.4','删基础介绍突出新意',
 [heading('O','\\section{Introduction}'),heading('O','Notations:')],
 [heading('I','The analysis is organized'),heading('I','\\paragraph{Notation.}')],[],[],
 '阶段已压缩通用背景并明示已有机制。',
 '编辑项，无外部定理；新贡献定位未确认，不能以精简替代新颖性。',[IMPL])
row('R5.5','补比较仿真',
 [heading('O','\\section{Simulation}')],
 [heading('F','Comparative simulation results are not included'),heading('X','Reproducible calculation scope')],
 [rloc('danger_aware_watermark_screen_20261114/报告3_水印与空间检测的统一强基线.md','#')],['Zhou','Ferrari','Gallo','He22'],
 '已有正确同模型单拍编码/空间强基线的补充分析和明确负结果。',
 '完整算法基线、原50节点受攻与正常精度/误报/隔离公平比较仍未完成。静态理论表不是实际50节点仿真。',[IMPL])
row('R6.1','真实物理类似攻击事件与数学构造联系',
 [heading('O','Construction of Strictly Stealthy Attacks against')],
 [heading('X','A finite attack with no start-up')],
 [rloc('有限资源自主轨迹攻击_启动结束与基准核验_20261104.md','#')],[],
 '双积分器/跟踪或循环例可以解释虚假轨迹和有限偏移，但均为明确的数学示意。',
 '没有来源真实事件证明它按本构造发生，不能把示意写成现场事故；真实案例或诚实范围说明待补。本条不适用水印定理。',[IMPL])
row('R6.2','需式(13)—(53)的软件实现',
 [oeq(13),oeq(53)],
 [lab('M','mod:algorithm'),heading('X','Reproducible calculation scope')],
 [rloc('original_estimator_gate_20261114/证据表.md','#'),rloc('danger_aware_watermark_screen_20261114/证据分级与阶段验收.md','#')],[],
 '名义缓冲/评分参考实现、独立有理恒等式、原共G矩阵/DARE误差桥与阶段脚本已有，具体输入哈希可追溯。',
 '局部数值认证和运行原脚本不等于原全部公式到图表端到端实现；隔离、随机配置、软件依赖和输出图数据映射未闭合。',[MATH,IMPL])
row('R6.3','说明仿真软件/数据处理流程',
 [heading('O','\\section{Simulation}'),heading('O','independent Monte Carlo runs')],
 [heading('X','Reproducible calculation scope')],
 [rloc('original_estimator_gate_20261114/报告一_原稿估计器与后续KF的严格对照.md','## 1.'),rloc('danger_aware_watermark_screen_20261114/README.md','#')],[],
 '各新分支已有依赖/输入/哈希与证据等级说明，QMC、物理采样和精确枚举已区分。',
 '原图声称1000次MC与最终源码/种子/图数据尚未唯一闭合；新分支Python流程不能追认成原仿真软件完整复现。',[IMPL])
row('R6.4','逐链路解释表1数据',
 [lab('O','tab:detection_metrics'),oeq(55)],
 [heading('X','The following examples')],
 [rloc('web_revision_control_20261112/29条意见_本机状态与剩余验收.md','|R6.4|')],[],
 '已有版本差异和图表追溯疑点保留；新研究单链结果明确对象。',
 '六条原受攻链最终数据必须先核准再解释；网络均值/另一0→1实例不能替代原表逐链路结果。本条无外部定理。',[IMPL])
row('R6.5','摘要结论给关键量化结果',
 [heading('O','\\begin{abstract}'),heading('O','\\section{Conclusion}')],
 [heading('F','This paper studies'),heading('C','Finite-window parity residuals')],
 [rloc('danger_aware_watermark_screen_20261114/报告5_审稿对应与本轮决定_C.md','## 5.')],[],
 '阶段摘要已写有限条件、编码能量阶和检测极限，没有把弱候选百分比包装成成功。',
 '最终摘要需对应采用的完整论文和证据。新首包风险负界可作为分析限制，不能写成已实现新防御；编辑项，无独立定理要求。',[IMPL])

assert [r['id'] for r in rows]==list(raw_items)
assert len(rows)==29

# Snapshot all canonical completed first-paper report names/hashes/headings.
# This is a coverage inventory, not a claim that every word/proof was reread.
excluded={'historical','tmp','output','_build','publication','literature_work','__pycache__',H.name}
inventory=[]
for f in sorted(P.rglob('*.md')):
 if any(part in excluded for part in f.relative_to(P).parts):continue
 ls=f.read_text(encoding='utf-8-sig').splitlines()
 inventory.append(dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
   headings=[dict(line=i,text=l)for i,l in enumerate(ls,1)if re.match(r'^#{1,3} ',l)],
   scope='stage coverage inventory; no claim all proofs independently verified this round'))

out=dict(date='2026-11-14',requested_baseline='b9d057e',
 assessment='C recommended for current new-defense-method claim; not field-wide impossibility or accepted analysis-paper novelty',
 raw_review_source=dict(path=str(RAW),sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),
   editor_decision='declined; most reviewers questioned contribution/novelty; routine revision not sufficient in R1',
   actual_submission_pdf_obtained=False),
 source_files={k:dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for k,p in aliases.items()},
 unique_library_sha256=hashlib.sha256(lib_path.read_bytes()).hexdigest(),
 source_status='O is local preserved TeX, not authenticated submission-system final PDF; B is Nov2 analytical work revision; C reports are not merged',
 current_resource_boundary=dict(
   previous_resource_face='kappa=1 and Penc<=P0 imply unit orthogonal M under independence/full conditional rank',
   current_reference='original Gaussian Penc=nu*P0, not healthy unencoded P0',
   nu='3/4-(1/8)*exp(1/2)*E1(1/2) = 0.6346361709395336913958938280213869059272...',
   minimum_feasible_kappa='nu^(-1/2) = 1.2552712713820280439211036480825874199360...',
   equality_structure='M=kappa_star*O a.s. under independent current measurement/full conditional second moment',
   old_unitary_first_packet_miss_gt_048_not_transferred=True,
   kappa_gt_boundary='feasible freedom may remain; does not itself establish new mechanism, true performance or novelty',
   proof_source=dict(path=str((H/'数学补充_原Gaussian资源边界.md').relative_to(ROOT)),
     sha256=hashlib.sha256((H/'数学补充_原Gaussian资源边界.md').read_bytes()).hexdigest())),
 classifications_nonexclusive=True,no_closed_count_claim=True,rows=rows,literature=literature,
 completed_stage_inventory=inventory,protected_sources_written=False,
 no_new_paper_writing=True,no_old_scripts_run=True,no_new_literature_search=True)
(H/'reviewer_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def fmt_locations(items):
 return '<br>'.join(f"{v['alias']}:{v['line']} {v['description']}" for v in items)
def fmt_research(items):
 return '<br>'.join(str(Path(v['path']).relative_to('研究规划/第一篇小论文_重新投稿修订_20261102'))+':'+str(v['line']) for v in items) or '无新增研究定理；见相应B编辑/实现范围'
intro='''# 论文级创新性评估：29条真实JFI审稿意见逐项对应

日期：2026-11-14；指定研究基线`b9d057e`。本次只建立论文定位证据，不重写原稿、阶段TeX/PDF或既有报告，不运行旧核查脚本，不改唯一参考文献库，不启动新的查新或实验。

**当前建议C：停止以“提出独立新乘性水印防御机制”为中心的投稿主张。** 数学修复、攻击条件、性能边界和负结果值得保留；它们没有共同建立新机制、不可替代性质和同资源真实性能收益。此判断仅针对当前成果，不宣称所有有界可逆编码不存在改进，也不预先否定经过另行独立性验收的分析论文。

## 1. 版本、来源与分类口径

- **O：本机保留原稿** `研究规划/第一篇小论文/secure distributed safety.tex`。本轮按其实际行号、原式tag及label定位；本机TeX已有历史局部文字，不冒称取得投稿系统真实最终PDF。
- **B：11月2日16页阶段工作稿** `manuscript.tex`及`sections/01—06`。其首页明确“不含比较仿真、非投稿准备完成”；原定理2/3对应阶段标签`wm:fixed`/`wm:energy`（阶段Theorem3/4）。
- **C：随后独立研究记录**。旧F/g安全符号、GLRT、ρ、单拍边界及危险方向等报告均未回写O/B，不能将“报告中已证”计为正式正文完成。

下表源别名：M=`02_model_detector.tex`，A=`03_attack.tex`，W=`04_watermark.tex`，I=`01_introduction.tex`，X=`05_analytical_checks.tex`，C=`06_conclusion.tex`，F=`manuscript.tex`。每个位置都有机器证据中的真实路径、匹配行和输入SHA256。原审稿附件条目起始行也已重新解析，数量为4+3+6+6+5+5=29。

分类**不互斥**：M“数学核验可采用”仍附条件；E“实验实现可补”包括明确的编辑/引用/物理依据子项；N“防御新颖性未解决”。不将三类相加，不给“已解决29条”的虚假总计。编辑/实验意见若没有数学命题，明确不适用外部定理，不硬配论文。

**新旧资源面分开：** 上一轮以健康未编码功率P₀为参考的κ=1结论属于历史范围；本轮参考原Gaussian的精确νP₀，最小可行峰值κ*=ν⁻¹/²≈1.25527127138，等号才强制scaled orthogonal。旧单位正交首包漏警>.48不直接迁移。详见[数学补充](数学补充_原Gaussian资源边界.md)。

## 2. 29行公式—修订—近邻—剩余增量对照

外部文献短名对应第3节的具体定理与实际PDF/印刷页；空项为编辑/实现或标准数学工具，不表示已做全领域排重。

|意见/真实原附件起始行|原稿公式、label、行|B实际章节/定理label、行|C研究记录（未并稿）|最近已有结果/实际页入口|已有正确内容|剩余独立增量/验收|分类|
|---|---|---|---|---|---|---|---|
'''
table=[]
for r in rows:
 near='、'.join(r['nearest_literature']) or '不适用特定安全定理；见本行说明'
 tags=' / '.join('M' if c==MATH else 'E' if c==IMPL else 'N' for c in r['classifications'])
 table.append('|'+ '|'.join([r['id']+' / '+str(r['raw_review']['line'])+'：'+r['issue'],
   fmt_locations(r['original']),fmt_locations(r['stage_tex']),fmt_research(r['research']),near,
   r['correct_result'],r['remaining_increment'],tags])+'|')
lit=['\n## 3. 四项核心新颖性质疑的定理级近邻基础\n',
 '本轮复用实际指定页核读，未计为重新通读全文。文献缺全文是证据缺口，不是A或独立创新的支持。\n',
 '|近邻/固定ID|具体已有结果|实际阅读页|对本稿的约束与不能扩大之处|\n|---|---|---|---|']
for key in ['Zhou','Ferrari','Miao','Gallo','Huang','He25','He22','Niu','Parity19','Parity24','Parity23']:
 v=literature[key]
 lit.append(f"|{key} / {v['id']} / {v['doi']}|{v['theorem']}|{v['pages']}|{v['scope']}|")
conclusion='''

### R1.2、R2.2、R3.5、R5.3分别还差什么？

**R1.2（机制身份）**：阶段已承认采用Zhou矩阵和互逆编码。Miao已给旋转及刷新，Ferrari已有带状态互逆切换，Gallo已有最坏损害设计。单拍符号、Haar旋转、满秩、时间相关、公开参数更新不构成新的机制身份。Huang/He提醒统计参数未知、当前实现未知和载荷不可读是不同权限。当前没有经同权限强基线检验的不可替代机制。

**R2.2（困难和优势）**：非Gaussian KL、时间块非退化、编码/解码能量区别、P3重选与首次危险事件是具体分析困难。修复这些使理论可信，但真实优势不能由困难存在推出。12词相关方案仅微小实际功效变化，ρ候选未过正常精度—风险强基线；普通空间一致性已利用原可逆H实例的可信本地量测。当前没有严格新增能力来支撑优势陈述。

**R3.5（已有水印以外的新性质）**：有限能量和真实报警事件与既有文献不完全同式，但Gallo已联合残差能量与最坏损害，Miao已有有限损害/隐蔽时长；换风险名称或有限时域记号不足。上一轮κ=1且Penc≤健康未编码P₀的资源面仅允许单位正交；**本轮参考原Gaussian的精确νP₀，最小可行κ*=ν⁻¹/²≈1.25527127138，在独立当前量测/条件二阶矩满秩假设下，边界等号强制M=κ*O**。这由[原Gaussian资源边界补充](数学补充_原Gaussian资源边界.md)独立有理认证；不是沿用κ=1。旧单位正交首包漏警>.48不能原样迁移到κ*O。κ>κ*仍有可行自由度，但自由度本身不等于新机制、真实性能或A；当前C也不扩大为任意κ的全领域不可能性。

**R5.3（公式与实际贡献）**：核空间、可达性、Gaussian矩、广义特征值、PF最大熵、GLRT、NP引理和范数封顶是工具。可审查的增量只能是：在固定原共G模型、同一攻击权限及实际资源下，某个新能力产生简单基线不能达到的已证性质和真实性能。当前尚未得到。更宽因果攻击章节可能保留为分析候选，但其与直接奇偶近邻定理的独立性尚未证成，且不能静默替换用户固定原自主主攻击。

## 4. 已完成分支如何影响论文级判断

下表汇总各阶段的有效结果和停止边界，不把报告数、代码数或证明长度当原创。全部规范Markdown报告的路径/哈希/标题清单在JSON中；这是阶段覆盖清单，**不是声明本轮逐字重审每份报告及全部证明**。既有已读范围沿用，各关键结论依据实际主报告及本次原文/标签核对。

|阶段|可保留结果|对创新/采用的限制|
|---|---|---|
|11月2日阶段攻击与KL修订|健康历史/重叠窗因果条件、有害性、临界谱边界、c1、非Gaussian KL、名义χ²评分|更宽攻击不是当前主族；正文仅分析例，未含比较仿真；修复不能变新水印|
|11月3日风险联合水印/融合|正确风险模型、攻击与融合接口、有限强基线对照|共同最优未显示不可替代联合收益；异构堆叠KF不等于原共G|
|11月3—4日局部修复/可信验证|Niu求和与区域可观核对；共享认证/旁证资源诊断|区域可观不等于纠错冗余；候选等同强选择性认证＋标准匹配，未有独立新机制|
|11月4日有限自主攻击/充要条件/P1|真实编码能量、支持/启动/停止/有效性/有害性分开|完整窗隐蔽不能扩大为全过程，旧数值绑定旧F/g；标准核不列原创|
|11月5日P3和12词相关水印|实际损害Sπ、最低损害边界、有限码本逐实现秩、概率求积|真实最坏功效仍低；受限重复码本上限仅冻结类有效，停止调概率|
|11月5日正交时序混合及路线收束|单标量通信实际等待、参数知情预补偿、简单符号强基线|保能量不等于更安全；原D门槛不可行不等于100%检出；暂停时序混合|
|11月12日条件GLRT|标准匹配检验、联合误报与危险词精确上界|不是新GLRT；known-onset末拍诊断不是未知起点全事件风险优势|
|11月13日安全符号过程与独立对抗审查|冻结模型安全词/尾部/图过程数学核验|旧堆叠KF、单次L10/零残余窄；公开交替主指标强基准、最小扩展反例，停止秘密方法主张|
|11月14日原估计器回归|共m/先求和/共G映射、补充聚合DARE、原相容五节点与误差桥|原50节点最终配置/图表未闭合，不能移植旧安全词数字或冒称原实验复现|
|11月14日ρ候选|即时范数封顶、正确正常/偏移递推、能量界|普通固定缩放/裁剪强基线未支持不可替代耦合，C停止；不重新命名串联|
|11月14日固定单拍编码边界|真实Pd/P1/P3、资源向量、公开基线、早期普通检测风险反例|无新机制＋真实性能＋不可替代定理三者，C停止随机矩阵/码本搜索|
|11月14日上一轮危险方向/空间基线|共同危险集完整角/径向几何、原信息空间检验；历史κ=1且Penc≤P₀时的正交门槛、单位编码首包oracle界|该历史资源面不是本轮原Gaussian的νP₀；旧>.48不迁移。危险全角且简单基线已有可检测性；NP界标准，未证明任意κ最优性|
|11月14日本轮原Gaussian资源补充|精确ν、κ*=ν⁻¹/²的可行阈值及条件等号scaled orthogonal，由新目录有理证书支持|κ>κ*仍有自由度，未证明检测—损害分布最优性，也未形成独立新机制；不因可行就判A|

## 5. 原50节点实验与正式采用状态

原稿O第1111行起写50节点、1000次MC和原图/表；第1146行起有Q/R/H随机范围、第1155行起有T=150、s=4、θ=39。**文本给参数范围不等于恢复最终抽样H/邻接矩阵/增益/种子/评分/门控和原图数据的一致来源。** 既有追溯与11月14日原估计器报告确认该链仍未闭合。

新五节点实例忠实原求和/共G，并显式补入聚合MMSE设计，可作独立补充实例；它不是原50节点子图，也不是最终图重跑。旧异构分别堆叠KF和960安全符号证书不能移植到原架构。研究报告中存在结果不能自动算作原稿或11月2日阶段TeX完成，更不能认定期刊已接受其回应。

## 6. 论文级采用建议

**数学材料可采用**：保留各自假设的有限支持/全过程条件、P1/P3、谱与停止边界、非Gaussian KL及块非退化、名义评分、编码损害和信息权限分离。采用前仍需依当前主攻击选择版本、统一预测/后验时序及符号，不能直接拼接所有分支。

**实现与编辑可补**：原配置/图表溯源、同协议强算法比较、实际门控/同步/误报、临界或高维实验证据、准确真实案例、引用与摘要精简。这些尚未执行，本评估不替用户完成论文重写，也不以“可补”暗示补完必录用。

**防御新颖性仍未解决**：四项核心意见缺的是有价值的新能力而非一条尚未写出的漂亮不等式。现有最强结果支持重新讨论“有限资源检测—损害边界分析”定位；分析稿本身相对直接攻击与编码近邻的独立增量尚须另审。不能把改标题、降低期刊或尚缺全文排重当作A理由。

所以当前建议C，停止新的防御方法主张与无目标矩阵搜索；保留正确成果、负结果与证据缺口供导师作论文定位决定。此处没有替用户决定改稿主题，也没有把所有研究劳动判为无价值。
'''
(H/'论文级创新性评估_29条审稿对应.md').write_text(intro+'\n'.join(table)+'\n'+'\n'.join(lit)+conclusion,encoding='utf-8')
print(json.dumps(dict(reviewer_items=len(rows),stage_inventory=len(inventory),
    classified_nonexclusively=True,outputs=['reviewer_evidence.json','论文级创新性评估_29条审稿对应.md']),ensure_ascii=False))
