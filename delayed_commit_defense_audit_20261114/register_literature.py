"""One-time additions to the user's unique library; no parallel bibliography.

Existing bibliographic fields remain intact; append only this reading's notes.
Three obtained PDFs are copied to the canonical library with verified SHA256.
"""
from pathlib import Path
import json,hashlib,shutil,subprocess
H=Path(__file__).resolve().parent;ROOT=H.parents[2];LIB=ROOT/'参考文献'
catalog=LIB/'文献库.json'; data=json.loads(catalog.read_text(encoding='utf-8'))
assert len(data['records'])==176,'Do not re-register or overwrite external changes'
before=json.loads(catalog.read_text(encoding='utf-8'))
(H/'library_before.json').write_text(json.dumps(before,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
a=json.loads((H/'literature_1_sources.json').read_text(encoding='utf-8'))
b=json.loads((H/'literature_2_sources.json').read_text(encoding='utf-8'))
report='../研究规划/第一篇小论文_重新投稿修订_20261102/delayed_commit_defense_audit_20261114/01_直接近邻与创新性排重.md'
def base(rid,key,title,author,year,kind,doi,source,scope):
 return dict(id=rid,citation_key=key,title=title,author=author,bib_author=author.replace(';',' and'),year=year,
  kind=kind,doi=doi,source_page=source,pdf_url='',file='',workspace_file='',pages=0,sha256='',
  institution='',degree='非学位论文',language='英文',topics=['第一篇延迟确认立项审查'],
  reading_status='核心全文未得；不据摘要判定定理等价',read_scope=scope,
  citation_status='第一关审查来源；未写入正式论文',read_reports=[report],
  duplicate_of='',added_date='2026-11-14')
items=[]
li=a[0]
r=base('DCA01','Paper_Li2023_AsynchronousSecure',li['title'],'; '.join(li['authors']),2023,'会议论文',li['doi'],li['url'],
 '取得8页原文；定向核PDF2-6/印刷7074-7078、PDF8参考文献，视觉PDF3、5；Assumptions1-5、Algorithm1、Theorems2-3。普通时间戳排序缓冲不等于攻击确认。')
r.update(booktitle='2023 62nd IEEE Conference on Decision and Control (CDC)',page_range='7073-7080')
items.append((r,'Li2023_primary.pdf',8))
c=b['Chen'];r=base('DCA02','Paper_Chen2026_ProlongedAnomalies',c['title'],c['author'],2026,'期刊论文',c['doi'],c['source_page'],
 '取得正式13页；定向核PDF2、5-7/1302、1305-1307及参考文献12-13，视觉PDF6-7。Algorithm1、Theorems2-3，健康初始段/迟滞界等条件不可移植为Gaussian零风险。')
r.update(journal=c['journal'],volume='13',number='6',page_range='1301-1313',pdf_url=c['pdf_url'])
items.append((r,'Chen_final.pdf',13))
c=b['Bhattacharya'];r=base('DCA03','Paper_Bhattacharya2026_AdaptiveFDI',c['title'],c['author'],2026,'期刊论文',c['doi'],c['source_page'],
 '官方检索正文片段；直接HTML/PDF403，未取得可连续核读全文；不填写完整定理假设或PDF页码。')
r.update(journal=c['journal'],volume='25',number='1',page_range='17:1-17:30');items.append((r,None,0))
for rid,key,row in [('DCA04','Paper_Cheng2023_DelayedDSE',a[1]),('DCA05','Paper_Xu2023_SpatialTemporalAttacks',a[2])]:
 r=base(rid,key,row['title'],'; '.join(row['authors']),2023,'期刊论文',row['doi'],row['url'],
  '核题录/作者摘要及有限官方全文路径；核心PDF未取得，不据摘要填写定理或认定覆盖。')
 for k in ['journal','volume','number','page_range']:
  if row.get(k):r[k]=row[k]
 r['journal']=row['venue']
 if row.get('issue'):r['number']=str(row['issue'])
 r['page_range']=row.get('printed_pages',row.get('article_number',''))
 items.append((r,None,0))
r=base('DCA06','Paper_Kong2018_CPSCheckpoint','Cyber-Physical System Checkpointing and Recovery',
 'Fanxin Kong; Meng Xu; James Weimer; Oleg Sokolsky; Insup Lee',2018,'会议论文',
 '10.1109/ICCPS.2018.00011','https://repository.upenn.edu/cis_papers/839',
 '取得12PDF页（2机构前页+10正文）。实际核PDF5-7的II.B/D、III及IV.A，式3-10，Fig3；视觉PDF6-7；参考文献PDF12。PDF无连续印刷页脚，会议题录22-31，不冒称可见印刷页。')
r.update(booktitle='2018 ACM/IEEE 9th International Conference on Cyber-Physical Systems (ICCPS)',page_range='22-31',
 pdf_url='https://esnl.hnu.edu.cn/ecps2018/2019.3.19.1.pdf')
items.append((r,'checkpoint2018_hnu.pdf',12))
r=base('DCA07','Paper_Moore1973_FixedLag','Discrete-time fixed-lag smoothing algorithms','John B. Moore',1973,'期刊论文',
 '10.1016/0005-1098(73)90071-X','https://users.cecs.anu.edu.au/~john/papers/JOUR/033.PDF',
 '通过网页工具取得作者原始PDF的全文解析，定向读PDF1-5/163-167、式10-16/23；本地下载超时，无本地PDF。真正平滑以未来新息改过去值，与仅延迟交付不同。非全篇证明审证。')
r.update(journal='Automatica',volume='9',number='2',page_range='163-173',
 reading_status='作者原PDF网页解析定向核读；无本地PDF，非仅摘要')
items.append((r,None,0))
migrations=[]
for r,name,pages in items:
 assert all(x['id']!=r['id'] and x.get('citation_key')!=r['citation_key'] and x.get('doi','').lower()!=r['doi'].lower()for x in data['records'])
 if name:
  src=(H/'literature_work'/name).resolve(); dst=(LIB/'02_期刊与会议论文'/(r['id']+'_'+str(r['year'])+'.pdf')).resolve()
  assert src.is_relative_to(H.resolve())and dst.is_relative_to(LIB.resolve())and not dst.exists()
  blob=src.read_bytes();assert blob.startswith(b'%PDF-');digest=hashlib.sha256(blob).hexdigest()
  shutil.copy2(src,dst);assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
  r.update(file=dst.relative_to(LIB).as_posix(),workspace_file=dst.relative_to(ROOT).as_posix(),
   pages=pages,bytes=len(blob),sha256=digest,reading_status='已取得原文并定向核读；非全部证明链审证')
  if not r.get('pdf_url'):r['pdf_url']=r['source_page']
  migrations.append(dict(id=r['id'],temporary_source=str(src),canonical_file=r['file'],sha256=digest,bytes=len(blob)))
 data['records'].append(r)
for rid in ['FPN10','SRA01','SRA02']:
 row=next(x for x in data['records']if x['id']==rid)
 scope={'FPN10':'2026-11-14仍仅期刊预览，未取得核心全文；不据空间残差名称判等价。',
 'SRA01':'2026-11-14四作者期刊终版核心全文未得；未以三作者旧v3冒充终版。',
 'SRA02':'2026-11-14定向核旧三作者v3 PDF3-4、6-8与参考文献，Theorem3与式28-33；仅旧版本，非2026四作者终版。'}[rid]
 row.setdefault('citation_notes',[]).append(dict(date='2026-11-14',report=report,scope=scope,use='延迟确认第一关；未写入正式论文'))
 if report not in row.setdefault('read_reports',[]):row['read_reports'].append(report)
# Trace entries remain annotations on the actually read source, not fictitious
# acquired fulltexts or a second bibliography.
next(x for x in data['records']if x['id']=='DCA01')['related_reference_traces']=[
 dict(reference='[10]',title='Unbiased FIR Filtering for Time-Stamped Discretely Delayed and Missing Data',doi='10.1109/TAC.2019.2937850',fulltext_read=False),
 dict(reference='[17]',title='Data synchronization strategies for multi-sensor fusion',year=2003,fulltext_read=False),
 dict(reference='[18]',title='State and existence estimation with out-of-sequence measurements for a collision avoidance system',year=2013,fulltext_read=False)]
next(x for x in data['records']if x['id']=='DCA06')['related_reference_traces']=[
 dict(reference='[21]',title='A survey of rollback-recovery protocols in message-passing systems',year=2002,fulltext_read=False)]
next(x for x in data['records']if x['id']=='DCA02')['related_reference_traces']=[
 dict(reference='[49]',title='Cascade observer design for a class of uncertain nonlinear systems with delayed outputs',year=2018,fulltext_read=False)]
next(x for x in data['records']if x['id']=='SRA02').setdefault('related_reference_traces',[]).extend([
 dict(reference='[40]',title='Information bounds and quick detection of parameter changes in stochastic systems',year=1998,fulltext_read=False),
 dict(reference='[41]',title='Continuous inspection schemes',year=1954,fulltext_read=False),
 dict(reference='[42]',title='Procedures for reacting to a change in distribution',year=1971,fulltext_read=False),
 dict(reference='[45]',title='Optimal detection of a change in distribution',year=1985,fulltext_read=False)])
data['updated']='2026-11-14'
data['summary']['local_pdf_records']=sum(bool(r.get('file'))for r in data['records'])
data['summary']['metadata_only']=sum(not r.get('file')for r in data['records'])
data['summary']['delayed_commit_gate1_new_sources']=7
catalog.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
python='D:/conda/envs/zx/python.exe'
outputs=[]
for action in [('export',),('check','--hash')]:
 proc=subprocess.run([python,'-X','utf8',str(LIB/'管理工具.py'),*action],capture_output=True,text=True,encoding='utf-8',check=True)
 outputs.append(dict(command=list(action),output=proc.stdout))
out=dict(before_records=len(before['records']),after_records=len(data['records']),
 before_local_pdf_records=before['summary']['local_pdf_records'],after_local_pdf_records=data['summary']['local_pdf_records'],
 independent_full_theses_unchanged=46,new_ids=[r['id']for r,_,_ in items],
 existing_read_note_updates=['FPN10','SRA01','SRA02'],migrations=migrations,checks=outputs,
 canonical_library_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
 no_third_party_fulltext_uploaded=True,no_parallel_library=True)
(H/'library_update.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items()if k not in ['migrations','checks']},ensure_ascii=False))
