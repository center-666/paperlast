from pathlib import Path
import json,hashlib,shutil,subprocess
H=Path(__file__).resolve().parent;ROOT=H.parents[2];LIB=ROOT/'参考文献';f=LIB/'文献库.json'
before=f.read_bytes();d=json.loads(before);scope=json.loads((H/'scope_before_results.json').read_text())
assert hashlib.sha256(before).hexdigest()==scope['library_before_sha256']
(H/'library_before.json').write_bytes(before)
sources=json.loads((H/'literature_sources.json').read_text(encoding='utf-8'))['sources']
rows=[];report='../研究规划/第一篇小论文_重新投稿修订_20261102/realtime_admission_feasibility_20261115/literature_audit.md'
for rid,pages in [('FPN18',13),('FPN19',15)]:
 r=next(x for x in d['records']if x['id']==rid);a=next(x for x in sources if x['id']==rid)
 assert not r.get('file')
 src=ROOT/a['path'];dst=(LIB/'02_期刊与会议论文'/(rid+'_formal_'+str(r['year'])+'.pdf')).resolve()
 assert src.resolve().is_relative_to(H.resolve())and dst.is_relative_to(LIB.resolve())and not dst.exists()
 blob=src.read_bytes();assert blob.startswith(b'%PDF-')and hashlib.sha256(blob).hexdigest()==a['sha256']
 shutil.copy2(src,dst);assert hashlib.sha256(dst.read_bytes()).hexdigest()==a['sha256']
 r.update(file=dst.relative_to(LIB).as_posix(),workspace_file=dst.relative_to(ROOT).as_posix(),pages=pages,
  sha256=a['sha256'],bytes=len(blob),pdf_url=a['primary_url'],
  reading_status='已取得正式全文并定向核读；非全部证明独立审证')
 note='2026-11-15实时准入审查：PDF'+','.join(map(str,a['read_pdf_pages']))+'；关键页视觉核对。详见本轮报告，不能将文献保证移植为当前风险/精度结论。'
 r['read_scope']=r.get('read_scope','')+'\n'+note
 r.setdefault('read_reports',[]).append(report);r.setdefault('citation_notes',[]).append(dict(date='2026-11-15',report=report,scope=note))
 rows.append(dict(id=rid,temporary_source=str(src),canonical_file=r['file'],sha256=r['sha256'],pages=pages))
for rid in ['FPN12','FPN16']:
 r=next(x for x in d['records']if x['id']==rid)
 r.setdefault('citation_notes',[]).append(dict(date='2026-11-15',report=report,scope='本轮按literature_sources.json所列指定页复核；非新通读或全部证明链认证。'))
 if report not in r.setdefault('read_reports',[]):r['read_reports'].append(report)
d['updated']='2026-11-15';d['summary']['local_pdf_records']=sum(bool(r.get('file'))for r in d['records'])
d['summary']['metadata_only']=sum(not r.get('file')for r in d['records'])
f.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
logs=[]
for cmd in [('export',),('check','--hash')]:
 p=subprocess.run(['D:/conda/envs/zx/python.exe','-X','utf8',str(LIB/'管理工具.py'),*cmd],capture_output=True,text=True,encoding='utf-8',check=True)
 logs.append(dict(command=list(cmd),output=p.stdout))
out=dict(before_records=len(d['records']),after_records=len(d['records']),before_pdf=122,after_pdf=d['summary']['local_pdf_records'],
 new_records=0,existing_fulltexts_added=rows,existing_read_notes_updated=['FPN12','FPN16'],
 canonical_library_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),checks=logs,third_party_fulltexts_not_for_publication=True)
(H/'library_update.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items()if k not in ['existing_fulltexts_added','checks']},ensure_ascii=False))
