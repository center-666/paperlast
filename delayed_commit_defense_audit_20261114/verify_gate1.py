"""Scope-aware first-gate audit. No original scripts or simulations executed."""
from pathlib import Path
import ast,hashlib,json,re,zipfile
from urllib.parse import unquote
H=Path(__file__).resolve().parent;P=H.parent;ROOT=H.parents[2];LIB=ROOT/'参考文献'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=read(H/'initial_freeze.json');changed=[]
for n,h in f['prior_files'].items():
 p=ROOT/n
 if sha(p)!=h:changed.append(n)
assert changed==['参考文献\\文献库.json'] or changed==['参考文献/文献库.json'],changed
old=read(H/'library_before.json');new=read(LIB/'文献库.json');upd=read(H/'library_update.json')
assert len(old['records'])==176 and len(new['records'])==183
old_by={r['id']:r for r in old['records']};new_by={r['id']:r for r in new['records']}
assert set(new_by)-set(old_by)==set(upd['new_ids'])
allowed={'citation_notes','read_reports','related_reference_traces'}
for rid,r in old_by.items():
 nr=new_by[rid]
 if rid not in upd['existing_read_note_updates']:assert nr==r,rid
 else:
  assert {k:v for k,v in r.items()if k not in allowed}=={k:v for k,v in nr.items()if k not in allowed},rid
  for key in allowed:
   original=r.get(key,[]);assert nr.get(key,[])[:len(original)]==original,(rid,key)
assert sha(LIB/'文献库.json')==upd['canonical_library_sha256']
for r in new['records']:
 if r.get('file'):
  p=(LIB/r['file']).resolve();assert p.is_relative_to(LIB.resolve())and p.read_bytes().startswith(b'%PDF-')
  if r.get('sha256'):assert sha(p)==r['sha256'],r['id']
for r in upd['migrations']:
 assert sha(LIB/r['canonical_file'])==r['sha256']==sha(Path(r['temporary_source']))
assert sum(bool(r.get('file'))for r in new['records'])==122
assert sum(r['kind']=='完整学位论文'for r in new['records'])==46
for key in [new_by[rid]['citation_key']for rid in upd['new_ids']]:
 assert (LIB/'参考文献引用库.bib').read_text(encoding='utf-8').count('{'+key+',')==1,key
legacy=read(P/'checks/修改前保护哈希.json');legacy={n:h for n,h in legacy.items()if '文献库.json'not in n}
for n,h in legacy.items():assert sha(ROOT/n)==h,n
archive=read(ROOT/'研究规划/原稿归档/第一篇小论文_初始原稿_20261026_清单.json')
with zipfile.ZipFile(ROOT/archive['archive'])as z:
 for r in archive['files']:
  assert sha(ROOT/r['source'])==r['sha256']
  assert hashlib.sha256(z.read(r['archive_member'])).hexdigest()==r['sha256']
stage=read(P/'checks/攻击补强交付核查.json')
stagefiles=[r for r in stage['artifacts']if r['file'].endswith(('.tex','.pdf'))]
for r in stagefiles:assert sha(P/r['file'])==r['sha256']
checks=read(H/'equivalence_checks.json')
assert not any(checks[k]for k in ['gate2_entered','gate3_entered','gate4_entered','delay_scan_performed'])
assert sha(P/'original_estimator_gate_20261114/five_node_aggregate_model.json')==checks['model_sha256']
one=read(H/'literature_1_sources.json');two=read(H/'literature_2_sources.json')
assert one[0]['is_pdf']and not one[1]['is_pdf']and not one[2]['is_pdf']
assert two['Chen']['pages']==13 and all(not two[k]['pdf_obtained']for k in ['Bhattacharya','Zhang','Gupta'])
required=['01_直接近邻与创新性排重.md','候选与标准缓存_固定滞后方法的逐式等价性审查.md',
          '06_立项决策.md','evidence_table.md','README.md']
for n in required:assert(H/n).is_file()
assert not any(p.name.startswith(('02_','04_','05_'))for p in H.glob('*.md'))
for p in H.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
for p in H.glob('*.md'):
 for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
  if link.startswith(('https:','http:','#')):continue
  target=unquote(link.split('#')[0].strip('<>'))
  assert(p.parent/target).exists(),(p.name,target)
out=dict(baseline=f['baseline'],decision='C: stop current mechanism at gate1',
 task_sha256=f['instruction_sha256'],first_gate_deliverables=required,
 first_gate_conditional_stop_rule_satisfied=True,
 direct_neighbors_requested=7,direct_primary_fulltexts_obtained=2,direct_core_fulltexts_missing=5,
 all_seven_theorem_comparisons_complete=False,
 mathematical_claim='conditional same-policy pathwise equivalence; not impossibility for all delayed policies',
 identities=checks['identities'],new_network_samples=0,new_algorithms=0,delay_scan=False,
 later_gates='not entered by explicit gate1-C instruction; no placeholder deliverables',
 prior_paths_checked=len(f['prior_files']),prior_paths_unchanged_excluding_authorized_library=len(f['prior_files'])-1,
 legacy_sources_checked=len(legacy),archived_original_sources_and_zip_members=len(archive['files']),
 stage_TeX_PDF_checked=len(stagefiles),original_and_formal_sources_unchanged=True,
 library_before_records=176,library_after_records=183,library_after_local_pdf_records=122,
 new_PDF_records=3,independent_full_theses=46,old_library_metadata_and_files_preserved=True,
 source_library_only_allowed_append_fields=True,export_and_full_hash_check_passed=True,
 no_third_party_fulltexts_for_publication=True,
 artifacts={p.name:sha(p)for p in H.iterdir()if p.is_file()and p.suffix in ['.md','.py','.json']and p.name not in ['verification.json','publish_receipt.json','remote_readback.json','publication_manifest.json']})
(H/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items()if k!='artifacts'},ensure_ascii=False))
