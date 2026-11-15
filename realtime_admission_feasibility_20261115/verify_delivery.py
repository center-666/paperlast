from pathlib import Path
import json,hashlib,ast,re,zipfile
from urllib.parse import unquote
H=Path(__file__).resolve().parent;P=H.parent;ROOT=H.parents[2];LIB=ROOT/'参考文献'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=read(H/'scope_before_results.json');changed=[]
for n,h in f['prior_file_hashes'].items():
 if sha(ROOT/n)!=h:changed.append(n)
assert len(changed)==1 and changed[0].replace('\\','/')=='参考文献/文献库.json',changed
old=read(H/'library_before.json');new=read(LIB/'文献库.json');oldby={r['id']:r for r in old['records']};newby={r['id']:r for r in new['records']}
assert oldby.keys()==newby.keys()and len(newby)==183
fields={'file','workspace_file','pages','bytes','sha256','pdf_url','reading_status','read_scope','read_reports','citation_notes'}
for rid,r in oldby.items():
 nr=newby[rid]
 if rid in ['FPN18','FPN19']:
  assert {k:v for k,v in r.items()if k not in fields}=={k:v for k,v in nr.items()if k not in fields},rid
  assert nr['read_scope'].startswith(r['read_scope'])
 elif rid in ['FPN12','FPN16']:
  ex={'citation_notes','read_reports'}
  assert {k:v for k,v in r.items()if k not in ex}=={k:v for k,v in nr.items()if k not in ex},rid
 else:assert r==nr,rid
 if rid in ['FPN18','FPN19','FPN12','FPN16']:
  for k in ['citation_notes','read_reports']:assert nr.get(k,[])[:len(r.get(k,[]))]==r.get(k,[])
for r in new['records']:
 if r.get('file')and r.get('sha256'):assert sha(LIB/r['file'])==r['sha256'],r['id']
lu=read(H/'library_update.json');assert sha(LIB/'文献库.json')==lu['canonical_library_sha256']
assert sum(bool(r.get('file'))for r in new['records'])==124
legacy=read(P/'checks/修改前保护哈希.json');legacy={n:h for n,h in legacy.items()if '文献库.json'not in n}
for n,h in legacy.items():assert sha(ROOT/n)==h,n
archive=read(ROOT/'研究规划/原稿归档/第一篇小论文_初始原稿_20261026_清单.json')
with zipfile.ZipFile(ROOT/archive['archive'])as z:
 for r in archive['files']:
  assert sha(ROOT/r['source'])==r['sha256']
  assert hashlib.sha256(z.read(r['archive_member'])).hexdigest()==r['sha256']
stage=read(P/'checks/攻击补强交付核查.json');st=[r for r in stage['artifacts']if r['file'].endswith(('.tex','.pdf'))]
for r in st:assert sha(P/r['file'])==r['sha256']
orig50=read(P/'central_theory_final_audit_20261114/reproduction_evidence.json')
for n,h in orig50['input_hashes'].items():assert sha(ROOT/n)==h,n
for n in ['fixed_gain_numbers.json','gate_normal_moments.json','admission_limits.json']:
 r=read(H/n);assert r.get('source_sha256',r.get('model_sha256'))==f['model_sha256']
a=read(H/'admission_limits.json')
assert a['binary_target_only']['healthy_neighbor2_always_accepted']and a['ordinary_clip']['gaussian_all_radii_excluded']is False
assert a['spatial_gate']['Au_gaussian_failure_at_1percent_not_proven']
assert all(v['risk_upper_zero_by_strict_norm_bound']for v in a['clip_zero_risk_example'].values())
required=['01_安全目标与真实权限.md','02_拒包闭环与稳定性.md','03_强基线与真实闭环比较.md',
 '04_近邻理论覆盖矩阵.md','05_非平凡权衡_必要界与适用限制.md','06_立项决定_C.md','README.md','evidence_table.md']
for n in required:assert(H/n).is_file()
for p in H.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
for p in H.glob('*.md'):
 for v in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
  if v.startswith(('http:','https:','#')):continue
  assert(p.parent/unquote(v.split('#')[0].strip('<>'))).exists(),(p.name,v)
out=dict(date='2026-11-15',baseline=f['baseline_commit'],decision='C: no new real-time admission mechanism project',
 predeclared_risk=.01,normal_allowance=.2,L=10,E=10,D=.1,new_physical_samples=0,new_algorithms=0,
 prior_files_checked=len(f['prior_file_hashes']),prior_files_unchanged_excluding_library=len(f['prior_file_hashes'])-1,
 legacy_protected_sources=len(legacy),archive_sources_and_members=len(archive['files']),stage_TeX_PDF=len(st),
 old50_source_and_outputs=len(orig50['input_hashes']),protected_files_unchanged=True,
 library_records=183,library_PDF_records=124,new_fulltexts=2,new_records=0,
 all_admission_policies_infeasible_proven=False,all_saturation_baselines_excluded=False,
 no_global_probability_optimum_claim=True,core_missing=['FPN31','FPN10','FPN32'],
 required_reports=required,manuscript_or_thesis_modified=False,
 own_artifacts={p.name:sha(p)for p in H.iterdir()if p.is_file()and p.suffix in ['.md','.json','.py']and p.name not in ['verification.json','publish_receipt.json','publication_manifest.json','remote_readback.json']})
(H/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items()if k!='own_artifacts'},ensure_ascii=False))
