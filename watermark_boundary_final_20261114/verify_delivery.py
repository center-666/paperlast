"""Read-only old-input/protected-file checks. Writes only this directory.

Requires the full local thesis workspace, not the compact public review tree.
Does not invoke old verification scripts or regenerate any old certificate.
"""
from pathlib import Path
import ast, hashlib, json, re, zipfile
from urllib.parse import unquote

H=Path(__file__).resolve().parent
P=H.parent
ROOT=H.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
scope=load(H/'frozen_scope.json')
bad=[]
for n,v in scope['protected_files'].items():
 p=ROOT/n
 if not p.is_file()or sha(p.read_bytes())!=v:bad.append(n)
assert not bad,bad
old=load(P/'checks/修改前保护哈希.json')
legacy=[]
for n,v in old.items():
 if n.startswith('参考文献'):continue # Current library hash is frozen above.
 assert sha((ROOT/n).read_bytes())==v,n
 legacy.append(n)
manifest=load(ROOT/'研究规划/原稿归档/第一篇小论文_初始原稿_20261026_清单.json')
with zipfile.ZipFile(ROOT/manifest['archive'])as z:
 for f in manifest['files']:
  assert sha(z.read(f['archive_member']))==f['sha256'],f
  assert sha((ROOT/f['source']).read_bytes())==f['sha256'],f
stage=load(P/'checks/攻击补强交付核查.json')
stagefiles=[]
for f in stage['artifacts']:
 if f['file'].endswith('.tex')or f['file'].endswith('.pdf'):
  assert sha((P/f['file']).read_bytes())==f['sha256'],f
  stagefiles.append(f['file'])
syntax=[]
for p in H.glob('*.py'):
 ast.parse(p.read_text(encoding='utf-8-sig'),filename=p.name);syntax.append(p.name)
links=[]
for p in H.glob('*.md'):
 text=p.read_text(encoding='utf-8')
 assert '\ufffd'not in text,p.name
 for target in re.findall(r'\]\(([^)]+)\)',text):
  if target.startswith(('http:','https:','#','mailto:')):continue
  target=unquote(target.split('#')[0]).strip('<>')
  assert (p.parent/target).exists(),(p.name,target)
  links.append([p.name,target])
power=load(H/'baseline_comparison.json')
events=load(H/'event_risk_screen.json')
early=load(H/'unitary_early_boundary.json')
assert early['risk_lower_rational']=='49/50'
assert early['actual_energy_float']<10
assert early['conditional_miss_lower_float']>.98
assert power['estimator_unchanged']and power['normal_fusion_unchanged']
assert not events['global_worst_risk_solved']
assert all(m['common_corpus_count']==11 for m in events['models'].values())
assert all(m['no_noise_window_independence_assumed']for m in events['models'].values())
assert load(H/'event_risk_config.json')['validation_paths']==4096
result=dict(date='2026-11-14',passed=True,baseline_commit=scope['baseline_commit'],
 protected_previous_files=len(scope['protected_files']),protected_previous_mismatches=bad,
 legacy_nonlibrary_protected=len(legacy),original_sources_and_zip_members=len(manifest['files']),
 old_stage_TeX_PDF=len(stagefiles),python_ast_files=syntax,local_markdown_links_checked=len(links),
 unique_library_current_hash_unchanged=True,new_defense_algorithm=False,
 formal_manuscript_modified=False,new_large_physical_simulation=False,
 joint_residual_noise_watermark_MC_disclosed=True,finite_corpus_not_global_risk_certificate=True,
 input_sha256={n:sha((H/n).read_bytes())for n in ['frozen_scope.json','baseline_comparison.json',
  'event_risk_config.json','event_risk_selection_locked.json','event_risk_screen.json',
  'event_risk_p1p3.json','unitary_early_boundary.json','power_comparison_checks.json',
  'moment_implementation_crosscheck.json','probability_nonimplication.json']})
(H/'delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k]for k in ['passed','protected_previous_files','legacy_nonlibrary_protected',
 'original_sources_and_zip_members','old_stage_TeX_PDF','local_markdown_links_checked']},ensure_ascii=False))
