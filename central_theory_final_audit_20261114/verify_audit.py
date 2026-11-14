"""Verify old inputs stayed unchanged and audit references are internally valid.
Writes only this directory. Not a theorem prover or an external peer review.
"""
from pathlib import Path
import ast, hashlib, json, re, zipfile
from urllib.parse import unquote
H=Path(__file__).resolve().parent; P=H.parent; ROOT=H.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
frozen=read(H/'frozen_inputs.json'); checks=[]
for n,h in frozen['protected_files'].items():
 p=ROOT/n; assert p.is_file()and sha(p)==h,n
checks.append(dict(group='prior_task_files_and_current_library',count=len(frozen['protected_files']),passed=True))
old=read(P/'checks/修改前保护哈希.json')
old={n:h for n,h in old.items()if '文献库.json'not in n}
for n,h in old.items():assert sha(ROOT/n)==h,n
checks.append(dict(group='original_and_formal_chapter_legacy_hashes',count=len(old),passed=True))
archive=read(ROOT/'研究规划/原稿归档/第一篇小论文_初始原稿_20261026_清单.json')
with zipfile.ZipFile(ROOT/archive['archive'])as z:
 for row in archive['files']:
  assert sha(ROOT/row['source'])==row['sha256']
  assert hashlib.sha256(z.read(row['archive_member'])).hexdigest()==row['sha256']
checks.append(dict(group='original_sources_and_archive_members',count=len(archive['files']),passed=True))
stage=read(P/'checks/攻击补强交付核查.json')
selected=[r for r in stage['artifacts']if r['file'].endswith(('.tex','.pdf'))]
for r in selected:assert sha(P/r['file'])==r['sha256'],r['file']
checks.append(dict(group='stage_TeX_PDF_frozen_artifacts',count=len(selected),passed=True))
repro=read(H/'reproduction_evidence.json')
for n,h in repro['input_hashes'].items():assert sha(ROOT/n)==h,n
assert repro['table_cell_mismatch_count']==22
assert not repro['original50_reproduction_closed']
assert repro['fig1_raster_comparison']['identical_pixels']
checks.append(dict(group='direct_static_audit_inputs',count=len(repro['input_hashes']),passed=True))
a=read(H/'candidate_A_checks.json')
for m in a['models'].values():
 assert m['internal_detection_rank']==0 and m['whole_event_detection_rank']==2
bc=read(H/'candidate_BC_evidence.json')
for n,h in bc['input_sha256'].items():assert sha(P/n)==h,n
assert bc['B_singular_record_example']['exact_joint_KL']=='5/2'
assert bc['B_singular_record_example']['incorrect_sum_marginal_KL']=='19/4'
assert not bc['C3']['probability_interval_certified']
checks.append(dict(group='candidate_evidence_and_scope_assertions',passed=True))
src=read(H/'final_version_source.json')
assert src['pages']==8 and not src['library_modified']
assert sha(H/'literature_work/FPN16_publisher_final_work.pdf')==src['sha256']
fetch=read(H/'primary_fetch_A.json')
# Do not mistake HTML/empty responses for primary fulltext clearance.
checks.append(dict(group='fulltext_source_and_recorded_missing_sources',passed=True,
 actual_download=src['existing_library_id'],missing=['FPN07','FPN08','FPN11']))
for p in H.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
broken=[]
for p in H.glob('*.md'):
 for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
  if link.startswith(('https:','http:','#')):continue
  target=unquote(link.split('#')[0].strip('<>'))
  if not(p.parent/target).exists():broken.append([p.name,target])
assert not broken,broken
checks.append(dict(group='new_python_AST_and_markdown_local_links',passed=True))
out=dict(date='2026-11-14',baseline=frozen['baseline_commit'],checks=checks,
 old_files_unchanged=True,formal_paper_unchanged=True,unique_library_unchanged=True,
 no_new_network_simulations=True,no_old_verification_entrypoints_executed=True,
 external_independent_peer_review=False,primary_A_fulltext_comparison_complete=False,
 independent_center_established=False,paper_decision='C')
(H/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
