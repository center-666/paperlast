"""Verify final positioning artifacts; never execute or write old scripts."""
from pathlib import Path
from fractions import Fraction as Q
import ast,hashlib,json,re,zipfile
from urllib.parse import unquote
import mpmath as mp
H=Path(__file__).resolve().parent;P=H.parent;ROOT=H.parents[2]
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fr=load(H/'frozen_inputs.json');bad=[]
for n,v in fr['protected_files'].items():
 p=ROOT/n
 if not p.exists()or sha(p)!=v:bad.append(n)
assert not bad,bad
legacy=load(P/'checks/修改前保护哈希.json');count=0
for n,v in legacy.items():
 if n.startswith('参考文献'):continue
 assert sha(ROOT/n)==v,n
 count+=1
arc=load(ROOT/'研究规划/原稿归档/第一篇小论文_初始原稿_20261026_清单.json')
with zipfile.ZipFile(ROOT/arc['archive'])as z:
 for f in arc['files']:
  assert hashlib.sha256(z.read(f['archive_member'])).hexdigest()==f['sha256']
  assert sha(ROOT/f['source'])==f['sha256']
stage=load(P/'checks/攻击补强交付核查.json');stage_count=0
for f in stage['artifacts']:
 if f['file'].endswith(('.tex','.pdf')):
  assert sha(P/f['file'])==f['sha256'];stage_count+=1
review=load(H/'reviewer_evidence.json');rows=review['rows']
expected={f'R{r}.{i}'for r,n in enumerate([4,3,6,6,5,5],1)for i in range(1,n+1)}
assert len(rows)==29 and{r['id']for r in rows}==expected
anchors=0
for r in rows:
 assert r['correct_result']and r['remaining_increment']
 if r['id']in ['R1.2','R2.2','R3.5','R5.3']:assert r['nearest_literature']
 assert not r['stage_tex_is_submission_ready']and not r['research_is_merged_into_tex']
 for a in r['original']+r['stage_tex']:
  lines=(ROOT/a['path']).read_text(encoding='utf-8-sig').splitlines()
  assert lines[a['line']-1].strip()==a['matched'].strip(),(r['id'],a)
  anchors+=1
for s in review['source_files'].values():assert sha(ROOT/s['path'])==s['sha256']
raw=Path(review['raw_review_source']['path']);assert sha(raw)==review['raw_review_source']['sha256']
assert review['classifications_nonexclusive']and review['no_closed_count_claim']
math=load(H/'resource_boundary_math.json');ref=load(H/'scaled_orthogonal_reference.json')
nu_lo=Q(math['rational_enclosure']['nu_lower']);nu_hi=Q(math['rational_enclosure']['nu_upper'])
k_lo=Q(math['optimal_peak']['certified_decimal_lower']);k_hi=Q(math['optimal_peak']['certified_decimal_upper'])
assert nu_lo<nu_hi and nu_hi-nu_lo<Q(127,10**41)
assert k_lo*k_lo*nu_hi<1<k_hi*k_hi*nu_lo
assert nu_lo>Q(634636,10**6)
mp.mp.dps=85
nu=mp.mpf(3)/4-mp.exp(mp.mpf(1)/2)*mp.e1(mp.mpf(1)/2)/8
val=lambda x:mp.mpf(x.numerator)/x.denominator
assert val(nu_lo)<nu<val(nu_hi)
assert abs(float(nu)-ref['nu'])<1e-14
assert abs(float(1/mp.sqrt(nu))-ref['kappa_star'])<1e-14
assert ref['no_parameter_search']and ref['no_direction_optimization']and ref['physical_noise_samples']==0
assert ref['new_QMC_nodes']==0
assert math['script_sha256']==sha(H/'resource_boundary_math.py')
syntax=[];links=[]
for p in H.glob('*.py'):
 ast.parse(p.read_text(encoding='utf-8-sig'),filename=p.name);syntax.append(p.name)
for p in H.glob('*.md'):
 t=p.read_text(encoding='utf-8');assert '\ufffd'not in t
 for target in re.findall(r'\]\(([^)]+)\)',t):
  if target.startswith(('http:','https:','#','mailto:')):continue
  target=unquote(target.split('#')[0]).strip('<>')
  assert (p.parent/target).exists(),(p.name,target)
  links.append([p.name,target])
for p in H.glob('*.json'):load(p)
out=dict(date='2026-11-14',passed=True,baseline_commit=fr['baseline_commit'],decision='C',
 protected_prior_files=len(fr['protected_files']),old_mismatches=bad,
 legacy_original_and_formal_nonlibrary=count,original_zip_and_sources=len(arc['files']),old_stage_TeX_PDF=stage_count,
 reviewer_rows=29,original_and_stage_line_anchors_verified=anchors,reviewer_classifications_not_exclusive=True,
 no_claim_all_reviewers_solved=True,core_novelty_issues_still_open=['R1.2','R2.2','R3.5','R5.3'],
 rational_nu_enclosure_and_kappa_bracket_verified=True,independent_special_function_crosscheck=True,
 rounded_reference_not_mixed=True,new_physical_noise_samples=0,new_parameter_search=0,new_algorithms=0,
 formal_manuscript_changed=False,unique_library_changed=False,
 python_AST_checked=syntax,local_markdown_links_checked=len(links),
 data_hashes={n:sha(H/n)for n in ['frozen_inputs.json','reviewer_evidence.json','resource_boundary_math.json',
 'scaled_orthogonal_reference.json']})
(H/'delivery_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:out[k]for k in ['passed','decision','protected_prior_files','reviewer_rows',
 'original_and_stage_line_anchors_verified','rational_nu_enclosure_and_kappa_bracket_verified',
 'local_markdown_links_checked']},ensure_ascii=False))
