"""Read old files, validate new delivery; write ONLY this working directory."""
from pathlib import Path
from fractions import Fraction
import ast,hashlib,json,re,zipfile
from urllib.parse import unquote
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent;ROOT=H.parents[2]
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frozen=load(H/'frozen_inputs.json');bad=[]
for n,v in frozen['protected_files'].items():
 p=ROOT/n
 if not p.exists()or sha(p)!=v:bad.append(n)
assert not bad,bad
legacy=load(P/'checks/修改前保护哈希.json');legacy_count=0
for n,v in legacy.items():
 if n.startswith('参考文献'):continue
 assert sha(ROOT/n)==v,n
 legacy_count+=1
archive=load(ROOT/'研究规划/原稿归档/第一篇小论文_初始原稿_20261026_清单.json')
with zipfile.ZipFile(ROOT/archive['archive'])as z:
 for f in archive['files']:
  assert hashlib.sha256(z.read(f['archive_member'])).hexdigest()==f['sha256']
  assert sha(ROOT/f['source'])==f['sha256']
stage=load(P/'checks/攻击补强交付核查.json');stages=[]
for f in stage['artifacts']:
 if f['file'].endswith(('.tex','.pdf')):
  assert sha(P/f['file'])==f['sha256'];stages.append(f['file'])
syntax=[];links=[]
for p in H.glob('*.py'):
 ast.parse(p.read_text(encoding='utf-8-sig'),filename=p.name);syntax.append(p.name)
for p in H.glob('*.md'):
 t=p.read_text(encoding='utf-8');assert '\ufffd'not in t,p.name
 for target in re.findall(r'\]\(([^)]+)\)',t):
  if target.startswith(('http:','https:','#','mailto:')):continue
  target=unquote(target.split('#')[0]).strip('<>')
  assert (p.parent/target).exists(),(p.name,target)
  links.append([p.name,target])
det=load(H/'danger_detection_comparison.json');geo=load(H/'danger_geometry.json')
sp=load(H/'spatial_baseline.json');early=load(H/'first_packet_detection_bound.json')
first=load(H/'first_alarm_bounds.json');diff={}
for name in ['unstable','stable']:
 a=geo['models'][name]['links']['0->1'];b=det['models'][name]
 S=np.array(a['S_by_bias_state']);Sb=np.array(b['no_watermark_prefix_moment_matrices'])
 diff[name]=float(np.max(np.abs(S-Sb)));assert diff[name]<1e-12
 c=geo['models'][name]['main_link_rational_certificate']
 assert c['shared_peak_state']==(10 if name=='unstable'else 1)
 assert all(Fraction(x)>0 for x in c['all_energy_boundary_directions_strictly_dangerous_leading_minors'])
 V=np.array(sp['models'][name]['V_spatial']);assert np.allclose(V,np.diag([221/242,61/72]),atol=1e-15)
 for row in sp['models'][name]['deterministic_comparisons'].values():
  assert abs(row['common_danger_min']['Pd']-row['closed_form_common_danger_check']['Pd'])<1e-10
 for scheme,levels in first['models'][name]['schemes'].items():
  for level in levels:
   for probe,v in level['probes'].items():
    lo=np.array(v['parity_first_alarm_CDF_lower']);hi=np.array(v['parity_first_alarm_CDF_upper'])
    assert np.all(lo<=hi+1e-12)and np.all(np.diff(lo)>=-1e-12)and np.all(np.diff(hi)>=-1e-12)
    assert v['parity_danger_before_alarm_lower']<=v['parity_danger_before_alarm_upper']+1e-12
    assert v['parity_no_independence_assumption']and v['spatial_probability_integrated_after_conditional_product']
assert early['strict_risk_lower']=='12/25'and early['energy_float']<10
assert det['physical_noise_samples']==0 and first['physical_noise_samples']==0
for p in H.glob('*.json'):load(p)
out=dict(date='2026-11-14',passed=True,baseline_commit=frozen['baseline_commit'],
 protected_prior_files=len(frozen['protected_files']),mismatches=bad,
 legacy_original_and_formal_nonlibrary=legacy_count,original_zip_and_sources=len(archive['files']),
 old_stage_TeX_PDF=len(stages),python_AST_checked=syntax,local_links_checked=len(links),
 independent_geometry_moment_maxdiff=diff,spatial_closed_form_crosscheck=True,
 first_alarm_bound_invariants=True,strict_oracle_bound='risk>12/25 under declared assumptions',
 physical_noise_samples=0,new_network_trajectories=0,new_watermark_algorithms=0,new_attack_mechanisms=0,
 manuscript_changed=False,unique_library_changed=False,
 figure_visual_QA=dict(viewed=['danger_geometry_main.png','detection_damage_map.png'],
  figures_are_owned_scientific_plots=True,observed_clipping_or_missing_labels=False),
 C_decision_scope='current exogenous unit-resource dangerous-direction watermark route; not all kappa/detectors no-go',
 certificate_hashes={n:sha(H/n)for n in ['frozen_inputs.json','danger_geometry.json','danger_detection_comparison.json',
 'danger_detection_checks.json','spatial_baseline.json','bounds_review.json','first_packet_detection_bound.json',
 'first_alarm_bounds.json','comparison_summary.json']})
(H/'delivery_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:out[k]for k in ['passed','protected_prior_files','legacy_original_and_formal_nonlibrary',
 'original_zip_and_sources','old_stage_TeX_PDF','local_links_checked','independent_geometry_moment_maxdiff']},ensure_ascii=False))
