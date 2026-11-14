"""Read-only preservation and consistency checks for this screening package."""
from pathlib import Path
from fractions import Fraction
import json,hashlib,ast,zipfile
H=Path(__file__).resolve().parent;P=H.parent;ROOT=H.parents[2]
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
scope=load(H/'frozen_scope.json')
for name,digest in scope['protected_files'].items():assert sha(ROOT/name)==digest,name
for name,digest in load(P/'checks/修改前保护哈希.json').items():
 if name=='参考文献\\文献库.json':continue
 assert sha(ROOT/name)==digest,name
archive=load(ROOT/'研究规划/原稿归档/第一篇小论文_初始原稿_20261026_清单.json')
with zipfile.ZipFile(ROOT/archive['archive'])as z:
 for r in archive['files']:
  assert sha(ROOT/r['source'])==r['sha256']
  assert hashlib.sha256(z.read(r['archive_member'])).hexdigest()==r['sha256']
stage=load(P/'checks/攻击补强交付核查.json');assert sha(Path(stage['pdf']))==stage['pdf_sha256']
tex=[r for r in stage['artifacts']if r['file'].endswith('.tex')]
for r in tex:assert sha(P/r['file'])==r['sha256']
theory=load(H/'rho_theory_check.json')
assert Fraction(theory['uniform_energy_gain']['all_time_squared_bias_upper'])<Fraction(1,10)
assert scope['E']==10 and scope['D']==.1 and scope['normal_MSE_relative_budgets']==[.01,.05,.1,.2]
normal=load(H/'rho_normal_screen.json');cl=load(H/'clip_baseline_results.json')
net=load(H/'clip_risk_net.json');candidate=load(H/'candidate_first_mc.json')
cfg=load(H/'clip_risk_net_config.json');basecfg=load(H/'clip_baseline_config.json')
assert cfg['net_validation_seed']!=basecfg['validation_seed']
assert cfg['net_watermark_seed']!=basecfg['watermark_seed']
assert cfg['candidate_seed']not in [cfg['net_validation_seed'],cfg['net_watermark_seed'],basecfg['validation_seed']]
for n in net['rows']:
 c=next(r for r in candidate['rows']if r['model']==n['model'])
 assert n['fresh_seeds_after_initial_grid_screen']
 assert n['continuous_L1_risk_upper_qualified']<c['L1_worst_risk_lower_qualified']
 assert not candidate.get('full_variable_duration_worst_risk_computed',False)
for name,m in normal['models'].items():
 assert [r['eta']for r in m['rows']]==scope['normal_MSE_relative_budgets']
 for r in m['rows']:
  assert abs(r['fixed_trace']-r['target_trace'])<1e-10
  assert abs(r['aware_trace']-r['target_trace'])<1e-10
for p in H.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
for p in H.glob('*.md'):
 import re
 for href in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
  if not href.startswith(('https:','http:','#')):
   assert (p.parent/href.split('#')[0].strip('<>')).exists(),(p.name,href)
out=dict(date='2026-11-14',baseline_commit=scope['baseline_commit'],
    old_protected_files_unchanged=len(scope['protected_files']),
    original_and_archive_files_unchanged=len(archive['files']),stage_tex_unchanged=len(tex),stage_pdf_unchanged=True,
    original_manuscript_changed=False,reference_library_unchanged=True,
    watermarked_aggregate_estimator_preserved=True,original_50_node_reproduction=False,
    E=10,D=.1,eta_budgets=scope['normal_MSE_relative_budgets'],
    candidate_distribution_changed=False,MAC_or_scheduling_or_RL_added=False,
    normal_linear_moments='deterministic quadrature and linear covariance equations',
    ordinary_clip_Monte_Carlo=True,
    independent_initial_noise_paths=basecfg['training_paths']+basecfg['validation_paths'],
    independent_confirmatory_noise_paths=cfg['net_validation_paths'],
    physical_noise_scalar_variates=cl['noise_samples_generated']+cfg['net_validation_paths']*(basecfg['burn_steps']+basecfg['evaluation_steps'])*8,
    first_packet_grid_samples_fresh=True,full_variable_duration_risk_solved=False,
    normal_clip_MSE_exact_budget_certified=False,large_50_node_experiment=False,
    decision='C: stop this rho candidate; innovation not validated',
    files=[dict(file=p.name,sha256=sha(p))for p in sorted(H.iterdir())
           if p.is_file()and p.name not in ['scope_check.json','publish_receipt.json']])
(H/'scope_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items()if k!='files'},ensure_ascii=False,indent=2))
