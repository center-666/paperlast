"""Audit consistency and protected artifacts. Not a substitute for proofs."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,zipfile,ast
HERE=Path(__file__).resolve().parent;P=HERE.parent;ROOT=P.parents[1]
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
snap=load(HERE/'inputs_sha256.json')
for name,digest in snap.items():assert sha(ROOT/name)==digest,name
ind=load(HERE/'independent_cert.json');ce=load(HERE/'counterexamples_exact.json')
assert ind['classification_input_old_results'] is False
assert ind['old_scripts_run']==0
assert [ind['safe_count'],ind['boundary_count'],ind['danger_count']]==[960,0,64]
assert ind['word_time_checks']==13312
assert Fraction(ind['uniform_all_time_damage_strict_upper_using_Wmax13'])<Fraction(1,10)
for key in ['residual','repeated','duration_9','duration_11','small_perturbation','piecewise_autonomous']:
    assert Fraction(ce[key]['damage'])>Fraction(1,10),key
    e=ce[key].get('energy',ce[key].get('energy_per_event'))
    assert Fraction(e)<10,key
assert Fraction(ce['nonzero_residual_and_small_waveform_robust_set']['safe_damage_upper'])<Fraction(1,10)
pub=load(HERE/'public_period_audit.json')
assert pub['exact_public_alternation']['global_minimax_pathwise_damage_attained_conditional_on_tail_power_certificate']
assert pub['total_enumerated_period_words']==2046
assert pub['total_legal_period_words']==1626
physical=load(HERE/'physical_origin_audit.json')
for key in ['exact_F_operator_total_error_upper','exact_g_Euclidean_total_error_upper']:
    assert Fraction(physical['reconstruction'][key])<Fraction(1,10**12)
assert physical['mathematical_parameter_model_certified']
assert not physical['original_paper_full_model_certified']
bridge=load(HERE/'bridge_margin_check.json')
assert bridge['certified_for_exact_declared_stacked_KF']
assert Fraction(bridge['remaining_Q_margin_lower'])>Fraction(1,5000)
assert Fraction(bridge['all_tail_damage_upper'])<Fraction(1,10)
assert not bridge['engineering_parameters_certified']
dp=load(HERE/'detection_privacy_boundary.json')
if 'fresh_safe_indices' in dp:assert set(dp['fresh_safe_indices'])==set(ind['safe_indices'])
protected=load(P/'checks/修改前保护哈希.json')
for name,digest in protected.items():
    if name=='参考文献\\文献库.json':continue
    assert sha(ROOT/name)==digest,name
archive=load(ROOT/'研究规划/原稿归档/第一篇小论文_初始原稿_20261026_清单.json')
with zipfile.ZipFile(ROOT/archive['archive']) as z:
    for r in archive['files']:
        assert sha(ROOT/r['source'])==r['sha256']
        assert hashlib.sha256(z.read(r['archive_member'])).hexdigest()==r['sha256']
stage=load(P/'checks/攻击补强交付核查.json')
assert sha(Path(stage['pdf']))==stage['pdf_sha256']
tex=[r for r in stage['artifacts'] if r['file'].endswith('.tex')]
for row in tex:assert sha(P/row['file'])==row['sha256']
reg=load(P/'glrt_screening_20261112/registration.json')
assert sha(ROOT/'参考文献/文献库.json')==reg['library_sha256_after']
for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
result=dict(date='2026-11-13',prior_first_paper_files_unchanged=len(snap),
    archive_sources_and_members_unchanged=len(archive['files']),stage_tex_unchanged=len(tex),
    stage_pdf_unchanged=True,library_unchanged=True,
    independent_implementation='SymPy powers/closed convolution; raw inputs, not old word classes',
    original_scripts_run=0,independent_safe_count=960,independent_danger_count=64,
    exact_violating_witnesses=6,original_DARE_architecture_certified=False,
    declared_stacked_KF_DARE_roundoff_certified=True,
    DARE_roundoff_to_safety_independent_bridge_certified=True,
    public_alternation_global_pathwise_minimax_certified=True,
    unknown_onset_GLRT_performance_verified=False,
    physical_trajectories=0,noise_samples=0,independent_external_peer_review=False,
    mathematical_axis='附加条件下通过',publication_axis='应停止独立方法主张',
    artifacts=[dict(file=p.name,sha256=sha(p)) for p in sorted(HERE.iterdir())
               if p.is_file() and p.name!='scope_check.json'])
(HERE/'scope_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items()if k!='artifacts'},ensure_ascii=False,indent=2))
