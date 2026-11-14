"""Verify this gate's results and preservation; no original research scripts run."""
from pathlib import Path
from fractions import Fraction
import json,hashlib,ast,zipfile
H=Path(__file__).resolve().parent;P=H.parent;ROOT=H.parents[2]
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
snap=load(H/'input_snapshot.json');reg=load(H/'library_registration.json')
libkey='参考文献\\文献库.json'
for name,digest in snap.items():
 if name==libkey:
  assert reg['old_library_sha256']==digest and reg['new_library_sha256']==sha(ROOT/name)
 else:assert sha(ROOT/name)==digest,name
protected=load(P/'checks/修改前保护哈希.json')
for name,digest in protected.items():
 if name==libkey:continue
 assert sha(ROOT/name)==digest,name
archive=load(ROOT/'研究规划/原稿归档/第一篇小论文_初始原稿_20261026_清单.json')
with zipfile.ZipFile(ROOT/archive['archive'])as z:
 for r in archive['files']:
  assert sha(ROOT/r['source'])==r['sha256']
  assert hashlib.sha256(z.read(r['archive_member'])).hexdigest()==r['sha256']
stage=load(P/'checks/攻击补强交付核查.json')
assert sha(Path(stage['pdf']))==stage['pdf_sha256']
tex=[r for r in stage['artifacts']if r['file'].endswith('.tex')]
for r in tex:assert sha(P/r['file'])==r['sha256']
model=load(H/'five_node_aggregate_model.json');num=load(H/'variable_attack_envelopes.json')
np=load(H/'first_packet_limit.json');cert=load(H/'aggregate_gap_certificate.json')
assert num['input_sha256']==np['instance_sha256']==hashlib.sha256((H/'five_node_aggregate_model.json').read_text(encoding='utf-8-sig').replace('\r\n','\n').encode('utf-8')).hexdigest()
assert model['gain_regime'].startswith('steady-state fixed')
for key in ['unstable','stable']:
 m=num['models'][key]
 assert len(m['one_segment'])==11 and len(m['two_segment'])==55
 assert m['first_packet_detection']['unknown_start_event_detection_probability']is None
 assert m['first_packet_detection']['danger_before_alarm_probability']is None
for r in cert['rows']:
 assert Fraction(r['exact_DARE_witness_damage_lower'])>Fraction(1,10)
 assert Fraction(r['witness_total_energy'])<Fraction(3,2)
 assert Fraction(r['exact_F_norm_upper'])<1
 assert Fraction(r['saved_F_G_operator_error_upper'])<Fraction(1,10**10)
for p in H.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
lib=load(ROOT/'参考文献/文献库.json')
assert len(lib['records'])==176
assert sum(bool(r.get('file'))for r in lib['records'])==119
assert sum(r['kind']=='完整学位论文'for r in lib['records'])==46
for r in lib['records']:
 if r.get('file')and r.get('sha256'):assert sha(ROOT/'参考文献'/r['file'])==r['sha256'],r['id']
out=dict(date='2026-11-14',baseline_commit='fbabe2a0dca8200f5c2bff947f40c24728d19b8b',
    previous_frozen_files=len(snap),previous_files_unchanged_except_registered_library=len(snap)-1,
    old_original_and_archive_members_unchanged=len(archive['files']),stage_tex_unchanged=len(tex),
    stage_pdf_unchanged=True,formal_manuscript_changed=False,
    library_update='one FPN12 full PDF under existing ID, 2 new metadata-only records, 5 read-scope appends',
    library_counts=dict(records=176,local_PDF=119,complete_theses=46),
    original_estimator_structure_recovered=True,original_50_node_execution_reproduced=False,
    newly_declared_5_node_aggregate_instance=True,declared_target_DARE_certified=True,
    arbitrary_unknown_onset_closed_loop_risk_solved=False,
    physical_noise_samples=0,new_defense_algorithm=False,large_scale_experiments=False,
    task4_5_development_started=False,decision='B: conditional problem retention, not method approval',
    files=[dict(file=str(p.relative_to(H)),sha256=sha(p))for p in sorted(H.rglob('*'))
           if p.is_file()and '__pycache__'not in p.parts and p.name not in ['scope_check.json','publish_receipt.json']])
(H/'scope_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items()if k!='files'},ensure_ascii=False,indent=2))
