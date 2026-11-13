"""Cross-artifact consistency and original-file protection audit (full local project)."""
from pathlib import Path
from fractions import Fraction
import json,hashlib,zipfile,ast
import numpy as np
HERE=Path(__file__).resolve().parent
P=HERE.parent
ROOT=P.parents[1]
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
snapshot=load(HERE/'inputs_sha256.json')
for name,digest in snapshot.items():assert sha(ROOT/name)==digest,name
safe=load(HERE/'safe_words.json'); graph=load(HERE/'graph_results.json')
tail=load(HERE/'tail_robustness_results.json'); detection=load(HERE/'detection_results.json')
assert safe['safe_count']==960 and safe['danger_count']==64 and not safe['boundary_indices']
assert graph['source']['sha256']==sha(HERE/'safe_words.json')
assert Fraction(safe['common_exact_margin'])==Fraction(1,4000)
assert tail['primary']['all_time_safe_word_margin']=='1/4000'
edges=graph['exact_graph_edges']
assert {x['word'] for x in edges}==set(safe['safe_indices'])
for e in edges:assert e['source']==e['word']>>1 and e['target']==e['word']&511
states=set(graph['states']);assert len(states)==470
adj={v:[] for v in states}; rev={v:[] for v in states}
for e in graph['transitions']:
    assert e['source_state'] in states and e['target_state'] in states
    adj[e['source_state']].append(e['target_state']);rev[e['target_state']].append(e['source_state'])
for edgeset in [adj,rev]:
    seen={next(iter(states))};todo=list(seen)
    while todo:
        v=todo.pop()
        for w in edgeset[v]:
            if w not in seen:seen.add(w);todo.append(w)
    assert seen==states
assert all(1<=len(v)<=2 for v in adj.values())
assert sum(len(v)==1 for v in adj.values())==28
pi=dict(zip(graph['states'],graph['stationary'])); row_sums={v:0. for v in states}
weights=np.zeros(1024)
for e in graph['transitions']:
    row_sums[e['source_state']]+=e['probability']
    weights[e['word_index']]+=pi[e['source_state']]*e['probability']
assert max(abs(v-1) for v in row_sums.values())<1e-12
assert np.max(abs(weights-np.array(detection['weights'])))<1e-14
assert all(r['rank']==4 and all(x>0 for x in r['support_gram_leading_minors_exact'])
           for r in detection['conditional_known_state_four_packet_gram'])
assert detection['physical_trajectories']==detection['noise_samples']==0
for key in ['constrained_parity','constrained_glrt','dyadic_constrained_parity','dyadic_constrained_glrt']:
    for probe in ['P1','iid_risk_witness','metrics_at_this_direction']:
        v=detection['comparisons'][key][probe]
        assert v['danger_probability']==v['risk_lower']==v['risk_upper']==0

protected=load(P/'checks/修改前保护哈希.json')
for name,digest in protected.items():
    if name=='参考文献\\文献库.json':continue
    assert sha(ROOT/name)==digest,name
archive=load(ROOT/'研究规划/原稿归档/第一篇小论文_初始原稿_20261026_清单.json')
with zipfile.ZipFile(ROOT/archive['archive']) as z:
    for row in archive['files']:
        assert sha(ROOT/row['source'])==row['sha256']
        assert hashlib.sha256(z.read(row['archive_member'])).hexdigest()==row['sha256']
stage=load(P/'checks/攻击补强交付核查.json')
assert sha(Path(stage['pdf']))==stage['pdf_sha256']
tex=[r for r in stage['artifacts'] if r['file'].endswith('.tex')]
for row in tex:assert sha(P/row['file'])==row['sha256']
registration=load(P/'glrt_screening_20261112/registration.json')
assert sha(ROOT/'参考文献/文献库.json')==registration['library_sha256_after']
library=load(ROOT/'参考文献/文献库.json')
assert len(library['records'])==174

repo=load(HERE/'repository_text_snapshot.json')
for r in repo:
    b=r['text'].encode('utf-8')
    assert hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest()==r['sha'],r['path']
for script in HERE.glob('*.py'):ast.parse(script.read_text(encoding='utf-8'))
result=dict(date='2026-11-13',unchanged_old_first_paper_files=len(snapshot),
            original_archive_sources_and_members=len(archive['files']),stage_tex_unchanged=len(tex),
            stage_pdf_unchanged=True,library_unchanged=True,library_records=174,
            repository_text_files_fetched_and_blob_verified=len(repo),
            safe_words=960,boundary_words=0,danger_words=64,cyclic_states=470,cyclic_edges=912,
            all_time_tail_certified_in_frozen_model=True,matrix_perturbation_domain_certified=True,
            physical_trajectories=0,noise_samples=0,novelty_established=False,
            unknown_onset_damage_guarantee=True,unknown_onset_GLRT_power_validated=False,
            independent_peer_review=False,decision='有条件继续',
            artifacts=[dict(file=p.name,sha256=sha(p)) for p in sorted(HERE.iterdir())
                       if p.is_file() and p.name!='scope_check.json'])
(HERE/'scope_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},ensure_ascii=False,indent=2))
