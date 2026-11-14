"""Static source/data/hash audit. No candidate simulation is imported or run."""
from pathlib import Path
import ast,csv,hashlib,json,re,subprocess,zipfile
import numpy as np
from PIL import Image,ImageChops
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
ORIG=ROOT/'研究规划/第一篇小论文'
CODE=ROOT/'mypaper/非华理学报投稿件_归档/privacyandsaety/code'
OUT=ROOT/'mypaper/学报投稿/历史归档/根目录散落实验输出/simulation_outputs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rel=lambda p:str(p.relative_to(ROOT))
tex=ORIG/'secure distributed safety.tex';ot=tex.read_text(encoding='utf-8')
main=CODE/'simulation_revised_detection_metrics_complete_v2.py';ct=main.read_text(encoding='utf-8')
lines=ct.splitlines()
def anchors(path,terms):
 ls=path.read_text(encoding='utf-8').splitlines()
 return [dict(line=i,text=t)for i,t in enumerate(ls,1)if any(q in t for q in terms)]
terms=['MC_runs =','KLD_threshold =','A_stable =','watermark_lambda =','seed_val =','radius=',
 'G.add_edge','np.random.seed(42)','H[0, 0, i]','H[1, 1, i]','np.random.normal(0.0, np.sqrt(var)',
 'self.M[','self.M_inv[','transmitted = transmitted +','watermark.decrypt(',
 'for receiver, sender in Params.attack_links:', 'covariance_hat =','np.cov(',
 '+ 1e-6 *','kld_stats[link][k] =','isolated_links[link][k] =','H_stack =','R_block =',
 'gain = P_pred','innovation =','x_hat[:, k, receiver] =','for _mc in range',
 'attack_duration=Params.attack_duration','plot_rmse_with_defense(']
code_files={}
for p in CODE.glob('*.py'):
 text=p.read_text(encoding='utf-8');tree=ast.parse(text)
 code_files[p.name]=dict(path=rel(p),sha256=sha(p),bytes=p.stat().st_size,
  static_markers=anchors(p,['MC_runs =','mc_runs:','n_nodes:','KLD_threshold =','A_stable =',
  'H_stack =','gain = P_pred','K_gain =','q_std:','r_std:']))
figs=[]
for p in sorted(ORIG.glob('Fig*.pdf')):
 q=OUT/p.name
 figs.append(dict(original=rel(p),sha256=sha(p),bytes=p.stat().st_size,
  archived_output=rel(q)if q.exists()else None,
  archive_bytes=q.stat().st_size if q.exists()else None,
  exact_archived_copy=q.exists()and sha(p)==sha(q),
  source_run_identified=False))
# Equal length or a differing PDF hash alone does not establish different
# topology: metadata/object IDs can differ. Check the two Fig1 images as well.
tmp=H/'tmp';tmp.mkdir(exist_ok=True)
fig1paths=[ORIG/'Fig1_Topology_Clean.pdf',OUT/'Fig1_Topology_Clean.pdf']
ims=[]
for i,p in enumerate(fig1paths):
 prefix=tmp/f'fig1_source_{i}'
 subprocess.run(['pdftoppm','-f','1','-singlefile','-scale-to','1600','-png',str(p),str(prefix)],check=True)
 ims.append(Image.open(prefix.with_suffix('.png')).convert('RGB'))
fig1_raster=dict(scale_to_pixels=1600,same_size=ims[0].size==ims[1].size,
 identical_pixels=ims[0].size==ims[1].size and ImageChops.difference(ims[0],ims[1]).getbbox()is None,
 not_a_raw_graph_or_H_snapshot=True)
def csvrows(name):
 with (OUT/name).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))
t1=csvrows('Table1_No_Watermark_Per_Link_FPR_FNR.csv')
t2=csvrows('Table2_Watermark_Per_Link_FPR_FNR.csv')
t3=csvrows('Table3_Watermark_Mean_FPR_FNR.csv')
pat=r'\$(\d+)\\to\s*(\d+)\$\s*&\s*([\d.]+)\\%\s*&\s*([\d.]+)\\%\s*&\s*([\d.]+)\\%\s*&\s*([\d.]+)\\%'
table=[]
for match in re.finditer(pat,ot):
 sender,receiver=map(int,match.group(1,2));o=list(map(float,match.group(3,4,5,6)))
 a=next(r for r in t1 if int(r['sender_node'])==sender and int(r['receiver_node'])==receiver)
 b=next(r for r in t2 if int(r['sender_node'])==sender and int(r['receiver_node'])==receiver and float(r['watermark_variance'])==.5)
 c=[100*float(a['fpr_clean']),100*float(a['fnr_strictly_stealthy_attack']),100*float(b['fpr']),100*float(b['fnr'])]
 table.append(dict(sender=sender,receiver=receiver,original_percent=o,saved_csv_percent=c,
  saved_csv_rounded_2dp=[float(f'{x:.2f}')for x in c],
  matches_original_at_display_precision=[abs(x-y)<=.005000001 for x,y in zip(o,c)],
  original_tex_line=ot[:match.start()].count('\n')+1))
assert len(table)==6
npz=OUT/'Fig6_Optimal_Arbitrary_Attack_Sequences.npz'
headers=[]
with zipfile.ZipFile(npz)as z:
 for name in z.namelist():
  with z.open(name)as f:
   version=np.lib.format.read_magic(f)
   shape,fortran,dtype=np.lib.format._read_array_header(f,version)
  headers.append(dict(member=name,shape=list(shape),dtype=str(dtype),object_dtype=bool(dtype.hasobject)))
saved=[dict(path=rel(p),sha256=sha(p),bytes=p.stat().st_size)for p in sorted(OUT.iterdir())if p.is_file()]
metrics=CODE/'results/metrics.json';md=json.loads(metrics.read_text(encoding='utf-8'))
run_scope=dict(path=rel(metrics),sha256=sha(metrics),config=md.get('config'),
 distinct_from_original50=True,reason='n_nodes=5, mc_runs=100, different H/A/noise and privacy-envelope experiment')
# One immutable evidence snapshot includes candidate sources and every directly
# inspected artifact, beyond the earlier protected P/ snapshot.
input_hashes={rel(p):sha(p)for p in [tex,*CODE.glob('*.py'),metrics,*OUT.iterdir()]if p.is_file()}
out=dict(date='2026-11-14',baseline='c1dbba3',audit_type='independent static read; no import/execute candidate code',
 search_scope=['canonical preserved first paper','named archived privacyandsaety/code candidates',
 'historical root loose outputs/simulation_outputs; not every E-drive file'],
 canonical_tex=rel(tex),canonical_tex_sha256=sha(tex),candidate_complete_v2=rel(main),
 candidate_code_files=code_files,candidate_complete_v2_anchors=anchors(main,terms),
 original_tex_anchors=anchors(tex,['M=1000','q=50','\\theta=39','tab:detection_metrics','\\tag{3}','\\tag{4}','\\tag{11}','\\tag{12}']),
 figure_copy_checks=figs,fig1_raster_comparison=fig1_raster,cited_figures=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}',ot),
 table_comparison=table,table_cell_mismatch_count=sum(not b for row in table for b in row['matches_original_at_display_precision']),
 saved_output_inventory=saved,saved_table1=t1,saved_table2=t2,saved_table3=t3,
 saved_arbitrary_attack_npz=dict(path=rel(npz),sha256=sha(npz),headers=headers,not_autonomous_primary_run_evidence=True),
 neighboring_results_run=run_scope,
 MC1000_status='default code value and loop exist; env override allowed; no unique matching execution log/seed/raw per-run manifest establishes original figures',
 model_status='candidate complete_v2 and two older candidates use separately stacked gain, not common-G summed-neighbor estimator',
 stable_A_status='candidate .95,.02/.02,.97; canonical .8,.05/.01,.9',
 detector_status='candidate 4 residual plug-in Gaussian KL with regularized covariance, threshold42; canonical says39; not corrected nominal chi-square interface',
 plugin_covariance_rank_bound=dict(residual_dimension=8,number_residuals=4,max_unregularized_rank=3,regularization=1e-6),
 topology_H_status='candidate algorithm/seeds present; no final original-run H/topology snapshot tied to figure',
 synchronization_status='same in-memory dictionary(node,k), ideal access; not key/packet-loss synchronization implementation',
 isolation_status='instantaneous score gate only for six preconfigured links; stacked KF recomputed after exclusion; not original fixed common-G recursion',
 defense_duration_status='candidate Fig5 uses finite duration60; canonical isolation prose says through end; not evidence same final source',
 no_evidence_of_fabrication_claim=True,raw_data_never_existed_claim=False,
 original50_reproduction_closed=False,new5node_is_original50=False,
 new_network_or_noise_simulation=False,input_hashes=input_hashes)
(H/'reproduction_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(original_figure_copies=len(figs),byte_identical_copies=sum(x['exact_archived_copy']for x in figs),
 table_rows=len(table),mismatching_table_cells=out['table_cell_mismatch_count'],npz_members=len(headers),
 candidate_python_files=len(code_files),original50_reproduction_closed=False),ensure_ascii=False))
