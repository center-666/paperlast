"""Independent read-only historical source/CSV/version audit.

Parses source as text/AST; never imports or runs any old simulation entrypoint.
The only arithmetic is table aggregation, quantization compatibility and hashes.
"""
from pathlib import Path
import ast,csv,json,hashlib,re,collections
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,data):
    (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n','utf8')
def rel(p):return str(p.relative_to(ROOT))
def records(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    before={}
    code=[]
    roots=[ROOT/'mypaper',ROOT/'仓库论文资料/mypaper']
    keep={'simulation_arbitrary_attack_solver.py','simulation_revised_detection_metrics_direction_fixed.py','simulation_revised_detection_metrics_complete_v2.py','run_experiments.py','run_journal_experiments.py','run_journal_experiments_第十六轮.py','run_journal_experiments_第十七轮.py'}
    pats=['MC_runs =','mc_runs:','n_nodes:','sensorNum =','timeSize =','KLD_threshold =','KLD_window =','watermark_lambda =','watermark_vars_test =','attack_start =','attack_duration =','impact_attack_duration =','A_stable =','A_unstable =','H_stack =','K_gain =','gain = P_pred','np.vstack','Theta =','pinv(Theta)','covariance_hat =','KLD_window - 1','attack_eval_start =','clean_eval_start =','clean_eval_end =','attack_eval_end =','SIM_OUTPUT_DIR','"Table1','"Table2','"Table3','"Detection_Metrics','0.96, 0.14','0.95, 0.02','0.8, 0.05']
    for root in roots:
        for p in root.rglob('*.py'):
            if 'skills' in p.parts or p.name not in keep:continue
            before[str(p)]=sha(p);tx=p.read_text('utf-8-sig');ln=tx.splitlines();tree=ast.parse(tx)
            params={}
            for cls in [n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name in {'Params','SimConfig'}]:
                for n in cls.body:
                    if isinstance(n,ast.Assign):
                        for target in n.targets:
                            if isinstance(target,ast.Name):params[target.id]={'line':n.lineno,'expression':ast.unparse(n.value)}
                    if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name):
                        params[n.target.id]={'line':n.lineno,'expression':ast.unparse(n.value) if n.value else ''}
            anchors=[{'line':i+1,'text':s.strip()} for i,s in enumerate(ln) if any(t in s for t in pats)]
            code.append({'file':rel(p),'sha256':sha(p),'bytes':p.stat().st_size,'lines':len(ln),'params':params,'anchors':anchors,'AST_only':True})
    groups=collections.defaultdict(list)
    for r in code:groups[r['sha256']].append(r['file'])

    original=ROOT/'研究规划/第一篇小论文/secure distributed safety.tex'
    tx=original.read_text('utf-8-sig');before[str(original)]=sha(original)
    original_rows=[]
    for i,s in enumerate(tx.splitlines()):
        if '\\to' in s and s.lstrip().startswith('$'):
            link=re.search(r'\$(\d+)\\to(\d+)\$',s)
            vals=re.findall(r'([0-9.]+)\\%',s)
            if link and len(vals)==4:original_rows.append({'line':i+1,'sender':int(link[1]),'receiver':int(link[2]),'percent':[float(x) for x in vals]})
    assert len(original_rows)==6
    source_dir=ROOT/'mypaper/学报投稿/历史归档/根目录散落实验输出'
    out=source_dir/'simulation_outputs'
    t1,t2,t3=[records(out/f) for f in ['Table1_No_Watermark_Per_Link_FPR_FNR.csv','Table2_Watermark_Per_Link_FPR_FNR.csv','Table3_Watermark_Mean_FPR_FNR.csv']]
    csv_paths=list(out.glob('Table*.csv'))+[source_dir/'Table1_Strictly_Stealthy_Attack_Metrics.csv',source_dir/'Detection_Metrics_Summary.csv',source_dir/'Detection_Metrics_Per_Link.csv']
    csv_evidence=[{'file':rel(p),'sha256':sha(p),'rows':len(records(p)),'columns':list(records(p)[0])} for p in csv_paths]
    for p in csv_paths:before[str(p)]=sha(p)
    index1={(int(v['receiver_node']),int(v['sender_node'])):v for v in t1}
    index2={(float(v['watermark_variance']),int(v['receiver_node']),int(v['sender_node'])):v for v in t2}
    comparisons=[]
    for row in original_rows:
        key=(row['receiver'],row['sender']);without=index1[key];half=index2[(.5,*key)];const=index2[(0.,*key)]
        combo=[float(without['fpr_clean'])*100,float(without['fnr_strictly_stealthy_attack'])*100,float(half['fpr'])*100,float(half['fnr'])*100]
        combo_const=[float(const['fpr'])*100,float(const['fnr'])*100,float(half['fpr'])*100,float(half['fnr'])*100]
        rounded=lambda z:[round(v,2) for v in z]
        comparisons.append({**row,'CSV_no_encoding_plus_beta05':rounded(combo),'CSV_constant_lambda_plus_beta05':rounded(combo_const),'different_cells_no_encoding':sum(a!=b for a,b in zip(row['percent'],rounded(combo))),'different_cells_constant_lambda':sum(a!=b for a,b in zip(row['percent'],rounded(combo_const)))})
    mean_checks=[]
    for r in t3:
        v=float(r['watermark_variance']);subset=[x for x in t2 if float(x['watermark_variance'])==v]
        diffs={key:sum(float(z[key]) for z in subset)/len(subset)-float(r['mean_'+key]) for key in ['fpr','fnr','tpr']}
        diffs['delay']=sum(float(z['mean_detection_delay_samples']) for z in subset)/len(subset)-float(r['mean_detection_delay_samples'])
        assert max(abs(d) for d in diffs.values())<1e-12
        mean_checks.append({'variance':v,'per_link_rows':len(subset),'mean_differences':diffs})
    legacy_summary=records(source_dir/'Detection_Metrics_Summary.csv')
    legacy_per=records(source_dir/'Detection_Metrics_Per_Link.csv')
    legacy_index={(row['scenario'],float(row['watermark_variance']) if row['watermark_variance'] else None,int(row['receiver_node']),int(row['sender_node'])):row for row in legacy_per}
    legacy_comparisons=[]
    for row in original_rows:
        key=(row['receiver'],row['sender'])
        a=legacy_index[('No watermarking',None,*key)]
        b=legacy_index[('Dynamic watermark',.5,*key)]
        alt=legacy_index[('Only lambda',0.,*key)]
        vals=[100*float(a['fpr']),100*float(a['fnr']),100*float(b['fpr']),100*float(b['fnr'])]
        vals_alt=[100*float(alt['fpr']),100*float(alt['fnr']),100*float(b['fpr']),100*float(b['fnr'])]
        legacy_comparisons.append({**row,'old_CSV_no_encoding_plus_beta05':[round(v,2) for v in vals],'old_CSV_constant_lambda_plus_beta05':[round(v,2) for v in vals_alt],'different_cells_no_encoding':sum(round(a,2)!=b for a,b in zip(vals,row['percent'])),'different_cells_constant_lambda':sum(round(a,2)!=b for a,b in zip(vals_alt,row['percent']))})
    legacy_mean_checks=[]
    for row in legacy_summary:
        var=float(row['watermark_variance']);scenario='Only lambda' if var==0 else 'Dynamic watermark'
        subset=[r for r in legacy_per if r['scenario']==scenario and r['watermark_variance'] and float(r['watermark_variance'])==var]
        d={k:sum(float(s[k]) for s in subset)/len(subset)-float(row['mean_'+k]) for k in ['fpr','fnr','tpr']}
        d['delay']=sum(float(s['detection_delay_samples']) for s in subset)/len(subset)-float(row['mean_detection_delay_samples'])
        assert max(abs(x) for x in d.values())<1e-12
        legacy_mean_checks.append({'variance':var,'differences':d})
    quant=[]
    for label,rs,fp,fnr,den0,dena in [
        ('complete_v2_default1000_geometry',t1,'fpr_clean','fnr_strictly_stealthy_attack',130000,53000),
        ('direction_fixed_default5_geometry',records(source_dir/'Table1_Strictly_Stealthy_Attack_Metrics.csv'),'fpr_without_attack','fnr_under_strictly_stealthy_attack',150,480)]:
        data=[]
        for row in rs:
            floats=[float(row[fp])*den0,(1-float(row[fnr]))*dena]
            ints=[round(x) for x in floats]
            assert max(abs(a-b) for a,b in zip(floats,ints))<1e-7
            data.append({'link':row['directed_link_sender_to_receiver'],'compatible_false_alarms':ints[0],'compatible_true_alarms':ints[1],'residual':max(abs(a-b) for a,b in zip(floats,ints))})
        quant.append({'hypothesis':label,'clean_denominator':den0,'attack_denominator':dena,'data':data,'actual_run_count_proved':False,'reason':'compatible rational counts are not a signed run manifest; other multiples may also fit'})

    figures=[]
    canonical=original.parent
    for f in sorted(canonical.glob('Fig*.pdf')):
        h=sha(f);before[str(f)]=h;matches=[]
        for root in roots:
            for other in root.rglob(f.name):
                if other.is_file() and 'skills' not in other.parts:
                    matches.append({'file':rel(other),'sha256':sha(other),'byte_equal_to_preserved':sha(other)==h})
                    before[str(other)]=sha(other)
        figures.append({'preserved_file':rel(f),'sha256':h,'same_name_candidates':matches,'same_bytes_do_not_prove_model_or_run_config':True})
    configs=[]
    for root in roots:
        for f in root.rglob('simulation_parameters.json'):
            if 'skills' in f.parts:continue
            a=json.loads(f.read_text('utf8'));before[str(f)]=sha(f)
            configs.append({'file':rel(f),'sha256':sha(f),'top_level_fields':list(a),'state_matrix_A':a.get('state_matrix_A'),'config':a.get('config'),'is_original_50_node_run_manifest':False})
    npzs=[]
    for f in source_dir.rglob('*Optimal*Attack*Sequences.npz'):
        before[str(f)]=sha(f)
        with np.load(f,allow_pickle=False) as data:
            npzs.append({'file':rel(f),'sha256':sha(f),'arrays':{k:{'shape':list(data[k].shape),'dtype':str(data[k].dtype)} for k in data.files},'contains_all_physical_MC_runs':False})
    # This is a bounded search over named paper/code archives, not the whole E drive.
    forbidden_missing=[]
    for row in code:
        for a in row['anchors']:
            if re.search(r'KLD_threshold\s*=\s*39\b',a['text']) or ('A_stable' in a['text'] and '.8' in a['text'] and '.05' in a['text']):forbidden_missing.append((row['file'],a))
    protected_changed=[p for p,h in before.items() if sha(Path(p))!=h]
    assert not protected_changed
    report={'date':'2026-11-15','baseline':'72a972ccd4c3dc5fb024ef1b178769073b10c422','scope':'mypaper and mirrored 仓库论文资料/mypaper named historical experiment sources; skills excluded; no second-paper data examined','simulations_imported_or_run':False,'Monte_Carlo':0,'code_records':code,'exact_duplicate_source_groups':[{'sha256':k,'files':v} for k,v in groups.items() if len(v)>1],'csv_evidence':csv_evidence,'original_combined_table_active_rows':original_rows,'original_vs_CSV':comparisons,'different_display_cells_no_encoding':sum(x['different_cells_no_encoding'] for x in comparisons),'different_display_cells_constant_lambda':sum(x['different_cells_constant_lambda'] for x in comparisons),'Table3_is_mean_of_Table2_checks':mean_checks,'original_vs_legacy_CSV':legacy_comparisons,'legacy_different_display_cells_no_encoding':sum(x['different_cells_no_encoding'] for x in legacy_comparisons),'legacy_different_display_cells_constant_lambda':sum(x['different_cells_constant_lambda'] for x in legacy_comparisons),'legacy_summary_is_mean_of_per_link':legacy_mean_checks,'quantization_compatibility':quant,'preserved_figures':figures,'other_configuration_candidates':configs,'stored_attack_arrays':npzs,'found_50node_source_with_original_theta39_or_Astable08':forbidden_missing,'protected_static_inputs':len(before),'protected_changed':protected_changed,'input_hashes':{rel(Path(p)):h for p,h in before.items()}}
    save('historical_source_evidence.json',report)
    print(json.dumps({'source_files':len(code),'unique_code_hashes':len(groups),'CSV_files':len(csv_evidence),'figures':len(figures),'other_configs':len(configs),'different_cells_no_encoding':report['different_display_cells_no_encoding'],'different_cells_constant_lambda':report['different_display_cells_constant_lambda'],'Table3_mean_checks':len(mean_checks),'old_inputs_unchanged':len(before)},ensure_ascii=False))

if __name__=='__main__':main()
