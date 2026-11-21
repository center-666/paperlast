"""Exact rational certificate for registered 50-node local gain modes.

No simulation is imported or run. The certificate covers the exact decimal or
IEEE coefficients supplied by physical_configuration.json and gain_table.json;
it does not prove the gains solve the DARE exactly or bound runtime rounding.
"""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import platform

HERE=Path(__file__).resolve().parent


def read_inputs():
    cfg_bytes=(HERE/'physical_configuration.json').read_bytes()
    gain_bytes=(HERE/'gain_table.json').read_bytes()
    cfg_decimal=json.loads(cfg_bytes.decode('utf-8'),parse_float=Fraction)
    gains_decimal=json.loads(gain_bytes.decode('utf-8'),parse_float=Fraction)
    cfg_ieee=json.loads(cfg_bytes.decode('utf-8'))
    gains_ieee=json.loads(gain_bytes.decode('utf-8'))
    manifest=json.loads((HERE.parent/'paper1_original50_source_20261121/source_manifest.json').read_text(encoding='utf-8'))
    files={Path(row['path']).name:row['sha256'] for row in manifest['files']}
    expected_attacks=[[0,1],[2,3],[15,16],[17,31],[14,49],[17,34]]
    assert gains_decimal['attack_links_receiver_sender']==expected_attacks
    raw_config=HERE.parent/'paper1_original50_source_20261121/configuration_snapshot.json'
    config_hash=hashlib.sha256(raw_config.read_bytes()).hexdigest()
    assert config_hash==hashlib.sha256(cfg_bytes).hexdigest()
    assert config_hash==gains_decimal['physical_configuration_sha256']
    assert cfg_decimal['source_sha256']==files['simulation_revised_detection_metrics_complete_v2.py']
    input_bytes={'physical_configuration.json':cfg_bytes,'gain_table.json':gain_bytes}
    return cfg_decimal,gains_decimal,cfg_ieee,gains_ieee,manifest,files,config_hash,input_bytes


def validate_mode_coverage(cfg,gains):
    links=[tuple(map(int,x)) for x in gains['attack_links_receiver_sender']]
    assert cfg['node_count']==50 and cfg['state_dimension']==cfg['measurement_dimension']==2
    base=[set(map(int,x)) for x in cfg['graph']['neighbors']]
    expected=set()
    for i in range(50):
        incoming=[sender for receiver,sender in links if receiver==i]
        for mask in range(1<<len(incoming)):
            dropped={sender for k,sender in enumerate(incoming) if mask>>k&1}
            expected.add((i,tuple(sorted(base[i]-dropped))))
    assert len(expected)==57
    for name,model in gains['models'].items():
        assert model['A']==cfg['A_'+name]
        actual=set();byrow={}
        for row in model['rows']:
            i=int(row['receiver']);active=list(map(int,row['active_neighbors']))
            assert len(active)==len(set(active)) and set(active)<=base[i] and i not in active
            assert len(active)==row['retained_neighbor_count']
            key=(i,tuple(sorted(active)))
            assert key not in actual
            actual.add(key);idx=int(row['row_index']);assert idx not in byrow;byrow[idx]=key
        assert actual==expected and set(byrow)==set(range(57))
        mapping=model['global_mask_to_local_row']
        assert len(mapping)==64
        for mask,rows in enumerate(mapping):
            assert len(rows)==50
            for i,index in enumerate(rows):
                dropped={sender for k,(receiver,sender) in enumerate(links) if receiver==i and mask>>k&1}
                assert byrow[int(index)]==(i,tuple(sorted(base[i]-dropped)))


def matrix(data,convert):
    return [[convert(x) for x in row] for row in data]


def eye(n):return [[Fraction(int(i==j)) for j in range(n)] for i in range(n)]


def transpose(a):return [list(row) for row in zip(*a)]


def multiply(a,b):
    return [[sum((a[i][k]*b[k][j] for k in range(len(b))),Fraction(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def subtract(a,b):
    return [[a[i][j]-b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def determinant_2(a):return a[0][0]*a[1][1]-a[0][1]*a[1][0]


def frac_text(x):return f'{x.numerator}/{x.denominator}'


def certify_model(name,cfg,table,convert,coefficient_label):
    model=table['models'][name]
    # The physical file is the source of A/H, and the table copy must agree.
    A=matrix(cfg['A_'+name],convert)
    rows=[]
    for source_row in model['rows']:
        i=int(source_row['receiver'])
        active=[int(j) for j in source_row['active_neighbors']]
        Hi=matrix(cfg['H'][i],convert)
        Hactive=[matrix(cfg['H'][j],convert) for j in active]
        Hsum=[[sum((Hj[r][c] for Hj in Hactive),Fraction(0)) for c in range(2)] for r in range(2)]
        K=matrix(source_row['K'],convert)
        G=matrix(source_row['G'],convert)
        # F=(I-K H_i-G sum_j H_j)A. F_diagnostic is deliberately not read.
        closed=subtract(subtract(eye(2),multiply(K,Hi)),multiply(G,Hsum))
        F=multiply(closed,A)
        D=subtract([[Fraction(1,100) if r==c else Fraction(0) for c in range(2)] for r in range(2)],
                   multiply(transpose(F),F))
        d11,d22=D[0][0],D[1][1]
        det=determinant_2(D)
        if not (d11>0 and d22>0 and det>0):
            raise ArithmeticError(f'{name} receiver={i} row={source_row["row_index"]}: c=.1 strict PD certificate failed')
        rows.append(dict(receiver=i,active_neighbors=active,local_row=int(source_row['row_index']),
                         D11=frac_text(d11),D22=frac_text(d22),det_D=frac_text(det),
                         D11_float=float(d11),D22_float=float(d22),det_D_float=float(det)))
    if len(rows)!=57:raise ValueError(f'Expected 57 registered local rows, found {len(rows)}')
    return dict(model=name,coefficients=coefficient_label,c_exact='1/10',formula='F=(I-KH_i-G sum(H_j))A; D=c^2 I-F^T F',
                modes=len(rows),all_modes_strictly_certified=True,rows=rows)


def run():
    cfg_d,gain_d,cfg_f,gain_f,manifest,source_hashes,configuration_sha,input_bytes=read_inputs()
    validate_mode_coverage(cfg_d,gain_d)
    certificate=[]
    for name in ('unstable','stable'):
        certificate.append(certify_model(name,cfg_d,gain_d,Fraction,'literal_decimal_rational'))
        certificate.append(certify_model(name,cfg_f,gain_f,Fraction.from_float,'exact_binary_rational_of_parsed_IEEE_float'))
    summary=[]
    for item in certificate:
        summary.append(dict(model=item['model'],coefficients=item['coefficients'],modes=item['modes'],
                            min_D11_exact=frac_text(min(Fraction(row['D11']) for row in item['rows'])),
                            min_D22_exact=frac_text(min(Fraction(row['D22']) for row in item['rows'])),
                            min_det_D_exact=frac_text(min(Fraction(row['det_D']) for row in item['rows'])),
                            min_D11=min(row['D11_float'] for row in item['rows']),
                            min_D22=min(row['D22_float'] for row in item['rows']),
                            min_det_D=min(row['det_D_float'] for row in item['rows']),
                            all_modes_strictly_certified=item['all_modes_strictly_certified']))
    inputs={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)) for name,data in input_bytes.items()}
    script=Path(__file__).read_bytes()
    result=dict(evidence_version='common-g-rational-v1',
                certificate_type='Exact rational Sylvester test for c^2 I-F^T F from serialized K/G/H/A. This is not a DARE optimality or runtime floating-point execution certificate.',
                arithmetic='Python fractions.Fraction; signs checked on exact integers, no floating test used for acceptance',
                certificate_runtime=dict(python=platform.python_version(),script='certify_common_g.py',script_sha256=hashlib.sha256(script).hexdigest()),
                input_files=inputs,uses_F_diagnostic=False,
                contraction_c_exact='1/10',contraction_c_squared_exact='1/100',
                coverage=dict(models=['unstable','stable'],local_modes_per_model=57,distinct_local_mode_matrices=114,
                              coefficient_models_per_matrix=2,exact_rational_checks=228,
                              global_joint_masks_per_model=64,registered_mode_lookup_verified=True),
                source_commit=manifest['commit'],source_hashes=source_hashes,
                configuration_snapshot_sha256=configuration_sha,
                gain_table_input_baseline=gain_d['input_baseline_commit'],
                gain_table_numpy_scipy_versions=gain_d['versions'],
                attack_links_receiver_sender=gain_d['attack_links_receiver_sender'],
                modes_per_dynamics=57,summary=summary,certificates=certificate)
    for name,data in input_bytes.items():
        assert (HERE/name).read_bytes()==data,f'{name} changed during certification; rerun on stable inputs'
    path=HERE/'rational_certificate.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(all_passed=True,input_files=inputs,
          summary=[{k:v for k,v in row.items() if not k.endswith('_exact')} for row in summary]),indent=2))


if __name__=='__main__':run()
