"""Regenerate historical 50-node defaults without importing/running simulations.

Only hash-pinned, explicitly whitelisted AST nodes are executed: Params,
graph/matrix initialization and the original pure parity-matrix function.
No Monte Carlo, plotting, data loading, isolation or attack optimization runs.
Regenerated values are NOT recovered original-run input-array snapshots.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import platform
import numpy as np
import networkx as nx
import scipy
from scipy.linalg import block_diag, null_space, pinv

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source/simulation_revised_detection_metrics_complete_v2.py'
EXPECTED='2f4694e6180680ccd83ecc4ae60cb587b76276625c9d34c5946cfeca2d3e8a4a'


def load_initialization():
    raw=SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED:
        raise ValueError('Recovered historical source hash changed; do not execute an unreviewed AST')
    tree=ast.parse(raw.decode('utf-8-sig'))
    names={'seed_val','W','neighbors_in','A_unstable','A_stable','Q','H','R'}
    selected=[]
    for node in tree.body:
        keep=False
        if isinstance(node,ast.ClassDef) and node.name=='Params':keep=True
        elif isinstance(node,ast.Assign):
            keep=any(isinstance(t,ast.Name) and t.id in names for t in node.targets)
        elif isinstance(node,ast.While) and node.lineno==159:keep=True
        elif isinstance(node,ast.For) and node.lineno in (169,262):keep=True
        elif isinstance(node,ast.Expr) and node.lineno==259:keep=True
        elif isinstance(node,ast.FunctionDef) and node.name=='build_parity_space':keep=True
        if keep:selected.append(node)
    expected_lines={57,158,159,169,173,174,255,256,257,259,260,261,262,297}
    if {n.lineno for n in selected}!=expected_lines:
        raise ValueError('Whitelisted source structure no longer matches reviewed historical version')
    scope={'np':np,'nx':nx,'Path':Path,'os':SimpleNamespace(environ={}),
           'null_space':null_space,'block_diag':block_diag,'pinv':pinv}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'),scope)
    return scope,sorted(expected_lines)


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return value.as_posix()
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def extract():
    scope,lines=load_initialization();P=scope['Params'];G=scope['G'];H=scope['H'];R=scope['R']
    adjacency=scope['W'];neighbors=scope['neighbors_in'];Q=scope['Q']
    params={k:jsonable(v) for k,v in vars(P).items() if not k.startswith('_')}
    config=dict(status='regenerated_from_hash_pinned_historical_source; NOT original logged run snapshot',
       node_count=P.sensorNum,state_dimension=P.dataDim,measurement_dimension=P.meaDim,
       source_sha256=EXPECTED,whitelisted_top_level_lines=lines,
       versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,networkx=nx.__version__),
       graph=dict(seed_start=42,seed_selected=scope['seed_val'],radius=.31,connected=nx.is_connected(G),
                  edge_count=G.number_of_edges(),positions={str(i):list(G.nodes[i]['pos']) for i in range(P.sensorNum)},
                  adjacency=adjacency.tolist(),neighbors=neighbors),
       A_unstable=scope['A_unstable'].tolist(),A_stable=scope['A_stable'].tolist(),Q=Q.tolist(),
       true_initial_state_covariance=np.eye(P.dataDim).tolist(),
       estimator_initial_covariance=(5*np.eye(P.dataDim)).tolist(),
       communication_loop_first_time_index=1,
       H_layout='node-major [50,2,2], source layout was [measurement,state,node]',
       H=np.moveaxis(H,2,0).tolist(),R=np.moveaxis(R,2,0).tolist(),defaults=params,
       H_rank_each=[int(np.linalg.matrix_rank(H[:,:,i])) for i in range(P.sensorNum)],
       attack_links=[dict(receiver=int(i),sender=int(j),physical=f'{j}->{i}') for i,j in P.attack_links],
       estimator='unconstrained stacked-measurement KF, with isolation-dependent active stack; NOT equal common-G blocks',
       common_G_available=False,original_MC_execution_recovered=False)
    out=HERE/'configuration_snapshot.json'
    out.write_text(json.dumps(config,indent=2,ensure_ascii=False,default=jsonable)+'\n',encoding='utf-8')
    parity_arrays={};parity_summary={};gain_arrays={};gain_summary={};all_zerr=0.;all_mineig=np.inf
    for label in ('unstable','stable'):
        A=scope['A_'+label];Zs,Thetas=scope['build_parity_space'](A)
        zerr=0.;mineig=np.inf;pernode=[]
        for sender in range(P.sensorNum):
            Z,Theta=Zs[sender],Thetas[sender]
            if Z is None:raise ValueError('Unexpected empty parity subspace')
            O=np.vstack([H[:,:,sender]@np.linalg.matrix_power(A,t) for t in range(P.s+1)])
            zerr=max(zerr,float(np.max(np.abs(Z@O))))
            mineig=min(mineig,float(np.linalg.eigvalsh(Theta)[0]))
            parity_arrays[f'{label}_Z_sender{sender}']=Z
            parity_arrays[f'{label}_Theta_sender{sender}']=Theta
            pernode.append(dict(sender=sender,parity_shape=list(Z.shape),theta_shape=list(Theta.shape)))
        parity_summary[label]=dict(max_abs_ZO=zerr,min_theta_eigenvalue=mineig,senders=pernode,
                                   legacy_theta_diagonal_regularizer=1e-8)
        all_zerr=max(all_zerr,zerr);all_mineig=min(all_mineig,mineig)
        gains=[]
        for receiver in range(P.sensorNum):
            order=list(neighbors[receiver])+[receiver]
            Hstack=np.vstack([H[:,:,j] for j in order]);Rstack=block_diag(*[R[:,:,j] for j in order])
            Ppred=A@(5*np.eye(P.dataDim))@A.T+Q
            gain=Ppred@Hstack.T@pinv(Hstack@Ppred@Hstack.T+Rstack)
            gain_arrays[f'{label}_initial_stack_gain_receiver{receiver}']=gain
            blocks=[gain[:,j*P.meaDim:(j+1)*P.meaDim] for j in range(len(order))]
            neighbor_blocks=blocks[:-1]
            spread=max((float(np.linalg.norm(x-neighbor_blocks[0],'fro')) for x in neighbor_blocks),default=0.)
            gains.append(dict(receiver=receiver,stack_sender_order=[int(j) for j in order],
                         self_block_index=len(order)-1,gain_shape=list(gain.shape),
                         maximum_neighbor_block_difference=spread))
        gain_summary[label]=dict(status='derived first update k=1, no isolation; NOT recovered stored gains or common-G',
                                P0=(5*np.eye(P.dataDim)).tolist(),nodes=gains,
                                common_G_equality_violations=sum(x['maximum_neighbor_block_difference']>1e-10 for x in gains),
                                maximum_neighbor_block_difference=max(x['maximum_neighbor_block_difference'] for x in gains))
    np.savez_compressed(HERE/'parity_matrices_regenerated.npz',**parity_arrays)
    np.savez_compressed(HERE/'initial_stacked_gains_regenerated.npz',**gain_arrays)
    (HERE/'parity_verification.json').write_text(json.dumps(parity_summary,indent=2)+'\n',encoding='utf-8')
    (HERE/'gain_interface_audit.json').write_text(json.dumps(gain_summary,indent=2)+'\n',encoding='utf-8')
    assert P.sensorNum==50 and H.shape==(2,2,50)
    assert np.array_equal(adjacency,adjacency.T) and not np.any(np.diag(adjacency))
    assert nx.is_connected(G) and all_zerr<1e-10 and all_mineig>0
    print(json.dumps(dict(nodes=P.sensorNum,edges=G.number_of_edges(),graph_seed=scope['seed_val'],
                         versions=config['versions'],max_ZO=all_zerr,min_Theta_eigenvalue=all_mineig,
                         gain_type=config['estimator'],common_G_available=False),indent=2))


if __name__=='__main__':extract()
