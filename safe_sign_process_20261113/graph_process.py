"""Finite exact safe-word graph and numerical maximum-entropy Markov chain.

The graph input is the independently supplied safe_words.json when available;
otherwise it is the 2026-11-12 integer certificate, clearly marked provisional.
Graph/SCC/symmetry counts are finite combinatorial results. Perron vectors and
the Markov chain are floating-point representations; spectral-radius bounds
use a strictly positive rational test vector and exact Collatz inequalities.
No physical/noise simulation, no old-file modification.
"""
from pathlib import Path
from fractions import Fraction
from hashlib import sha256
from math import gcd, log2
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs

HERE = Path(__file__).resolve().parent
P = HERE.parent

def bits(index, n):
    return [1 if (index >> j) & 1 else -1 for j in range(n-1,-1,-1)]

def choose_words():
    candidate = HERE/'safe_words.json'
    if candidate.exists():
        raw = candidate.read_bytes()
        obj = json.loads(raw.decode('utf-8-sig'))
        # Accept explicit index keys only; do not guess that an arbitrary list
        # is a zero-based binary index list.
        for key in ['safe_word_indices','strict_safe_word_indices','safe_indices']:
            if key in obj:
                return sorted(obj[key]), dict(file=candidate.name,sha256=sha256(raw).hexdigest(),
                                              provisional_old_input=False,key=key)
        raise ValueError('safe_words.json needs an explicit safe index field')
    candidate = P/'glrt_screening_20261112/exact_sign_danger_cap.json'
    raw = candidate.read_bytes()
    obj = json.loads(raw.decode('utf-8-sig'))
    unsafe = set(obj['danger_capable_word_union_indices'])
    return [i for i in range(1024) if i not in unsafe], dict(
        file=str(candidate.relative_to(P)),sha256=sha256(raw).hexdigest(),
        provisional_old_input=True,key='complement of danger_capable_word_union_indices')

def period_of_component(outgoing):
    # For a strongly connected directed graph, gcd of distance differences
    # over all directed edges equals the gcd of directed cycle lengths.
    distance = [-1]*len(outgoing)
    distance[0] = 0
    queue = [0]
    for i in queue:
        for j in outgoing[i]:
            if distance[j] < 0:
                distance[j] = distance[i]+1
                queue.append(j)
    assert all(x>=0 for x in distance)
    d = 0
    for i, row in enumerate(outgoing):
        for j in row:
            d = gcd(d, abs(distance[i]+1-distance[j]))
    return d

def entropy(prob):
    return -sum(float(p)*log2(float(p)) for p in prob if p>0)

def main():
    safe, source = choose_words()
    assert len(safe)==len(set(safe)) and all(0<=i<1024 for i in safe)
    edges = [(w>>1, w&511, w) for w in safe]
    edge_set = {(i,j) for i,j,w in edges}
    assert all((511-i,511-j) in edge_set for i,j,w in edges)
    A = csr_matrix((np.ones(len(edges), dtype=np.int64),
                    ([i for i,j,w in edges],[j for i,j,w in edges])),shape=(512,512))
    number, labels = connected_components(A,directed=True,connection='strong')
    components=[]
    for label in range(number):
        states=np.flatnonzero(labels==label).tolist()
        cyclic = len(states)>1 or bool(A[states[0],states[0]])
        components.append(dict(label=label,states=states,size=len(states),cyclic=cyclic))
    cyclic=[c for c in components if c['cyclic']]
    assert cyclic
    chosen=max(cyclic,key=lambda c:c['size'])
    states=chosen['states']
    local={s:i for i,s in enumerate(states)}
    sub_edges=[(local[i],local[j],w) for i,j,w in edges if i in local and j in local]
    Ac = csr_matrix((np.ones(len(sub_edges)),
                     ([i for i,j,w in sub_edges],[j for i,j,w in sub_edges])),shape=(len(states),len(states)))
    vals, vr=eigs(Ac,k=1,which='LR',tol=1e-13,v0=np.ones(len(states)))
    rho=float(vals[0].real)
    right=vr[:,0].real
    if right.sum()<0:right=-right
    assert right.min()>0
    right/=right.max()
    vals_l,vl=eigs(Ac.T,k=1,which='LR',tol=1e-13,v0=np.ones(len(states)))
    left=vl[:,0].real
    if left.sum()<0:left=-left
    assert left.min()>0 and abs(vals_l[0].real-rho)<1e-10
    left/=left@right
    pi=left*right
    pi/=pi.sum()
    transitions=[]
    outgoing=[[] for _ in states]
    probabilities=[[] for _ in states]
    for i,j,w in sub_edges:
        prob=float(right[j]/(rho*right[i]))
        outgoing[i].append(j)
        probabilities[i].append(prob)
        transitions.append(dict(source_state=states[i],target_state=states[j],
                                word_index=w,next_sign=1 if w&1 else -1,probability=prob,
                                stationary_edge_mass=float(pi[i]*prob)))
    row_sums=np.array([sum(row) for row in probabilities])
    Pm=csr_matrix((np.array([t['probability'] for t in transitions]),
                  ([i for i,j,w in sub_edges],[j for i,j,w in sub_edges])),shape=Ac.shape)
    assert np.max(np.abs(row_sums-1))<1e-11
    assert np.max(np.abs(pi@Pm-pi))<1e-11
    local_entropy=np.array([entropy(row) for row in probabilities])
    entropy_rate=float(pi@local_entropy)
    forced=[i for i,row in enumerate(outgoing) if len(row)==1]
    period=period_of_component(outgoing)
    # A finite-precision implementation need only preserve the support. A
    # 16-bit draw per clock tick gives explicit rational transitions; this is
    # not claimed to be the exact PF maximum-entropy chain.
    dyadic_rows=[]
    for i,row in enumerate(outgoing):
        if len(row)==1:
            dyadic_rows.append([65536])
        else:
            first=min(65535,max(1,round(probabilities[i][0]*65536)))
            dyadic_rows.append([first,65536-first])
    dyadic_transitions=[]
    for i,row in enumerate(outgoing):
        for z,j in enumerate(row):
            dyadic_transitions.append(dict(source_state=states[i],target_state=states[j],
                                          numerator=dyadic_rows[i][z],denominator=65536))
    dyadic_by_edge={(t['source_state'],t['target_state']):t['numerator'] for t in dyadic_transitions}
    assert all(dyadic_by_edge[(511-i,511-j)]==n for (i,j),n in dyadic_by_edge.items())
    Pd=csr_matrix((np.array([t['numerator']/65536 for t in dyadic_transitions]),
                   ([i for i,row in enumerate(outgoing) for _ in row],
                    [j for row in outgoing for j in row])),shape=Ac.shape)
    pi_d=pi.copy()
    for _ in range(10000):
        nxt=pi_d@Pd
        if np.max(np.abs(nxt-pi_d))<1e-15:
            pi_d=nxt;break
        pi_d=nxt
    assert np.max(np.abs(pi_d@Pd-pi_d))<1e-13
    dyadic_entropy=float(sum(pi_d[i]*entropy([p/65536 for p in row])
                             for i,row in enumerate(dyadic_rows)))
    forced_chain_lengths=[]
    for i in forced:
        visited=set();q=i;nforced=0
        while len(outgoing[q])==1:
            assert q not in visited,'A deterministic closed cycle would be a separate SCC'
            visited.add(q);nforced+=1;q=outgoing[q][0]
        forced_chain_lengths.append(nforced)
    # Full-graph viability: states that can reach the unique cyclic component.
    reverse=[[] for _ in range(512)]
    for i,j,w in edges:reverse[j].append(i)
    viable=set(states);queue=list(states)
    for j in queue:
        for i in reverse[j]:
            if i not in viable:viable.add(i);queue.append(i)
    # Exact Collatz lower/upper bounds with a rational positive test vector.
    test=[Fraction(str(float(x))) for x in right]
    ratios=[sum(test[j] for j in outgoing[i])/test[i] for i in range(len(states))]
    low,high=min(ratios),max(ratios)
    # Flip symmetry is exact at graph level. Uniqueness of the stationary
    # maximum-entropy measure gives exact unbiasedness; numbers cross-check it.
    assert {511-s for s in states}==set(states)
    flip_error=max(abs(pi[i]-pi[local[511-s]]) for i,s in enumerate(states))
    pplus=np.zeros(len(states))
    for t in transitions:
        if t['next_sign']==1:pplus[local[t['source_state']]]+=t['probability']
    prediction=[]
    for r in range(10):
        mask=(1<<r)-1
        groups={}
        for i,s in enumerate(states):
            suffix=s&mask
            if suffix not in groups:groups[suffix]=[0.,0.]
            groups[suffix][0]+=float(pi[i])
            groups[suffix][1]+=float(pi[i]*pplus[i])
        rows=[]
        for suffix,(weight,plus_mass) in sorted(groups.items()):
            pp=plus_mass/weight
            rows.append(dict(history=bits(suffix,r),probability=weight,
                             next_plus_probability=pp,
                             best_prediction=max(pp,1-pp),conditional_entropy=entropy([pp,1-pp])))
        prediction.append(dict(known_past_sign_count=r,history_count=len(rows),
            average_optimal_next_prediction=sum(x['probability']*x['best_prediction'] for x in rows),
            worst_history_next_prediction=max(x['best_prediction'] for x in rows),
            conditional_entropy=sum(x['probability']*x['conditional_entropy'] for x in rows),
            minimum_history_conditional_entropy=min(x['conditional_entropy'] for x in rows),histories=rows))
    future=[]
    bit_prob=pplus.copy()
    # Guess probability of an entire h-symbol future path, via max-product DP.
    path_guess=np.ones(len(states))
    for horizon in range(1,11):
        next_path=np.array([max(probabilities[i][z]*path_guess[j]
                               for z,j in enumerate(outgoing[i])) for i in range(len(states))])
        future.append(dict(horizon=horizon,
            average_optimal_single_bit_prediction=float(pi@np.maximum(bit_prob,1-bit_prob)),
            worst_known_state_single_bit_prediction=float(np.maximum(bit_prob,1-bit_prob).max()),
            average_optimal_block_guess=float(pi@next_path),worst_known_state_block_guess=float(next_path.max())))
        bit_prob=Pm@bit_prob
        path_guess=next_path
    # At least one directed cycle is given explicitly, found via a walk until
    # a state repeats. Choice is deterministic and is not the random generator.
    walk=[]; seen={}; q=0
    while q not in seen:
        seen[q]=len(walk);walk.append(q);q=outgoing[q][0]
    cycle=[states[x] for x in walk[seen[q]:]]+[states[q]]
    word_probabilities=np.zeros(1024)
    for t in transitions:word_probabilities[t['word_index']]=t['stationary_edge_mass']
    assert abs(word_probabilities.sum()-1)<1e-12
    words=np.array([bits(i,10) for i in range(1024)])
    sign_correlation=np.einsum('w,wi,wj->ij',word_probabilities,words,words)
    result=dict(date='2026-11-13',source=source,input_word_count=len(safe),
        state_encoding='9 bits, -1=0,+1=1, oldest bit most significant',
        word_encoding='10 bits, -1=0,+1=1, itertools.product([-1,1],repeat=10)',
        total_graph_states=512,total_graph_edges=len(edges),
        full_graph_outdegree_counts={str(n):int((np.diff(A.indptr)==n).sum()) for n in [0,1,2]},
        infinitely_extendible_states=sorted(viable),infinitely_extendible_state_count=len(viable),
        strong_component_count=number,cyclic_component_count=len(cyclic),components=components,
        selected_component=chosen,selected_component_edges=len(sub_edges),period=period,
        explicit_cycle_states=cycle,sign_flip_graph_invariant=True,
        rho=rho,topological_entropy_bits=log2(rho),
        collatz_rational_lower=str(low),collatz_rational_upper=str(high),
        collatz_lower_float=float(low),collatz_upper_float=float(high),
        collatz_width_float=float(high-low),
        pf_right=right.tolist(),pf_left=left.tolist(),stationary=pi.tolist(),
        states=states,transitions=transitions,
        stationary_word_probabilities=word_probabilities.tolist(),
        stationary_ten_sign_correlation=sign_correlation.tolist(),
        row_sum_max_error=float(np.max(np.abs(row_sums-1))),
        stationarity_max_error=float(np.max(np.abs(pi@Pm-pi))),
        sign_flip_stationary_max_error=float(flip_error),
        next_plus_probability=float(pi@pplus),next_sign_mean=float(2*(pi@pplus)-1),
        entropy_rate_bits=entropy_rate,entropy_rate_vs_logrho_difference=float(entropy_rate-log2(rho)),
        minimum_state_conditional_entropy=float(local_entropy.min()),
        maximum_state_conditional_entropy=float(local_entropy.max()),
        forced_next_state_count=len(forced),forced_next_states=[states[i] for i in forced],
        longest_consecutive_forced_transitions=max(forced_chain_lengths,default=0),
        stationary_forced_state_probability=float(pi[forced].sum()),
        worst_state_next_prediction=float(np.maximum(pplus,1-pplus).max()),
        average_state_next_prediction=float(pi@np.maximum(pplus,1-pplus)),
        prediction_by_known_past=prediction,prediction_by_future_horizon=future,
        exact_graph_edges=[dict(source=i,target=j,word=w) for i,j,w in edges],
        exact_scc_counts=True,pf_vectors_numerical=True,collatz_bound_exact_rational=True,
        dyadic_16bit_chain=dict(transitions=dyadic_transitions,stationary=pi_d.tolist(),
            entropy_rate_bits=dyadic_entropy,
            max_transition_difference=max(abs(t['probability']-u['numerator']/65536)
                                         for t,u in zip(transitions,dyadic_transitions)),
            sign_flip_invariance_exact=True,
            maximum_branch_probability_numerator=max(n for row in dyadic_rows if len(row)==2 for n in row),
            maximum_branch_probability_denominator=65536,
            any_two_steps_contain_branch=max(forced_chain_lengths,default=0)==1,
            raw_secret_uniform_bits_per_clock_tick=16,
            exact_support_preserved=True,exact_maximum_entropy=False),
        physical_trajectories=0,noise_samples=0,uniform_positive_per_step_conditional_entropy=False)
    (HERE/'graph_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['source','input_word_count','strong_component_count','cyclic_component_count',
        'selected_component_edges','period','rho','topological_entropy_bits','collatz_width_float',
        'entropy_rate_bits','next_plus_probability','next_sign_mean','forced_next_state_count',
        'stationary_forced_state_probability','worst_state_next_prediction','average_state_next_prediction']}
        |{'selected_states':len(states)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
