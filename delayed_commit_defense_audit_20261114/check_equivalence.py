"""First-gate exact identities only; not a candidate/B4 algorithm experiment.

No attack response optimization, detector, simulation, DARE or confirmation-delay
scan is implemented. Matrix symbols are noncommutative; dimensions stay two.
"""
from pathlib import Path
import hashlib,json
import sympy as s
H=Path(__file__).resolve().parent; P=H.parent
n=2
F=[s.MatrixSymbol('F'+str(i),n,n)for i in range(1,4)]
c=[s.MatrixSymbol('c'+str(i),n,1)for i in range(1,4)]
x=s.MatrixSymbol('checkpoint',n,1); delta=s.MatrixSymbol('checkpoint_error',n,1)
# Chronological substitution and independently unrolled affine composition.
chronological=F[2]*(F[1]*(F[0]*x+c[0])+c[1])+c[2]
unrolled=F[2]*F[1]*F[0]*x+F[2]*F[1]*c[0]+F[2]*c[1]+c[2]
assert s.expand(chronological-unrolled)==s.ZeroMatrix(n,1)
dirty=F[2]*(F[1]*(F[0]*(x+delta)+c[0])+c[1])+c[2]
assert s.expand(dirty-chronological-F[2]*F[1]*F[0]*delta)==s.ZeroMatrix(n,1)
A=s.MatrixSymbol('A',n,n); K=s.MatrixSymbol('K',n,n); G=s.MatrixSymbol('G',n,n)
Hi=s.MatrixSymbol('Hi',n,n);Hj=s.MatrixSymbol('Hj',n,n)
y=s.MatrixSymbol('local_y',n,1);v=s.MatrixSymbol('neighbor_y',n,1)
raw=A*x+K*(y-Hi*A*x)+G*(v-Hj*A*x)
affine=(A-K*Hi*A-G*Hj*A)*x+K*y+G*v
assert s.expand(raw-affine)==s.ZeroMatrix(n,1)
# Removing a whole neighbor innovation changes the state coefficient.
removed=A*x+K*(y-Hi*A*x)
assert s.expand(removed-raw+G*(v-Hj*A*x))==s.ZeroMatrix(n,1)
p,q=s.symbols('phi1 central_probability',positive=True)
truncated_U_second=2*(1-2*p/q)
cross=s.simplify((truncated_U_second-2)/4)
assert cross==-p/q
model=P/'original_estimator_gate_20261114/five_node_aggregate_model.json'
d=json.loads(model.read_text(encoding='utf-8'))
assert len(d['models'])==2
out=dict(baseline='917a46804c1eabe08d94ec54579acae6f68ccfbb',
 model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
 scope='exact first-gate semantic identities; valid for arbitrary matrices and labels, stable or unstable',
 identities=dict(noncommutative_affine_composition=True,
 dirty_checkpoint_not_removed_by_replay=True,original_common_G_affine_form=True,
 dropping_entire_innovation_changes_state_coefficient=True,
 future_acceptance_cross_moment='-phi(1)/(2 Phi(1)-1), nonzero'),
 retained_problem=dict(main_attack='a_t=H_j A^t xi, t=0,...,L-1',L=10,E=10,D=.1,
 estimator='original common-G neighbor-sum; no new gains'),
 not_a_candidate_algorithm_test=True,no_physical_samples=True,no_false_alarm_or_risk_experiment=True,
 delay_scan_performed=False,gate2_entered=False,gate3_entered=False,gate4_entered=False,
 theoretical_result='same checkpoint/update/labels/commit schedule implies same output path; all-accept no-smoothing implies historical baseline value',
 limitation='does not prove all different policies equal or all delayed-security theorems are old')
(H/'equivalence_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
