"""Independent elementary resource/spatial bounds; no new watermark design.

Read-only inputs: original/phase TeX, aggregate model, unique reference library.
All generated output is confined to this new study directory. No prior research
script is imported or executed, and no physical Monte Carlo is performed.
"""
from pathlib import Path
import hashlib
import json
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P = ROOT / '研究规划/第一篇小论文_重新投稿修订_20261102'
R = sp.Rational

def ser(matrix):
    return [[str(v) for v in row] for row in matrix.tolist()]

def energy(M, Sigma):
    C = M.inv().T * M.inv()
    return sp.simplify(sp.trace(Sigma * C))

inputs = [
    ROOT/'研究规划/第一篇小论文/secure distributed safety.tex',
    P/'sections/02_model_detector.tex', P/'sections/04_watermark.tex',
    P/'original_estimator_gate_20261114/five_node_aggregate_model.json',
    P/'original_estimator_gate_20261114/报告一_原稿估计器与后续KF的严格对照.md',
    P/'web_revision_control_20261112/29条意见_本机状态与剩余验收.md',
    ROOT/'参考文献/文献库.json',
    P/'watermark_boundary_final_20261114/theorem_novelty_evidence.json',
]
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in inputs}

# κ > 1 admits an anisotropic deterministic decoder at exactly healthy energy.
Sigma = sp.diag(4, 1)
M = sp.diag(2, R(1, 2))
assert energy(M, Sigma) == sp.trace(Sigma) == 5
assert M.T*M != sp.eye(2)
anisotropic = dict(Sigma=ser(Sigma), M=ser(M), kappa=2,
    encoded_energy=str(energy(M, Sigma)), healthy_energy='5',
    decoder_singular_values=['2','1/2'],
    scope='counterexample to orthogonality without contraction; not a candidate algorithm')

# κ=1 and positive energy slack allow contraction in one direction, at cost.
M_slack = sp.diag(1, R(1, 2))
assert energy(M_slack, Sigma) == 8

# Positive definiteness is essential: zero-energy coordinates are unconstrained.
Sigma_singular = sp.diag(1, 0)
assert energy(M_slack, Sigma_singular) == sp.trace(Sigma_singular) == 1

# Equality for every fixed direction in expectation does not imply each matrix
# is orthogonal unless the decoder is a contraction.
M1, M2 = sp.sqrt(2)*sp.eye(2), sp.sqrt(R(2,3))*sp.eye(2)
mean_inverse_gram = sp.simplify((M1.inv().T*M1.inv()+M2.inv().T*M2.inv())/2)
assert mean_inverse_gram == sp.eye(2)

# Current-data-dependent decoder can preserve the realized y direction while
# contracting its orthogonal complement: prohibited by the original model.
x, y = sp.symbols('x y', real=True)
v = sp.Matrix([x, y])
projector = v*v.T/(x*x+y*y)
M_data = projector + R(1,2)*(sp.eye(2)-projector)
assert sp.simplify(M_data*v-v) == sp.zeros(2,1)
assert sp.simplify(M_data.det()) == R(1,2)

# Exact spatial covariance in the declared supplementary model (edge 0 -> 1).
Hi = sp.diag(R(11,10), R(6,5))
Hj = sp.eye(2)
Ri = Rj = R(1,2)*sp.eye(2)
C = Hj*Hi.inv()
V = sp.simplify(Rj + C*Ri*C.T)
assert V == sp.diag(R(221,242),R(61,72))
condition = sp.simplify(V[0,0]/V[1,1])
assert condition == R(7956,7381)

# Signs preserve the spatial Mahalanobis norm for this diagonal covariance.
sign_checks=[]
for s0 in [-1,1]:
    for s1 in [-1,1]:
        S = sp.diag(s0,s1)
        assert S.T*V.inv()*S == V.inv()
        sign_checks.append([s0,s1])

Au=sp.Matrix([[R(108,100),R(5,100)],[R(1,100),R(105,100)]])
As=sp.Matrix([[R(8,10),R(5,100)],[R(1,100),R(9,10)]])
assert Au.det()!=0 and As.det()!=0

library = json.loads((ROOT/'参考文献/文献库.json').read_text(encoding='utf-8-sig'))
records={rec['id']:rec for rec in library['records']}
prior_literature=[]
for rid in ['INTRO06','INTRO05','FPN04','FPN16','FPN06']:
    item=records[rid]
    prior_literature.append(dict(id=rid,citation_key=item['citation_key'],
        doi=item['doi'],read_scope_reused=item.get('read_scope',''),
        new_full_text_read_this_subtask=False,
        theorem_equivalence_audit_this_subtask=False))

out = dict(
    date='2026-11-14',
    scope='first-stage resource/spatial mathematical gate, not watermark mechanism design',
    input_sha256=hashes,
    strict_results=[
      'Independent/predictable M with positive conditional physical second moment: Penc>=tr(Sigma_y)/kappa^2.',
      'Equality at that lower bound forces M=kappa*O almost surely when Sigma_y is positive definite.',
      'kappa=1 and no increase over healthy encoded energy force orthogonality; anisotropic Sigma_y does not evade this.',
      'Orthogonality does not exclude nonuniform distributions on O(2), deterministic rotations or directional orientation effects.',
      'At same time, spatial consistency cancels true state exactly if H_j=C H_i and local measurements are trustworthy.',
      'With fixed independent original attacks and independent sensor noises, spatial nominal covariance is R_j+C R_i C^T.',
      'The instantaneous decoder/sign cannot be inferred to improve spatial test power solely by breaking a temporal parity kernel.',
      'For known onset and independent-time spatial residuals, cumulative spatial noncentrality controls accumulated bias through a standard finite operator Gramian.',
    ],
    finite_exact_checks=dict(
      anisotropic_same_energy_kappa_gt_one=anisotropic,
      contraction_energy_slack=dict(M=ser(M_slack),Sigma=ser(Sigma),Penc='8',Phealthy='5'),
      singular_Sigma_counterexample=dict(M=ser(M_slack),Sigma=ser(Sigma_singular),Penc='1',Phealthy='1'),
      directionwise_mean_equality_counterexample=dict(
        M1=ser(M1),M2=ser(M2),probabilities=['1/2','1/2'],
        expected_inverse_gram=ser(mean_inverse_gram),kappa='sqrt(2)'),
      current_data_dependent_exception=dict(
        decoder='P_y+(1/2)(I-P_y), P_y=y y^T/(y^T y)',
        acts_as_identity_on_realized_y=True,determinant='1/2',
        scope='not allowed by original independent-secret model'),
      spatial=dict(C=ser(C),covariance=ser(V),condition_exact=str(condition),
        condition_float=float(condition),coordinate_signs_preserving_ncp=sign_checks,
        Au_det=str(Au.det()),As_det=str(As.det()),
        meaning='all prescribed nonzero autonomous directions are spatially nonblind; not a probability/delay guarantee')
    ),
    evidence_classes=dict(
      algebraic_propositions='strict matrix proofs in bounds_review.md',
      rational_examples='exact finite SymPy checks, no sampling',
      power_risk_claims='not computed in this subtask; parent performs actual power comparison',
      literature='prior designated-page evidence only; second-stage novelty audit not started',
      physical_covariance='original R_j positive definite guarantees positivity; no missing initial state prior invented'),
    prior_literature_evidence=prior_literature,
    protected_files_written=False,new_algorithm=False,new_library=False,
)
(HERE/'bounds_review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_spatial_covariance=ser(V),condition_number=float(condition),
    finite_exact_checks_passed=True,source_files=len(inputs)),ensure_ascii=False))
