"""Exact finite-dimensional parity certificates and counterexamples."""
from pathlib import Path
import json,sys,itertools
sys.dont_write_bytecode=True
import sympy as s
H=Path(__file__).resolve().parent;P=H.parent
def sm(a):return s.Matrix([[s.Rational(str(v)) for v in row] for row in a])
def col(rows,n=2):return s.Matrix.vstack(*rows) if rows else s.zeros(0,n)
def check(Oj,Oc,Q=None):
    n=Oj.cols;Q=Oj if Q is None else Q
    O=Oj.col_join(Oc);D=Q.col_join(s.zeros(Oc.rows,Q.cols))
    Z=col([v.T for v in O.T.nullspace()],O.rows);R=Z*D
    return dict(rank_attack=Q.rank(),rank_residual=R.rank(),
      kernel_condition=bool(Oc.rank()==Oc.col_join(Oj).rank()),
      all_effective_directions_visible=bool(R.rank()==Q.rank()),
      invisible_initial_basis=[str(v) for v in R.nullspace()],left_null_rows=Z.rows)
def main():
    source=json.loads((P/'common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json').read_text('utf8'));rows={}
    for name in ('unstable','stable'):
        A=sm(source['models'][name]['A']);Hj=sm(source['H'][0]);Hi=sm(source['H'][1]);Hh=sm(source['H'][2])
        Oj=col([Hj*A**t for t in range(5)]);local=col([Hi*A**t for t in range(5)])
        cs={'none':s.zeros(0,2),'local_trusted':local,'local_plus_other':local.col_join(col([Hh*A**t for t in range(5)])),
          'trusted_repeated_Hj':Oj}
        for c,label in ((s.Matrix([[1,0]]),'coordinate1'),(s.Matrix([[0,1]]),'coordinate2')):
            for times in ((0,),(4,),(0,4)):
                cs[label+'_'+str(times)]=col([c*A**t for t in times])
        out={label:check(Oj,Oc) for label,Oc in cs.items()}
        assert not out['none']['all_effective_directions_visible']
        assert all(out[z]['all_effective_directions_visible'] for z in ('local_trusted','local_plus_other','trusted_repeated_Hj'))
        assert not out['coordinate1_(0,)']['all_effective_directions_visible']
        assert out['coordinate1_(0, 4)']['all_effective_directions_visible'] and out['coordinate2_(0, 4)']['all_effective_directions_visible']
        for z in out.values():assert z['kernel_condition']==z['all_effective_directions_visible']
        # Every truncated support pattern is visible using local trusted data.
        mixed=[]
        for mask in itertools.product((0,1),repeat=5):
            Q=col([mask[t]*Hj*A**t for t in range(5)]);r=check(Oj,local,Q)
            assert r['all_effective_directions_visible'];mixed.append(dict(mask=list(mask),**r))
        node=source['models'][name]['nodes'][1]
        K=sm(node['K']);G=sm(node['G']);F=(s.eye(2)-K*Hi-G*(Hj+Hh))*A
        frob2=s.trace(F.T*F)
        assert frob2<s.Rational(1,25)
        rows[name]=dict(full_window=out,truncated_patterns_with_trusted_local=mixed,
          original_Hj_rank=Hj.rank(),trusted_current_Hi_rank=Hi.rank(),
          stop_tail_certificate=dict(F_exact_decimal_inputs=str(F),F_frobenius_squared=str(frob2),
             exact_less_than_1_over_25=True,consequence='after injection stops, ||b_stop+r||<=5**(-r)||b_stop|| for all integer r>=0; all-accept fixed gains only'))
    Oj=s.Matrix([[1,0]]);Oc=s.Matrix([[1,-1]]);R=s.Matrix([1,0]);v=s.Matrix([1,1])
    assert Oc*v==s.zeros(1,1) and Oj*v==Oj*R
    restricted=check(Oj,Oc);assert not restricted['all_effective_directions_visible']
    assert (Oc*R).rank()==(Oc*R).col_join(Oj*R).rank()
    mixed=check(s.ones(3,1),s.zeros(0,1),s.Matrix([1,0,0]));assert not mixed['kernel_condition'] and mixed['all_effective_directions_visible']
    duplicate=dict(A='1',H='1',T='2',corrupted_pair='[y+a;2(y+a)]',joint_residual_attack='0',
      protected_pre_tamper_copy='[y+a;y]',protected_difference='a',interpretation='the latter adds a trusted channel, not independent physical sensing; noise is singular and must not be falsely pseudowhitened away')
    out=dict(date='2026-11-16',models=rows,restricted_subspace_counterexample=dict(Oj=str(Oj),Oc=str(Oc),R=str(R),v=str(v),
      naive_restricted_kernel_condition=True,effective_allowed_attack_is_invisible=True),
      boundary_nonnecessity_counterexample=mixed,scalar_trusted_same_direction=dict(Oj='ones(3,1)',Oc='ones(3,1)',
      rank_joint_state=1,rank_attack_residual=1,interpretation='independent trusted repeat measures same state direction but exposes uncompensated tampering'),
      deterministic_duplicate=duplicate,all_checks_exact=True,new_physical_samples=0)
    (H/'structural_certificates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps(dict(all_checks_exact=True,existing_local_covers_effective_attack=True,mixed_support_checks=64)))
if __name__=='__main__':main()
