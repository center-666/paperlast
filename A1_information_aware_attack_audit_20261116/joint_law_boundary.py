"""Standard Gaussian cross-information boundary, not a new detector."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import sympy as s
H=Path(__file__).resolve().parent;P=H.parent
def sm(x):return s.Matrix([[s.Rational(str(v)) for v in row] for row in x])
def main():
    src=json.loads((P/'common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json').read_text('utf8'))
    out={}
    for name in ('unstable','stable'):
        A=sm(src['models'][name]['A']);Q=sm(src['Q']);Hi=sm(src['H'][1]);Hj=sm(src['H'][0])
        T=s.zeros(10)
        for t in range(5):
            T[2*t:2*t+2,:2]=A**t
            for r in range(t):T[2*t:2*t+2,2+2*r:4+2*r]=A**(t-1-r)
        statecov=T*s.diag(s.eye(2),Q,Q,Q,Q)*T.T
        cross=s.diag(*([Hi]*5))*statecov*s.diag(*([Hj.T]*5))
        assert T.det()==1 and cross.rank()==10
        assert cross.det()==Hi.det()**5*Hj.det()**5*Q.det()**4
        out[name]=dict(prefix=5,raw_target_dimension=10,exact_cross_rank=cross.rank(),
          determinant=str(cross.det()),state_covariance_determinant=str(statecov.det()),
          general_prefix_argument='for any N, state map is unit block triangular; x0 covariance I and Q positive; invertible H0/H1 imply cross rank2N',
          scope='full finite raw trusted+target history, known Gaussian initial law; equality also joint with secret M',
          not_a_claim_about='sliding parity statistic, finite power, alarm before danger, or arbitrary partial-observation networks')
    result=dict(date='2026-11-17',models=out,
      theorem='For joint Gaussian X,Y with Cov(X,Y) full column rank, any g=f(Y,U) with independent U preserving the joint law of(X,g) equalsY a.s. Conditional on independent secret matrices, this applies to all target-q causal policies.',
      proof='L=Cov(X,Y)Cov(Y)^-1 has full column rank. E[XgT]=L E[YgT]=Cov(X,Y)=L Cov(Y) implies E[YgT]=Cov(Y). Equal second moments give E[(g-Y)(g-Y)T]=0.',
      relation_to_prior='standard Gaussian regression/moment argument and block lifting of the safe-suspicious cross-covariance obstruction; closest Wei eq20,81 and Remark10; not an independent central novelty',
      new_detector=False,new_physical_paths=0)
    (H/'joint_law_boundary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps({'exact_cross_ranks':{n:z['exact_cross_rank'] for n,z in out.items()},'novelty_claim':False}))
if __name__=='__main__':main()
