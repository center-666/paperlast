"""Exact small algebraic checks for the review-only theory corrections.

No physical simulation, no old entrypoint and no search for an algorithm.
These examples do not replace the frozen five-node implementation tests.
"""
from pathlib import Path
import hashlib,json
import sympy as s
H=Path(__file__).resolve().parent
def zero(M):return all(s.simplify(x)==0 for x in M)
def main():
 checks=[]
 A=s.Matrix([[0]]);Hj=s.Matrix([[1]]);O=Hj.col_join(Hj*A);Z=s.Matrix([[0,1]])
 xi=s.Matrix([s.Rational(2)]);delta=O*xi;D=s.diag(2*delta[0]**2,2*delta[1]**2)
 assert O.rank()==1 and Z.rank()==1 and D!=s.zeros(2) and zero(Z*D*Z.T)
 checks.append({'id':'TP1','claim':'locally observable A=0 counterexample to full-row-rank strict positivity','O':str(O),'Z':str(Z),'Delta':str(delta),'c1':'0','exact':True})
 A=s.Matrix([[1,s.Rational(1,5)],[0,s.Rational(9,10)]])
 J=s.Matrix([[s.Rational(3,10),s.Rational(1,10)],[s.Rational(1,20),s.Rational(2,5)]])
 G=s.Matrix([[s.Rational(1,5),s.Rational(1,10)],[0,s.Rational(1,4)]])
 F=J*A;Phi=A*J
 assert F!=Phi and F.charpoly().as_expr()==Phi.charpoly().as_expr()
 a0=s.Matrix([2,-1]);a1=s.Matrix([1,3]);b0=G*a0;p1=A*b0;b1=J*p1+G*a1
 assert b1==F*b0+G*a1 and A*b1==Phi*p1+A*G*a1
 checks.append({'id':'TP2','claim':'same-time posterior/prediction identity; F and Phi differ but have equal characteristic polynomial','F':str(F),'Phi':str(Phi),'posterior_0':str(b0),'prediction_1':str(p1),'exact':True})
 # Independent fourth-moment expansion, not a call to the old watermark lemma.
 x,y,a,b,q,lam=s.symbols('x y a b q lam',real=True)
 def emonom(n):return 0 if n%2 else s.factorial2(n-1)*q**(n//2) if n else 1
 def expectation(poly):return s.expand(sum(c*emonom(i)*emonom(j) for (i,j),c in s.Poly(s.expand(poly),x,y).terms()))
 beta=s.Matrix([x,y]);v=s.Matrix([a,b]);u=beta*(beta.T*v)[0]
 mean=u.applyfunc(expectation);second=(u*u.T).applyfunc(expectation);cov=s.simplify(second-mean*mean.T)
 expected=q*q*(v*v.T+(v.dot(v))*s.eye(2))
 assert zero(mean-q*v) and zero(cov-expected)
 mv=lam*v+u;decoded=expectation(mv.dot(mv))
 assert s.expand(decoded-(lam*lam+2*lam*q+4*q*q)*(a*a+b*b))==0
 checks.append({'id':'TP3','claim':'independent m=2 Gaussian polynomial moment expansion and decoded-energy factor','covariance':str(cov),'decoded_energy':str(decoded),'exact':True})
 # Structured D trace identity, including a rank-deficient but nonzero time block.
 c,d=s.symbols('c d',real=True);z0=s.Matrix([[1,0],[0,0]]);z1=s.Matrix([[0,1],[1,0]])
 th=s.diag(2,3);w=th.inv();v0=s.Matrix([a,b]);v1=s.Matrix([c,d])
 ZZ=z0.row_join(z1);DD=s.diag(v0*v0.T+v0.dot(v0)*s.eye(2),v1*v1.T+v1.dot(v1)*s.eye(2))
 lhs=s.trace(w*ZZ*DD*ZZ.T)
 rhs=sum((v.T*z.T*w*z*v)[0]+v.dot(v)*s.trace(w*z*z.T) for z,v in [(z0,v0),(z1,v1)])
 assert s.expand(lhs-rhs)==0
 checks.append({'id':'TP4','claim':'structured time-block trace identity; full column rank per block not required','trace':str(s.expand(lhs)),'exact':True})
 shifts=[]
 for k in range(3):
  window=s.Matrix([1 if 0<=t<2 else 0 for t in [k-1,k]])
  shifts.append((s.Matrix([[-1,1]])*window)[0])
 assert shifts==[1,0,-1]
 checks.append({'id':'TP5','claim':'finite support autonomous attack has exposed onset/termination despite zero interior shift','L':2,'s':1,'shifts_at_k_0_1_2':[int(v) for v in shifts],'exact':True})
 A=s.Matrix([[1,1],[0,1]]);J=s.zeros(2);G=s.eye(2);xi=s.Matrix([0,1]);F=J*A
 for t in range(6):
  assert A**t*xi==s.Matrix([t,1])
  if t>=1:assert A*(A**(t-1)*xi)==s.Matrix([t,1])
 last=A*xi;after=F*last;assert after==s.zeros(2,1)
 checks.append({'id':'TP6','claim':'unit-circle Jordan excitation and distinct finite-stop decay','posterior_at_t':'(t,1)^T','prediction_at_t_ge_1':'(t,1)^T','first_posterior_after_stop':'(0,0)^T','exact':True})
 t=s.symbols('t',nonnegative=True);f=t-s.log(1+t)
 assert s.simplify(s.diff(f,t)-t/(1+t))==0
 checks.append({'id':'TP7','claim':'KL trace-bound inverse uses increasing f; q^-2 is sigma_beta^-4','f_derivative':str(s.diff(f,t)),'exact':True})
 out={'date':'2026-11-15','purpose':'review-only mathematical corrections, not new method or five-node performance proof','checks':checks,'exact_checks_passed':len(checks),'physical_samples':0,'Monte_Carlo':0,'old_scripts_executed':False,'review_tex_sha256':hashlib.sha256((H/'theory_review.tex').read_bytes()).hexdigest()}
 (H/'theory_patch_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8');print(json.dumps({'exact_checks_passed':len(checks),'physical_samples':0},ensure_ascii=False))
if __name__=='__main__':main()
