"""Analytic resource audit, not a watermark generator or parameter search.

Original m=2 M=I+beta beta', beta~N(0,I), q=lambda=1. Independent
rational Gauss/Radau Laguerre bounds certify its inverse-square moment.
All writes are local. No old script is imported or executed.
"""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, localcontext
import hashlib,json,math,sys
import mpmath as mp
import sympy as sp

HERE=Path(__file__).resolve().parent
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(30000)
mp.mp.dps=100

def laguerre_resolvent(n,radau=False):
    """e0'(I+2 J_n)^-1 e0, exactly rational.

    Gauss: diag(J)=(1,3,...,2n-1), offdiag=(1,...,n-1).
    Zero-node Radau changes its last diagonal to n-1. The two spectral
    quadratures respectively bound integral exp(-t)/(1+2t) dt below/above.
    """
    z=F(2*n-1 if radau else 4*n-1)
    for k in range(n-2,-1,-1):
        z=F(4*k+3)-F(4*(k+1)**2)/z
    return 1/z

def dec(x,digits=80):
    with localcontext()as ctx:
        ctx.prec=digits
        return str(Decimal(x.numerator)/Decimal(x.denominator))

def floor_grid(x,digits):
    z=10**digits
    return F((x.numerator*z)//x.denominator,z)

def ceil_grid(x,digits):
    z=10**digits
    return F(-((-x.numerator*z)//x.denominator),z)

def main():
    # X=||beta||^2~Exp(1/2). I1=E[1/(1+X)]=exp(.5)E1(.5)/2.
    # Integration by parts: E[1/(1+X)^2]=1/2-I1/2;
    # rotational symmetry: nu=1/2+1/2 E[1/(1+X)^2]=3/4-I1/4.
    nu=mp.mpf(3)/4-mp.exp(mp.mpf(1)/2)*mp.e1(mp.mpf(1)/2)/8
    kstar=1/mp.sqrt(nu)
    n=1024;g=laguerre_resolvent(n);r=laguerre_resolvent(n,True)
    assert 0<g<r<1
    nulo=F(3,4)-r/4;nuhi=F(3,4)-g/4
    assert nuhi-nulo<F(13,10**40)
    # Decimal output endpoints are rounded outward; proof uses fractions.
    lo36=floor_grid(nulo,36);hi36=ceil_grid(nuhi,36)
    kfloor=mp.floor(kstar*10**36)
    klo=F(int(kfloor),10**36);khi=klo+F(1,10**36)
    assert klo*klo*nuhi<1 and khi*khi*nulo>1
    assert mp.mpf(nulo.numerator)/nulo.denominator<nu<mp.mpf(nuhi.numerator)/nuhi.denominator
    reference=F(634636,10**6)
    assert reference<nulo
    rounded_k=1/mp.sqrt(mp.mpf(reference.numerator)/reference.denominator)

    # Symbolic check of the data-dependent equality counterexample. No new
    # physical kappa is selected; kappa and c remain symbolic (0<c<kappa).
    y=sp.Matrix([3,4]);u=y/5;I=sp.eye(2)
    kap,c=sp.symbols('kappa c',positive=True);proj=u*u.T
    M=kap*proj+c*(I-proj)
    assert sp.simplify(M*y-kap*y)==sp.zeros(2,1)
    assert sp.simplify(M.inv()*y-y/kap)==sp.zeros(2,1)
    assert sp.simplify(M.T*M-(kap**2*proj+c**2*(I-proj)))==sp.zeros(2)
    assert M.T*M!=kap**2*I
    delta,epsilon=sp.symbols('delta epsilon',positive=True)
    cost=kap**-2+delta*epsilon**2*(epsilon**-2-kap**-2)/2
    assert sp.simplify(cost-(kap**-2+delta/2-delta*epsilon**2/(2*kap**2)))==0

    # Exact no-detection implication counterexample: constant scalar decoding
    # leaves every autonomous internal parity sequence in the same nullspace.
    blind=[]
    for name,A in [('unstable',sp.Matrix([[sp.Rational(27,25),sp.Rational(1,20)],[sp.Rational(1,100),sp.Rational(21,20)]])),
                   ('stable',sp.Matrix([[sp.Rational(4,5),sp.Rational(1,20)],[sp.Rational(1,100),sp.Rational(9,10)]]))]:
        O=sp.Matrix.vstack(*(A**t for t in range(5)));Z=sp.zeros(8,10)
        for k in range(4):Z[2*k:2*k+2,2*k:2*k+2]=-A;Z[2*k:2*k+2,2*k+2:2*k+4]=I
        assert Z*O==sp.zeros(8,2)
        blind.append(dict(model=name,Z_O_exact_zero=True,arbitrary_constant_scalar_multiplier_still_blind=True))

    out=dict(date='2026-11-14',task='analytic resource audit only; no new algorithm',
        gaussian=dict(dimension=2,lambda_value=1,q_value=1,
            law='M=I+beta beta^T, beta~N(0,I2), independent of current y',
            eigenvalues='1 and 1+X, X~chi-square2 = Exponential(rate1/2)',
            inverse_square_mean='E[M^-T M^-1]=nu I',
            nu_exact='3/4-exp(1/2)*E1(1/2)/8',nu_high_precision=mp.nstr(nu,90),
            encoder_operator_peak='1, pointwise',decoder_operator_essential_supremum='infinity',
            decoder_peak_exceedance='Pr(||M||2>k)=exp(-(k-1)/2), k>=1',
            mean_normal_signal_energy_ratio_exact='nu'),
        rational_enclosure=dict(method='Gauss lower / zero-node Radau upper, Laguerre weight exp(-t), exact continued fractions',
            order=n,I1_gauss_lower=str(g),I1_radau_upper=str(r),nu_lower=str(nulo),nu_upper=str(nuhi),
            nu_interval_width=str(nuhi-nulo),nu_interval_width_decimal=dec(nuhi-nulo,30),
            nu_outward_decimal_lower=dec(lo36,50),nu_outward_decimal_upper=dec(hi36,50)),
        optimal_peak=dict(reference_power_ratio='nu, exact original Gaussian moment',
            theorem='kappa_min=nu^(-1/2) over nonsingular square encoders with Penc<=nu E||y||^2',
            kappa_star_high_precision=mp.nstr(kstar,90),
            certified_decimal_lower=dec(klo,50),certified_decimal_upper=dec(khi,50),
            equality='M=kappa_star O a.s., O orthogonal, if E[yy^T|M] positive definite a.s.; current-measurement independence plus positive definite second moment is sufficient',
            sufficiency='constant kappa_star I, or any exogenous orthogonal process times kappa_star',
            necessity_does_not_require_independence=True,
            positive_finite_reference_measurement_energy_required=True,
            no_claim_inside_fixed_rank_one_Gaussian_parametric_family=True),
        rounded_user_number=dict(literal_budget='.634636',difference_from_exact_nu=mp.nstr(nu-mp.mpf('.634636'),70),
            literal_budget_strictly_smaller_than_original=True,
            literal_budget_kappa_min_high_precision=mp.nstr(rounded_k,80),
            warning='If .634636 is an exact decimal constraint, the unscaled original Gaussian itself slightly exceeds it; use p^(-1/2), not exact-nu kappa_star.'),
        counterexamples=dict(full_support_y_is_not_enough=dict(
            general='For any full-support continuous y, u=y/||y||; M(y)=kappa uu^T+c(I-uu^T), 0<c<kappa; M^-1 y=y/kappa, but M is not scaled orthogonal.',
            conditional_second_moment='rank1 on the line selected by M; hence excluded by conditional positive definiteness',
            concrete_y=['3','4'],concrete_M=[[str(v)for v in row]for row in M.tolist()],
            symbolic_eigenvalues_of_MtM=['kappa^2','c^2, with 0<c<kappa'],
            not_a_proposed_coding_or_synchronization_mechanism=True),
            minimum_peak_does_not_ensure_detection=blind,
            decoder_cap_does_not_bound_encoder_peak=dict(
                scope='kappa>kappa_star; y second moment I2; M independent of y',
                delta='nu-kappa^(-2)>0',
                finite_family='M=kappa I with probability 1-p; M=diag(epsilon,kappa) with p=delta epsilon^2; epsilon<min(kappa,1/sqrt(delta))',
                power_ratio='kappa^(-2)+delta/2*(1-epsilon^2/kappa^2)<kappa^(-2)+delta/2<nu',
                encoder_peak='1/epsilon, arbitrarily large',condition_number_peak='kappa/epsilon, arbitrarily large',
                unbounded_distribution='epsilon_n=epsilon0 2^(-n), p_n=delta epsilon_n^2 2^(-n), n>=1; epsilon0<min(kappa,1/sqrt(delta)); remaining mass at kappa I',
                total_small_singular_value_probability='delta epsilon0^2/7<1',
                unbounded_distribution_power_ratio='kappa^(-2)+delta/2*(1-epsilon0^2/(7 kappa^2))<nu',
                unbounded_distribution_encoder_and_condition_essential_suprema='infinity',
                not_a_proposed_watermark_design=True)),
        resource_comparison=[
            dict(scheme='original Gaussian',power_ratio='nu',encoder_peak='1',decoder_peak='unbounded'),
            dict(scheme='existing scalar-scaled orthogonal baseline',power_ratio='nu, pointwise',encoder_peak='sqrt(nu)',decoder_peak='nu^(-1/2)'),
            dict(scheme='normalized Gaussian Mnorm=sqrt(nu) M',power_ratio='1',encoder_peak='nu^(-1/2)',decoder_peak='unbounded')],
        interpretation=['No global distribution or detection-risk optimum is proved.',
            'At equality, orientation, means and temporal correlations remain free, but singular values do not.',
            'Finite decoder peak eliminates Gaussian essential-unbounded amplification, already attained by simple scaled orthogonal/sign baselines.',
            'Larger kappa or extra signal power only creates possible singular-value freedom; no performance improvement follows.',
            'The prior unit-orthogonal >.48 oracle miss bound does not transfer to kappa_star O; its known-shift NCP is multiplied by kappa_star^2.',
            'kappa is an external physical budget; no new numeric kappa is selected.'],
        no_physical_samples=True,no_old_scripts_executed=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'resource_boundary_math.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(nu=out['gaussian']['nu_high_precision'],kappa_star=out['optimal_peak']['kappa_star_high_precision'],
        rational_nu_width=out['rational_enclosure']['nu_interval_width_decimal'],
        certified_kappa=[dec(klo,45),dec(khi,45)],rounded_budget_kappa=out['rounded_user_number']['literal_budget_kappa_min_high_precision']),indent=2))

if __name__=='__main__':main()
