"""Explicit conditional instance of original summed-neighbor estimator.

Not a reproduction of original 50-node figures. No original script imported.
DARE values are floating diagnostics; stability follows analytically from
positive Q/R, observable aggregate measurement and the stabilizing solution.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import sympy as sp
from scipy.linalg import block_diag,solve_discrete_are
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
S=sp.Rational
Aexact={'unstable':sp.Matrix([[S(108,100),S(5,100)],[S(1,100),S(105,100)]]),
        'stable':sp.Matrix([[S(80,100),S(5,100)],[S(1,100),S(90,100)]])}
Hexact=[sp.diag(S(a,10),S(b,10)) for a,b in [(10,10),(11,12),(12,14),(13,11),(14,13)]]
neighbors=[[1,4],[0,2],[1,3],[2,4],[3,0]]
Q=np.eye(2);Rs=[.5*np.eye(2) for _ in range(5)]
H=[np.array(h,float) for h in Hexact]
rows={}
for name,Ax in Aexact.items():
    A=np.array(Ax,float);nodes=[]
    for i,N in enumerate(neighbors):
        Hsum=sum((H[j] for j in N),start=np.zeros((2,2)))
        C=np.vstack([H[i],Hsum]);R=block_diag(Rs[i],sum((Rs[j]for j in N),start=np.zeros((2,2))))
        prior=solve_discrete_are(A.T,C.T,Q,R)
        L=np.linalg.solve(C@prior@C.T+R,C@prior).T
        K=L[:,:2];G=L[:,2:]
        T=np.eye(2)-L@C
        post=T@prior@T.T+L@R@L.T
        F=T@A;Phi=A@T;Psi=A@G
        residual=prior-(A@post@A.T+Q)
        Cstack=np.vstack([H[j]for j in [i]+N]);Rstack=block_diag(*(Rs[j]for j in [i]+N))
        Ps=solve_discrete_are(A.T,Cstack.T,Q,Rstack)
        Ls=np.linalg.solve(Cstack@Ps@Cstack.T+Rstack,Cstack@Ps).T
        Ts=np.eye(2)-Ls@Cstack
        Posts=Ts@Ps@Ts.T+Ls@Rstack@Ls.T
        # Exact equal-noise aggregation loss for this declared instance.
        hs=sum((Hexact[j]for j in N),start=sp.zeros(2))
        Jloss=sum((2*Hexact[j].T*Hexact[j]for j in N),start=sp.zeros(2))-hs.T*hs
        assert Jloss==Jloss.T
        assert Jloss[0,0]>=0 and Jloss[1,1]>=0 and Jloss.det()>=0
        assert max(abs(np.linalg.eigvals(F)))<1
        assert np.linalg.norm(Phi-A@F@np.linalg.inv(A))<1e-12
        assert np.max(abs(residual))<1e-12
        assert np.linalg.norm(post-np.linalg.inv(np.linalg.inv(prior)+C.T@np.linalg.solve(R,C)))<1e-12
        # Direct random-free equation crosscheck for posterior/predictor injection.
        b=np.array([.37,-.2]);a=np.array([.1,-.6])
        assert np.max(abs((T@A@b+G@a)-(A@b-K@H[i]@A@b-G@Hsum@A@b+G@a)))<1e-15
        nodes.append(dict(node=i,neighbors=N,aggregate_C=C.tolist(),aggregate_R=R.tolist(),
            prior_covariance=prior.tolist(),posterior_covariance=post.tolist(),K=K.tolist(),G=G.tolist(),
            posterior_F=F.tolist(),posterior_attack_G=G.tolist(),predictor_Phi=Phi.tolist(),predictor_Psi=Psi.tolist(),
            posterior_spectral_radius=float(max(abs(np.linalg.eigvals(F)))),
            DARE_residual_max=float(np.max(abs(residual))),
            local_observability_rank=int(sp.Matrix.vstack(Hexact[i],Hexact[i]*Ax).rank()),
            aggregate_observability_rank=int(sp.Matrix.vstack(sp.Matrix(C),sp.Matrix(C)*Ax).rank()),
            aggregation_information_loss_exact=[[str(v)for v in row]for row in Jloss.tolist()],
            separate_stack_posterior_trace=float(np.trace(Posts)),aggregate_posterior_trace=float(np.trace(post)),
            covariance_loss_min_eigenvalue=float(np.linalg.eigvalsh(post-Posts).min()),
            separate_neighbor_gain_blocks=[Ls[:,2:4].tolist(),Ls[:,4:6].tolist()],
            separate_gain_block_difference_norm=float(np.linalg.norm(Ls[:,2:4]-Ls[:,4:6]))))
    rows[name]=dict(A=A.tolist(),nodes=nodes)
sources=[ROOT/'研究规划/第一篇小论文/secure distributed safety.tex',
         ROOT/'第二篇小论文/韧性恢复/references/niu2023/paper.txt',
         HERE.parent/'异构模型与主动可信验证_设计及最小核验_20261103.md',
         ROOT/'硕士毕业论文/完整论文准备_20261029/全篇复审_20261029/复现追溯_20261030/第三章实验复现追溯与待修订项.md']
out=dict(date='2026-11-14',status='explicit supplementary conditional instance; not original 50-node reproduction',
    estimator='original equations(3)(4): neighbor measurements summed before common G',
    gain_design='explicitly supplied Niu2023 equations(9)-(13), not printed in protected original TeX',
    n=2,m=2,nodes=5,H=[h.tolist()for h in H],Q=Q.tolist(),R=[r.tolist()for r in Rs],
    graph='declared undirected five-cycle; not original 50-node graph',neighbors=neighbors,
    attacked_link=dict(sender=0,receiver=1),
    gain_regime='steady-state fixed nominal gains; no alarm gating, dropouts or covariance reinitialization',
    initial_attack_bias='zero',models=rows,
    exact_statements=['all local H_i invertible over rationals; original local observability satisfied',
                      'aggregate measurement compression preserves original common-G architecture; not sufficient for full neighbor data',
                      'DARE stabilizing solution exists; Q positive and C full-column rank',
                      'Joseph identity yields Schur predictor A(I-LC); posterior F similar since A invertible',
                      'no cross-node bias coupling because raw measurements, not estimates, are exchanged'],
    numeric_statements='SciPy displayed values are diagnostics; separate aggregate_gap_certificate.json encloses target exact DARE and selected gap/norm claims',
    original_experiment_reproduced=False,old_four_dimensional_F_g_reused=False,
    physical_samples=0,
    sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)for p in sources])
(HERE/'five_node_aggregate_model.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({name:[dict(node=n['node'],rho=n['posterior_spectral_radius'],
      trace=n['aggregate_posterior_trace'],stacktrace=n['separate_stack_posterior_trace'])
      for n in v['nodes']]for name,v in rows.items()},ensure_ascii=False,indent=2))
