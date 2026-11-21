"""Exact second-moment and Gaussian-equivalent-KL reproduction for Paper 1.

Dependencies: numpy, scipy.
The script generates generic observable (A,H) cases, constructs a parity
basis, and evaluates the exact covariance under the angular-coupled
rank-one multiplicative watermark. Monte Carlo is not used for the
reported covariance/KL values.
"""
import json
import numpy as np
import scipy.linalg as la
from scipy.special import hyp1f1
from scipy.stats import t as student_t

SIGMA_B2 = 0.5
RHO = 0.7
KAPPAS = [0.0, 0.5, 1.0, 2.0, 4.0]
L = 5
ENERGY = 10.0
N_CASES = 100


def sphere_constants(m, kappa, rho=RHO):
    z = float(hyp1f1(0.5, m / 2.0, -kappa))
    tilted_c2 = float((1.0 / m) * hyp1f1(1.5, m / 2.0 + 1.0, -kappa) / z)
    c1 = (1.0 - rho) / m + rho * tilted_c2
    eig2 = (c1 - 1.0 / m) / (1.0 - 1.0 / m)
    return z, c1, eig2


def mutual_information_bound(m, kappa, rho=RHO, order=600):
    # Gauss-Legendre quadrature for C~Beta(1/2,(m-1)/2).
    from scipy.special import roots_legendre, beta as beta_fn
    x, w = roots_legendre(order)
    c = (x + 1.0) / 2.0
    w = w / 2.0
    a, b = 0.5, (m - 1.0) / 2.0
    density = c**(a-1) * (1-c)**(b-1) / beta_fn(a, b)
    z, _, _ = sphere_constants(m, kappa, rho)
    g = (1.0-rho) + rho*np.exp(-kappa*c)/z
    return float(np.sum(w*density*g*np.log(g)))


def parity_model(A, H, L):
    n, m = A.shape[0], H.shape[0]
    H0 = np.vstack([H @ np.linalg.matrix_power(A, ell) for ell in range(L)])
    Z = la.null_space(H0.T).T
    Hw = np.zeros((L*m, (L-1)*n))
    for ell in range(L):
        for j in range(L-1):
            if ell > j:
                Hw[ell*m:(ell+1)*m, j*n:(j+1)*n] = H @ np.linalg.matrix_power(A, ell-j-1)
    E = Hw @ np.kron(np.eye(L-1), np.eye(n)) @ Hw.T + np.eye(L*m)
    Theta = Z @ E @ Z.T
    return H0, Z, Theta


def attack_vector(A, H, xi, L, energy=ENERGY):
    d = np.concatenate([H @ np.linalg.matrix_power(A, ell) @ xi for ell in range(L)])
    return d * np.sqrt(energy) / np.linalg.norm(d)


def cross_block(delta_l, delta_r, m, h, kappa, rho=RHO):
    _, c1, eig2 = sphere_constants(m, kappa, rho)
    c_h = 1.0/m + (1.0-1.0/m)*(eig2**h)
    b_h = (c_h-1.0/m)/((m-1.0)*(m+2.0))
    return (SIGMA_B2**2)*(m**2)*b_h * (
        np.dot(delta_l, delta_r)*np.eye(m)
        + np.outer(delta_r, delta_l)
        - (2.0/m)*np.outer(delta_l, delta_r)
    )


def exact_residual_covariance(Z, d, m, L, kappa, rho=RHO):
    ds = d.reshape(L, m)
    C = np.zeros((Z.shape[0], Z.shape[0]))
    for ell in range(L):
        Zl = Z[:, ell*m:(ell+1)*m]
        dl = ds[ell]
        Cll = SIGMA_B2**2 * (np.outer(dl, dl) + np.dot(dl, dl)*np.eye(m))
        C += Zl @ Cll @ Zl.T
        for r in range(ell+1, L):
            Zr = Z[:, r*m:(r+1)*m]
            Clr = cross_block(dl, ds[r], m, r-ell, kappa, rho)
            C += Zl @ Clr @ Zr.T + Zr @ Clr.T @ Zl.T
    return C


def gaussian_equivalent_kl(C, Theta):
    # KL(N(0,Theta+C) || N(0,Theta)); lower-bounds actual KL for
    # a zero-mean non-Gaussian residual with the same covariance.
    whitened = la.solve(Theta, C, assume_a="pos")
    sign0, logdet0 = np.linalg.slogdet(Theta)
    sign1, logdet1 = np.linalg.slogdet(Theta+C)
    assert sign0 > 0 and sign1 > 0
    return float(0.5*(np.trace(whitened) + logdet0 - logdet1))


def generate_case(n, m, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n,n)))
    eig = np.linspace(1.03, 1.09, n)
    A = Q @ np.diag(eig) @ Q.T
    for _ in range(1000):
        H = rng.normal(size=(m,n))
        H0 = np.vstack([H @ np.linalg.matrix_power(A, ell) for ell in range(L)])
        if np.linalg.matrix_rank(H0) == n:
            break
    else:
        raise RuntimeError("Could not generate observable case")
    xi = rng.normal(size=n)
    return A, H, xi


def run():
    dimensions = [(2,2),(4,2),(4,3),(4,4),(6,2),(6,3),(6,4)]
    results = {"settings": {
        "L":L, "energy":ENERGY, "rho":RHO, "kappas":KAPPAS,
        "cases_per_dimension":N_CASES,
        "Q_process":"I_n", "R_measurement":"I_m",
        "attack":"H_0,s xi, normalized to fixed energy",
        "metric":"Gaussian-equivalent KL from exact residual covariance"
    }, "results":[]}
    for n,m in dimensions:
        values = []
        for j in range(N_CASES):
            A,H,xi = generate_case(n,m,20000+100*m+10*n+j)
            H0,Z,Theta = parity_model(A,H,L)
            d = attack_vector(A,H,xi,L)
            assert np.linalg.norm(Z@d) < 1e-8
            row = []
            for kappa in KAPPAS:
                C = exact_residual_covariance(Z,d,m,L,kappa,RHO)
                row.append(gaussian_equivalent_kl(C,Theta))
            values.append(row)
        V = np.asarray(values)
        means = V.mean(axis=0)
        entry = {"n":n,"m":m,"means":means.tolist(),"paired_differences":{}}
        for j,kappa in enumerate(KAPPAS[1:],start=1):
            diff = V[:,j]-V[:,0]
            half = student_t.ppf(0.975,N_CASES-1)*diff.std(ddof=1)/np.sqrt(N_CASES)
            entry["paired_differences"][str(kappa)] = {
                "mean":float(diff.mean()),
                "ci95":[float(diff.mean()-half),float(diff.mean()+half)],
                "fraction_positive":float(np.mean(diff>0))
            }
        results["results"].append(entry)
    results["leakage"] = {}
    for m in [2,3,4]:
        for kappa in KAPPAS[1:]:
            z,c1,eig2 = sphere_constants(m,kappa,RHO)
            results["leakage"][f"m{m}_k{kappa}"] = {
                "mean_squared_cosine":c1,
                "iid_reference":1.0/m,
                "pair_mutual_information_upper_bound":mutual_information_bound(m,kappa,RHO),
                "window_L5_total_correlation_upper_bound":(L-1)*mutual_information_bound(m,kappa,RHO)
            }
    with open("angular_coupled_watermark_results_recomputed.json","w") as f:
        json.dump(results,f,indent=2)
    print("Wrote angular_coupled_watermark_results_recomputed.json")


if __name__ == "__main__":
    run()
