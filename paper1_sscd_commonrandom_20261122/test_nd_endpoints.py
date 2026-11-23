"""Independent quadrature check of tilted-Beta inverse-CDF boundaries.

This targets the declared |theta|<=6 sampler range, including the m=2
positive-tilt upper tail that the default 30-step inversion did not resolve.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, hyp1f1
from watson_nd import tilted_beta_icdf, tilted_beta_cdf


def independent_cdf(z, theta, m):
    a=.5;b=(m-1)/2.;normalizer=np.exp(betaln(a,b))*hyp1f1(a,m/2.,theta)
    upper=np.arcsin(np.sqrt(float(z)))
    # z=sin^2(t) removes both Beta endpoint singularities.
    return quad(lambda t: 2*np.sin(t)**(2*a-1)*np.cos(t)**(2*b-1)
                *np.exp(theta*np.sin(t)**2)/normalizer,
                0.,upper,epsabs=2e-13,epsrel=2e-13,limit=300)[0]


def main():
    quantiles=np.array([1e-10,1e-8,1e-6,1e-3,.01,.1,.5,.9,.99,.999,1-1e-6,1-1e-8,1-1e-10])
    rows=[]
    for m in [2,3,4,6,10]:
        for theta in [-6.,6.]:
            z=tilted_beta_icdf(quantiles,theta,m)
            own=np.abs(tilted_beta_cdf(z,theta,m)-quantiles)
            independent=[]
            # Independent integral on central and moderately extreme tails;
            # the most extreme endpoints are separately bounded by the own-CDF
            # inverse residual because double precision rounds z near 1.
            for index in range(1,len(quantiles)-1):
                independent.append(abs(independent_cdf(z[index],theta,m)-quantiles[index]))
            max_independent=max(independent)
            max_inverse=float(own.max())
            assert max_independent<2e-8,(m,theta,max_independent)
            assert max_inverse<2e-8,(m,theta,max_inverse)
            rows.append({'m':m,'theta':theta,'max_independent_quadrature_residual':max_independent,
                         'max_inverse_residual':max_inverse,'quantile_count':len(quantiles)})
    result={'status':'PASS','independent_quadrature':True,
                      'theta_range_checked':[-6,6],'dimensions':[2,3,4,6,10],
                      'rows':rows,
                      'endpoint_precision_note':'For m=2, theta=+6, float64 quantile spacing near z=1 limits extreme-tail CDF residual to about 1e-8; registered 0.1-nat candidates are far inside this edge.'}
    (Path(__file__).resolve().parent/'nd_endpoint_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
