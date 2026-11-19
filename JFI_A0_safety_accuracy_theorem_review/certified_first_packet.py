"""Deterministic first-packet bounds on the frozen paired-risk functional."""

from dataclasses import dataclass
from math import fsum

from flint import arb
import numpy as np

from certified_cost import cholesky2, eigen_pair, outward, real


def down(x):
    return np.nextafter(x, -np.inf)


def up(x):
    return np.nextafter(x, np.inf)


@dataclass(frozen=True)
class Band:
    lo: np.ndarray | float
    hi: np.ndarray | float

    def __getitem__(self, key):
        return Band(self.lo[key], self.hi[key])

    def __add__(self, other):
        return Band(down(self.lo+other.lo), up(self.hi+other.hi))

    def __sub__(self, other):
        return Band(down(self.lo-other.hi), up(self.hi-other.lo))

    def __mul__(self, other):
        p = (self.lo*other.lo, self.lo*other.hi, self.hi*other.lo, self.hi*other.hi)
        return Band(down(np.minimum(np.minimum(p[0],p[1]),np.minimum(p[2],p[3]))),
                    up(np.maximum(np.maximum(p[0],p[1]),np.maximum(p[2],p[3]))))

    def squared(self):
        a, b = np.abs(self.lo), np.abs(self.hi)
        near = np.where((self.lo>0)|(self.hi<0), np.minimum(a,b),0.)
        far = np.maximum(a,b)
        return Band(np.maximum(0., down(near*near)), up(far*far))

    def sqrt(self):
        return Band(np.maximum(0.,down(np.sqrt(np.maximum(0.,self.lo)))),
                    up(np.sqrt(np.maximum(0.,self.hi))))


def point(v):
    return Band(v,v)


def norm(x):
    return (x[...,0].squared()+x[...,1].squared()).sqrt()


def multiply(matrix, x):
    """The only matrix shape here is two by two."""
    bounds = [[Band(*outward(v)) for v in row] for row in matrix]
    first, second = [a*x[...,0]+b*x[...,1] for a,b in bounds]
    return Band(np.stack([first.lo,second.lo],axis=-1),
                np.stack([first.hi,second.hi],axis=-1))


def box_center(band):
    center = point((band.lo+band.hi)*0.5)
    error = norm(band-center).hi
    return center,error


def point_norm(x):
    return np.sqrt(x[...,0]*x[...,0]+x[...,1]*x[...,1])


def point_projection(x, size, r):
    return x*(r/np.maximum(r,size))[...,None]


def cdf(x, twice_variance):
    return (1+(real(x)/real(twice_variance).sqrt()).erf())/2


def cells(x, y, two_var):
    """Disjoint boxes and their Gaussian mass enclosures."""
    lo_x, lo_y = np.meshgrid(x[:-1],y[:-1], indexing="ij")
    hi_x, hi_y = np.meshgrid(x[1:],y[1:], indexing="ij")
    geometry = Band(np.column_stack([lo_x.ravel(),lo_y.ravel()]),
                    np.column_stack([hi_x.ravel(),hi_y.ravel()]))

    def mass(edges):
        cumul = [cdf(edge,two_var) for edge in edges]
        a = np.array([outward(b-a) for a,b in zip(cumul,cumul[1:])])
        return Band(a[:,0],a[:,1])

    mx, my = mass(x), mass(y)
    joined = mx[:,None]*my[None,:]
    return geometry, Band(joined.lo.ravel(),joined.hi.ravel())


def left_product(a,b):
    return [[sum((a[i][m]*b[m][j] for m in range(2)),arb(0)) for j in range(2)]
            for i in range(2)]


def first_packet_bounds(health, start, template_first, factor_sq, radii, config):
    """Integrate box inclusions and a BF survival minorant, without new data."""
    gap, history, v, ky = health.stages[start]
    k = [[real(ky[i,j]) for j in range(2)] for i in range(2)]
    chol = cholesky2(v.tolist())
    a,b,c = chol[0][0],chol[1][0],chol[1][1]
    whiten = [[1/a,arb(0)],[-b/(a*c),1/c]]
    x,y = [real(u)*real(factor_sq).sqrt() for u in template_first]
    along = [[x,-y],[y,x]]
    decoder_to_white = left_product(whiten,along)
    decoder_to_gap = left_product(k,along)
    conditional_to_gap = left_product(k,chol)

    bside = config["beta_mesh_per_unit"]
    uside = config["standard_mesh_per_unit"]
    bcut, ucut = 2.75, 3.5
    corners,m_b = cells(np.arange(0,bcut+1/bside,1/bside),
                        np.arange(-bcut,bcut+1/bside,1/bside),1)
    u_axis = np.arange(-ucut,ucut+1/uside,1/uside)
    u_box,m_u = cells(u_axis,u_axis,2)
    keys_x, keys_y = corners[:,0],corners[:,1]
    along_a, along_turn = point(.5)+keys_x.squared(),keys_x*keys_y
    basis = Band(np.column_stack([along_a.lo,along_turn.lo]),
                 np.column_stack([along_a.hi,along_turn.hi]))
    m_b = m_b*point(2.)
    shift_white = multiply(decoder_to_white,basis)
    h_box = multiply(decoder_to_gap,basis)
    h0,dh = box_center(h_box)
    h_norm = norm(h_box)
    g_marginal, dg_marginal = box_center(multiply(cholesky2(gap.tolist()),u_box))
    g_terminal, dg_terminal = box_center(multiply(conditional_to_gap,u_box))
    covered = (2*cdf(bcut,1)-1)**2 * (2*cdf(ucut,2)-1)**2
    tail_hi = outward(1-covered)[1]
    c_lo,c_hi = outward(real("1/10").sqrt())
    speed = eigen_pair(history)[1]

    # Rounded survival factors. The tabulated h_H event is nested in radius.
    history_edges = np.arange(65)/512
    history_lower = np.array([outward(1-(-real(t)**2/(2*speed)).exp())[0]
                              for t in history_edges])
    history_lower[0] = 0.
    power_bins = np.arange(97)/16
    fail_upper = np.array([outward((1+(real(t)/2).exp())/20)[1] for t in power_bins])

    # Nonnegative masses, no cancellation. All product roundings are explicit.
    # At < 50 million addends a 1e-7 guard exceeds the total binary64 summation error.
    addends = len(corners.lo)*len(u_box.lo)
    assert addends < 50_000_000
    mass_b_upper = float(np.sum(m_b.hi))
    mass_u_upper = float(np.sum(m_u.hi))
    assert mass_b_upper*mass_u_upper <= 1.00001
    error_guard = 1e-7
    index = np.flatnonzero(h_norm.hi >= c_lo)
    for center in (h0,g_marginal,g_terminal):
        assert np.max(np.abs(center.lo)) <= 8.
    # Under the asserted coordinate/radius bounds this exceeds the binary64
    # error of a midpoint projection distance (see the arithmetic audit).
    point_guard = 1e-8
    points = {}
    for label,r in radii.items():
        assert .24 <= r <= .35
        points[label] = (point_projection(g_marginal.lo,point_norm(g_marginal.lo),r),
                         point_projection(g_terminal.lo,point_norm(g_terminal.lo),r),[],[])
    gmin = down(down(point_norm(g_marginal.lo)-point_guard)-dg_marginal)
    gmax = up(up(point_norm(g_terminal.lo)+point_guard)+dg_terminal)
    for batch in np.array_split(index,max(1,(len(index)+31)//32)):
        weight = m_b[batch,None]*m_u[None,:]
        h = h0.lo[batch,None,:]
        e_h = dh[batch,None]
        b_m = g_marginal.lo[None,:,:]+h
        b_t = g_terminal.lo[None,:,:]+h
        size_m,size_t = point_norm(b_m),point_norm(b_t)
        err_m = up(up(up(2*dg_marginal[None,:])+e_h)+point_guard)
        err_t = up(up(up(2*dg_terminal[None,:])+e_h)+point_guard)
        min_sum = down(gmin[None,:]+down(down(size_m-point_guard)-up(dg_marginal[None,:]+e_h)))
        both_inside = np.maximum(gmax[None,:],up(up(size_t+point_guard)+up(dg_terminal[None,:]+e_h)))
        after = shift_white[batch,None,:]+u_box[None,:,:]
        energy = (after[...,0].squared()+after[...,1].squared()).hi
        zidx = np.ceil(16*energy).astype(int)
        fail = fail_upper[np.clip(zidx,0,len(power_bins)-1)]
        for label,r in radii.items():
            marginal_projected,terminal_projected,sums_lo,sums_hi = points[label]
            distance = point_norm(point_projection(b_m,size_m,r)-marginal_projected[None,:,:])
            can_reach = up(distance+err_m) >= c_lo
            # The sum of distances to the two foci is restricted on danger.
            max_sum = up(up(2*r*h_norm.hi[batch,None])/c_lo)
            can_reach &= min_sum <= max_sum
            sums_hi.append(float(np.sum(np.where(can_reach,weight.hi,0.))))
            distance = point_norm(point_projection(b_t,size_t,r)-terminal_projected[None,:,:])
            allowance = down(down(down(distance-err_t)-c_hi)/2)
            remaining = np.where(h_norm.lo[batch,None]>c_hi,down(r-both_inside),0.)
            allowance = np.maximum(allowance,remaining)
            hidx = np.searchsorted(history_edges,allowance,side="right")-1
            good = history_lower[np.clip(hidx,0,len(history_edges)-1)]
            value = np.maximum(0.,down(good-fail))
            value[zidx>=len(power_bins)] = 0.
            sums_lo.append(float(np.sum(np.maximum(0.,down(weight.lo*value)))))
    result = {}
    for label,r in radii.items():
        sums_lo,sums_hi = points[label][2:]
        lower = max(0., float(down(fsum(sums_lo)-error_guard)))
        upper = min(1., float(up(up(fsum(sums_hi)+error_guard)+tail_hi)))
        result[label] = {"radius":r,"joint_risk_lower":lower,"joint_risk_upper":upper}
    return {"box_pairs_per_radius":addends,"gaussian_outside_box_upper":tail_hi,
            "binary64_mass_sum_guard":error_guard,"binary64_midpoint_guard":point_guard,
            "bounds":result}
