"""Exact Fraction verification of rounded, published counterexample witnesses.

This checks explicit attacks, independent of the original safety algorithms.
"""
from fractions import Fraction as Q
from decimal import Decimal
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;P=HERE.parent
raw=(P/'checks/水印收束_理论与映射核验_20261105.json').read_bytes()
src=json.loads(raw,parse_float=Decimal)
F=[[Q(v)for v in row]for row in src['F']];g=[Q(v)for v in src['g']]
search=json.loads((HERE/'counterexample_search.json').read_text(encoding='utf-8'))
fresh=json.loads((HERE/'independent_cert.json').read_text(encoding='utf-8'))
safe=set(fresh['safe_indices']);states=set(fresh['graph']['cyclic_states'])
def wordid(s):return sum(int(x>0)<<(len(s)-1-i)for i,x in enumerate(s))
def extendible(s):
    assert len(s)>=9 and wordid(s[:9]) in states
    for t in range(len(s)-9):
        w=wordid(s[t:t+10]);assert w in safe and w&511 in states
    return True
W=lambda L:[[sum((Q(1),Q(t,10))[i]*(Q(1),Q(t,10))[j]for t in range(L))for j in range(2)]for i in range(2)]
vector=lambda values:[Q(str(round(float(x),8)))for x in values]
def step(b,u):return [sum(F[i][j]*b[j]for j in range(4))+g[i]*u for i in range(4)]
def norm2(b):return sum(x*x for x in b)
def energy(x,L):return sum(x[i]*W(L)[i][j]*x[j]for i in range(2)for j in range(2))
def amplify(x,t):return x[0]+Q(t,10)*x[1]
def follow(s,a,initial=None,N=None):
    b=[Q(0)]*4 if initial is None else initial
    trace=[]
    for k in range(len(a) if N is None else N):
        b=step(b,Q(s[k])*a[k]if k<len(a)else Q(0));trace.append(norm2(b))
    return trace
out={}
rr=search['residual_witness'];x=[Q('-0.5207'),Q('-0.9759')]
b0=[Q('.0018'),Q('.0047'),Q(0),Q(0)]
a=[amplify(x,t)for t in range(10)]
tr=follow(rr['signs'],a,b0)
assert extendible(rr['signs'])
assert energy(x,10)<10 and tr[9]>Q(1,10) and norm2(b0)<Q(1,100)**2
out['residual']=dict(xi=list(map(str,x)),b_start=list(map(str,b0)),
                    b_start_norm_float=float(norm2(b0))**.5,signs=rr['signs'],state=10,
                    energy=str(energy(x,10)),damage=str(tr[9]),damage_float=float(tr[9]))
two=search['two_attack_witness'];x=vector(two['xi_first']);Ta=two['second_start'];N=two['evaluation_state']
s=(two['period']*20)[:N];a=[Q(0)]*N
assert extendible(two['period']*10)
for start in [0,Ta]:
    for t in range(min(10,N-start)):a[start+t]+=amplify(x,t)
tr=follow(s,a);assert energy(x,10)<10 and tr[-1]>Q(1,10)
out['repeated']=dict(period=two['period'],starts=[0,Ta],xi=list(map(str,x)),state=N,
                    energy_per_event=str(energy(x,10)),total_event_energy=str(2*energy(x,10)),
                    damage=str(tr[-1]),damage_float=float(tr[-1]))
for entry in search['duration_witnesses']:
    L=entry['L'];x=vector(entry['xi']);a=[amplify(x,t)for t in range(L)]
    assert extendible(entry['word'])
    tr=follow(entry['word'],a)
    assert energy(x,L)<10 and tr[-1]>Q(1,10)
    out[f'duration_{L}']=dict(signs=entry['word'],xi=list(map(str,x)),energy=str(energy(x,L)),
                             damage=str(tr[-1]),damage_float=float(tr[-1]),state=L)
pe=search['small_additive_perturbation'];x=vector(pe['autonomous_xi']);e=vector([1.05*v for v in pe['additive_perturbation']])
assert extendible(pe['signs'])
a=[amplify(x,t)+e[t]for t in range(10)]
tr=follow(pe['signs'],a)
assert norm2(a)<10 and tr[-1]>Q(1,10)
out['small_perturbation']=dict(signs=pe['signs'],xi=list(map(str,x)),perturbation=list(map(str,e)),
                              coded_sequence=list(map(str,a)),energy=str(norm2(a)),
                              perturbation_L2_float=float(norm2(e))**.5,
                              perturbation_Linf=max(map(abs,e)).__str__(),
                              state=10,damage=str(tr[-1]),damage_float=float(tr[-1]),
                              predetermined_same_sequence_for_all_secret_paths=True)
piece=search['piecewise_autonomous'];x1=vector(piece['xi_first']);x2=vector(piece['xi_second'])
a=[amplify(x1,t) for t in range(5)]+[amplify(x2,t)for t in range(5)]
assert extendible(piece['signs'])
tr=follow(piece['signs'],a)
assert norm2(a)<10 and tr[-1]>Q(1,10)
out['piecewise_autonomous']=dict(signs=piece['signs'],split_after=5,xi_first=list(map(str,x1)),
    xi_second=list(map(str,x2)),energy=str(norm2(a)),state=10,damage=str(tr[-1]),damage_float=float(tr[-1]))
# Nonzero admissible residual AND small waveform disturbance set. Existing
# common margin yields forced displacement <=.316; ||F^t||<1.015 all t.
# Thus any ||b_start||<=1/5000 and coded error ||e||2<=1/10000 obeys conservative
# ||b||<=.316+1.015*.0002 + .18*.0001 <sqrt(.1).
assert (13-W(10)[0][0])*(13-W(10)[1][1])-W(10)[0][1]**2>0
assert Q(1,10)-Q(10,4000*13)<Q(316,1000)**2
assert Q(103,100)<Q(203,200)**2
assert Q(56,1000)**2>sum(v*v for v in g)
# use sqrt10<3.163, so finite-10 input convolution norm <=1.015*.056*3.163 <.18
assert Q(10)<Q(3163,1000)**2
assert Q(203,200)*Q(56,1000)*Q(3163,1000)<Q(18,100)
radius=Q(316,1000)+Q(203,200)*Q(1,5000)+Q(18,100)*Q(1,10000)
assert radius**2<Q(1,10)
out['nonzero_residual_and_small_waveform_robust_set']=dict(
    base_autonomous_energy_upper='10',residual_L2_radius='1/5000',
    added_waveform_L2_radius='1/10000',safe_amplitude_upper=str(radius),
    safe_damage_upper=str(radius**2),safe_damage_upper_float=float(radius**2),
    total_coded_energy_may_need_separate_constraint=True,
    scope='fixedF/g, single10 autonomous component; joint residual/waveform conservative sufficient set')
out['input_sha256']=hashlib.sha256(raw).hexdigest()
out['certification']='Exact rational explicit witness comparisons, not global minimum failure radii'
(HERE/'counterexamples_exact.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:{n:v for n,v in row.items()if not n in ['damage','coded_sequence']}if isinstance(row,dict)else row
                  for k,row in out.items()},ensure_ascii=False,indent=2))
