"""Ideal authentication interface around the frozen common-G implementation.

No cryptographic performance/security claim. The oracle is an in-process
unforgeable-token model; it returns a Boolean, never a source packet copy.
"""
from pathlib import Path
from dataclasses import dataclass,replace
import sys,json
sys.dont_write_bytecode=True
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
sys.path.insert(0,str(P/'common_g_reference_revision_20261115'))
import common_g_reference as ref
sys.path.insert(0,str(P/'A1_information_aware_attack_audit_20261116'))
from screen_attack_classes import detectors
CFG=json.loads((H/'registration.json').read_text('utf8'))
INF=10**9
@dataclass(frozen=True)
class Packet:
    sender:int
    receiver:int
    epoch:int
    time:int
    payload:bytes
    tag:object=None
    def value(self):return np.frombuffer(self.payload,dtype='<f8').copy()
def bind(p):return (p.sender,p.receiver,p.epoch,p.time,p.payload)
class IdealVerifier:
    __slots__=('__bindings','__consumed')
    def __init__(self):self.__bindings={};self.__consumed=set()
    def issue(self,p):
        t=object();q=replace(p,tag=t);self.__bindings[t]=bind(q);return q
    def verify(self,p,expected_time):
        b=self.__bindings.get(p.tag)
        if (b is None or b!=bind(p) or p.sender!=0 or p.receiver!=1 or
            p.epoch!=1 or p.time!=expected_time or p.tag in self.__consumed):return False
        self.__consumed.add(p.tag);return True
@dataclass(frozen=True)
class AttackView:
    k:int
    past_q:tuple
    current_q:tuple|None
    public_gamma:tuple|None
class CausalActor:
    """Only coded target packets and optional public schedule are provided."""
    __slots__=('plan','remaining','history','energy','actions')
    def __init__(self,plan):self.plan=plan;self.remaining=10.*(1-1e-12);self.history=[];self.energy=0.;self.actions=[]
    def act(self,k,current_q=None,public_gamma=None):
        kind=self.plan['kind'];family=self.plan.get('family','OL')
        assert family=='C' or current_q is None
        view=AttackView(k,tuple(self.history),tuple(current_q) if current_q is not None else None,public_gamma)
        a=np.zeros(2)
        if 7<=k<17:
            if kind in ('plan','signed'):
                sign=1.
                if kind=='signed':
                    obs=self.plan['obs']
                    if obs==k:
                        assert view.current_q is not None
                        x=np.array(view.current_q)
                    else:
                        assert obs<k;x=np.array(view.past_q[obs])
                    sign=1. if np.array(self.plan['ell'])@x>=0 else -1.
                a=sign*np.array(self.plan['word'][k-7])
            elif kind=='pulse' and k==self.plan.get('time',7):a=np.array(self.plan['packet'])
            elif kind=='none':pass
            else:assert kind in ('pulse','none')
            # Explicitly a protected-timing-constrained waveform, NOT A0.
            if self.plan.get('avoid_public_protected'):
                assert view.public_gamma is not None
                if view.public_gamma[k]:a=np.zeros(2)
            cost=a@a
            if cost>self.remaining:a*=np.sqrt(self.remaining*(1-1e-14)/cost)
        cost=float(a@a);self.remaining=max(0.,self.remaining-cost);self.energy+=cost
        self.actions.append(a.copy())
        assert self.energy<=10
        return a,view
    def observe_after_decision(self,q):self.history.append(tuple(q))
def bank_path(m,index):
    seeds=json.loads((P/'attack_gain_switch_audit_20261115/sampling_seed_map.json').read_text('utf8'))[m.name]
    path=ref.draw_physical_path(m,32,seeds['physical'][index])
    M,_=ref.original_watermarks(m,32,seeds['watermark'][index])
    return path,M
def run(m,path,M,plan,check_plan,delay=0,private=False):
    """check_plan is a verification APPOINTMENT, not timely protection gamma.
    All outputs are real-time. Late failures quarantine detector history only.
    The effective timely gamma is check_plan when delay0, and zero otherwise.
    """
    T=32;gamma=np.asarray(check_plan,bool);assert gamma.shape==(T,) and delay in (0,1,2)
    actor=CausalActor(plan);oracle=IdealVerifier();prev=np.zeros((5,2));out=np.zeros((T,5,2))
    q=np.linalg.solve(M,path['y'][...,None])[...,0]
    stream=[];available=[];pending={};log=[];scores={};tstat={lab:INF for lab in ('temporal','joint_GLRT')};tauth=INF;checked=failures=tagged=0
    def event(k,phase,**kw):log.append(dict(time=k,phase=phase,**kw))
    def verdict(k,t,p):
        nonlocal tauth,checked,failures
        ok=oracle.verify(p,t);checked+=1
        if not ok:tauth=min(tauth,k);failures+=1
        event(k,'verify',packet_time=t,passed=ok,deadline=t,late=k>t,
              receiver_result_only=True)
        return ok
    # Construct only the fixed nominal whitening maps from the public model.
    # Actual scores below use actual received packets, never clean-source
    # residual plus an oracle-computed attack shift.
    monitor=detectors(m,{'y':np.zeros((1,T,5,2)),'x':np.zeros((1,T,2))})
    neighbor2=[]
    for k in range(T):
        for t,p in pending.pop(k,[]):
            ok=verdict(k,t,p)
            if not ok:available[t]=False
        event(k,'physical',sender_measurement_not_visible_to_attacker=True)
        p=Packet(0,1,1,k,np.asarray(q[k,0],dtype='<f8').tobytes())
        # Private receiver schedule needs indistinguishable tagged wire packets.
        if private or gamma[k]:p=oracle.issue(p);tagged+=1
        event(k,'encode',packet_time=k,tag_present=p.tag is not None)
        a,view=actor.act(k,q[k,0] if plan.get('family')=='C' else None,
                         None if private else tuple(bool(v) for v in gamma))
        actor.observe_after_decision(q[k,0])
        pa=replace(p,payload=np.asarray(p.value()+a,dtype='<f8').tobytes()) if np.any(a) else p
        event(k,'attack',family=plan.get('family','OL'),energy_spent=float(a@a),
              accessible_fields=list(view.__dataclass_fields__),private_schedule_hidden=private)
        accept=True
        if gamma[k]:
            if delay==0:accept=verdict(k,k,pa)
            else:
                pending.setdefault(k+delay,[]).append((k,pa))
                event(k,'verification_pending',packet_time=k,result_due=k+delay)
        # Decoder never gets the oracle's stored source payload.
        yd=M[k,0]@pa.value() if accept else None
        stream.append(yd);available.append(accept)
        neighbor2.append(M[k,2]@q[k,2])
        event(k,'decode_admission',accepted=accept,provisional=bool(gamma[k] and delay),
              rejected_is_missing_not_zero=True)
        if k>=4:
            if all(available[k-4:k+1]):
                for lab,d in monitor.items():
                    if lab=='temporal':observed=np.array(stream[k-4:k+1]).reshape(-1)
                    else:
                        observed=np.concatenate([np.r_[path['y'][t,1],stream[t],neighbor2[t]]
                                                 for t in range(k-4,k+1)])
                    r=d['V']@observed
                    val=float(r@r)
                    tau=float(CFG['temporal_threshold'] if lab=='temporal' else CFG['joint_threshold'])
                    scores.setdefault(lab,[]).append(dict(end=k,value=val,alarm=val>tau))
                    if val>tau:tstat[lab]=min(tstat[lab],k)
                event(k,'statistical',complete_window=True)
            else:event(k,'statistical',complete_window=False,action='skip, do not zero-fill')
        event(k,'admission_final',accepted=accept,
              rule='authentication eligibility only; statistical alarm logged without extra gating')
        for i in range(5):
            neigh=[];Hs=[]
            for j in m.neighbors[i]:
                if (i,j)==(1,0):
                    if accept:neigh.append(yd);Hs.append(m.H[j])
                else:
                    neigh.append(M[k,j]@q[k,j]);Hs.append(m.H[j])
            _,out[k,i]=ref.common_step(prev[i],path['y'][k,i],neigh,m.A,m.H[i],Hs,m.K[i],m.G[i])
        event(k,'update',output_time=k,uses_only_verdicts_received_by=k)
        prev=out[k].copy()
    # Audit delayed results arriving after the last published output as well.
    for k in sorted(pending):
        for t,p in pending[k]:verdict(k,t,p)
    return dict(posterior=out,actions=np.array(actor.actions),available=np.array(available),
      logs=log,scores=scores,tstat=tstat,tauth=tauth,
      resource=dict(generated_tags=tagged,verification_checks=checked,failures=failures,
        timely_protected_packets=int(gamma.sum()) if delay==0 else 0,
        scheduled_checks=int(gamma.sum()),numeric_tag_bytes_not_implemented=True),
      energy=actor.energy)
