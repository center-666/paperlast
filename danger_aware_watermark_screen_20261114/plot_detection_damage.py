"""Static scientific figure: actual xi geometry and true mixture powers."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import chi2,ncx2
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import danger_detection_compare as comp
from spatial_baseline import setup
H=Path(__file__).resolve().parent
d=json.loads((H/'danger_detection_comparison.json').read_text(encoding='utf-8'))
fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
names={'Gaussian':'Gaussian','normalized_Gaussian':'normalized Gaussian','whole_sign':'whole sign',
 'coordinate_sign':'coordinate sign','Haar_SO2':'Haar SO(2)','public_scalar_alternation':'public alternating'}
summary={}
for col,name in enumerate(['unstable','stable']):
 m=d['models'][name];p=setup(name);W=p['W'];S=p['Ss'][9 if name=='unstable'else 0]
 ev,U=eigh(W);Wi=(U/np.sqrt(ev))@U.T
 ang=np.linspace(0,2*np.pi,721);v=np.array([np.cos(ang),np.sin(ang)])
 peak=np.einsum('in,ij,jn->n',v,Wi@S@Wi,v)
 outer=Wi@(np.sqrt(10)*v);inner=Wi@(np.sqrt(.1/peak)*v)
 lo=outer.min(axis=1)*1.06;hi=outer.max(axis=1)*1.06
 xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],170),np.linspace(lo[1],hi[1],170))
 X=np.stack([xx,yy],axis=-1);energy=np.einsum('...i,ij,...j->...',X,W,X)
 damage=np.einsum('...i,ij,...j->...',X,S,X);mask=(energy<=10)&(damage>=.1)
 Qs,_=comp.base.path_Qs(p['A'],'whole_sign',32)
 lam=np.maximum(np.einsum('...i,nij,...j->...n',X,Qs,X),0)
 power=ncx2.sf(comp.base.TAU,8,lam).mean(axis=-1);power[~mask]=np.nan
 ax=axs[0,col];im=ax.pcolormesh(xx,yy,100*power,shading='auto',cmap='viridis')
 ax.plot(outer[0],outer[1],color='black',lw=1.2,label='energy = 10')
 ax.plot(inner[0],inner[1],color='firebrick',lw=1.2,label='unencoded peak = 0.1')
 row=m['schemes']['whole_sign']['probes'][-1]
 marks=[(m['geometry']['peak_P1']['initial_state'],'*','peak-most harmful E10'),
  (row['full_energy_minimum']['initial_state'],'x','weakest E10 detection'),
  (row['danger_set_minimum']['initial_state'],'o','weakest danger-set detection')]
 for xi,marker,label in marks:
  xi=np.array(xi);ax.scatter(*xi,marker=marker,s=55,c='white',edgecolors='black'if marker!='x'else None,
   linewidths=.8,label=label,zorder=5)
 ax.set(title=f'{name}: whole-sign true parity power',xlabel='xi coordinate 1',ylabel='xi coordinate 2',aspect='equal')
 ax.legend(fontsize=7,loc='lower right');fig.colorbar(im,ax=ax,label='single-test Pd (%)')
 ax=axs[1,col]
 for label,text in names.items():
  curves=m['schemes'][label]['curves'];tt=np.array([a['angle']for a in curves])
  vals=np.array([a['minimum_danger']['Pd']for a in curves])
  ax.plot(tt*180/np.pi,100*vals,label=text,lw=1.5)
 # Same common danger radius, unencoded spatial data, no new watermarker.
 vals=[];ordinary=[]
 for t in tt:
  u=Wi@np.array([np.cos(t),np.sin(t)]);xi=np.sqrt(.1/(u@S@u))*u
  lam=float(xi@p['G5']@xi)
  vals.append(ncx2.sf(chi2.isf(1/300,2),2,lam))
  ordinary.append(ncx2.sf(chi2.isf(1/300,10),10,lam))
 ax.plot(tt*180/np.pi,100*np.array(vals),label='no encoding + standard spatial match',color='black',ls='--',lw=2)
 ax.plot(tt*180/np.pi,100*np.array(ordinary),label='no encoding + spatial energy',color='grey',ls=':',lw=2)
 ax.set(xlim=(0,180),xlabel='energy-whitened projective angle (degrees)',ylabel='single-test Pd on danger boundary (%)',
  title='same packets 0..4; per-test alpha=1/300')
 ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
 summary[name]=dict(heatmap_grid=170,heatmap_watermark_paths=32,physical_noise_samples=0,
   all_heatmap_powers_are_actual_noncentral_mixtures=True,
   boundary_spatial_match_min_grid=float(min(vals)),boundary_spatial_energy_min_grid=float(min(ordinary)),
   no_event_probability_or_global_optimization_claim=True)
fig.savefig(H/'detection_damage_map.png',dpi=190)
fig.savefig(H/'detection_damage_map.pdf')
(H/'detection_damage_figure_metadata.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
