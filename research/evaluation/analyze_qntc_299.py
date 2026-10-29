#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('csv',type=Path);p.add_argument('--out',type=Path,default=Path('QNTC_299_statistics'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(a.csv)
assert d.frame.nunique()==299 and set(d.stage)==set(range(4))
assert d.groupby(['frame','stage','color']).size().eq(6).all()
rows=[]
for (stage,color),group in d[d.stage<3].groupby(['stage','color']):
    sal=group[group.ordering=='saliency'].set_index('frame').sort_index()
    ran=group[group.ordering!='saliency'].groupby('frame')['translation_relative_rms'].mean().sort_index()
    s=sal['translation_relative_rms'];r=ran;diff=r-s
    # Frame-paired differences, random seed average within each frame.
    mean=float(diff.mean());se=float(stats.sem(diff));ci=stats.t.interval(.95,len(diff)-1,loc=mean,scale=se)
    t,pval=stats.ttest_rel(r,s)
    reduction=100*(r-s)/r
    rows.append(dict(stage=stage,color=color,frames=len(diff),mean_saliency_pct=100*s.mean(),mean_random_pct=100*r.mean(),mean_paired_absolute_reduction_pp=100*mean,ci95_low_pp=100*ci[0],ci95_high_pp=100*ci[1],mean_framewise_relative_reduction_pct=reduction.mean(),ratio_of_mean_errors_reduction_pct=100*(1-s.mean()/r.mean()),paired_t=t,paired_p=pval,paired_cohens_dz=mean/diff.std(ddof=1),winning_frames=int((diff>0).sum()),median_payload_bytes=float(sal.stage_payload_bytes.median())))
result=pd.DataFrame(rows).sort_values(['stage','color']);result.to_csv(a.out/'paired_translation_statistics.csv',index=False)
colors={'red':'tab:red','orange':'tab:orange','yellow':'goldenrod','blue':'tab:blue'}
for color in ['red','orange','yellow','blue']:
    sub=d[d.color==color]
    fig,ax=plt.subplots(figsize=(7,4.5))
    for name,ls in [('saliency','-'),('random','--')]:
        q=sub[sub.ordering=='saliency'] if name=='saliency' else sub[sub.ordering!='saliency']
        z=q.groupby('stage').agg(error=('translation_relative_rms','mean'),bytes=('stage_payload_bytes','mean'))
        ax.plot(z['bytes']/1024,100*z['error'],marker='o',linestyle=ls,label=name)
    ax.set(xlabel='Cumulative stage payload (KiB)',ylabel='Mean relative translation RMS error (%)',title=f'{color.capitalize()} region: 299 frames')
    ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(a.out/f'translation_{color}.png',dpi=200);plt.close(fig)
print(result.to_string(index=False,float_format=lambda v:f'{v:.5g}'))
print('\nSaved:',a.out)
