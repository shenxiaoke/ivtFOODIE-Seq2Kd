from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib.patches import Patch

import argparse

parser = argparse.ArgumentParser(description='Draw batch-specific overexpression vs GO set overlaps')
parser.add_argument('--classifications', type=Path, required=True)
parser.add_argument('--go', type=Path, required=True)
parser.add_argument('--eligibility', type=Path, required=True)
parser.add_argument('--targets', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
paths = [args.classifications, args.go, args.eligibility, args.targets]
d = pd.read_csv(paths[0], sep='\t')
g = pd.read_csv(paths[1])
raw = pd.read_csv(paths[2], sep='\t')
universe = set(pd.read_csv(paths[3], sep='\t').TF)
roles = ['activator', 'repressor']
terms = dict(zip(roles, ['GO:0001228', 'GO:0001227']))
go = {r: set(g.loc[g.GOALL.eq(t), 'SYMBOL'].dropna()) for r,t in terms.items()}
conditions = [('overexpression_only', .25, 'All eligible genes'),
              ('overexpression_only', .5, 'All eligible genes'),
              ('overexpression_ChIP_targets', .25, 'ChIP-Atlas targets'),
              ('overexpression_ChIP_targets', .5, 'ChIP-Atlas targets')]
colors = ('#77A9CE', '#E5B858')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'pdf.fonttype':42})
summary, members, geometry = [], [], []

def oe(batch, mode, threshold, role):
    z=d[d.batch.eq(batch)&d['mode'].eq(mode)&d.abs_log2FC_threshold.eq(threshold)&d.label.eq(role)]
    assert not z.TF.duplicated().any()
    return set(z.TF)

def panel(ax, a, b, labels, title, key):
    counts=(len(a-b),len(b-a),len(a&b))
    assert counts[0]+counts[2]==len(a) and counts[1]+counts[2]==len(b)
    v=venn2(subsets=counts, set_labels=(f'{labels[0]} (n={len(a)})', f'{labels[1]} (n={len(b)})'),
            set_colors=colors, alpha=.65, ax=ax)
    for patch in v.patches:
        if patch is not None: patch.set_edgecolor('#555555');patch.set_linewidth(.7)
    for lab in v.set_labels:
        if lab is not None: lab.set_fontsize(10)
    for lab in v.subset_labels:
        if lab is not None: lab.set_fontsize(12); lab.set_weight('bold')
    # Move small nonempty lens counts outside the circles with a leader line.
    lab=v.get_label_by_id('11')
    if lab is not None and counts[2] and (counts[2]/max(1,min(len(a),len(b)))<.25 or min(len(a),len(b))/max(len(a),len(b))<.12):
        xy=lab.get_position();lab.set_visible(False)
        top=max(c.y+r for c,r in zip(v.centers,v.radii))
        ax.annotate(str(counts[2]),xy=xy,xytext=(xy[0],top+.13),ha='center',va='bottom',
                    fontsize=12,weight='bold',arrowprops={'arrowstyle':'-','color':'#555555','lw':.8})
        ax.set_ylim(ax.get_ylim()[0],top+.25)
    ax.set_title(title, fontsize=10.5, pad=13)
    ax.text(.5,-.20,f'Intersection: {len(a&b)} TFs',transform=ax.transAxes,ha='center',fontsize=9.5)
    if not a or not b:
        empty_label = v.get_label_by_id('10' if not a else '01')
        if empty_label is not None: empty_label.set_visible(False)
    summary.append(dict(comparison=key,n_left=len(a),n_right=len(b),left_only=counts[0],right_only=counts[1],overlap=counts[2],overlap_TFs=';'.join(sorted(a&b))))
    for region,s in [('left_only',a-b),('overlap',a&b),('right_only',b-a)]:
        members.extend(dict(comparison=key,region=region,TF=tf) for tf in sorted(s))
    if a and b:
        r0,r1=v.radii; dist=float(np.hypot(v.centers[0].x-v.centers[1].x,v.centers[0].y-v.centers[1].y))
        if dist>=r0+r1: lens=0.
        elif dist<=abs(r0-r1):lens=np.pi*min(r0,r1)**2
        else:
            lens=(r0*r0*np.arccos(np.clip((dist*dist+r0*r0-r1*r1)/(2*dist*r0),-1,1))+
                  r1*r1*np.arccos(np.clip((dist*dist+r1*r1-r0*r0)/(2*dist*r1),-1,1))-
                  .5*np.sqrt(max(0,(-dist+r0+r1)*(dist+r0-r1)*(dist-r0+r1)*(dist+r0+r1))))
        assert np.isclose(r0*r0/(r1*r1),len(a)/len(b),rtol=1e-5)
        assert np.isclose(lens/(np.pi*r0*r0),len(a&b)/len(a),atol=1e-5)
        geometry.append(dict(comparison=key,area_ratio=float(r0*r0/(r1*r1)),intersection_fraction_left=float(lens/(np.pi*r0*r0))))

def save(fig,stem,note):
    fig.text(.07,.035,note,fontsize=8.5,linespacing=1.6,va='bottom',color='#454545')
    fig.savefig(OUT/(stem+'.pdf'))
    fig.savefig(OUT/(stem+'.png'),dpi=140)
    plt.close(fig)

for batch in [0,1]:
    fig,axs=plt.subplots(4,2,figsize=(10,13))
    fig.subplots_adjust(left=.08,right=.95,top=.86,bottom=.12,hspace=.8,wspace=.35)
    fig.suptitle(f'Batch{batch} TF overexpression vs GO annotation',y=.978,fontsize=17,weight='bold')
    fig.text(.5,.95,'Activator: GO:0001228     |     Repressor: GO:0001227',ha='center',fontsize=11)
    fig.legend(handles=[Patch(color=colors[0],label='TF overexpression (OE)'),Patch(color=colors[1],label='GO annotation')],
               loc='upper center',bbox_to_anchor=(.5,.937),ncol=2,frameon=False)
    for j,role in enumerate(roles): fig.text(.29 if j==0 else .75,.905,role.capitalize(),ha='center',fontsize=14,weight='bold')
    for i,(mode,t,label) in enumerate(conditions):
        for j,role in enumerate(roles):
            panel(axs[i,j],oe(batch,mode,t,role),go[role],('OE','GO'),f'{label} | |log2FC| ≥ {t:g}',f'batch{batch}__{mode}__{t}__{role}__GO')
    save(fig,f'batch{batch}_OE_vs_GO_area_venn','Circle and intersection areas are proportional to counts within each panel; panels are normalized separately.\n'
         'Original full GO symbol sets are retained, including symbols annotated to both terms; no OE coverage filter.\n'
         'An empty set has zero area. OE direction sets follow the original analysis eligibility criteria.')

fig=plt.figure(figsize=(10,16))
grid=fig.add_gridspec(5,2,left=.08,right=.95,top=.91,bottom=.10,hspace=.9,wspace=.35)
fig.suptitle('Batch0 vs Batch1: TF set overlap',y=.98,fontsize=18,weight='bold')
fig.legend(handles=[Patch(color=colors[0],label='Batch0'),Patch(color=colors[1],label='Batch1')],
           loc='upper center',bbox_to_anchor=(.5,.967),ncol=2,frameon=False)
all_sets={b:set(raw.loc[raw.batch.eq(b)&raw.n_cells.gt(0),'TF'])&universe for b in [0,1]}
for b in [0,1]: assert set(d.loc[d.batch.eq(b),'TF'])<=all_sets[b]
panel(fig.add_subplot(grid[0,:]),all_sets[0],all_sets[1],('Batch0','Batch1'),
      'All experimentally observed TFs (before direction-analysis filtering)','all_experimental_TFs')
for i,(mode,t,label) in enumerate(conditions):
    for j,role in enumerate(roles):
        panel(fig.add_subplot(grid[i+1,j]),oe(0,mode,t,role),oe(1,mode,t,role),('Batch0','Batch1'),
              f'{role.capitalize()} | {label}\n|log2FC| ≥ {t:g}',f'batch0_vs_batch1__{mode}__{t}__{role}')
save(fig,'batch0_vs_batch1_area_venn','Circle and intersection areas are proportional to counts within each panel; panels are normalized separately.\n'
     'Top panel: TFs with ≥1 observed cell in raw ORF records, restricted to the project TF universe (ORFs deduplicated by TF).\n'
     'Direction panels use full original direction sets per batch, including TFs absent from the other batch.\n'
     'Uncertain TFs are included in the top panel. An empty set has zero area.')
pd.DataFrame(summary).to_csv(OUT/'overlap_summary.tsv',sep='\t',index=False)
pd.DataFrame(members).to_csv(OUT/'venn_memberships.tsv',sep='\t',index=False)
(OUT/'validation_and_sources.json').write_text(json.dumps({'inputs':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
     'geometry_validation':geometry,'area_tolerance':1e-5,'all_experimental_TFs':{b:len(s) for b,s in all_sets.items()},
     'all_experimental_overlap':len(all_sets[0]&all_sets[1])},indent=2)+'\n')
print(pd.DataFrame(summary).drop(columns='overlap_TFs').to_string(index=False))
