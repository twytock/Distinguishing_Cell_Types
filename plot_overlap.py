#!/usr/bin/env python
import sys
sys.path = sys.path = [p for p in sys.path if '.local' not in p]
import numpy as np
import cPickle as cp
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter as ig
from matplotlib.colors import Normalize,LinearSegmentedColormap,LogNorm
import os,os.path as osp
from collections import defaultdict
e = 1/8.
RR,GG,BB = 0.9,0.7,0.9
cdict = {'red': ((0,RR,RR),
               (.9*e,RR,0),
               (3.9*e,0,RR),
               (6.9*e,RR,0),
               (1,0,0)),
        'green': ((0,RR,RR), ## use same number as red to ensure grey
                 (0.9*e,GG,0),
                 (1.9*e,0,GG),
                 (3.9*e,GG,0),
                 (5.9*e,0,GG),
                 (6.9*e,GG,0),
                 (1,0,0)),
        'blue': ((0,RR,RR),## use same number as red to ensure grey
                (1.9*e,BB,0),
                (2.9*e,0,BB),
                (3.9*e,BB,0),
                (4.9*e,0,BB),
                (5.9*e,BB,0),
                (1,0,0))}

clrNORM =Normalize(vmin=0,vmax=7)
ccm = LinearSegmentedColormap('custom1',cdict)
ccm.set_bad('white',1)

kind_l = ['GeneExp','HiC']
OUT = 'Analysis'
NORM = 'Normalization'
FIG = 'Figures'
FS = 'Feature_Selection'
DAT = 'Data'
alg_l = ['UU','WC','WU']
alg_const_d = {'UU':4,'WC':1,'WU':2}

def order_cts(kind,ct_grp_d={}):
    all_ct_l = []
    data = pd.read_pickle('%s/%s/%s_corr_data.pkl' % (kind,NORM,kind))
    vcs = pd.Categorical(data.index.get_level_values(1)).value_counts()
    if ct_grp_d == {}:
        vcs = vcs.sort_values(ascending=False)
        return all_ct_l, vcs.index.tolist()        
    grp_len_d = {}
    for grpId,ct_l in ct_grp_d.items():
        sel = data.index.get_level_values(1).isin(ct_l)
        grp_len_d[grpId] = data[sel].shape[0]

    grp_len_ser = pd.Series(grp_len_d).sort_values(ascending=False)
    len_l = []
    for grpId,L in grp_len_ser.iteritems():
        srt_l = sorted(ct_grp_d[grpId],key=lambda x: vcs.loc[x])
        all_ct_l.extend(srt_l)
        len_l.append(len(srt_l))
    return all_ct_l,grp_len_ser.index.tolist(),len_l


def group_by_block(data,grp_d):
    from itertools import combinations
    from collections import defaultdict
    grp_cnt_d = defaultdict(dict)
    for g1,g2 in combinations(grp_d.keys(),2):
        ct_l1 = grp_d[g1]
        ct_l2 = grp_d[g2]
        sel1 = data.index.get_level_values(0).isin(ct_l1)
        sel2 = data.index.get_level_values(0).isin(ct_l2)
        grp_cnt_d[g1][g2] = data[sel1].T[sel2].sum().sum().astype(float)
        grp_cnt_d[g2][g1] = data[sel2].T[sel1].sum().sum().astype(float)
    return pd.DataFrame(dict(grp_cnt_d))

def plot_colormap(data,ccm,norm,labels=False,ytls=[],xtls=[]):
    if not labels:
        fig = plt.figure(figsize=(1.75,1.5),dpi=300)
        ax = fig.add_axes([0.05,0.05,0.8,0.8])
    else:
        fig = plt.figure(figsize=(6,5.15),dpi=300)
        ax = fig.add_axes([0.01,0.01,0.8,0.8])
        
    if isinstance(data,dict):
        ylabel = 'Predicted cell type'
        xlabel = 'Actual cell type'        
        for alg,dat in data.items():
            ndf = (dat/dat.sum(axis=0))
            ax.imshow(ndf,cmap=ccm[alg],origin='lower',
                       interpolation=None,norm=norm,alpha=0.5)
        nct = ndf.shape[0]
    else:
        ylabel = 'Query cell type'
        xlabel = 'Test cell type'
        for ii in range(data.shape[0]):
            if isinstance(data,pd.DataFrame):
                data.iloc[ii,ii] = np.nan
            else:
                data[ii,ii] = np.nan
        coll = ax.imshow(data,cmap=ccm,norm=clrNORM,alpha=0.7,
                     origin='lower',aspect='auto',interpolation='none')
        nct = data.shape[0]
    ax.set_yticks(np.arange(0,nct,5))
    ax.set_xticks(np.arange(0,nct,5))
    if labels:
        from matplotlib.ticker import FixedLocator
        mtloc = FixedLocator([x for x in range(nct) if x % 5 >0])
        ax.xaxis.set_minor_locator(mtloc)
        ax.yaxis.set_minor_locator(mtloc)
        ax.set_yticklabels([])
        ax.set_xticklabels([])        
        #ax.set_yticklabels(ytls,size=6)
        #ax.set_xticklabels(xtls,size=6,rotation=90)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_xlim(-0.5,nct-0.5)
    ax.set_ylim(-0.5,nct-0.5)
    ax.tick_params(axis='x',which='both',top=False,bottom=True,direction='out')
    ax.tick_params(axis='y',which='both',right=False,left=True,direction='out')
    ax.set_ylabel(ylabel,size=10,family='Serif')
    ax.set_xlabel(xlabel,size=10,family='Serif')
    return fig

def plot_bar_counts(arr,y2min,y2max):
    fig2 = plt.figure(figsize=(4,1))
    ax2 = fig2.add_axes([0.2,0.4,0.75,0.55])
    ax2_ticks = np.arange(7)+0.5
    ax2_ticklabels = ['WC','WU','WC&WU','UU','WC&UU','WU&UU','All']
    clr_l = ['#0000e6','#00b300','#00b3e6','#e60000',
             '#e600e6','#e6b300','#000000']
    hists = np.histogram(arr[~arr.mask] ,bins=np.arange(9)-.5)
    ## how to bin:
    groupsum_d = {'WC':[1,3,5,7],'WU':[2,3,6,7],'UU':[4,5,6,7]}
    grp_sum_d = {}
    for ext,its in groupsum_d.items():
        grp_sum_d[ext] = np.sum(ig(*its)(hists[0]))
    
    ax2.bar(hists[1][1:-1],hists[0][1:],width=0.75,log=True,align='center',
            color=clr_l,alpha=0.7)
    ax2.set_xlim(-0.15,7.15)
    ax2.set_ylim(y2min,y2max)
    ax2.set_xticks(ax2_ticks)
    ax2.set_xticklabels(ax2_ticklabels,size=6,family='Serif',rotation=15,
                        horizontalalignment='right')
    plt.setp(ax2.get_yticklabels(),size=6,family='Serif')
    ax2.tick_params(axis='x',which='both',top=False,bottom=True,direction='out')
    ax2.tick_params(axis='y',which='both',right=True,left=True,direction='in')
    ax2.set_xlabel('Version(s)',size=10,family='Serif')
    ax2.set_ylabel('Total pairs',size=10,family='Serif')
    return fig2,grp_sum_d

def plot_overlap_counts(grp_sum_d,y3min,y3max,xlbl):
    fig3 = plt.figure(figsize=(3.375,2))
    ax3 = fig3.add_axes([0.2,0.2,0.75,0.75])
    ax3.bar(np.linspace(.25,.75,3),height=ig(*['UU','WC','WU'])(grp_sum_d),width=.2,align='center',color=['#1f77b4','#ff7f0e','#2ca02c'],edgecolor='none')
    ax3.set_xticks(np.linspace(.25,.75,3))
    ax3.set_xlim(0,1)
    ax3.set_ylim(y3min,y3max)
    ax3.set_xticklabels(['UU','WC','WU'],size=6,family='Serif')
    plt.setp(ax3.get_yticklabels(),size=6)
    ax3.set_ylabel('Number of overlaps',size=10,family='Serif')
    ax3.set_xlabel(xlbl,size=10,family='serif')
    return fig3
    

def plot_minsep(kind):
    if kind=='GeneExp':
        y3min,y3max = 0,1300
        y2min,y2max = 1,1500
        nfeat = 4
        cat_ct_d = pd.read_pickle('%s/%s/ct_groupings_d.pkl' % (kind,DAT))
        all_ct_l,grp_ct_l,len_l = order_cts(kind,cat_ct_d)
    else:
        y3min,y3max = 0,51
        y2min,y2max = 1,100
        nfeat = 3
        cat_ct_d = {}
        ___,all_ct_l = order_HiC_cts(kind)
    alg_data_d = {}
    alg_grpdata_d = {}
    for alg in alg_l:
        dat = pd.read_pickle('%s/%s/top_dists_%s.pkl' % (kind,OUT,alg))
        dat_feat = dat.loc[:,nfeat].unstack(level=1)
        df_inds = dat_feat.index
        datL = dat_feat.shape[0]
        flat_vals = dat_feat.values.ravel()
        nan_inds = np.where(np.isnan(flat_vals))[0]
        gt_zero = (flat_vals>0).astype(float)
        gt_zero[nan_inds] = np.nan
        gt_zero = gt_zero.reshape(datL,datL)
        alg_data_d[alg] = pd.DataFrame(gt_zero,index=df_inds,columns=df_inds)
        if kind=='GeneExp':
            block_df = group_by_block(dat_feat,cat_ct_d)
            blockL = block_df.shape[0]
            bl_inds = block_df.index
            flat_block = block_df.values.ravel()
            nan_inds = np.where(np.isnan(flat_block))[0]
            gt_zero = (flat_block>0).astype(float)
            gt_zero[nan_inds] = np.nan
            gt_zero = gt_zero.reshape(blockL,blockL)
            alg_grpdata_d[alg]=pd.DataFrame(gt_zero,index=bl_inds,columns=bl_inds)
    comb_df = sum([C*alg_data_d[alg] for alg,C in alg_const_d.items()])
    comb_df = comb_df.loc[all_ct_l,all_ct_l]
    #comb_df = 4*uudf + 2*wudf + wcdf
    arr = np.ma.masked_less(comb_df.fillna(-1).values,0)
    #col_srt = np.argsort(colsums.values)
    #row_srt = np.argsort(rowsums.values)
    #rowsums.order(inplace=True)
    fig =  plot_colormap(arr,ccm,clrNORM,labels=True)
    fig.savefig('%s/%s/minsep_%s_alldata.svg' % (kind,FIG,kind))
    barr = np.ma.masked_less(comb_df.loc[all_ct_l,all_ct_l].fillna(-1).values,0)
    fig2,grp_sum_d = plot_bar_counts(barr.ravel(),y2min,y2max)
    fig2.savefig('%s/%s/minsep_%s_bars.svg'% (kind,FIG,kind))
    xlbl = '%s: Minimal separation' % kind
    fig3 = plot_overlap_counts(grp_sum_d,y3min,y3max,xlbl)
    fig3.savefig('%s/%s/minsep_%s_overlaps.svg' % (kind,FIG,kind))
    if kind=='GeneExp':
        comb_block_df = sum([C*alg_grpdata_d[alg] for alg,C in alg_const_d.items()])
        comb_block_df = comb_block_df.loc[grp_ct_l,grp_ct_l]

def process_loo_results(res):
    alg_ml_d = defaultdict(dict)
    allcts = res.index.get_level_values(1).unique()
    for actId,grp in res.groupby('ActualId'):
        vc = pd.Categorical(grp.PredictedId).value_counts()
        rem_cts = set(allcts)-set(vc.index.tolist())
        for pId,val in vc.iteritems():
            alg_ml_d[actId][pId] = val
        for pId in rem_cts:
            alg_ml_d[actId][pId] = 0
    ## columns are the actual cts, rows are the estimated ones
    return pd.DataFrame(dict(alg_ml_d)).fillna(0)
    
def plot_ML(kind,method='loo'):
    if kind=='GeneExp':
        y3min,y3max = 0,1300
        y2min,y2max = 1,1500
        nfeat = 4
        cat_ct_d = pd.read_pickle('%s/%s/ct_groupings_d.pkl' % (kind,DAT))
        all_ct_l,grp_ct_l,len_l = order_cts(kind,cat_ct_d)
        clrNORM2 = LogNorm(vmin=1e-2,vmax=1)
        alpha=1
        thr =0.05
    else:
        y3min,y3max = 0,51
        y2min,y2max = 1,100
        nfeat = 3
        cat_ct_d = {}
        ___,all_ct_l = order_cts(kind)
        clrNORM2 = LogNorm(vmin=1e-1,vmax=1)
        alpha=0.5
        thr=0.1
        
    alg_data_d = {}
    alg_grpdata_d = {}
    alg_tier_d = {}
    alg_cmap_d = {'UU':'Blues','WC':'Oranges','WU':'Greens'}
    for alg in alg_l:
        if method == 'loo':
            loo = pd.read_pickle('%s/%s/nfeat_loo_%s_d.pkl' % (kind,OUT,alg))[nfeat]
            alg_data_d[alg] = process_loo_results(loo).loc[all_ct_l,all_ct_l]
            if kind=='GeneExp':
                tier=pd.read_pickle('%s/%s/nfeat_tiered_%s_d.pkl'% (kind,OUT,alg))[nfeat]
                alg_tier_d[alg] = process_loo_results(tier).loc[grp_ct_l,grp_ct_l]
        elif method=='ncvfreq':
            ncv = pd.read_pickle('%s/%s/ncvfreq_%s_%d.pkl' % (kind,OUT,alg,nfeat))
            ncv = pd.DataFrame(ncv)
            alg_data_d[alg] = ncv
            thr = 0.001 if kind=='GeneExp' else 0.001
    cml_df = sum([C*(alg_data_d[alg]/alg_data_d[alg].sum()>thr).astype(float) for alg,C in alg_const_d.items()])
    fig_old = plot_colormap(cml_df,ccm,clrNORM2,labels=True)
    fig_old.savefig('%s/%s/ml_%s-%s_alldata.svg' % (kind,FIG,kind,method))
    
    fig = plot_colormap(alg_data_d,alg_cmap_d,LogNorm(vmin=thr,vmax=1),labels=True)
    fig.savefig('%s/%s/loo_%s-%s_alldata.svg' % (kind,FIG,kind,method))

    hcnts = cml_df.loc[all_ct_l,all_ct_l].values - np.eye(cml_df.shape[0])
    hcnts = np.ma.masked_less(hcnts,0).ravel()
    hcnts = np.ma.masked_invalid(hcnts)
    fig2,grp_sum_d = plot_bar_counts(hcnts,y2min,y2max)
    fig2.savefig('%s/%s/ml_%s-%s_bars.svg' % (kind,FIG,kind,method))

    xlbl = '%s: Nearest neighbors' % kind
    fig3 = plot_overlap_counts(grp_sum_d,y3min,y3max,xlbl)
    fig3.savefig('%s/%s/ml_%s-%s_overlaps.svg' % (kind,FIG,kind,method))
    if kind=='GeneExp' and method=='loo':
        ctier_df = sum([C*(alg_tier_d[alg]/alg_tier_d[alg].sum()>thr/5.).astype(float) for alg,C in alg_const_d.items()])
        fig_oldtier = plot_colormap(ctier_df,ccm,clrNORM2,labels=True)
        fig_oldtier.savefig('%s/%s/mltier_%s-%s_alldata.svg' % (kind,FIG,kind,method))
        
        figt = plot_colormap(alg_tier_d,alg_cmap_d,LogNorm(vmin=thr/5.,vmax=1),labels=True)
        figt.savefig('%s/%s/tiered_%s-%s_alldata.svg' % (kind,FIG,kind, method))
        
        hcntst = ctier_df.loc[grp_ct_l,grp_ct_l].values - np.eye(ctier_df.shape[0])
        hcntst = np.ma.masked_less(hcntst,0).ravel()
        hcntst = np.ma.masked_invalid(hcntst)
        
        fig2t,grp_sum_dt = plot_bar_counts(hcntst,y2min,y2max)
        fig2t.savefig('%s/%s/mltier_%s-%s_bars.svg' % (kind,FIG,kind, method))
        
        fig3t = plot_overlap_counts(grp_sum_dt,y3min,y3max,xlbl)
        fig3t.savefig('%s/%s/mltier_%s-%s_overlaps.svg' % (kind,FIG,kind, method))

if __name__ == '__main__':
    for kind in ['GeneExp','HiC']:
        plot_minsep(kind)
        for meth in ['loo','ncvfreq']:
            plot_ML(kind,meth)
