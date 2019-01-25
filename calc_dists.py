#!/usr/bin/env python
# encoding: utf-8
"""
calc_dists.py

Created by Thomas Wytock on 2019-01-21.

"""

import sys
sys.path = [p for p in sys.path if not '.local' in p]
import pandas as pd
import os.path as osp
from glob import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from itertools import product,combinations as combs
from functools import partial
from scipy.stats import scoreatpercentile as sap
from scipy.special import comb
import sys
from scoop import futures
from utilities import weights

try:
    isw,isc,kind,isknn,NN = sys.argv[1:]
except ValueError:
    isw,isc,kind,isknn,NN = 'W','C','GeneExp','d','1'

ext = '%s%s' % (isw,isc)
OUT = '%s/Analysis' % kind
NORM = '%s/Normalization' % kind
FS = '%s/Feature_Selection' % kind
DAT = '%s/Data' % kind

if kind=='HiC':
    n_neighbors=7
    if not 'C' in ext:
        data_df = pd.read_pickle('%s/%s_data_selection_test.pkl' % (NORM,kind)).fillna(0)
    else:
        data_df = pd.read_pickle('%s/%s_corr_data.pkl' % (NORM,kind)).fillna(0)
    if NN!='all':
        #assume that NN in coercible to an integer
        feat_l = pd.read_pickle('%s/%s_feat_l_%s_all.pkl' % (FS,kind,ext))
        if int(NN) > len(feat_l):
            NN = str(len(feat_l))
        data_df = data_df.loc[:,feat_l[:int(NN)]]
elif kind=='GeneExp':
    n_neighbors=9
    if not 'C' in ext:
        data_df = pd.read_pickle('%s/nonseq_batch_corrected_data.pkl' % (NORM))
    else:
        data_df = pd.read_pickle('%s/%s_corr_data.pkl' % (NORM,kind))
    if NN!='all':
        #assume that NN in coercible to an integer
        feat_l = pd.read_pickle('%s/%s_feat_l_%s_all.pkl' % (FS,kind,ext))
        if int(NN) > len(feat_l):
            NN = str(len(feat_l))
        data_df = data_df.loc[:,feat_l[:int(NN)]]
sel_cts = data_df.index.get_level_values(1).unique()
ctGB = data_df.groupby(level=1)

def samp_pairwise_dist(ctf_cti,MAX_PAIRS=10000,NREP=25):
    ## make the experiments the columns and rows genes
    ## requirement: take in ctf,cti as a pair
    (ctf,dff),(cti,dfi) = [(ct,ctGB.get_group(ct)) for ct in ctf_cti]
    if isw == 'W':
        WT = weights(dff)
    else:
        if isinstance(dff,pd.DataFrame):
            WT = 1/np.sqrt(dff.shape[1])
        else:
            WT = 1/np.sqrt(dff.shape[0])
    PTILES = np.linspace(5,95,10,dtype=int)
    NPT = PTILES.shape[0]
    N_pairs = dff.shape[0]*dfi.shape[0] if ctf!=cti else comb(dff.shape[0],2,exact=True)
    SAMP = N_pairs > MAX_PAIRS
    if SAMP:
        rep_stats = np.zeros((NREP,NPT))
        for _i in range(NREP):
            sel_inds = np.random.choice(N_pairs,MAX_PAIRS,replace=False)
            pairIter = product(dff.index,dfi.index) if ctf!=cti else combs(dff.index,2)
            pairs=[(acc1,acc2) for _n,(acc1,acc2) in enumerate(pairIter) if _n in sel_inds]
            _X,_Y = zip(*pairs);
            MI_X = pd.MultiIndex.from_tuples(_X)
            MI_Y = pd.MultiIndex.from_tuples(_Y)
            if isinstance(WT,np.float):
                DV = ((dff.loc[MI_X,:].values-dfi.loc[MI_Y,:].values)*WT)
            else:
                DV = ((dff.loc[MI_X,:].values-dfi.loc[MI_Y,:].values)*WT.values)
            D = np.sum(np.square(DV),axis=1)
            ## calculate stats mean,min/median/max
            rep_stats[_i,:] = np.asarray(map(lambda pt: sap(D,pt),PTILES))
        IND = [(ctf,cti,pt) for pt in PTILES]
        MI = pd.MultiIndex.from_tuples(IND)
        opser = pd.Series(np.mean(rep_stats,axis=0),index=MI)
        return opser
    else:
        dat = np.zeros(NPT)
        pairIter = product(dff.index,dfi.index) if ctf!=cti else combs(dff.index,2)
        _X,_Y = zip(*pairIter)
        MI_X = pd.MultiIndex.from_tuples(_X)
        MI_Y = pd.MultiIndex.from_tuples(_Y)
        if isinstance(WT,np.float):
            DV = ((dff.loc[MI_X,:].values-dfi.loc[MI_Y,:].values)*WT)
        else:
            DV = ((dff.loc[MI_X,:].values-dfi.loc[MI_Y,:].values)*WT.values)
        D = np.sum(np.square(DV),axis=1)
        dat[:] = map(lambda pt: sap(D,pt),PTILES)
        IND = pd.MultiIndex.from_tuples([(ctf,cti,pt) for pt in PTILES])
        opser = pd.Series(dat,index=IND)
        return opser

def all_overlap_dist(ctf_cti,NREP=5,nf=3):
    ## need to change the unpacking to go back to genes.
    ctf,cti = ctf_cti
    clf = KNeighborsClassifier(n_neighbors,weights='distance')
    ctf_df = ctGB.get_group(ctf)
    nn = int(ctf_df.shape[0]*(nf-1)/float(nf))
    cti_df = ctGB.get_group(cti)
    n_spl = np.amin([cti_df.shape[0],ctf_df.shape[0],nf]).astype(int)
    yvs = np.asarray([0]*(cti_df.shape[0]) + [1]*(ctf_df.shape[0]))
    tmp_df = pd.concat([cti_df,ctf_df],axis=0)
    REP_D = {}
    for _nrep in range(NREP):
        score_l = []
        skf = StratifiedKFold(n_splits=n_spl,shuffle=True)
        for train_index,test_index in skf.split(tmp_df,yvs):
            train_data = tmp_df.iloc[train_index]
            test_data = tmp_df.iloc[test_index]
            test_gsms = tmp_df.iloc[test_index].index.tolist()
            train_labels = yvs[train_index]
            test_labels = yvs[test_index]
            if isw=='W':
                WT = weights(train_data)
            else:
                WT = 1/np.sqrt(train_data.shape[1])
            if np.any(np.isnan(WT)): continue
            clf=KNeighborsClassifier(min(n_neighbors,nn),weights='distance')
            clf.fit(train_data*WT,train_labels)
            scr = clf.predict_proba(test_data*WT)
            rinds = np.arange(scr.shape[0])
            cinds = (test_labels==1).astype(int)
            scr_ser = pd.Series(np.abs(scr[rinds,cinds]),index=test_gsms)
            score_l.append(scr_ser)
        ser = pd.concat(score_l,axis=0)
        REP_D[_nrep]=ser
    MU = pd.DataFrame(REP_D).mean(axis=1).groupby(level=1).mean()
    SER = pd.Series(MU.loc[[ctf,cti]].values,index=['CTF','CTI'])
    return (ctf,cti),SER

def pairwise_distance_results(feats,MAX_PAIRS=10000,NREP=16):
    inv_d = {}
    kw_order = ['max','mean','median','min']
    spd = partial(samp_pairwise_dist,feats=feats,MAX_PAIRS=MAX_PAIRS,NREP=NREP)
    for ctf in sel_cts:
        L = list(futures.map(spd,[(ctf,cti) for cti in sel_cts]))
        df = pd.concat(L,axis=0)
        df.index = pd.MultiIndex.from_tuples(df.index)
        TF = df.index.get_level_values(1)==ctf
        A = df[TF]
        B = df[~TF]
        comp_data = {}
        for stat in kw_order:
            bv = B.xs(stat,level=2)
            av = A.xs(stat,level=2).values
            comp_data[stat] = (bv < av)
        res = np.dstack(comp_data.values()).sum(axis=2) ## array
        res_df = pd.DataFrame(res,index=bv.index,columns=df.columns)
        inv_d[ctf] = (res_df>0).sum(axis=0)
    ft_pwdr_df = pd.DataFrame(inv_d)
    ft_pwdr_ser = ft_pwdr_df.sum(axis=1)
    return ft_pwdr_ser

def calc_dist(ML=False):
    PTILES = np.linspace(5,95,10,dtype=int)
    numfeat = 'all' if NN=='all' else 'top%s' % NN
    if ML:
        ctps = [(ctf,cti) for ctf in sel_cts  for cti in sel_cts if ctf!=cti]
        L = list(futures.map(all_overlap_dist,ctps))
        alldists_df = pd.DataFrame(dict(L)).T
        alldists_df.to_pickle('%s/%s_dists_%s_ML.pkl' % (OUT,numfeat,ext))
        return alldists_df
    else:
        ctps = [(ctf,cti) for ctf in sel_cts  for cti in sel_cts]
        L = list(futures.map(samp_pairwise_dist, ctps))
        alldists_ser = pd.concat(L,axis=0)
        alldists_ser = alldists_ser.unstack(level=2)
        alldists_ser.columns = alldists_ser.columns.astype(int)
        alldists_ser = alldists_ser.loc[:,PTILES]
        alldists_ser.to_pickle('%s/%s_dists_%s.pkl' % (OUT,numfeat,ext))
        return alldists_ser

def clean_ML_arr(fn):
    X = pd.read_pickle(fn)
    X=X[~np.isnan(X.iloc[:,0])]
    X=X.iloc[:,:2]
    X.index=pd.MultiIndex.from_tuples(X.index)
    X.index.names = ['ACC_i','CTI','CTF']
    X.to_pickle(fn)
    return fn,X

def gather_distances():
    from glob import glob
    ML_list = glob('%s/top*_dists_%s_ML.pkl' % (OUT,ext))
    all_list = glob('%stop*_dists_%s.pkl' % (OUT,ext))
    dist_list = filter(lambda x: '_ML.pkl' not in x, all_list)
    ext_ML_d = {}
    ext_dist_d = {}
    for MLfn in ML_list:
        ml_ser = pd.read_pickle(MLfn)
        nfeat = MLfn.split('_')[3]
        ext_ML_d[nfeat]=ml_ser
    ext_ML_df = pd.DataFrame(ext_ML_d)
    ext_ML_df.to_pickle('%s/top_ML_%s.pkl' % (OUT,ext))
    for distfn in dist_list:
        dist_ser = pd.read_pickle(distfn)
        nfeat = distfn.split('_')[3].split('.')[0]
        ext_dist_d[nfeat]=dist_ser
    ext_dist_df = pd.DataFrame(ext_dist_d)
    ext_dist_df.to_pickle('%s/top_dists_%s.pkl' % (OUT,ext))

if __name__ == '__main__':
    calc_dist(ML=isknn.startswith(('K','k')))

