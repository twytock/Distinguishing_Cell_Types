#!/usr/bin/env python
# encoding: utf-8
"""
nonconvexity_tests.py

Created by Thomas Wytock on 2019-01-21.
"""
import sys
sys.path = [p for p in sys.path if not '.local' in p]
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from scoop import futures
#from scoop import shared
from functools import partial
from scipy.sparse import dok_matrix
from itertools import combinations as combs
from utilities import weights,SAVE

# set up some parameters that can be set from the command line if need be.
try:
    isw,isc,kind = sys.argv[1:]
except ValueError:
    isw,isc,kind = 'W','C','GeneExp'

OUT = '%s/Analysis' % kind
NORM = '%s/Normalization' % kind
FS = '%s/Feature_Selection' % kind
DAT = '%s/Data' % kind
ext = '%s%s' % (isw,isc)

if kind=='HiC':
    n_neighbors=7
    if not 'C' in ext:
        data_df = pd.read_pickle('%s/%s_data_selection_test.pkl' % (NORM,kind)).fillna(0)
    else:
        data_df = pd.read_pickle('%s/%s_corr_data.pkl' % (NORM,kind)).fillna(0)
elif kind=='GeneExp':
    n_neighbors=9
    if not 'C' in ext:
        data_df = pd.read_pickle('%s/nonseq_batch_corrected_data.pkl' % (NORM))
    else:
        data_df = pd.read_pickle('%s/%s_corr_data.pkl' % (NORM,kind))

feat_l = pd.read_pickle('%s/%s_feat_l_%s_all.pkl' % (FS,kind,ext))

ctGB = data_df.groupby(level='CellType')
CAT = pd.Categorical(data_df.index.get_level_values(level='CellType'))
code_cat_ser = pd.Series(dict([(CAT.categories.tolist().index(ct),ct) for ct in CAT.categories]))
sel_cts = data_df.index.get_level_values(level='CellType').unique()
ct_counts = ctGB.count().iloc[:,0]

def generate_pseudodata_sparse(ctf,M=10000):
    N = ct_counts.loc[ctf]
    npair = N*(N-1)/2
    MM = dok_matrix((M,N),dtype=np.float64)
    ## generate the pseudodata first, then transform according to
    ## the variances?
    #np.random.seed(987654321)
    if npair > M:
        ## sample pairs randomly
        S = np.random.choice(npair,size=M,replace=False)
        sel_prs=[(x,y) for ii,(x,y) in enumerate(combs(range(N),2)) if ii in S]
        ## make dok_matrix
        for ii,(x,y) in enumerate(sel_prs):
            MM[ii,x] = 0.5
            MM[ii,y] = 0.5
    else:
        ind = 0
        npts = M//npair + 1
        XP = npts * npair - M
        S = np.random.choice(npair,size=XP,replace=False)
        for ii,(x,y) in enumerate(combs(range(N),2)):
            if ii in S:
                pts = np.arange(npts)[1:]/float(npts)
            else:
                pts = (np.arange(npts)+1.)/(npts+1.)
            for zz,pt in enumerate(pts):
                MM[ind+zz,x] = 1-pt
                MM[ind+zz,y] = pt
            ind += pts.shape[0]
    return MM.tocsr()

sparse_ind_d = dict([(ct,generate_pseudodata_sparse(ct)) for ct in sel_cts]) ## need to change

def quick_pseudodata(ctf,feats):
    ctf_df = ctGB.get_group(ctf)
    if isw=='W':
        WT = weights(ctf_df.loc[:,feats])
        wt_data = data_df.loc[:,feats]*WT
    else:
        WT = 1/np.sqrt(len(feats))
        wt_data = data_df.loc[:,feats]*WT
    X = np.dot(sparse_ind_d[ctf].todense(),wt_data.xs(ctf,level='CellType'))
    return pd.DataFrame(X,columns = feats),wt_data

def ctf_test_pseudodata(ctf,feats):
    ps_data,wt_data = quick_pseudodata(ctf,feats)
    yvs = CAT.codes #inds_class_ser.loc[wt_data.index]
    clf=KNeighborsClassifier(n_neighbors,weights='distance')
    clf.fit(wt_data.values,yvs)
    ctf_code = CAT.categories.tolist().index(ctf)
    counts = code_cat_ser.loc[clf.predict(ps_data.values)].value_counts()/float(ps_data.shape[0])
    scr = clf.score(ps_data.values,np.array([ctf_code for __ in range(ps_data.shape[0])]))
    return ctf,scr,counts

def test_pseudodata(feats,ext):
    '''test_pseudodata
    This function tests the predictive ability of pseudodata
    on the chords linking two measurements of the same cell type.
Argument: feats -- a list of features to classify the cell type
returns: a series, indexed by final cell type with the accuracy statistic'''
    ctf_tp=partial(ctf_test_pseudodata,feats=feats)
    OPL = list(futures.map(ctf_tp, sel_cts)) ## this needs to be changed...
    A,B,C = zip(*OPL)
    res_d = dict(zip(A,B)) #CHECK
    freq_d = dict(zip(A,C))
    SAVE(freq_d,'%s/ncvfreq_%s.pkl' % (OUT,ext))
    ## take the minimum score to ensure maximum convexity in all cases
    ncv_ser = pd.Series(res_d)
    ncv_ser.to_pickle('%s/ncv_%s.pkl' % (OUT,ext))
    return 0

if __name__ == '__main__':
    #num_arr = np.hstack([np.arange(1,50,dtype=int),np.array([150000])])
    fin = False
    if kind=='GeneExp':
        sel_num = [4]
    elif kind=='HiC':
        sel_num = [3]
    for _n in sel_num:
        if int(_n) > len(feat_l):
            _n = str(len(feat_l))
            fin = True
        ext = '%s%s_%s' % (isw,isc,_n)
        test_pseudodata(feat_l[:int(_n)],ext)
        if fin:
            break

