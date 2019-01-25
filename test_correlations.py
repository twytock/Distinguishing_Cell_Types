#!/usr/bin/env python

import sys,os,os.path as osp
sys.path = [p for p in sys.path if not '.local' in p]
from collections import defaultdict
from operator import itemgetter as ig
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from scoop import futures
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT = 'Analysis'
NORM = 'Normalization'
FS = 'Feature_Selection'
DAT = 'Data'
FIG = 'Figures'
kind = sys.argv[1]
n_neighbors = 9 if kind =='GeneExp' else 7

## instead want to use the nci60 batch corrected data
## buchhain paths
def shuffle_each_row(ARR):
    X = ARR.copy()
    return np.random.shuffle(X)

def construct_data_set():
    if kind == 'GeneExp':
        rawdata_df = pd.read_pickle('%s/%s/nonseq_batch_corrected_data.pkl' %(kind,NORM))
    else:
        rawdata_df=pd.read_pickle('%s/%s/%s_data_selection.pkl' % (kind,NORM,kind)).fillna(0)
    corrdata_df=pd.read_pickle('%s/%s/%s_corr_data.pkl' % (kind,NORM,kind)).fillna(0)
    evcs = pd.read_pickle('%s/%s/%s_eigenvectors.pkl' % (kind,NORM,kind)).fillna(0)
    dat_cpy =rawdata_df.values.copy()
    cdat_cpy = corrdata_df.values.copy()
    np.random.shuffle(dat_cpy)
    np.random.shuffle(cdat_cpy)
    rand_uncorr_data = np.dot(dat_cpy,evcs)
    ALL_DATA_DF = np.vstack([corrdata_df.values,rand_uncorr_data,cdat_cpy])
    LL = corrdata_df.shape[0]
    YVALS = np.hstack([np.ones(LL)*_i for _i in range(3)])
    return ALL_DATA_DF,YVALS

ALL_DATA_DF,YVALS = construct_data_set()

def classify_data(inds_tup):
    train_inds,test_inds = inds_tup 
    clf = KNeighborsClassifier(n_neighbors, weights='distance')
    dat,lbls = ALL_DATA_DF[train_inds],YVALS[train_inds]
    clf.fit(dat,lbls)
    tdat,tlbls = ALL_DATA_DF[test_inds],YVALS[test_inds]
    pred = clf.predict(tdat)
    return confusion_matrix(tlbls,pred)

def plot_confusion_matrix():
    mat_fns = glob('%s/%s/corr_conf_mat_*_%s.npy' % (kind,OUT,kind))
    A = np.dstack([np.load(fn).mean(axis=2) for fn in mat_fns])
    mu = np.mean(A,axis=2)
    xticks = np.arange(3)+0.5
    yticks = np.arange(3)+0.5
    xtls = ['Actual','Random','Correlated']
    ytls = xtls
    NORM = mpl.colors.LogNorm(vmin=1e-3,vmax=1)
    fig = plt.figure(figsize=(3.375,3),dpi=300)
    ax = fig.add_subplot(111)
    coll = ax.pcolor(mu/np.mean(mu,axis=0),cmap=cm.Blues,norm=NORM)
    cb = plt.colorbar(coll)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xtls,size=6)
    ax.set_yticklabels(ytls,size=6)
    ax.set_xlabel('Predicted state',size=8)
    ax.set_ylabel('Actual state',size=8)
    cb.set_label('Fraction',size=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(),size=6)
    fig.savefig('%s/%s/%s_confusion_matrix.svg' % (kind,FIG,kind))

def main():
    SSS = StratifiedShuffleSplit(n_splits=10,test_size=1/3.)
    L = list(futures.map(classify_data, SSS.split(ALL_DATA_DF,YVALS) ))
    RES = np.dstack(L)
    np.save('%s/%s/corr_conf_mat_%d_%s.npy' % (kind,OUT,sys.argv[2],kind),RES)

if __name__ == '__main__':
    if sys.argv[2]=='plot':
        plot_confusion_matrix()
    else:
        main()

