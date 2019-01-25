#!/usr/bin/bash/

import sys,os.path as osp
sys.path = [p for p in sys.path if not '.local' in p]
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,Binarizer
from sklearn.model_selection import StratifiedKFold

#from utilities import SAVE

def filter_genes(X,kind):
    SS = StandardScaler().fit(X.values)
    X2 = SS.transform(X)
    BIN = Binarizer()
    X2 = BIN.transform(X2)
    X2 = 2*X2 - 1
    X2 = pd.DataFrame(X2,index=X.index,columns=X.columns)
    X_grp = X2.groupby(level='CellType').median().T
    zX = SS.transform(X)
    zX = pd.DataFrame(zX,index=X.index,columns=X.columns)
    X_grptrans = zX.groupby(level='CellType').mean().T
    X_grptrans = pd.DataFrame(X_grptrans,index=X_grp.index,
                              columns=X_grp.columns)
    myL = []
    for xx,col in X_grp.iteritems():
        for yy,val in col[col==0].iteritems():
            val2 = np.sign(X_grptrans.loc[yy,xx])
            myL.append((xx,yy,val2))
    for xx,yy,val2 in myL:
        X_grp.loc[yy,xx]=val2
    rowsums = X_grp.sum(axis=1)
    sel_d = {}
    for gn,S in rowsums.iteritems():
        if S == 0 or S == X_grp.shape[1]:
            sel_d[gn]=False
        else:
            sel_d[gn]=True
    ser = pd.Series(sel_d)
    return X_grp[ser]
kind_l = ['HiC','GeneExp']

def Glauber_stochastic(X,J,num):
    projections = np.empty((num,X.shape[0],X.shape[1]))
    Jp = J.copy()
    np.fill_diagonal(Jp,0)
    for ii in range(num):
        Xp = X.values.copy()
        order = np.random.permutation(np.arange(X.shape[0]))
        for jj in order:
            h = np.dot(Xp,Jp[jj,:])
            Xp[:,jj] = np.sign(h)
        projections[ii,:,:] = Xp
    return np.mean(projections,axis=0)

def Glauber_deterministic(X,J):
    Jp = J.copy()
    np.fill_diagonal(Jp,0)
    Xp = X.copy()
    h = np.dot(Xp,Jp)
    return np.sign(h)

def evaluate_alignment(X,sigs,kind):
    act = X.dot(sigs)
    op_ser = act.idxmax(axis=1)
    matches = op_ser.index.get_level_values(level='CellType')==op_ser.values
    return pd.Series(matches,index=op_ser.index)
    #act_l=np.asarray([L.index(ct) for ct in X.columns.get_level_values(level='CellType')])
    #pred_l = np.argmax(act.values,axis=1)
    #return pd.Series(pred_l==act_l,index=X.columns)


def classify_data(X_tr,X_te,Y_tr,Y_te,kind):
    ct_sigs = filter_genes(X_tr.loc[:,sel_gns],kind)
    A = np.dot(ct_sigs.values.T,ct_sigs.values)/float(ct_sigs.shape[0])
    AI = np.linalg.inv(A)
    ## set AI diagonal to zero
    #for ii in range(A.shape[0]):
    #    AI[ii,ii]=0
    J = np.dot(np.dot(ct_sigs.values,AI),ct_sigs.values.T)/float(ct_sigs.shape[0])
    ## now reduce X_te to the genes composing ct_sigs
    X_te = X_te.loc[:,ct_sigs.index]
    ## now apply Glauber Dynamics
    stsc = Glauber_stochastic(X_te,J,num=100)
    stsc = pd.DataFrame(stsc,index=X_te.index,columns=X_te.columns)
    dsc = Glauber_deterministic(X_te,J)
    dsc = pd.DataFrame(dsc,index=X_te.index,columns=X_te.columns)
    ## evaluate alignment
    stcorr = evaluate_alignment(stsc,ct_sigs,kind)
    dcorr = evaluate_alignment(dsc,ct_sigs,kind)
    return stcorr,dcorr

if __name__ == '__main__':
    for kind in kind_l:
        OUT = '%s/Analysis' % kind
        NORM = '%s/Normalization' % kind
        FS = '%s/Feature_Selection' % kind
        DAT = '%s/Data' % kind
        if kind == 'GeneExp':
            data = pd.read_pickle('%s/nonseq_batch_corrected_data.pkl' % (NORM))
            ## will need the probe/gsym mapping
            ps_gs_map = pd.read_pickle('%s/probe_gsym_mapping.pkl' % (DAT))
            sel_gns = data.columns.tolist()
        elif kind =='HiC':
            data = pd.read_pickle('%s/%s_data_selection.pkl' % (NORM,kind))
            sel_gns = data.columns.tolist()
        CAT = pd.Categorical(data.index.get_level_values(level='CellType'))
        yvs = CAT.codes
        counts = CAT.describe().counts
        st_l, d_l = [], []
        skf = StratifiedKFold(n_splits=3,shuffle=True)
        for tr_i,te_i in skf.split(data,yvs):
            X_tr,X_te = data.iloc[tr_i,:].fillna(0),data.iloc[te_i,:].fillna(0)
            Y_tr,Y_te = yvs[tr_i],yvs[te_i]
            st,d =classify_data(X_tr,X_te,Y_tr,Y_te,kind)
            st_l.append(st)
            d_l.append(d)
        stoch_ser = pd.concat(st_l);
        det_ser = pd.concat(d_l)
        stoch_ser.to_pickle('%s/%s_stoch_ser.pkl' % (OUT,kind))
        det_ser.to_pickle('%s/%s_det_ser.pkl' % (OUT,kind))
