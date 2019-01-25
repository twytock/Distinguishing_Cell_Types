#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
calculate_correlations.py
Created on Jan 23 2019

@author: thomaspwytock
"""

import pandas as pd
import numpy as np
import os.path as osp

def construct_evecs(KIND,EPS=1e-12):
    NORM = osp.join(KIND,'Normalization')
    if KIND == 'GeneExp':
        uncorr_data = pd.read_pickle('%s/nonseq_batch_corrected_data.pkl' % (NORM) )
    elif KIND == 'HiC':
        uncorr_data = pd.read_pickle('%s/%s_data_selection.pkl' % (NORM,KIND))
        if len(uncorr_data.columns.names) != 4:
            (chrA,binA),(chrB,binB) = map(lambda _x: zip(*_x), zip(*uncorr_data.columns.tolist()))
            uncorr_data.columns = pd.MultiIndex.from_tuples(zip(chrA,binA,chrB,binB))
            uncorr_data.columns.names = ['chrA','binA','chrB','binB']
    uncorr_data = uncorr_data.loc[uncorr_data.index.sort_values()]
    uncorr_data = uncorr_data.loc[:,uncorr_data.columns.sort_values()]
    uncorr_data.index.names = ['ExpId','CellType']
    if KIND == 'GeneExp':
        uncorr_data.to_pickle('%s/nonseq_batch_corrected_data.pkl' % (NORM) )
    elif KIND == 'HiC':
        uncorr_data.to_pickle('%s/%s_data_selection.pkl' % (NORM,KIND))
    print "Calculating correlations."
    uncorr_values =uncorr_data.fillna(0).values
    if KIND=='GeneExp':
        vals = (uncorr_values - uncorr_values.mean(axis=0)) / uncorr_values.std(axis=0,ddof=1)
    else:
        vals = uncorr_values
    vals[np.isnan(vals)] = 0
    U,S,V = np.linalg.svd(vals.T,full_matrices=False)
    evecs = U[:,S>EPS]
    evals = S[S>EPS]
    evl_lbls = ['{:.6e}'.format(evl) for evl in evals]
    eval_ser = pd.Series(evals,evl_lbls)
    eval_ser.to_pickle('%s/%s_eigenvalues.pkl' % (NORM,KIND))
    cols = uncorr_data.columns
    evec_df = pd.DataFrame(evecs,index=cols,columns=evl_lbls)
    #evec_df = evec_df.loc[uncorr_data.columns]
    evec_df.to_pickle('%s/%s_eigenvectors.pkl' % (NORM,KIND))
    print("Projecting Eigenvectors.")
    pda = np.dot(uncorr_data.fillna(0).values,evec_df.values)
    proj_data = pd.DataFrame(pda,index=uncorr_data.index,columns=evec_df.columns)
    proj_data.to_pickle('%s/%s_corr_data.pkl' % (NORM,KIND))
    return proj_data


def main():
    EPS = 1e-12
    for KIND in ['HiC','GeneExp']:
        proj_data = construct_evecs(KIND,EPS)

if __name__ == '__main__':
    main()
