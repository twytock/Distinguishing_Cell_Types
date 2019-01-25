
'''split_and_reconstitute_large_files.py'''
import os,sys

import pandas as pd
import numpy as np
from glob import glob

def split_data_file_by_rows():
    max_file_size = 25 * 2 ** 20 ## in bytes
    bytes_per_elt = 2 ** 3
    for kind in ['GeneExp','HiC']:
        if kind == 'GeneExp':
            X = pd.read_pickle('%s/Normalization/nonseq_batch_corrected_data.pkl' % kind)
            X2 = pd.read_pickle('%s/Data/Exp_Factors_df.pkl' % kind)
        elif kind =='HiC':
            X = pd.read_pickle('%s/Normalization/HiC_data_selection.pkl' % kind)
        nrow = int(( max_file_size // (bytes_per_elt * X.shape[1]))*.9)
        A = np.arange(0,X.shape[0],nrow)
        B = A + nrow; B[-1] = X.shape[0]
        for jj,(lb,ub) in enumerate(zip(A,B)):
            fn = '%s/Normalization/data_part_%d.pkl' % (kind,jj)
            X.iloc[lb:ub].to_pickle(fn)
        if kind == 'GeneExp':
            nrow2 = int(( max_file_size // (bytes_per_elt * X2.shape[1]))*.9)
            A2 = np.arange(0,X2.shape[0],nrow2)
            B2 = A2 + nrow2; B2[-1] = X2.shape[0]
            for jj,(lb,ub) in enumerate(zip(A2,B2)):
                fn2 = '%s/Data/data_part_%d.pkl' % (kind,jj)
                X2.iloc[lb:ub].to_pickle(fn2)
            os.remove('%s/Normalization/nonseq_batch_corrected_data.pkl' % kind)
            os.remove('%s/Data/Exp_Factors_df.pkl' % kind)
        elif kind =='HiC':
            os.remove('%s/Normalization/HiC_data_selection.pkl' % kind)
    return 0

def reconstitute_data():
    for kind in ['GeneExp','HiC']:
        fn_l = glob('%s/Normalization/data_part_*.pkl' % kind)
        if kind =='GeneExp':
            fn_l2 = glob('%s/Data/data_part_*.pkl' % kind)
            if len(fn_l) > 0:
                df = pd.concat([pd.read_pickle(fn) for fn in fn_l])
                df.to_pickle('%s/Normalization/nonseq_batch_corrected_data.pkl' % kind)
                map(os.remove,fn_l)
            if len(fn_l2) > 0:
                df2 = pd.concat([pd.read_pickle(fn) for fn in fn_l2])
                df2.to_pickle('%s/Data/Exp_Factors_df.pkl' % kind)
                map(os.remove,fn_l2)
        elif kind == 'HiC':
            if len(fn_l) > 0:
                df = pd.concat([pd.read_pickle(fn) for fn in fn_l])
                df.to_pickle('%s/Normalization/HiC_data_selection.pkl' % kind)
                map(os.remove,fn_l)

if __name__ == '__main__':
    if sys.argv[1].startswith(('s','S')):
        split_data_file_by_rows()
    else:
        reconstitute_data()
    