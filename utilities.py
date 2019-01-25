import pandas as pd
import numpy as np
import pickle as P

def SAVE(obj,fn):
    with open(fn,'wb') as fh:
        P.dump(obj,fh,-1)

def weights(df):
    if isinstance(df,pd.DataFrame):
        mu = df.mean(axis=0)
        sig = df.std(axis=0,ddof=1)
        WT = pd.Series(np.zeros(sig.shape),index=sig.index)
        Ainds = np.where(sig>0)[0]
        WT.iloc[Ainds] = 1/sig.iloc[Ainds]
        ## if contacts are absent, then exclude from weights
        Binds = np.where((sig==0) & (mu==0))[0]
        WT.iloc[Binds] = sig.iloc[Binds]
        Cinds = np.where((sig==0) & (mu!=0))[0]
        M = WT.max()
        if M > 0:
            WT.iloc[Cinds] = 2*M
        else:
            ## if all weights are zero, then weight all directions equally
            WT.iloc[:] = 1
        WT = WT.div(np.linalg.norm(WT.astype(np.float64)))
    else:
        mu = df.mean()
        sig = df.std(ddof=1)
        if sig>0:
            WT = 1/sig
        else:
            WT = 1/np.sqrt(df.shape[0])
    return WT

def weights_unnormalized(df):
    if isinstance(df,pd.DataFrame):
        mu = df.mean(axis=0)
        sig = df.std(axis=0,ddof=1)
        WT = pd.Series(np.zeros(sig.shape),index=sig.index)
        Ainds = np.where(sig>0)[0]
        WT.iloc[Ainds] = 1/sig.iloc[Ainds]
        ## if contacts are absent, then exclude from weights
        Binds = np.where((sig==0) & (mu==0))[0]
        WT.iloc[Binds] = sig.iloc[Binds]
        Cinds = np.where((sig==0) & (mu!=0))[0]
        M = WT.max()
        if M > 0:
            WT.iloc[Cinds] = 2*M
        else:
            ## if all weights are zero, then weight all directions equally
            WT.iloc[:] = 1
        #WT = WT.div(np.linalg.norm(WT.astype(np.float64)))
    else:
        mu = df.mean()
        sig = df.std(ddof=1)
        if sig>0:
            WT = 1/sig
        else:
            WT = 1/np.sqrt(df.shape[0])
    return WT
