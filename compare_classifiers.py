#!/usr/bin/env python
# encoding: utf-8
"""
compare_classifiers.py

Created by Thomas Wytock on 2019-01-21.
"""
import sys,os,os.path as osp
sys.path = [p for p in sys.path if not '.local' in p]
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from scoop import futures
import pandas as pd
from utilities import weights,SAVE
from collections import defaultdict
from functools import partial

classifiers_d = {'KNN':KNeighborsClassifier(9,weights='distance'),
               'SVC':SVC(gamma=2,C=1,probability=True),
               'RF':RandomForestClassifier(max_depth=5,n_estimators=10,max_features=1)}

alldata_df_d = {}
cts_count_d = {}
cts_gb_d = {}
weights_d = {}
kind_ind_ct_d = {}
kind_gse_cols_d = {}
kind_feat_l = {}

OUT = 'Analysis'
NORM = 'Normalization'
FS = 'Feature_Selection'
DAT = 'Data'

## use all features for this step...
for _kind in ['GeneExp','HiC']:
    for EXT in ['KNN','SVC','RF']:
        if _kind == 'GeneExp':
            if EXT == 'KNN':
                alldata_df = pd.read_pickle('%s/%s/%s_corr_data.pkl' % (_kind,NORM,_kind))
            else:
                alldata_df = pd.read_pickle('%s/%s/nonseq_batch_corrected_data.pkl' % (_kind,NORM))
        elif _kind == 'HiC':
            if EXT == 'KNN':
                alldata_df=pd.read_pickle('%s/%s/%s_corr_data.pkl' % (_kind,NORM,_kind)).fillna(0)
            else:
                alldata_df=pd.read_pickle('%s/%s/%s_data_selection.pkl' % (_kind,NORM,_kind)).fillna(0)
        CAT = pd.Categorical(alldata_df.index.get_level_values(level=1))
        idx = CAT.categories
        ind_ct_d = dict(enumerate(idx))
        alldatgb = alldata_df.groupby(level=1)
        alldata_df_d[(_kind,EXT)] = alldata_df
        cts_gb_d[(_kind,EXT)] = alldatgb
    cts_count = alldatgb.count().iloc[:,0]
    cts_count_d[_kind] = cts_count
    kind_ind_ct_d[_kind]=ind_ct_d
    gse_cols_d = {}
    feat_l_d = {}
    if _kind == 'GeneExp':
        lbl_df = pd.read_pickle('%s/%s/Exp_Factors_df.pkl' % (_kind,DAT))
        GSE_cols = lbl_df.loc[:,[lbl for lbl in lbl_df.columns if lbl.startswith('GSE')]]
        for gse,col in GSE_cols.iteritems():
            st = gse.find('[')
            fn = gse.find(']')
            gse_kw = gse[st+1:fn]
            sel_gsms = col[(col!=0)].index
            common_gsms = alldata_df.index.get_level_values(0).isin(sel_gsms)
            if common_gsms.sum() > 0:
                gse_cols_d[gse_kw] = alldata_df.loc[common_gsms].index.get_level_values(0)        
        gse_cols_d['GSE5372'] = pd.Index([u'GSM122528', u'GSM122531', u'GSM122539', u'GSM122540',
                                          u'GSM122541', u'GSM122542', u'GSM122543', u'GSM122544',
                                          u'GSM122546'])
        gse_cols_d['GSE8050'] = pd.Index(['GSM198785'])
        gse_cols_d['GSE8699'] = pd.Index(['GSM215557'])
        gse_cols_d['GSE10934'] = pd.Index(['GSM277272'])
        gse_cols_d['GSE11166'] = pd.Index(['GSM281414'])
        gse_cols_d['GSE13300'] = pd.Index(['GSM335830'])
        gse_cols_d['GSE13736'] = pd.Index(['GSM345269'])
        gse_cols_d['GSE15949'] = pd.Index(['GSM400075'])
    else:
        lbl_df = pd.read_pickle('%s/%s/%s_run_info.pkl' % (_kind,DAT,_kind))
        for srp, acc_df in lbl_df.reset_index().groupby('Study Accession'):
            gse_cols_d[srp] = pd.Index(acc_df.loc[:,'Run Number'])

    for _e in ['UU','WU','WC']:
        targfn = '%s/%s/%s_feat_l_%s.pkl' % (_kind,FS,_kind,_e)
        if osp.exists(targfn):
            fl = pd.read_pickle(targfn)
            feat_l_d[_e] = fl
    feat_l_d['PCA'] = alldata_df_d[(_kind,'KNN')].columns.tolist()
    kind_gse_cols_d[_kind] = gse_cols_d
    kind_feat_l[_kind] = feat_l_d
CKWS = classifiers_d.keys()[::-1]

## should add 1vsall test here as well.
def all_overlap_1_vs_all(ctf_kind_ckw,NREP=25,isW=True,frac=0.1):
    ## need to change the unpacking to go back to genes.
    ctf,kind,ckw = ctf_kind_ckw
    N_NEIGHBORS = 7 if kind=='HiC' else 9
    ctf_df = cts_gb_d[(kind,EXT)].get_group(ctf)
    data_df = alldata_df_d[(kind,EXT)]
    rem_df = data_df[~data_df.index.isin(ctf_df.index)]
    nn = int(ctf_df.shape[0]*(1-frac))
    yvs = np.asarray([0]*(rem_df.shape[0]) + [1]*(ctf_df.shape[0]))
    tmp_df = pd.concat([rem_df,ctf_df],axis=0)
    score_l = []; sensitivity_d = defaultdict(list)
    SSS = StratifiedShuffleSplit(n_splits=NREP, test_size=frac)
    for train_index,test_index in SSS.split(tmp_df, yvs):
        if train_index.shape[0] < 3: continue
        train_data = tmp_df.iloc[train_index]
        test_data = tmp_df.iloc[test_index]
        test_gsms = tmp_df.iloc[test_index].index.tolist()
        train_labels = yvs[train_index]
        test_labels = yvs[test_index]
        if isW:
            WT = weights(train_data)
        else:
            if isinstance(train_data,pd.DataFrame):
                WT = 1/np.sqrt(train_data.shape[1])
            else:
                WT = 1/np.sqrt(train_data.shape[0])
        if np.any(np.isnan(WT)): continue
        if ckw=='KNN':
            clf = KNeighborsClassifier(min(N_NEIGHBORS,nn),weights='distance')
        else:
            clf = classifiers_d[ckw]
        clf.fit(train_data*WT,train_labels)
        scr = clf.predict_proba(test_data*WT)
        isCt = scr[:,1]>0.5
        rinds = np.arange(scr.shape[0])
        cinds = (test_labels==1).astype(int)
        sensitivity_d['TP'].append(((cinds)&(isCt)).sum())
        sensitivity_d['TN'].append(((~cinds)&(~isCt)).sum())
        sensitivity_d['FP'].append(((~cinds)&(isCt)).sum())
        sensitivity_d['FN'].append(((cinds)&(~isCt)).sum())
        sensitivity_d['Frac'].append(frac)
        scr_ser = pd.Series(1 - np.abs(cinds - scr[rinds,cinds]),index=test_gsms)
        score_l.append(scr_ser)
    sensitivity_d = dict(sensitivity_d)
    sense_df = pd.DataFrame(sensitivity_d)
    sense_ser = sense_df.mean(axis=0)
    SER = pd.concat(score_l,axis=0)
    return (ctf,ckw),SER.mean(),sense_ser

def compare_multiclass(feat_l,ckw,nreps=25,frac=.1,kind='GeneExp',alg='UU'):
    ind_ct_d = kind_ind_ct_d[kind]
    data = alldata_df_d[(kind,ckw)]
    if alg=='WC' and ckw=='KNN':
        data = alldata_df_d[(kind,ckw)]
    elif ckw=='KNN':
        data = alldata_df_d[(kind,'RF')]
    else:
        data = alldata_df_d[(kind,ckw)]
    if len(feat_l)>0:
        data = data.loc[:,feat_l].fillna(0)
    CAT = pd.Categorical(data.index.get_level_values(level='CellType'))
    Y = CAT.codes
    N_NEIGHBORS = 7 if kind=='HiC' else 9
    score_arr = np.empty(nreps)
    logprob_arr = np.empty(nreps)
    ct_success_d = defaultdict(list)
    SSS = StratifiedShuffleSplit(n_splits=nreps, test_size=frac)
    for ii,(train_inds,test_inds) in enumerate(SSS.split(data,Y)):
        Xtr,Xte = data.iloc[train_inds],data.iloc[test_inds]
        if ckw=='KNN':
            if alg.startswith('W'):
                WT = weights(pd.DataFrame(Xtr))
            else:
                L = Xtr.shape[1]
                WT = pd.Series(np.ones(L)/np.sqrt(L))
            Xtr = np.multiply(Xtr,WT.values)
            Xte = np.multiply(Xte,WT.values)
        else:
            SS = StandardScaler().fit(Xtr)
            Xtr = SS.transform(Xtr)
            Xte = SS.transform(Xte)
        Ytr,Yte = Y[train_inds],Y[test_inds]
        if ckw == 'KNN':
            clf = KNeighborsClassifier(N_NEIGHBORS,weights='distance')
        else:
            clf = classifiers_d[ckw]
        clf.fit(Xtr,Ytr)
        score_arr[ii] = clf.score(Xte,Yte)
        probs = clf.predict_proba(Xte)
        logprob_arr[ii] = np.log(np.sum((1-probs[np.arange(Xte.shape[0]),Yte])**2))
    return len(feat_l),np.mean(score_arr),np.std(score_arr),np.mean(logprob_arr),np.std(logprob_arr)

def select_gses_nexp(gse_cols_d,L=621):
    gses = np.random.choice(np.array(gse_cols_d.keys()),replace=False,size=len(gse_cols_d.keys()))
    num = 0; ii =0
    while num < L:
        num+=gse_cols_d[gses[ii]].shape[0]
        ii+=1
    return gses[:ii],gses[ii:]

def select_gses_with_constraints(gsm_ct_ser,gse_cols_d,L=621,tries_max=10000):
    satisfied = False
    all_cts = gsm_ct_ser.unique()
    n_tries = 0
    while not satisfied and n_tries<tries_max:
        gses_te,gses_tr = select_gses_nexp(gse_cols_d,L)
        gsms_l = np.hstack([cols for gse,cols in gse_cols_d.items() if gse in gses_tr])
        gsms_l = list(set(gsms_l) & set(gsm_ct_ser.index))
        train_cts = gsm_ct_ser.loc[gsms_l].unique()
        if all_cts.shape[0]==train_cts.shape[0]:
            satisfied=True
        n_tries+=1
    if n_tries>tries_max-1:
        import pdb
        pdb.set_trace()
    return gses_te,gsms_l

def random_loo_pct(feat_l=[],L=621,NREP=25,ckw='KNN',kind='GeneExp',isW=True,isC=True,ctgrp=False):
    unique_id = 'ExpId'
    gse_cols_d = kind_gse_cols_d[kind]
    if isC and ckw=='KNN':
        data = alldata_df_d[(kind,ckw)]
    elif ckw=='KNN':
        data = alldata_df_d[(kind,'RF')]
    else:
        data = alldata_df_d[(kind,ckw)]
    if len(feat_l)>0:
        data = data.loc[:,feat_l].fillna(0)
    loo_stat_l = []
    gse_te_l = []
    if ckw == 'KNN':
        k_neigh=9 if kind=='GeneExp' else 7
    if ctgrp:
        CAT2 = map_cts_2_categories(data.index.get_level_values(level='CellType'))
        code_cat_ser = pd.Series(dict( enumerate(CAT2.categories) ))
    else:
        CAT2 = pd.Categorical(data.index.get_level_values(level='CellType'))
    gsm_ct_ser = pd.Series(dict(zip(data.index.get_level_values(0),CAT2)))
    for _n in range(NREP):
        gses_te,gsm_tr = select_gses_with_constraints(gsm_ct_ser,gse_cols_d,L)
        ## if one ct is missing, then need to pad the training set with nonsensical data...
        A = data.index.get_level_values(0).isin(gsm_tr)
        train_data = data[A].values
        test_data = data[~A]
        T_i = test_data.index
        test_data = test_data.values
        train_labels = CAT2[A]
        test_labels = CAT2[~A]
        if isW and ckw=='KNN':
            WT = weights(train_data)
            train_data = np.multiply(train_data,WT)
            test_data = np.multiply(test_data,WT)
        elif ckw=='KNN':
            WT = 1/np.sqrt(train_data.shape[0])
            train_data = train_data*WT
            test_data = test_data*WT
        else:
            SS = StandardScaler().fit(train_data)
            train_data = SS.transform(train_data)
            test_data = SS.transform(test_data)
        if ckw =='KNN':
            clf = KNeighborsClassifier(k_neigh,weights='distance')
        else:
            clf = classifiers_d[ckw]
        clf.fit(train_data,train_labels.codes)
        probs = clf.predict_proba(test_data)
        test_probs = probs[np.arange(test_data.shape[0]),test_labels.codes]
        pred_codes = np.argmax(probs,axis=1).astype(int)
        if ctgrp:
            pred_ids = code_cat_ser.loc[pred_codes].values
        else:
            pred_ids = map(kind_ind_ct_d[kind].get,pred_codes)
        #sel = (pred_codes == test_labels.codes)#/float(pred_codes.shape[0])
        looDF = pd.DataFrame({'Predicted':pd.Series(pred_codes,index=T_i),
                      'Actual':pd.Series(test_labels.codes,index=T_i),
                      'PredictedId':pd.Series(pred_ids,index=T_i),
                      'ActualId':pd.Series(test_labels,index=T_i),
                      'Prob':pd.Series(test_probs,index=T_i),
                      'ExpId':pd.Series([gse for __ in range(test_data.shape[0])],index=T_i)})
        loo_stat_l.append({'Accuracy':(looDF.ActualId==looDF.PredictedId).mean(),
                         'Mean_Prob':looDF.Prob.mean()})
        gse_te_l.append(gses_te)
    return len(feat_l),{'GSE_list':gse_te_l,'Stats':pd.DataFrame(loo_stat_l)}

def map_cts_2_categories(cts,kind='GeneExp'):
    cat_ct_d = pd.read_pickle('%s/%s/ct_groupings_d.pkl' % (kind,DAT))
    my_d = defaultdict(dict)
    for ii,ct in enumerate(cts):
        cat_l = [k for k,ct_l in cat_ct_d.items() if ct in ct_l]
        if len(cat_l) == 1:
            cat = cat_l[0]
        else:
            print "The number of matching categories is %d" % len(cat_l)
            raise
        my_d['CellType'][ii] = ct
        my_d['Category'][ii] = cat
    df = pd.DataFrame(dict(my_d))
    return pd.Categorical(df.Category)

def leave_one_GSE_out(feat_l=[],ckw='KNN',kind='GeneExp',isW=True,isC=True,ctgrp=False):
    unique_id = 'ExpId'
    gse_cols_d = kind_gse_cols_d[kind]
    if isC and ckw=='KNN':
        data = alldata_df_d[(kind,ckw)]
    elif ckw=='KNN':
        data = alldata_df_d[(kind,'RF')]
    else:
        data = alldata_df_d[(kind,ckw)]
    if len(feat_l)>0:
        data = data.loc[:,feat_l].fillna(0)
    looDF_l = []
    if ckw == 'KNN':
        k_neigh=9 if kind=='GeneExp' else 7
    if ctgrp:
        CAT2 = map_cts_2_categories(data.index.get_level_values(level='CellType'))
        code_cat_ser = pd.Series(dict( enumerate(CAT2.categories)))
    else:
        CAT2 = pd.Categorical(data.index.get_level_values(level='CellType'))
    for gse,cols in gse_cols_d.items():
        ## if one ct is missing, then need to pad the training set with nonsensical data...
        rem_cols = data.index.get_level_values(level=unique_id).difference(cols)
        A = data.index.get_level_values(level=unique_id).isin(rem_cols)
        train_cts = CAT2[A].value_counts().index.unique()
        all_cts = CAT2.value_counts().index.unique()
        if train_cts.shape[0] < all_cts.shape[0]:
            xt_d = {}
            for xct in np.setdiff1d(all_cts,train_cts):
                kp = ('GSMXXXXXX',xct)
                xt_d[kp] = pd.Series([-1e20 for x in data.columns],index=data.columns)
            xt_df = pd.DataFrame(xt_d).T
            xt_df.index.names = data.index.names
            xt_CAT = pd.Categorical(xt_df.index.get_level_values(level='CellType'))
            train_data = pd.concat([data[A],xt_df])
            train_labels = pd.concat([CAT2[A],xt_CAT])
            train_data = train_data.values
        else:
            train_data = data[A].values
            train_labels = CAT2[A]
        test_data = data[~A]
        T_i = test_data.index
        test_data = test_data.values
        if isW and ckw=='KNN':
            WT = weights(train_data)
            train_data = np.multiply(train_data,WT)
            test_data = np.multiply(test_data,WT)
        elif ckw=='KNN':
            L = train_data.shape[1]
            WT = np.ones(L)/np.sqrt(L)
            train_data = np.multiply(train_data,WT)
            test_data = np.multiply(test_data,WT)
        else:
            SS = StandardScaler().fit(train_data)
            train_data = SS.transform(train_data)
            test_data = SS.transform(test_data)
        test_labels = CAT2[~A]
        if ckw =='KNN':
            clf = KNeighborsClassifier(k_neigh,weights='distance')
        else:
            clf = classifiers_d[ckw]
        clf.fit(train_data,train_labels.codes)
        probs = clf.predict_proba(test_data)
        test_probs = probs[np.arange(test_data.shape[0]),test_labels.codes]
        pred_codes = np.argmax(probs,axis=1).astype(int)
        if ctgrp:
            pred_ids = code_cat_ser.loc[pred_codes].values
        else:
            pred_ids = map(kind_ind_ct_d[kind].get,pred_codes)
        #sel = (pred_codes == test_labels.codes)#/float(pred_codes.shape[0])
        looDF = pd.DataFrame({'Predicted':pd.Series(pred_codes,index=T_i),
                      'Actual':pd.Series(test_labels.codes,index=T_i),
                      'PredictedId':pd.Series(pred_ids,index=T_i),
                      'ActualId':pd.Series(test_labels,index=T_i),
                      'Prob':pd.Series(test_probs,index=T_i),
                      'ExpId':pd.Series([gse for __ in range(test_data.shape[0])],index=T_i)})
        looDF_l.append(looDF)
    return len(feat_l),pd.concat(looDF_l,axis=0)


def repeat_comparisons(frac_l=[.1],ttype='loo',MAX_FEATS=51):
    for KIND in ['GeneExp','HiC']:# 'GeneExp','NCI60','HiC']:
        if ttype in ('loo','tiered'):
            CTGRP = True if ttype=='tiered' else False
            if CTGRP and KIND=='HiC':
                continue
            feat_l_d = kind_feat_l[KIND]
            for ckw in CKWS:
                loo_GSE = partial(leave_one_GSE_out,kind=KIND,ckw=ckw,ctgrp=CTGRP)
                if ckw =='KNN':
                    for alg in ['WC','PCA','WU','UU']:
                        if alg=='WU':
                            f = partial(loo_GSE,isW=True,isC=False)
                        elif alg=='UU':
                            f = partial(loo_GSE,isW=False,isC=False)
                        else:
                            f = loo_GSE
                        if alg in feat_l_d.keys():
                            fl = feat_l_d[alg]
                        else:
                            continue
                        ub = min((MAX_FEATS,len(fl)))
                        ii,op_l1= zip(*list(futures.map(f,[fl[:n] for n in range(1,ub)])))
                        _opd = dict(zip(ii,op_l1))
                        SAVE(_opd,'%s/%s/nfeat_%s_%s_d.pkl' % (KIND,OUT,ttype,alg))
                else:
                    if 'UU' in feat_l_d.keys():
                        fl = feat_l_d['UU']
                    else:
                        continue
                    ub = min((MAX_FEATS,len(fl)))
                    f = partial(loo_GSE,isW=False,isC=False)
                    ii,op_l=zip(*list(futures.map(f,[fl[:n] for n in range(1,ub)])))
                    ckwd = dict(zip(ii,op_l))
                    SAVE(ckwd,'%s/%s/nfeat_%s_%s_d.pkl' % (KIND,OUT,ttype,ckw))
                    ## need to output the results
        elif ttype in ('loopct','tieredpct'):
            CTGRP = True if ttype=='tieredpct' else False
            feat_l_d = kind_feat_l[KIND]
            N = alldata_df_d[(KIND,CKWS[0])].shape[0]
            if CTGRP and KIND=='HiC':
                continue
            for ckw in CKWS:
                for frac in frac_l:
                    rnd_loo = partial(random_loo_pct,L=int(frac*N),ckw=ckw,kind=KIND,ctgrp=CTGRP)
                    if ckw =='KNN':
                        for alg in ['WC','PCA','WU','UU']:
                            if alg=='WU':
                                f = partial(rnd_loo,isW=True,isC=False)
                            elif alg=='UU':
                                f = partial(rnd_loo,isW=False,isC=False)
                            else:
                                f = rnd_loo
                            if alg in feat_l_d.keys():
                                fl = feat_l_d[alg]
                            else:
                                continue
                            ub = min((MAX_FEATS,len(fl)))
                            ii,op_l1= zip(*list(futures.map(f,[fl[:n] for n in range(1,ub)])))
                            _opd = dict(zip(ii,op_l1))
                            SAVE(_opd,'%s/%s/nfeat_%s_%s-%.2f_d.pkl' % (KIND,OUT,ttype,alg,frac))
                    else:
                        if 'UU' in feat_l_d.keys():
                            fl = feat_l_d['UU']
                        else:
                            continue
                        ub = min((MAX_FEATS,len(fl)))
                        f = partial(rnd_loo,isW=False,isC=False)
                        ii,op_l=zip(*list(futures.map(f,[fl[:n] for n in range(1,ub)])))
                        ckwd = dict(zip(ii,op_l))
                        SAVE(ckwd,'%s/%s/nfeat_%s_%s-%.2f_d.pkl' % (KIND,OUT,ttype,ckw,frac))
                    ## need to output the results
        else:
            feat_l_d = kind_feat_l[KIND]
            frac_comp_d = {}
            frac_compstd_d = {}
            frac_av1_d = {}
            for frac in frac_l:
                if ttype=='avs1':
                    av1_ff = partial(all_overlap_1_vs_all,frac=frac)
                    arg_l = [(ct,KIND,ckw) for ct in cts_count_d[KIND].index.tolist()
                                           for ckw in CKWS]
                    ctf_l,score_l,sense_ser_l = zip(*list(futures.map(av1_ff,arg_l)))
                    av1_score_ser = pd.Series(dict(zip(ctf_l,score_l)))
                    av1_sense_df = pd.DataFrame(dict(zip(ctf_l,sense_ser_l)))
                    av1_score_ser.to_pickle('%s/%s/ml_avs1_score_%.2f_d.pkl'% (KIND,OUT,frac))
                    av1_sense_df.to_pickle('%s/%s/ml_avs1_sense_%.2f_d.pkl' % (KIND,OUT,frac))
                elif ttype=='multi':
                    for CKW in CKWS:
                        comp_multi_ff = partial(compare_multiclass,frac=frac,kind=KIND,ckw=CKW)
                        if CKW=='KNN':
                            for ALG in ['UU','WC','WU','PCA']:
                                if ALG in feat_l_d.keys():
                                    fl = feat_l_d[ALG]
                                else:
                                    continue
                                ub = min((MAX_FEATS,len(fl)))
                                fl = feat_l_d[ALG]
                                F = partial(comp_multi_ff,alg=ALG)
                                sel_feat_l = [fl[:n] for n in range(1,ub)]
                                nfeat_l,xscore_l,xstd_l,lp_l,lpstd_l = zip(*list(futures.map(F,sel_feat_l)))
                                ckw_scores = pd.Series(dict(zip(nfeat_l,xscore_l)))
                                ckw_stds = pd.Series(dict(zip(nfeat_l,xstd_l)))
                                ckw_scores.to_pickle('%s/%s/ml_%s_%.2f_d.pkl' % (KIND,OUT,ALG,frac))
                                ckw_stds.to_pickle('%s/%s/ml_%s-std_%.2f_d.pkl' % (KIND,OUT,ALG,frac))
                                ckw_lps = pd.Series(dict(zip(nfeat_l,lp_l)))
                                ckw_lpstds = pd.Series(dict(zip(nfeat_l,lpstd_l)))
                                ckw_lps.to_pickle('%s/%s/ml_%s_%.2f_logprob_d.pkl' % (KIND,OUT,ALG,frac))
                                ckw_stds.to_pickle('%s/%s/ml_%s-std_%.2f_logprob_d.pkl' % (KIND,OUT,ALG,frac))
                        else:
                            if 'UU' in feat_l_d.keys():
                                fl = feat_l_d['UU']
                            else:
                                continue
                            ub = min((MAX_FEATS,len(fl)))
                            F = comp_multi_ff
                            sel_feat_l = [fl[:n] for n in range(1,ub)]
                            nfeat_l,xscore_l,xstd_l,lp_l,lpstd_l=zip(*list(futures.map(F,sel_feat_l)))
                            ckw_scores = pd.Series(dict(zip(nfeat_l,xscore_l)))
                            ckw_stds = pd.Series(dict(zip(nfeat_l,xstd_l)))
                            ckw_scores.to_pickle('%s/%s/ml_%s_%.2f_d.pkl' % (KIND,OUT,CKW,frac))
                            ckw_stds.to_pickle('%s/%s/ml_%s-std_%.2f_d.pkl' % (KIND,OUT,CKW,frac))
                            ckw_lps = pd.Series(dict(zip(nfeat_l,lp_l)))
                            ckw_lpstds = pd.Series(dict(zip(nfeat_l,lpstd_l)))
                            ckw_lps.to_pickle('%s/%s/ml_%s_%.2f_logprob_d.pkl' % (KIND,OUT,CKW,frac))
                            ckw_stds.to_pickle('%s/%s/ml_%s-std_%.2f_logprob_d.pkl' % (KIND,OUT,CKW,frac))
    return

if __name__ == '__main__':
    method = sys.argv[1]
    if 'pct' not in method:
        repeat_comparisons(frac_l=[],ttype=method)
    else:
        repeat_comparisons(frac_l=np.linspace(0.05,0.2,4),ttype=method)
    #repeat_comparisons(np.linspace(0.25,0.75,3),ttype='multi')
    #repeat_comparisons(np.linspace(0.25,0.75,3),ttype='avs1')
    #plot_ctmap_results()

