#!/usr/bin/env python
# encoding: utf-8
"""
feature_selection.py

Created by Thomas Wytock on 2019-01-21.

"""
import sys, os, os.path as osp
sys.path = [p for p in sys.path if not '.local' in p]
import numpy as np
import pandas as pd
import cPickle as cp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from scoop import futures
#from scoop import shared
from functools import partial
from scipy.sparse import dok_matrix
from itertools import product,combinations as combs
from scipy.special import comb
from utilities import SAVE, weights, weights_unnormalized
from collections import defaultdict


# set up some parameters that can be set from the command line if need be.
try:
    isw,isc,KIND,max_feat= sys.argv[1:]
    max_feat = int(max_feat)
except ValueError:
    isw,isc,KIND,max_feat= 'W','C','HiC',10
n_neighbors=9 if KIND=='HiC' else 7
DAT = osp.join(KIND,'Data')
NORM = osp.join(KIND,'Normalization')
FS = osp.join(KIND,'Feature_Selection')


estr = '%s%s' % (isw,isc)
ct_counts_min = 10

if KIND == 'GeneExp':
    if isc=='C':
        proj_data = pd.read_pickle('%s/%s_corr_data.pkl' % (NORM,KIND))
    else:
        proj_data = pd.read_pickle('%s/nonseq_batch_corrected_data.pkl' % (NORM) )
elif KIND == 'HiC':
    if isc=='C':
        proj_data = pd.read_pickle('%s/%s_corr_data.pkl' % (NORM,KIND))
    else:
        proj_data = pd.read_pickle('%s/%s_data_selection.pkl' % (NORM,KIND))

def get_HiC_run_info(sel_data_df):
    HiC_details = pd.read_pickle('%s/HiC_run_info.pkl' % DAT)
    ct_run_pairs = []
    HiC_details = HiC_details.loc[sel_data_df.index.get_level_values(0)]
    for runId,row in HiC_details.iterrows():
        if row.loc['Library Name'] not in ('THP-1 macrophage','PrEC','RWPE1'):
            ct_run_pairs.append((runId,row.loc['Library Name']))
        else:
            if row.loc['Library Name']=='THP-1 macrophage':
                ct_run_pairs.append((runId,'MDM'))
            elif row.loc['Library Name'] in ('PrEC','RWPE1'):
                ct_run_pairs.append((runId,'prostate'))
    MI = pd.MultiIndex.from_tuples(ct_run_pairs)
    sel_data_df = sel_data_df.loc[MI.levels[0]]
    sel_data_df.index=MI
    #data_proj_df = data_proj_df.loc[MI.levels[0]]
    #data_proj_df.index = MI
    if isc=='C':
        svfn = '%s/HiC_corr_data.pkl' % NORM
    else:
        svfn = '%s/HiC_data_selection.pkl' % NORM
    with open(svfn,'wb') as fh:
        cp.dump(sel_data_df,fh,2)
    my_id_ser = pd.Series(dict(zip(MI.get_level_values(0),MI.get_level_values(1))))
    return sel_data_df, my_id_ser

def select_untreated_GSMS():
    nci_ds = pd.read_pickle('%s/gene-expression_details_df.pkl' % DAT)
    nci_ds = nci_ds[nci_ds.index.isin(proj_data.index.get_level_values('GSM'))]
    gn_ctrl_cols = np.array(['None', 'BRAF : WT ; NRAS : WT', 'BRAF : ND ; NRAS : ND' ], dtype=object)
    control_cols = np.array([ 'None', 'serum; passage 8', 'No', 'serum; passage 10',
       'passage 19', 'serum; passage 13', 'serum; passage 20', 'serum; passage 35',
       'serum; passage 3', 'passage 13', 'grade 4', 'DMSO 24h', 'wt', u'mock',
       u'hiPSC-RPE_59M8_SDIA_GSE50738', u'hiPSC-RPE_59SV3_SDIA_GSE50738',
       u'hiPSC-RPE_59SV4_SDIA_GSE50738', u'h-RPE1 (lot-0F3237, Lonza)',
       u'h-RPE2 (lot-0F32920, Lonza)', u'h-RPE3 (lot-181239, Lonza)',
       u'retinal pigment epithelial cell line', u'control conditioned medium; 3h',
       u'human-induced-NPC', u'used in the balanced background',
       'mesenchymal', 'stem-like', u'2h', u'24h', u'8h',
       u'human AB serum; 24h', u'human AB serum; 5d', u'before exercise',
       u'after exercise', u'DMSO; 4h', u'HFF', u'DMSO; 72h', 'monolayer', 'spheroid',
       'DMSO 4h', 'DMSO 10h', 'DMSO 18h', 'ethanol 0.1% 6h', 'ethanol 0.1% 24h',
       u'healthy', u'non-RPE', '1d', '3d', u'GFP; 0h',  u'GFP; 2h', u'GFP; 24h',
       u'GFP; 6h', u'GFP; 12h', u'mock; 5d',  u'mock; 12h', 'DMSO',
       'Synchronized; DMSO 0h', 'Synchronized; DMSO 2h',
       'Synchronized; DMSO 4h', 'Synchronized; DMSO 6h',
       'Synchronized; DMSO 7h', 'Synchronized; DMSO 8h',
       'Synchronized; DMSO 9h', 'Synchronized; DMSO 10h',
       u'ESC_derived_beta_cells', 'fetal', u'embryonic stem cells BG01',
       u'embryonic stem cells H9', u'H9_induced_hepatocyte', u'iPS_induced_hepatocyte',
       'DMSO 21d', 'mock 21d', u'embryonic stem cells H1', 'DMSO 2h', 'DMSO 6h',
       'DMSO 1h', u'primary_islet_cells', 'DMSO 72h', u'primary_hepatocyte',
       u'fetal native retinal pigment epithelial cell',
       u'adult native retinal pigment epithelial cell',
       u'retinal pigment epithelial cell line APRE-19',
       u'motor_neuron_control', 'grade 3', 'grade 1', 'grade 2', u'liver',
       u'naive iPSC', u'naive embryonic stem cells', 'mock 10h',
       'unselected; mock', u'embryonic stem cells BG03', u'embryonic stem cells WIBR1',
       u'embryonic stem cells WIBR2', u'embryonic stem cells WIBR3',
       u'embryonic stem cells WIBR7', 'metastatic', 'primary_tumor', 'mock 1uM 3d',
       'cancer-unenriched', 'PC-3/Mc invasive', 'PC-3/S non-invasive',
       'xenograft', 'sensitive', 'DMSO 48h', u'human-induced-cardiomyocytes',
       'glucose 1mM 7d', 'glucose 25mM 7d', 'ethanol 0.1% 3h',
       'unstable microsatellite', 'stable microsatellite', 'stem-like; spheroid', 'GFP',
       'GFP; monolayer', 'GFP; spheroid', u'embryonic stem cells Shef3',
       u'embryonic stem cells H14', u'embryonic stem cells Shef1',
       '2-dim culture', '3-dim culture' ], dtype=object)
    nci_sel=nci_ds[(nci_ds.Treatment.isin(control_cols)) &( nci_ds.GeneTarget.isin(gn_ctrl_cols))]
    my_id_d = {}
    for _id,row in nci_sel.iterrows():
        CellLine,CellType = row.loc['CellLineName'],row.loc['CellType']
        if not CellLine=='None':
            my_id_d[_id] = CellLine
        else:
            if row.loc['Histology'] == 'None':
                my_id_d[_id] = CellType
            elif row.loc['Subtype'] == 'None':
                my_id_d[_id] = '--'.join(row.loc[['CellType','Histology']].values)
            else:
                my_id_d[_id] ='--'.join(row.loc[['CellType','Histology','Subtype']].values)
    SER=pd.Series(my_id_d)
    return nci_sel,SER

if len(proj_data.index.levels) <2:
    if KIND =='GeneExp':
        nci_sel,id_num_ser_all = select_untreated_GSMS()
    else:
        proj_data,id_num_ser_all = get_HiC_run_info(proj_data)
else:
    id_num_ser_all = pd.Series(dict(zip(*map(lambda _i: proj_data.index.get_level_values(_i),[0,1]))))
CAT_all = pd.Categorical(id_num_ser_all)
## default is to do all cell types
## want to remove gene perturbations from the dataframe
ct_counts = CAT_all.value_counts() ##map(len,class_inds_l)
sel_cts = ct_counts[ct_counts >= ct_counts_min].index
id_num_ser = id_num_ser_all[id_num_ser_all.isin(sel_cts)]
allgsms = id_num_ser.index
CAT = pd.Categorical(id_num_ser)
if not isinstance(proj_data.index,pd.MultiIndex):
    new_MI = pd.MultiIndex.from_tuples([tuple(xx.tolist()) for __,xx in id_num_ser.reset_index().iterrows()])
    proj_data = proj_data.loc[allgsms]
    proj_data.index = new_MI
    proj_data.index.names=['ExpId','CellType']

def all_overlap_dist(ctf,feats,NREP=25):
    min_num = ct_counts.loc[ctf]
    #cat = pd.Categorical(nonseq_data.index.get_level_values('CellType')) ## need to change
    yvs = CAT.codes
    tot = CAT.value_counts().loc[ctf]
    selcol = CAT.categories.get_indexer([ctf])[0]
    nf = min(min_num,2)
    ddf = proj_data.loc[:,feats].fillna(0)
    ## for stability, need to repeat this several times.
    REP_D = {}
    for _nrep in range(NREP):
        score_l = []
        skf = StratifiedKFold(n_splits=nf,shuffle=True)
        for train_index,test_index in skf.split(ddf,yvs):
            train_data = ddf.iloc[train_index]
            test_data = ddf.iloc[test_index]
            test_inds = ddf.iloc[test_index].index.tolist()
            train_labels = yvs[train_index]
            test_labels = yvs[test_index]
            clf=KNeighborsClassifier(min(n_neighbors,int(tot/float(nf))),weights='distance')
            if (train_labels==selcol).sum()>2:
                WT = weights(train_data[train_labels==selcol])
            else:
                WT = 1/np.sqrt(train_data.shape[1])
            clf.fit(train_data*WT,train_labels)
            scr = clf.predict_proba(test_data*WT)
            scr_ser = pd.Series((test_labels==selcol) - scr[:,selcol],index=test_inds)
            score_l.append(scr_ser)
        ser = pd.concat(score_l,axis=0)
        REP_D[_nrep]=ser
    SER = pd.DataFrame(REP_D).mean(axis=1)
    return ctf,np.sum(np.square(SER))

def calc_dist(feats):
    AOD = partial(all_overlap_dist,feats=feats)
    X = dict(list(map(AOD,sel_cts)))
    strat_ser = pd.Series(X)
    return feats[-1],strat_ser

def samp_pairwise_dist(ctf_cti,feats,rem_feats,MAX_PAIRS=1000,NREP=5):
    ## make the experiments the columns and rows genes
    ## requirement: take in ctf,cti as a pair
    func_l = [np.amin,np.median,np.amax]
    func_lbl_l = ['min','median','max']
    sel_data_df = proj_data.loc[:,feats+rem_feats].fillna(0)
    (ctf,dff),(cti,dfi) = [(ct,sel_data_df.xs(ct,level='CellType').T) for ct in ctf_cti]
    #TF_SEL = proj_data.columns.isin(feats)
    #rem_feats = proj_data.T[~TF_SEL].index.tolist()
    if isw=='W':
        WT = weights_unnormalized(dff.T)
    else:
        WT = pd.Series([1/np.sqrt(dff.shape[0])])
    N_pairs = dff.shape[1]*dfi.shape[1] if ctf!=cti else comb(dff.shape[1],2,exact=True)
    SEL_INDS = np.array(map(dff.index.get_loc,feats),dtype=int)
    REM_INDS = np.array(map(dff.index.get_loc,rem_feats),dtype=int)
    if N_pairs > MAX_PAIRS:
        rep_stats = np.zeros((NREP,len(func_l),len(rem_feats)))
        for _i in range(NREP):
            sel_inds = np.random.permutation(np.arange(N_pairs))[:MAX_PAIRS]
            pairIter = product(dff.columns,dfi.columns) if ctf!=cti else combs(dff.columns,2)
            pairs=[(gsm1,gsm2) for _n,(gsm1,gsm2) in enumerate(pairIter) if _n in sel_inds]
            dat = np.zeros((MAX_PAIRS,len(rem_feats)))
            _X,_Y = zip(*pairs)
            DV = ((dff.loc[:,_X].values-dfi.loc[:,_Y].values).T*WT.values).T
            DV_UNSEL = np.square(DV[REM_INDS])
            if isw=='W':
                WT_UNSEL = np.square(WT[REM_INDS])
            if len(feats)>0:
                DV_SELSUM = np.sum(np.square(DV[SEL_INDS]),axis=0)
                if isw=='W':
                    WT_SELSUM = np.sum(np.square(WT[SEL_INDS].values),axis=0)
                    WT_TOT = np.sqrt(WT_UNSEL+WT_SELSUM)
                dat[:,:] = np.sqrt(DV_UNSEL+DV_SELSUM).T
            else:
                dat[:,:] = np.sqrt(DV_UNSEL.T)
                if isw=='W':
                    WT_TOT = np.sqrt(WT_UNSEL)
            if isw=='W':
                dat = np.nan_to_num(np.divide(dat,WT_TOT.values))
            ## calculate stats mean,min/median/max
            rep_stats[_i,:,:] = np.vstack(map(lambda fcn: fcn(dat,axis=0),func_l))
        IND = pd.MultiIndex.from_tuples([(ctf,cti,stat) for stat in func_lbl_l])
        opdf = pd.DataFrame(np.mean(rep_stats,axis=0),index=IND,columns=rem_feats)
        return opdf
    else:
        dat = np.zeros((N_pairs,len(rem_feats)))
        pairIter=product(dff.columns,dfi.columns) if ctf!=cti else combs(dff.columns,2)
        _X,_Y = zip(*list(pairIter))
        if ctf!=cti:
            DV = ((dff.loc[:,_X].values-dfi.loc[:,_Y].values).T*WT.values).T
        else:
            DV = ((dff.loc[:,_X].values-dff.loc[:,_Y].values).T*WT.values).T
        DV_UNSEL = np.square(DV[REM_INDS])
        if isw=='W':
            WT_UNSEL = np.square(WT[REM_INDS])
        if len(feats)>0:
            DV_SELSUM = np.sum(np.square(DV[SEL_INDS]),axis=0)
            if isw=='W':
                WT_SELSUM = np.sum(np.square(WT[SEL_INDS].values),axis=0)
                WT_TOT = np.sqrt(WT_UNSEL+WT_SELSUM)
            dat[:,:] = np.sqrt(DV_UNSEL+DV_SELSUM).T
        else:
            dat[:,:] = np.sqrt(DV_UNSEL.T)
            if isw=='W':
                WT_TOT = np.sqrt(WT_UNSEL)
        if isw=='W':
            dat = np.nan_to_num(np.divide(dat,WT_TOT.values))
        stats = map(lambda fcn: fcn(dat,axis=0),func_l)
        IND = pd.MultiIndex.from_tuples([(ctf,cti,stat) for stat in func_lbl_l ])
        opdf = pd.DataFrame(stats,index=IND,columns=rem_feats)
        return opdf

def pairwise_distance_results(feats,rem_feats,MAX_PAIRS=1000,NREP=16):
    inv_d = {}
    kw_order = ['max','median','min']
    spd = partial(samp_pairwise_dist,feats=feats,rem_feats=rem_feats,MAX_PAIRS=MAX_PAIRS,NREP=NREP)
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

def feature_selection_old(rem_feats,sel_feats):
    """Arguments:
    feat - "fixed" features
    sel_feats - pool of features to select from
    Returns:
    next set of features - DataFrame of selected features"""
    ## calculate distances, minsep and strat
    ft_pwdr_ser = pairwise_distance_results(sel_feats,rem_feats)
    #print("calculating KNN model")
    X2 = dict(list(futures.map(calc_dist,[sel_feats+[rf] for rf in rem_feats])))
    ft_knnprob_ser = pd.DataFrame(X2).sum(axis=0)
    ##heuristic for selecting feature check to make sure that it gives reasonable selection
    return pd.DataFrame({'PWD':ft_pwdr_ser,'KNNprob':ft_knnprob_ser})

def select_features(**kwargs):
    ext = 'None' if not 'ext' in kwargs.keys() else kwargs['ext']
    p = 0.5 if not 'p' in kwargs.keys() else kwargs['p']
    N_MAX_FEAT = 50 if not 'N_MAX_FEAT' in kwargs.keys() else kwargs['N_MAX_FEAT']
    all_features = proj_data.columns.tolist()
    if not 'selfeat_fn' in kwargs.keys():
        feature_list = []
        rem_features = list(set(all_features))
    else:
        feature_list = pd.read_pickle(kwargs['selfeat_fn'])
        rem_features = list(set(all_features)-set(feature_list))
    if isc=='C':
        rem_features = sorted(rem_features,key=lambda k: float(k),reverse=True)
    else:
        rem_features = sorted(rem_features)
    for ii in range(N_MAX_FEAT):
        res_df = feature_selection_old(rem_features,feature_list)
        res_df.to_pickle('test_%s_%d_%s.pkl' %(KIND,ii,ext))
        ft = (p * res_df.PWD + res_df.KNNprob).idxmin()
        feature_list.append(ft)
        rem_features = list(set(rem_features)-set([ft]))
        if isc=='C':
            rem_features = sorted(rem_features,key=lambda k: float(k),reverse=True)
        else:
            rem_features = sorted(rem_features)
        SAVE(feature_list,'%s/%s_feat_l_%s.pkl' % (FS,KIND,ext))
        print(ii)

def feature_selection_pair(f_fl):
    """Arguments:
    limdf - Global - DataFrame of gene expression data
Returns:
    sel_df - DataFrame of selected features"""
    ## heuristic for selecting feature check to make sure that it gives reasonable selection
    feat,feat_l = f_fl
    limdf = proj_data.loc[:,feat_l]
    allpair_d = {}
    GBct = limdf.groupby(level=1)
    c_df = GBct.count()
    for cti in sel_cts:
        #print cti
        M = c_df.loc[cti]
        cti_df = GBct.get_group(cti)
        for ctf in sel_cts:
            if cti==ctf: continue
            ctf_df = GBct.get_group(ctf)
            if isw=='W':
                WT = weights(ctf_df)
            else:
                WT = np.ones(ctf_df.shape[1])/np.sqrt(ctf_df.shape[1])
                WT = pd.Series(WT,index=ctf_df.columns)
            ctf_df = ctf_df.multiply(WT)
            cti_dfw = cti_df.multiply(WT)
            ctf_mu,ctf_var = ctf_df.mean(),ctf_df.var(ddof=1)
            cti_mu,cti_var = cti_dfw.mean(),cti_dfw.var(ddof=1) #axis=0
            N = c_df.loc[ctf]
            pooled = np.sqrt(((M-1)*cti_var + (N-1)*ctf_var)/(N+M))
            bimodality = np.abs(cti_mu-ctf_mu)/pooled
            allpair_d[(cti,ctf)] = bimodality.fillna(0)
    allpair_df = pd.DataFrame(allpair_d)
    return feat,allpair_df.values.mean()

def alternate_feature_selection(ptile_tup=([.6],'60%')):
    ptile_l,ptstr = ptile_tup
    ct_ordinds_d = {}
    denom = (proj_data>0).sum(axis=0)
    for ct,grp in proj_data.groupby(level='CellType'):
        ct_cnt = (grp>0).sum(axis=0)
        thr = ct_cnt.describe(percentiles=ptile_l).loc[ptstr]
        A = ct_cnt/denom
        INDS = proj_data.T[ct_cnt>thr].index
        ct_ordinds_d[ct] = A.loc[INDS].sort_values(ascending=False)
    return ct_ordinds_d

def select_features_uncorr(**kwargs):
    ext = 'None' if not 'ext' in kwargs.keys() else kwargs['ext']    
    p = 0.5 if not 'p' in kwargs.keys() else kwargs['p']
    SEARCH = 5000 if not 'SEARCH' in kwargs.keys() else kwargs['SEARCH']
    N_MAX_FEAT = 50 if not 'N_MAX_FEAT' in kwargs.keys() else kwargs['N_MAX_FEAT']
    ct_ordinds_d = alternate_feature_selection()
    cts_ord_by_len = ct_counts.sort_values(ascending=False).index.tolist()
    if not 'selfeat_fn' in kwargs.keys():
        feature_list = []
    else:
        feature_list = pd.read_pickle(kwargs['selfeat_fn'])
    for ii in range(N_MAX_FEAT):
        ct_ind = ii % ct_counts.shape[0]
        ct = cts_ord_by_len[ct_ind]
        rem_features = [i for i in ct_ordinds_d[ct].index if i not in feature_list]
        if len(rem_features)>SEARCH:
            rem_features=rem_features[:SEARCH]
        res_df = feature_selection_old(rem_features,feature_list)
        ft = (p * res_df.PWD + res_df.KNNprob).idxmin()
        feature_list.append(ft)
        SAVE(feature_list,'%s/HiC_feat_l_%s.pkl' % (FS,ext))

def order_features_bypair(**kwargs):
    ext = 'None' if not 'ext' in kwargs.keys() else kwargs['ext']
    feat_seq = defaultdict(list)
    if 'sel_feats' in kwargs.keys():
        feat_seq['Feature'] = kwargs['sel_feats']
        start = len(feat_seq['Feature'])
    else:
        start=0
    unselfeat = set(proj_data.columns.tolist()) - set(feat_seq['Feature'])
    #for N in range(start,start+3):#hic_data_df.shape[1]):
        #print(N)
    tmp_feat_l = [sorted(feat_seq['Feature']+[f]) for f in unselfeat]
    L = list(futures.map(feature_selection_pair,zip(unselfeat,tmp_feat_l)))
    feat_ser = pd.Series(dict(L))
    all_feats = kwargs['sel_feats'] + feat_ser.sort_values(ascending=False).index.tolist()
    SAVE(all_feats,'%s/%s_feat_l_%s_all.pkl' % (FS,KIND,ext))
    return #feat_seq


if __name__ == '__main__':
    if KIND=='HiC' and isc!='C':
        select_features_uncorr(p=0.5,N_MAX_FEAT=max_feat,ext=estr)
    else:
        select_features(p=0.5,N_MAX_FEAT=max_feat,ext=estr)
    feat_l = pd.read_pickle('%s/%s_feat_l_%s.pkl' % (FS,KIND,estr))
    order_features_bypair(sel_feats=feat_l,ext=estr)

