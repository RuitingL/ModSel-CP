#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import scipy.stats as stats
from quantile_forest import RandomForestQuantileRegressor

from random import sample
import time

import pandas as pd
from sklearn.model_selection import train_test_split

import mtds_func_cqrBreak as cf




# import data
df_protein = pd.read_csv('realData/CASP.csv', header=None)
df_protein = df_protein.iloc[1:, :] #rid of original header
X_full = df_protein.iloc[:, 1:]   # All rows, all columns except the first one
Y_full = df_protein.iloc[:, 0]   # All rows, only the first column

X_tr, X, Y_tr, Y = train_test_split(X_full, Y_full, train_size=300, shuffle=True, random_state=42) # n_tr = 300

# train the quantile regressors using the training data (pretrained models)
d = X_tr.shape[1] # the number of features
alpha = 0.1
Betas = np.linspace(alpha/10000, 4*alpha, 10)
split_features = np.linspace(d / 10, d, 10)
N_ESTIMATOR = np.linspace(100, 400, 4).astype(int)

Models = []
for N_EST in N_ESTIMATOR:
    for f in split_features:
        fr = max(1, round(f))
        rfqr = RandomForestQuantileRegressor(n_estimators=N_EST, max_features=fr)
        rfqr.fit(X_tr, Y_tr)
        Models.append(rfqr)




# prep data
def data_subsample(n_subsample):
    # Subsample n_subsample indices without replacement
    sampled_indices = np.random.choice(np.arange(X.shape[0]), size=n_subsample, replace=False)
    X_version = X.iloc[sampled_indices,:]
    Y_version = Y.iloc[sampled_indices]
    
    # split into calibration and test data
    X_cal, X_test, Y_cal, Y_test = train_test_split(X_version, Y_version, train_size=500, shuffle=True)
    return X_cal, X_test, Y_cal, Y_test




# experiment
N_rep = 100; n_subsample = 600; split_portion = 0.5

Coverage = np.zeros((5,N_rep)); Length = np.zeros((5,N_rep))
benchL = np.zeros((len(Models)*len(Betas),N_rep))

start_time = time.time()
for t in range(N_rep):
    X_cal, X_test, Y_cal, Y_test = data_subsample(n_subsample)
    X_cal = X_cal.to_numpy(dtype = float); Y_cal = Y_cal.to_numpy(dtype = float)
    X_test = X_test.to_numpy(dtype = float); Y_test = Y_test.to_numpy(dtype = float)
    
    # only D_cal interval construction (except LOO)
    VFCPtest_cov, VFCPave_length = cf.YKsplit_CQR(Models, Betas, X_cal, Y_cal, X_test, Y_test, 
                                                  alpha, split_portion)
    
    Coverage[4,t] = VFCPtest_cov; Length[4,t] = VFCPave_length
    
    benchL[:,t] = cf.Mmodels_length_CQR(Models, Betas, X_cal, Y_cal, X_test, alpha)
    
    ensQ, ens_sel_len_factor = cf.ModSel_CQRcal(Models, Betas, X_cal, Y_cal, alpha)
    efcpQ, efcp_sel_len_factor = cf.YKbaseline_CQRcal(Models, Betas, X_cal, Y_cal, alpha)
    efcp_adjQ, efcp_adj_sel_len_factor = cf.YK_adj_CQRcal(Models, Betas, X_cal, Y_cal, alpha)
    
    n_cal = len(Y_cal); n_test = len(Y_test)
    
    testCov = np.zeros((4,n_test)); testLen = np.zeros((4,n_test))
    for s in range(n_test):
        # reshape the test point into a 2D array/df
        x_test = X_test[s,:].reshape(1,-1)
        y_test = Y_test[s]
        
        coverE, lengthE, _, _ = cf.ModSel_CQRtest(Models, Betas, ensQ, ens_sel_len_factor, 
                                                     n_cal, x_test, y_test)
        coverL, lengthL, _, _ = cf.ModSelLOO_CQR(Models, Betas, X_cal, Y_cal, x_test, y_test, alpha)
        coverEFCP, lengthEFCP = cf.YKbaseline_CQRtest(Models, Betas, efcpQ, efcp_sel_len_factor, 
                                                      n_cal, x_test, y_test)
        coverEFCP_adj, lengthEFCP_adj = cf.YKbaseline_CQRtest(Models, Betas, 
                                                     efcp_adjQ, efcp_adj_sel_len_factor, n_cal, x_test, y_test)
        
        testCov[0,s] = coverE; testCov[1,s] = coverL
        testCov[2,s] = coverEFCP; testCov[3,s] = coverEFCP_adj
        
        testLen[0,s] = lengthE; testLen[1,s] = lengthL
        testLen[2,s] = lengthEFCP; testLen[3,s] = lengthEFCP_adj
    
    
    for j in range(4):
        Coverage[j,t] = np.mean(testCov[j,:])
        Length[j,t] = np.mean(testLen[j,:])
    
# calculate final average
cov = np.mean(Coverage, axis=1)
leng = np.mean(Length, axis=1)
min_single_md_len = np.min(np.mean(benchL, axis=1))
    
end_time = time.time()
print(f"Elapsed time: {(end_time - start_time)/60} minutes")
    
print(f"Coverage of ModSel, LOO, YKbaseline, YK_adj, YKsplit: {cov}")
print(f"Length of ModSel, LOO, YKbaseline, YK_adj, YKsplit: {leng}")
print(f"min single model conformal set length: {min_single_md_len}")

