#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import scipy.stats as stats



# YK-baseline

def YKbaseline_class(Mmodels, X_cal, Y_cal, x_test, y_test, alpha):
    M = len(Mmodels); n = len(Y_cal)
    S = np.zeros(M)
    select_length = np.zeros(M)
    k = math.ceil((n+1)*(1-alpha))
    
    for m in range(M):
        mdl = Mmodels[m]
        Cond_prob = mdl.predict_proba(X_cal) # n*K array
        Scores = np.zeros(n)
        for i in range(n):
            Scores[i] = 1-Cond_prob[i, int(Y_cal[i])]
        
        S[m] = np.sort(Scores)[k-1] # S^{(m)}_{(k)}
        pred_prob = mdl.predict_proba(x_test)
        select_length[m] = (np.sum(Cond_prob-(1-S[m])>=0) + np.sum(pred_prob-(1-S[m])>=0))/(n+1)
        
    
    mhat = np.argmin(select_length)
    select_mdl = Mmodels[mhat]
    select_pred = select_mdl.predict_proba(x_test)
    select_pred = select_pred.flatten()
    
    if (1-select_pred[int(y_test)])>S[mhat]:
        cover = 0
    else:
        cover = 1
    
    pred_set = np.where(select_pred >= 1-S[mhat])[0]
    card = len(pred_set)
    
    return cover, card, pred_set




#YK_adj: alpha adjusted version

def YK_adj_class(Mmodels, X_cal, Y_cal, x_test, y_test, alpha):
    M = len(Mmodels); n = len(Y_cal)
    til_alpha = n*alpha/(n+1) + 1/(n+1) - n*(1/(3*np.sqrt(n)) + np.sqrt(np.log(2*M)/(2*n)))/(n+1)
    
    cover, card, pred_set = YKbaseline_class(Mmodels, X_cal, Y_cal, x_test, y_test, til_alpha)
    return cover, card, pred_set




# YK-split

def YKsplit_class(Mmodels, X_cal, Y_cal, x_test, y_test, alpha, split_portion):
    M = len(Mmodels)
    n1 = math.ceil(len(Y_cal)*split_portion); n2 = len(Y_cal) - n1
    k1 = math.ceil((n1 + 1)*(1-alpha)); k2 = math.ceil((n2 + 1)*(1-alpha))
    select_length = np.zeros(M)
    
    X_sel = X_cal[:n1, :]; Y_sel = Y_cal[:n1]
    X_c = X_cal[n1:, :]; Y_c = Y_cal[n1:]
    
    for m in range(M):
        mdl = Mmodels[m]
        Cond_prob = mdl.predict_proba(X_sel) # n1*K array
        Scores = np.zeros(n1)
        for i in range(n1):
            Scores[i] = 1-Cond_prob[i, int(Y_sel[i])]
        
        S = np.sort(Scores)[k1-1] # S^{(m)}_{(k)}
        select_length[m] = np.sum(Cond_prob-(1-S)>=0)/(n1)
        
    # select the model
    mhat = np.argmin(select_length)
    model = Mmodels[mhat]
    # conformal
    Cond_prob_sel =  model.predict_proba(X_c)
    S = np.zeros(n2)
    for i in range(n2):
        S[i] = 1-Cond_prob_sel[i,int(Y_c[i])]
    
    Q = np.sort(S)[k2 - 1]
    pred_prob = model.predict_proba(x_test)
    pred_prob = pred_prob.flatten()
    
    if (1-pred_prob[int(y_test)])>Q:
        cover = 0
    else:
        cover = 1
    
    pred_set = np.where(pred_prob>=1-Q)[0]
    card = len(pred_set)
    
    return cover, card, pred_set




# ModSel-CP

def ModSel_class_def(Mmodels, X_cal, Y_cal, x_test, y_test, alpha):
    M = len(Mmodels); n = len(Y_cal)
    S = np.zeros((2,M))
    select_length = np.zeros((2,M))
    k = math.ceil((n+1)*(1-alpha))
    select_length_y = []
    
    for m in range(M):
        mdl = Mmodels[m]
        Cond_prob = mdl.predict_proba(X_cal) # n*K array
        Scores = np.zeros(n)
        for i in range(n):
            Scores[i] = 1-Cond_prob[i, int(Y_cal[i])]
        
        Scores_sort = np.sort(Scores)
        S[1,m] = Scores_sort[k-1] # S^{(m)}_{(k)}
        S[0,m] = Scores_sort[k-2] # S^{(m)}_{(k-1)}
        pred_prob = mdl.predict_proba(x_test)
        pred_prob = pred_prob.flatten()
        select_length[1,m] = (np.sum(Cond_prob-(1-S[1,m])>=0) + np.sum(pred_prob-(1-S[1,m])>=0))/(n+1)
        select_length[0,m] = (np.sum(Cond_prob-(1-S[0,m])>=0) + np.sum(pred_prob-(1-S[0,m])>=0))/(n+1)
        
        sly = np.zeros(len(pred_prob))
        for y in range(len(pred_prob)):
            sly[y] = (np.sum(Cond_prob>=pred_prob[y]) + np.sum(pred_prob>=pred_prob[y]))/(n+1)
        
        select_length_y.append(sly)
        
    # min select length
    min_L = np.min(select_length[1,:])
    
    # filter models
    calM = np.where(select_length[0,:]<= min_L)[0]
    effeM = len(calM)
    S = S[:,calM]; select_length = select_length[:,calM]
    select_length_y = np.array(select_length_y) #M*K array
    select_length_y = select_length_y[calM,:]
    
    pred_set = []
    K = len(select_length_y[0,:])
    L = np.zeros((effeM, K))
    for m in range(effeM):
        for y in range(K):
            L[m,y] = min(select_length[1,m], max(select_length[0,m], select_length_y[m,y]))
    
    for y in range(K):
        mhat_y = np.argmin(L[:,y])
        mdl = Mmodels[calM[mhat_y]]
        pred_prob = mdl.predict_proba(x_test)
        pred_prob = pred_prob.flatten()
        if (1-pred_prob[y])<=S[1,mhat_y]:
            pred_set.append(y)
    
    pred_set = np.unique(pred_set)
    card = len(pred_set)
    # check coverage
    ext_set = np.append(pred_set, y_test)
    if len(np.unique(ext_set))>card:
        cover = 0
    else:
        cover = 1
    
    return cover, card, pred_set, effeM




# ModSel-CP-LOO

def ModSelLOO_class(Mmodels, X_cal, Y_cal, x_test, y_test, alpha):
    M = len(Mmodels); n = len(Y_cal)
    k = math.ceil((n+1)*(1-alpha))
    
    S_all = np.zeros((n,M))
    transform_S_all = np.zeros((n,M))
    LOOselect_length = []
    select_length = np.zeros(M)
    
    # calculate the score
    for m in range (M):
        mdl = Mmodels[m]
        Cond_prob = mdl.predict_proba(X_cal) # n*K array
        pred_cond = mdl.predict_proba(x_test)
        pred_cond = pred_cond.flatten()
        
        K = len(pred_cond)
        sel_length = np.zeros((n,K))
        for i in range(n):
            p = Cond_prob[i, int(Y_cal[i])]
            S_all[i,m] = 1-p
            transform_S_all[i,m] = (np.sum(Cond_prob >= p) + np.sum(pred_cond >= p))/(n+1)
            
        q_hat = np.sort(S_all[:,m])[k-1]
        select_length[m] = (np.sum(Cond_prob >= (1-q_hat)) + np.sum(pred_cond >= (1-q_hat)))/(n+1)
        
        for i in range(n):
            tmp_S = S_all[:,m]
            for y in range(K):
                tmp_S[i] = 1-pred_cond[y]
                q = np.sort(tmp_S)[k-1]
                sel_length[i,y] = (np.sum(Cond_prob >= (1-q)) + np.sum(pred_cond >= (1-q)))/(n+1)
        
        LOOselect_length.append(sel_length)
    
    # select the model
    mhat = np.argmin(select_length)
    select_mdl = Mmodels[mhat]
    pred_prob = select_mdl.predict_proba(x_test)
    Cond_prob = select_mdl.predict_proba(X_cal)
    pred_prob = pred_prob.flatten()
    K = len(pred_prob) # the number of classes
    
    # quick check
    random_index = np.random.choice(M)
    rd_check = LOOselect_length[random_index]
    if len(rd_check[0,:]) != K:
        print("mistake: incorrect # of classes")
    
    # calculate the prediction set
    pred_set = []
    for y in range(K):
        LHS = (np.sum(Cond_prob >= pred_prob[y]) + np.sum(pred_prob >= pred_prob[y]))/(n+1)
        
        Lscore = np.zeros((n,M))
        for m in range(M):
            tmp = LOOselect_length[m]
            Lscore[:,m] = tmp[:,y]
        
        RHS = np.zeros(n)
        for i in range(n):
            mi_hat = np.where(Lscore[i,:] == np.min(Lscore[i,:]))[0]
            RHS[i] = np.max(transform_S_all[i,mi_hat])
        
        Q = np.sort(RHS)[k-1]
        if LHS <= Q:
            pred_set.append(y)
    
    # determine coverage, length
    pred_set = np.unique(pred_set)
    card = len(pred_set)
    # check coverage
    ext_set = np.append(pred_set, y_test)
    if len(np.unique(ext_set))>card:
        cover = 0
    else:
        cover = 1
    
    return cover, card, pred_set
