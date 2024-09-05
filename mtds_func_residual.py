#!/usr/bin/env python
# coding: utf-8


import numpy as np
import math
import scipy.stats as stats
from sklearn.linear_model import Ridge
from random import sample



#YK-baseline

def YKbaseline_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha):
    M = len(Mmodels); n = len(Y_cal)
    S = np.zeros(M)
    k = math.ceil((n+1)*(1-alpha))
    
    for m in range(M):
        mdl = Mmodels[m]
        Residuals = np.sort(np.abs(Y_cal - mdl.predict(X_cal)))
        S[m] = Residuals[k-1] # S^{(m)}_{(k)}
    
    u = np.min(S); mhat = np.argmin(S)
    mu_hat = Mmodels[mhat].predict(x_test)
    
    if np.abs(y_test - mu_hat)>u:
        cover = 0
    else:
        cover = 1
    
    length = 2*u
    interv = [mu_hat-u, mu_hat+u]
    
    return cover, length, interv




#YK_adj: alpha adjusted version

def YK_adj_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha):
    M = len(Mmodels); n = len(Y_cal)
    til_alpha = n*alpha/(n+1) + 1/(n+1) - n*(1/(3*np.sqrt(n)) + np.sqrt(np.log(2*M)/(2*n)))/(n+1)
    if math.ceil((n+1)*(1-til_alpha)) > n: # return the real line
        cover = 1
        length = np.inf
        interval = [-np.inf, np.inf] # more rigorously should be \mathcal{Y}
    else:
        cover, length, interval = YKbaseline_res(Mmodels, X_cal, Y_cal, x_test, y_test, til_alpha)
    return cover, length, interval




#YK-split: data-splitting

def YKsplit_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha, split_portion):
    M = len(Mmodels)
    n1 = math.ceil(len(Y_cal)*split_portion); n2 = len(Y_cal) - n1
    k1 = math.ceil((n1 + 1)*(1-alpha)); k2 = math.ceil((n2 + 1)*(1-alpha))
    S = np.zeros(M)
    
    X_sel = X_cal[:n1, :]; Y_sel = Y_cal[:n1]
    X_c = X_cal[n1:, :]; Y_c = Y_cal[n1:]
    
    for m in range(M):
        mdl = Mmodels[m]
        Residuals = np.sort(np.abs(Y_sel - mdl.predict(X_sel)))
        S[m] = Residuals[k1-1] # S^{(m)}_{(k)}
        
    mhat = np.argmin(S)
    model = Mmodels[mhat]
    Restmp = np.sort(np.abs(Y_c - model.predict(X_c))); Res = Restmp[k2 - 1]
    mu_test = model.predict(x_test)
    
    if np.abs(y_test - mu_test)>Res:
        cover = 0
    else:
        cover = 1
    
    length = 2*Res
    interval = [mu_test-Res, mu_test+Res]
    return cover, length, interval




# ModSel-CP

def ModSel_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha):
    M = len(Mmodels); n = len(Y_cal)
    k = math.ceil((n+1)*(1-alpha))
    
    S = np.zeros((M, 2))
    pred_values_filtered = []
    
    for m in range(M):
        mdl = Mmodels[m]
        tmpRes = np.sort(np.abs(Y_cal - mdl.predict(X_cal)))
        S[m, 0] = tmpRes[k-2] # S^{(m)}_{(k-1)}
        S[m, 1] = tmpRes[k-1] # S^{(m)}_{(k)}
    
    u = np.min(S[:,1])
    
    #filter models
    find_mdl = np.where(S[:,0]-u <=0)[0]
    # the number of models in calM
    calM = len(find_mdl)
    
    for m in find_mdl:
        pred_values_filtered.append(Mmodels[m].predict(x_test))
    
    mu_test = np.sort(np.array(pred_values_filtered))
    
    #check cover or not
    if np.min(np.abs(mu_test - y_test))>u:
        cover = 0
    else:
        cover = 1
    
    #calculate length, determine the intervals and check connectness
    if calM >1:
        if np.max(mu_test[1:] - mu_test[:-1])<= 2*u:
            connect = 1
            left = np.min(mu_test) - u; right = np.max(mu_test) +u
        else:
            connect = 0
            breakpoint = np.where(mu_test[1:] - mu_test[:-1] -2*u >0)[0]
            left = np.append(np.min(mu_test)-u, mu_test[(breakpoint+1)]-u)
            right = np.append(mu_test[breakpoint]+u, np.max(mu_test)+u)
    else:
        connect = 1
        left = mu_test - u; right = mu_test+u
    
    if isinstance(right, np.ndarray):
        length = np.sum(right-left)
    else:
        length = right-left
    
    Interv = [left, right]
    
    return cover, length, Interv, connect, calM




# ModSel-CP-LOO

## auxillary functions for LOO

def CALMi(i, S_all, S_q, pred_value):
    M = len(S_all[0,:])
    Lo = np.zeros(M); Up = np.zeros(M)
    
    for m in range(M):
        if S_all[i,m] < S_q[1,m]:
            Lo[m] = S_q[1,m]
            Up[m] = S_q[2,m]
        else:
            Lo[m] = S_q[0,m]
            if S_all[i,m] > S_q[1,m]:
                Up[m] = S_q[1,m]
            else:
                Up[m] = S_q[2,m]
    
    # filter models
    find_ind = np.where(Lo <= np.min(Up))[0]
    Li = Lo[find_ind]; Ui = Up[find_ind]; Mu_i = pred_value[find_ind]; Sis = S_all[i,find_ind]
    
    ui = np.min(Ui)
    m_ui = np.where(Ui == ui)[0]
    Si_ui = max(Sis[m_ui])
    return Li, ui, Mu_i, Sis, Si_ui



def intersection(mu_sub, L_sub):
    bmu = mu_sub[0]; bL = L_sub[0]
    Intercepts = np.zeros(len(mu_sub)-1); pointer = np.zeros(len(mu_sub)-1)
    
    for j in range(len(mu_sub)-1):
        if L_sub[j+1] > bL:
            pointer[j] = 1
            if bmu+L_sub[j+1] <= mu_sub[j+1] - L_sub[j+1]: #rl type of intersection (safe)
                Intercepts[j] = (bmu+mu_sub[j+1])/2 
            else: #rflat type of intersection (safe)
                Intercepts[j] = bmu+L_sub[j+1]
        else:
            if L_sub[j+1] < bL:
                pointer[j] = 1
                if mu_sub[j+1] - bL <= bmu + bL: #flatl type of intersection (safe)
                    Intercepts[j] = mu_sub[j+1] - bL
                else: #rl type of intersection (safe)
                    Intercepts[j] = (bmu+mu_sub[j+1])/2 
            else:
                if bmu+bL <= mu_sub[j+1] - bL:
                    pointer[j] = 1 #rl type of intersection (safe)
                    Intercepts[j] = (bmu+mu_sub[j+1])/2 
                else:
                    pointer[j] = 2 #flatflat type of intersection
                    Intercepts[j] = mu_sub[j+1] - bL
    return Intercepts, pointer



def pass_down(incpt_tmp, pt_tmp, S_tmp, cutoff, S_comp):
    ind_tmp = np.where(incpt_tmp <= cutoff)[0]
    if len(ind_tmp)>0 and min(min(pt_tmp[ind_tmp])-1.5,S_comp-max(S_tmp[ind_tmp]))<=0:
        lower_type = np.where(pt_tmp[ind_tmp]-1.5 <=0)[0]
        flat_all = np.where(pt_tmp[ind_tmp]-1.5 >0)[0]
        
        if len(lower_type) == 0:
            flat_type = np.where(S_comp-max(S_tmp[ind_tmp])<=0)[0]
            new_incpt = min(incpt_tmp[ind_tmp[flat_type]])
            n_md_plus = np.where(incpt_tmp == new_incpt)[0][0] #the index of input incpt_tmp
        else:
            if len(flat_all) == 0 or len(np.where(S_comp - max(S_tmp[ind_tmp[flat_all]])<=0)[0]) ==0:
                new_incpt = min(incpt_tmp[ind_tmp[lower_type]])
                n_md_plus = np.where(incpt_tmp == new_incpt)[0][0] #the index of input incpt_tmp
            else:
                flat_type = np.where(S_comp - max(S_tmp[ind_tmp[flat_all]])<=0)[0]
                new_incpt = min(min(incpt_tmp[ind_tmp[flat_all[flat_type]]]), min(incpt_tmp[ind_tmp[lower_type]]))
                n_md_plus = np.where(incpt_tmp == new_incpt)[0][0] #the index of input incpt_tmp
    else:
        new_incpt = cutoff
        n_md_plus = 0
    return new_incpt, n_md_plus



def q_i(Li, ui, Mu_i, Sis, Si_ui):
    if len(Mu_i) == 1:
        Si = Si_ui
    else:
        # sort input from left to right
        sort_ind = np.argsort(Mu_i)
        mu = Mu_i[sort_ind]; S = Sis[sort_ind]; Lo = Li[sort_ind]
        
        # identify indep chunks
        upl = mu-ui; upr = mu+ui
        chunk_breakpt = np.where(upr[:-1] - upl[1:] <0)[0] #could be empty array
        chunk_l = np.append(upl[0], upl[chunk_breakpt+1]) #at least len(chunk_l) >=1
        chunk_r = np.append(upr[chunk_breakpt], upr[-1])
        
        Si = []
        # start calculate S_i inside each chunk
        for c in range(len(chunk_l)):
            md_in_chunk = np.where((mu-chunk_l[c])*(mu-chunk_r[c]) <=0)[0]
            if len(md_in_chunk) == 1:
                Si.append([S[md_in_chunk], chunk_l[c], chunk_r[c]])
            else:
                mu_sub = mu[md_in_chunk]; S_sub = S[md_in_chunk]; L_sub = Lo[md_in_chunk]
                M_left = len(mu_sub)-1
                lendpt = chunk_l[c]
                while M_left>0:
                    Intercepts, pointer = intersection(mu_sub, L_sub)
                    next_md = np.argmin(Intercepts); min_intcp = np.min(Intercepts)
                    # note the len(Intercepts) = len(pointer) = len(S_sub)-1
                    if min_intcp > lendpt and min(pointer[next_md]-1.5, S_sub[0]-S_sub[next_md+1])<=0:
                        Si.append([S_sub[0], lendpt, min_intcp])
                        lendpt = min_intcp # then pass down to next_md
                    else:
                        incpt_tmp = Intercepts[next_md:]; pt_tmp = pointer[next_md:]; S_tmp = S_sub[next_md+1:]
                        cutoff = mu_sub[0]+L_sub[0]
                        new_incpt, n_md_plus = pass_down(incpt_tmp, pt_tmp, S_tmp, cutoff, S_sub[0])
                        next_md = next_md + n_md_plus # then pass down to next_md
                        if new_incpt > lendpt:
                            Si.append([S_sub[0], lendpt, new_incpt])
                            lendpt = new_incpt
                    
                    #delete previous models and pass down to the next model
                    next_md = next_md + 1 # amend the dim difference
                    mu_sub = mu_sub[next_md:]; S_sub = S_sub[next_md:]; L_sub = L_sub[next_md:]
                    M_left = len(mu_sub)-1
                
                Si.append([S_sub[0], lendpt, chunk_r[c]])
    return Si



def quantl_y(k, S_in_chunk, outside_chunk_S):
    n = len(outside_chunk_S)
    seg = []; Lefts = []; Rights = []
    # unfold the info
    for i in range(n):
        Si = S_in_chunk[i]
        if isinstance(Si, list):
            l_tmp = []; r_tmp = []
            for S in Si:
                seg.append(S[1])
                seg.append(S[2])
                l_tmp.append(S[1]); r_tmp.append(S[2])
            Lefts.append(l_tmp); Rights.append(r_tmp)
        else:
            Lefts.append([]); Rights.append([])

    # calculate the quantile function
    if len(seg) == 0: # all Si's are scalar
        quantl = np.sort(outside_chunk_S)[k-1]
    else:
        seg = np.unique(seg)
        quantl = np.zeros(len(seg)+1)
        
        for j in range(len(seg)):
            if j == 0:
                quantl[j] = np.sort(outside_chunk_S)[k-1]
                quantl[-1] = quantl[0]
            else:
                p = (seg[j-1]+seg[j])/2
                S_tmp = outside_chunk_S
                for i in range(n):
                    left = Lefts[i]; right = Rights[i]
                    if len(left) >0:
                        left = np.array(left); right = np.array(right)
                        loc = np.where((p - left)*(p-right)<0)[0] # check where does this seg point lie
                        if len(loc) >0:
                            S_tmp[i] = S_in_chunk[i][loc[0]][0]
                            if len(loc)>1:
                                print("a mistake in seg midpt!!")

                quantl[j] = np.sort(S_tmp)[k-1]   
    return quantl, seg



def pred_set(mu_hat, y_test, quantl, seg):
    true_res = abs(y_test - mu_hat)
    if len(seg) == 0:
        Interv = [mu_hat-quantl, mu_hat+quantl]
        length = 2*quantl
        connect = 1
        if true_res <= quantl:
            cover = 1
        else:
            cover = 0
    else:
        left = np.zeros(len(seg)+1); right = np.zeros(len(seg)+1)
        for j in range(len(seg)):
            if j == 0:
                left[-1] = max(seg[-1], mu_hat - quantl[-1]); right[-1] = mu_hat + quantl[-1]
                left[0] = mu_hat - quantl[0]; right[0] = min(seg[0], mu_hat + quantl[0])
                if y_test > seg[-1]:
                    if true_res <= quantl[-1]:
                        cover = 1
                    else:
                        cover = 0
                if y_test < seg[0]:
                    if true_res <= quantl[0]:
                        cover = 1
                    else:
                        cover = 0
            else:
                left[j] = max(seg[j-1], mu_hat-quantl[j]); right[j] = min(seg[j], mu_hat+quantl[j])
                if (y_test - seg[j-1])*(y_test - seg[j])<0:
                    if true_res <= quantl[j]:
                        cover = 1
                    else:
                        cover = 0
            
            if y_test == seg[j]:
                if true_res <= max(quantl[j], quantl[j+1]):
                    cover = 1
                else:
                    cover = 0
        
        # calculate the length and output prediction set
        valid_ind = np.where(left<right)[0]
        left = left[valid_ind]; right = right[valid_ind]
        length = np.sum(right-left)
        if len(left) == 1:
            connect = 1
            Interv = [left[0], right[0]]
        else:
            breakpt = np.where(left[1:] - right[-1]>0)[0]
            if len(breakpt)>0:
                connect = 0
                l = np.append(left[0], left[breakpt+1])
                r = np.append(right[breakpt], right[-1])
                Interv = [l, r]
            else:
                connect = 1
                Interv = [min(left), max(right)]
        
    return cover, length, connect, Interv

# main function of LOO

def ModSelLOO_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha):
    n = len(Y_cal); M = len(Mmodels)
    k = math.ceil((n+1)*(1-alpha))
    
    S_all = np.zeros((n,M)); pred_value = np.zeros(M)
    for m in range(M):
        mdl = Mmodels[m]
        S_all[:,m] = np.abs(Y_cal - mdl.predict(X_cal))
        pred_value[m] = mdl.predict(x_test)
    
    S_q = np.sort(S_all, axis=0)[(k-2):(k+1),:]

    # initial filtering
    find_ind = np.where(S_q[0,:] <= np.min(S_q[2,:]))[0]
    S_all = S_all[:,find_ind]; S_q = S_q[:,find_ind]; pred_value = pred_value[find_ind]
    M = len(find_ind)
    
    # select the model
    mhat = np.argmin(S_q[1,:])
    
    if M == 1: #only mhat remains
        if abs(y_test - pred_value[mhat]) <= S_q[1,mhat]:
            cover = 1
        else:
            cover = 0
        
        length = 2*S_q[1,mhat]
        connect = 1
        Interv = [pred_value[mhat]-S_q[1,mhat], pred_value[mhat]+S_q[1,mhat]]
    else:
        S_in_chunk = []; outside_chunk_S = np.zeros(n)
        for i in range(n):
            Li, ui, Mu_i, Sis, Si_ui = CALMi(i, S_all, S_q, pred_value)
            Si = q_i(Li, ui, Mu_i, Sis, Si_ui)
            S_in_chunk.append(Si)
            outside_chunk_S[i] = Si_ui
        
        quantl, seg = quantl_y(k, S_in_chunk, outside_chunk_S)
        cover, length, connect, Interv = pred_set(pred_value[mhat], y_test, quantl, seg)
    return cover, length, connect, Interv

