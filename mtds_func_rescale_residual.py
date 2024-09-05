#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import scipy.stats as stats



# YK-baseline

def YKbaseline_rescaleRes(Mmodels, X_cal, Y_cal, x_test, y_test, alpha):
    M = len(Mmodels); n = len(Y_cal)
    S = np.zeros(M); leng_factor = np.zeros(M)
    k = math.ceil((n+1)*(1-alpha))
    
    for m in range(M):
        mdl = Mmodels[m]
        Scores = np.sort((np.abs(Y_cal - mdl[0].predict(X_cal))/mdl[1].predict(X_cal)))
        S[m] = Scores[k-1] # S^{(m)}_{(k)}
        leng_factor[m] = (np.sum(mdl[1].predict(X_cal)) + mdl[1].predict(x_test))/(n+1)
        
    
    mhat = np.argmin(S*leng_factor)
    select_mdl = Mmodels[mhat]
    mu_hat = select_mdl[0].predict(x_test)
    sigma_hat = select_mdl[1].predict(x_test)
    
    if np.abs(y_test - mu_hat)/sigma_hat>S[mhat]:
        cover = 0
    else:
        cover = 1
    
    length = 2*S[mhat]*sigma_hat
    interv = [mu_hat-(S[mhat]*sigma_hat), mu_hat+(S[mhat]*sigma_hat)]
    
    return cover, length, interv




#YK-adj: alpha adjusted version

def YK_adj_rescaleRes(Mmodels, X_cal, Y_cal, x_test, y_test, alpha):
    M = len(Mmodels); n = len(Y_cal)
    til_alpha = n*alpha/(n+1) + 1/(n+1) - n*(1/(3*np.sqrt(n)) + np.sqrt(np.log(2*M)/(2*n)))/(n+1)
    if math.ceil((n+1)*(1-til_alpha)) > n: # return the real line
        cover = 1
        length = np.inf
        interval = [-np.inf, np.inf] # more rigorously should be \mathcal{Y}
    else:
        cover, length, interval = YKbaseline_rescaleRes(Mmodels, X_cal, Y_cal, x_test, y_test, til_alpha)

    return cover, length, interval




# YK-split

def YKsplit_rescaleRes(Mmodels, X_cal, Y_cal, x_test, y_test, alpha, split_portion):
    M = len(Mmodels)
    n1 = math.ceil(len(Y_cal)*split_portion); n2 = len(Y_cal) - n1
    k1 = math.ceil((n1 + 1)*(1-alpha)); k2 = math.ceil((n2 + 1)*(1-alpha))
    S = np.zeros(M)
    
    X_sel = X_cal[:n1, :]; Y_sel = Y_cal[:n1]
    X_c = X_cal[n1:, :]; Y_c = Y_cal[n1:]
    
    for m in range(M):
        mdl = Mmodels[m]
        Scores = np.sort((np.abs(Y_sel - mdl[0].predict(X_sel))/mdl[1].predict(X_sel)))
        S[m] = Scores[k1-1] # S^{(m)}_{(k)}
        S[m] = S[m]*np.mean(mdl[1].predict(X_sel))
        
    mhat = np.argmin(S)
    model = Mmodels[mhat]
    Restmp = np.sort((np.abs(Y_c - model[0].predict(X_c))/model[1].predict(X_c))); 
    Res = Restmp[k2 - 1]
    mu_test = model[0].predict(x_test); sigma_test = model[1].predict(x_test)
    
    if np.abs(y_test - mu_test)/sigma_test>Res:
        cover = 0
    else:
        cover = 1
    
    length = 2*Res*sigma_test
    interval = [mu_test-(Res*sigma_test), mu_test+(Res*sigma_test)]
    return cover, length, interval



# ModSel-CP

def ModSel_rescaleRes(Mmodels, X_cal, Y_cal, x_test, y_test, alpha):
    M = len(Mmodels); n = len(Y_cal)
    k = math.ceil((n+1)*(1-alpha))
    
    S = np.zeros((M, 2)); leng_factor = np.zeros(M)
    
    for m in range(M):
        mdl = Mmodels[m]
        tmpS = np.sort((np.abs(Y_cal - mdl[0].predict(X_cal))/mdl[1].predict(X_cal)))
        leng_factor[m] = (np.sum(mdl[1].predict(X_cal)) + mdl[1].predict(x_test))/(n+1)
        S[m, 0] = tmpS[k-2] # S^{(m)}_{(k-1)}
        S[m, 1] = tmpS[k-1] # S^{(m)}_{(k)}
    
    u = np.min(S[:,1]*leng_factor)
    
    #filter models
    find_mdl = np.where(S[:,0]*leng_factor-u <=0)[0]
    # the number of models in calM
    calM = len(find_mdl)
    leng_factor_filtered = leng_factor[find_mdl]
    mu_test = np.zeros(calM)
    sigma_filtered = np.zeros(calM)
    
    for m in range(calM):
        mdl = Mmodels[find_mdl[m]]
        mu_test[m] = mdl[0].predict(x_test)
        sigma_filtered[m] = mdl[1].predict(x_test)
    
    slope = leng_factor_filtered/sigma_filtered
    
    #check cover or not
    if np.min(np.abs(mu_test - y_test)*slope)>u:
        cover = 0
    else:
        cover = 1
    
    #calculate length, determine the intervals and check connectness
    quantl = u/slope #reduce to residual type
    Lefts = mu_test - quantl; Rights = mu_test + quantl
        
    sort_ind = np.argsort(Lefts)
    Lefts = Lefts[sort_ind]; Rights = Rights[sort_ind]
        
    loc = 0
    length = 0; connect = 1
    left = [Lefts[0]]; right = []
    while loc<len(Lefts):
        find_ind = np.where(Lefts - Rights[loc]>0)[0]
        if len(find_ind) == 0:
            length += np.max(Rights) - Lefts[loc]
            right.append(np.max(Rights))
            loc = len(Lefts)
        else:
            if find_ind[0] - 1 == loc:
                connect = 0
                right.append(Rights[loc])
                length += Rights[loc] - Lefts[loc]
                loc = find_ind[0]
                left.append(Lefts[loc])
            else:
                length += Lefts[find_ind[0]-1] - Lefts[loc]
                loc = find_ind[0] - 1
    
    Interv = [left, right]
    
    return cover, length, Interv, connect, calM




# ModSel-CP-LOO

def CALMi(i, S_all, S_q, pred_value, var_ratio):
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
    Li = Lo[find_ind]; Ui = Up[find_ind]; Mu_i = pred_value[find_ind]; Slope_i = var_ratio[find_ind]
    Sis = S_all[i,find_ind]
    
    sort_ind = np.argsort(Li)
    Li = Li[sort_ind]; Ui = Ui[sort_ind]; Mu_i = Mu_i[sort_ind]; Slope_i = Slope_i[sort_ind]; Sis = Sis[sort_ind]
    
    ui = np.min(Ui)
    m_ui = np.where(Ui == ui)[0]
    Si_ui = max(Sis[m_ui])
    
    Lowl = Mu_i - (Li/Slope_i); Lowr = Mu_i + (Li/Slope_i)
    # further filter models
    columns_to_delete = set()
    for j in range(len(Li)):
        for k in range(len(Li)):
            if j != k:
                if (Li[k] <= Li[j]) and (Slope_i[k] <= Slope_i[j]):
                    if (Mu_i[k]-(Li[j]/Slope_i[k]) <= Lowl[j]) and (Mu_i[k]+(Li[j]/Slope_i[k]) >= Lowr[j]):
                        if Li[k]<Li[j]:
                            columns_to_delete.add(j)
                        else:
                            if Sis[k] >= Sis[j]:
                                columns_to_delete.add(j)

    mask = [j not in columns_to_delete for j in range(len(Li))]
    # Apply the mask to each vector and return the filtered models
    Li = Li[mask]; Ui = Ui[mask]; Mu_i = Mu_i[mask]; Slope_i = Slope_i[mask]; Sis = Sis[mask]
            
    
    return Li, ui, Mu_i, Slope_i, Sis, Si_ui

def Intercept(muw, mun,slopew,slopen, Lw,Ln, Ulw, Uln, Urw, Urn):
    wncross = (slopew*muw + slopen*mun)/(slopew+slopen)
    Llw = muw - (Lw/slopew); Lrw = muw + (Lw/slopew)
    Lln = mun - (Ln/slopen); Lrn = mun + (Ln/slopen)
    
    intercept = []
    
    if (Uln - Ulw)*(Uln - Urw)<=0 and Urn>Urw: # left wide, right narrow, not contain
        if Lrw <= wncross:
            if wncross <= Lln:
                intercept = [wncross]
            else:
                intercept = [muw+(Ln/slopew)]
        else:
            if Ln < Lw:
                intercept = [mun-(Lw/slopen)]
            else:
                intercept = [muw+(Ln/slopew)]
    
    if (Urn - Ulw)*(Urn - Urw)<=0 and Uln<Ulw: # left narrow, right wide, not contain
        if Llw >= wncross:
            if wncross >= Lrn:
                intercept = [wncross]
            else:
                intercept = [muw-(Ln/slopew)]
        else:
            if Ln < Lw:
                intercept = [mun+(Lw/slopen)]
            else:
                intercept = [muw-(Ln/slopew)]
    
    if Ulw <= Uln and Urn <=Urw: # narrow is contained in the wide
        wnparal = (slopew*muw - slopen*mun)/(slopew-slopen) #if this happens, their slope should be different
        level_cross = slopew*abs(wncross-muw); level_paral = slopew*abs(wnparal-muw)
        if min(level_cross, level_paral) >= Lw:
            if Ln <= min(level_cross, level_paral):
                intercept = [wncross, wnparal]
            else:
                if (Ln - level_cross)*(Ln - level_paral) <= 0:
                    intercept = [muw+ np.sign(wnparal-wncross)*(Ln/slopew), wnparal]
                else:
                    print("mistake: incorrect model filtering done previously")
        else:
            if Lw >= max(level_cross, level_paral):
                intercept = [mun-(Lw/slopen), mun+(Lw/slopen)]
                if Ln > Lw:
                    print("mistake: incorrect model filtering done previously")
            else:
                if Ln<Lw:
                    intercept = [wnparal, mun + np.sign(wncross-wnparal)*(Lw/slopen)]
                else:
                    intercept = [muw+ np.sign(wnparal-wncross)*(Ln/slopew), wnparal]
                    if Ln >= max(level_cross, level_paral):
                        print("mistake: incorrect model filtering done previously")
    
    return intercept

def min_val(Mu_i, Slope_i, Li, lendpt, rendpt):
    min_each_mdl = np.zeros(len(Mu_i))
    
    for m in range(len(Mu_i)):
        Ll = Mu_i[m] - (Li[m]/Slope_i[m])
        Lr = Mu_i[m] + (Li[m]/Slope_i[m])
        if rendpt < Ll or lendpt > Lr:
            min_each_mdl[m] = min(Slope_i[m]*abs(Mu_i[m]-lendpt), Slope_i[m]*abs(Mu_i[m]-rendpt))
        else:
            min_each_mdl[m] = Li[m]
    
    return min_each_mdl

def qi_y(Li, ui, Mu_i, Slope_i, Sis, Si_ui):
    seg_i = []
    if len(Mu_i) == 1:
        Si_y = Si_ui
    else:
        Upleft = Mu_i - (ui/Slope_i); Upright = Mu_i + (ui/Slope_i)
        # sort from left to right by upper left endpoint
        sort_ind = np.argsort(Upleft)
        Lo = Li[sort_ind]; S = Sis[sort_ind]
        Mu = Mu_i[sort_ind]; Slope = Slope_i[sort_ind]
        Upleft = Upleft[sort_ind]; Upright = Upright[sort_ind]
        
        # identify indep chunks
        chunk_l = []; chunk_r = []
        chunk_l.append(Upleft[0])
        loc = 0
        while loc<len(Mu):
            find_ind = np.where(Upleft - Upright[loc]>0)[0]
            if len(find_ind) == 0:
                chunk_r.append(np.max(Upright))
                loc = len(Mu)
            else:
                if find_ind[0] - 1 == loc:
                    chunk_r.append(Upright[loc])
                    chunk_l.append(Upleft[find_ind[0]])
                    loc = find_ind[0]
                else:
                    loc = find_ind[0] - 1
        
        # calculate intercepts inside each chunk
        for c in range(len(chunk_l)):
            seg_i.append(chunk_l[c])
            seg_i.append(chunk_r[c])
            # take the models inside this chunk
            md_in_chunk = np.where((Mu-chunk_l[c])*(Mu-chunk_r[c]) <=0)[0]
            mu_sub = Mu[md_in_chunk]; Slope_sub = Slope[md_in_chunk]
            S_sub = S[md_in_chunk]; L_sub = Lo[md_in_chunk]; 
            Ul_sub = Upleft[md_in_chunk]; Ur_sub = Upright[md_in_chunk]
            # sort models based on slope from small to large
            tmp_sort = np.argsort(Slope_sub)
            mu_sort = mu_sub[tmp_sort]; slope_sort = Slope_sub[tmp_sort]
            S_sort = S_sub[tmp_sort]; L_sort = L_sub[tmp_sort]
            Ul_sort = Ul_sub[tmp_sort]; Ur_sort = Ur_sub[tmp_sort]
            
            for j in range(len(mu_sort)):
                k = j+1
                while k<len(mu_sort):
                    # calculate intercpts between mdl j and k
                    intercpts = Intercept(mu_sort[j], mu_sort[k],slope_sort[j],slope_sort[k],
                                          L_sort[j],L_sort[k], Ul_sort[j], Ul_sort[k], Ur_sort[j], Ur_sort[k])
                    if len(intercpts)>0:
                        for p in intercpts:
                            seg_i.append(p)
                    k = k+1
            
        
        # calculate step function Si_y
        seg_i = np.unique(seg_i)
        Si_y = np.zeros(len(seg_i)+1)
        for j in range(len(seg_i)):
            if j == 0:
                Si_y[0] = Si_ui; Si_y[-1] = Si_ui
            else:
                if np.min(np.abs(seg_i[j] - np.array(chunk_l)))>0: #interval inside a chunk
                    # find min of each model between seg_i[j-1] and seg_i[j]
                    min_each_mdl = min_val(Mu_i, Slope_i, Li, seg_i[j-1], seg_i[j])
                    find_ind = np.where(min_each_mdl == np.min(min_each_mdl))[0]
                    Si_y[j] = max(Sis[find_ind])
                else: #its corresponding interval is outside-chunk
                    Si_y[j] = Si_ui
        
    
    return seg_i, Si_y


def quantl_y(k, S_and_seg):
    seg = []; S_out_chunk = []
    # unfold info
    for i in S_and_seg:
        seg_i = i[0]
        tmp_Sis = i[1]
        if len(seg_i)>0:
            S_out_chunk.append(tmp_Sis[0])
            for j in seg_i:
                seg.append(j)
        else:
            S_out_chunk.append(tmp_Sis)
    
    # calculate the step function of quantile
    if len(seg) == 0:
        quantl_y = np.sort(S_out_chunk)[k-1]
    else:
        seg = np.unique(seg)
        quantl_y = np.zeros(len(seg)+1)
        for j in range(len(seg)):
            if j == 0:
                quantl_y[0] = np.sort(S_out_chunk)[k-1]
                quantl_y[-1] = np.sort(S_out_chunk)[k-1]
            else:
                midpt = (seg[j-1]+seg[j])/2
                tmp_S = np.array(S_out_chunk)
                for i in range(len(S_and_seg)):
                    seg_i = S_and_seg[i][0]
                    S_cor = S_and_seg[i][1]
                    if len(seg_i)>0:
                        loc = np.where(seg_i-midpt>0)[0]
                        if len(loc)>0:
                            tmp_S[i] = S_cor[loc[0]]
                
                quantl_y[j] = np.sort(tmp_S)[k-1]
    
    return quantl_y, seg


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


def ModSelLOO_rescaleRes(Mmodels, X_cal, Y_cal, x_test, y_test, alpha):
    n = len(Y_cal); M = len(Mmodels)
    k = math.ceil((n+1)*(1-alpha))
    
    S_all = np.zeros((n,M)); pred_value = np.zeros(M); var_ratio = np.zeros(M)
    for m in range(M):
        mdl = Mmodels[m] ####### maybe revise later: mdl include prediction mu(x) and estimated sigma(x)
        S_all[:,m] = np.abs(Y_cal - mdl[0].predict(X_cal))/mdl[1].predict(X_cal)
        factor = (np.sum(mdl[1].predict(X_cal)) + mdl[1].predict(x_test))/(n+1)
        S_all[:,m] = factor*S_all[:,m] #transformed score
        
        pred_value[m] = mdl[0].predict(x_test)
        var_ratio[m] = factor/mdl[1].predict(x_test) # the slope
    
    S_q = np.sort(S_all, axis=0)[(k-2):(k+1),:]

    # initial filtering
    find_ind = np.where(S_q[0,:] <= np.min(S_q[2,:]))[0]
    S_all = S_all[:,find_ind]; S_q = S_q[:,find_ind] 
    pred_value = pred_value[find_ind]; var_ratio = var_ratio[find_ind]
    M = len(find_ind)
    
    # select the model
    mhat = np.argmin(S_q[1,:])
    
    if M == 1: #only mhat remains
        if np.abs(y_test - pred_value[mhat])*var_ratio[mhat] <= S_q[1,mhat]:
            cover = 1
        else:
            cover = 0
        
        length = 2*S_q[1,mhat]/var_ratio[mhat]
        connect = 1
        Interv = [pred_value[mhat]-(S_q[1,mhat]/var_ratio[mhat]), pred_value[mhat]+(S_q[1,mhat]/var_ratio[mhat])]
    else:
        S_and_seg = []
        for i in range(n):
            Li, ui, Mu_i, Slope_i, Sis, Si_ui = CALMi(i, S_all, S_q, pred_value, var_ratio)
            seg_i, Si_y = qi_y(Li, ui, Mu_i, Slope_i, Sis, Si_ui)
            S_and_seg.append([seg_i, Si_y])
        
        # combine all segs and get quantile's step function
        quantl, seg = quantl_y(k, S_and_seg)
        rescale_q = quantl/var_ratio[mhat] #reduce to residual type
        cover, length, connect, Interv = pred_set(pred_value[mhat], y_test, rescale_q, seg)
    return cover, length, connect, Interv
