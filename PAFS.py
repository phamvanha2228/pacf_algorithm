# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:14:32 2020

@author: VanHa
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy.sparse import csr_matrix


## FUNTION CALCULATE DUAL GAP

def dual_gap_cal(c, b, x, y):
    cx =  np.dot(np.transpose(c), x)
    by = np.dot(np.transpose(b),y)
    dg =  (abs(cx-by))/(1.0 + abs(cx))
    return dg


## FUNTION CALCULATE Y

def calculate_y(aa, dd, cc):
    term1 = np.dot(np.dot(aa, dd), np.transpose(aa))
    term2 = np.dot(np.dot(aa, dd), cc)
    # ADAt = np.dot(np.dot(a,d), np.transpose(a))
    # ADC = np.dot(np.dot(a,d), c)
    y = np.linalg.solve(term1, term2)
    return y  


def optimization(A, B, C):
#### get number of column and row from the object functions and contraints
    m = A.shape[0] ##m rows.
    n = A.shape[1]  ##n columns ~ n variables
    opt_gap=1.0e-8
    rho=0.9995
    iteration= []
    f_obj = []
    x_val = []
    alpha_val =[]

### checking A, B , C matrices
    if C.shape[1] != 1 or B.shape[1]!= 1:
        raise Exception("Error: c and b must be column vectors")
    if C.shape[0] != n or B.shape[0] != m:
        raise Exception("Error: inconsistent dimensions for c, b and A")
    if np.linalg.matrix_rank(A) != m: 
        raise Exception("Error: matrix A is not full row rank")
     
    ## INITIALATION
    #### infeasibilites: 
    e = np.ones((n,1), dtype = int)
    r= B - np.dot(A,e)

    ##### Big-M method:
    
    M = n*max(abs(C))
    Aext = np.hstack((A,r))   
    Cext = np.vstack((C,M))
    
    X = np.ones((n+1, 1))
    D = np.eye(n+1)

    #### Calculate "y"
    Y= calculate_y(Aext, D, Cext)
    dual_gap = dual_gap_cal(Cext,B, X, Y)
 
    ## ITERATIVE PROCEDURE 
    k = 0                      
    while dual_gap > opt_gap:
        z= Cext - np.dot(np.transpose(Aext), Y)
        delta = np.dot(-D,z)
        if np.all(delta >= 0): 
            print("Unbounded problem.")
            break
              
        alpha = rho*np.min(-X[delta < 0] / delta[delta < 0])
        X = X + alpha*delta
        D = np.diag(np.ravel(X))**2   

        
        Y= calculate_y(Aext, D, Cext)
        
        dual_gap = dual_gap_cal(Cext, B, X, Y)
        obj_func = np.dot(np.transpose(Cext), X)
    
    
        ### calculate the result table:
        iteration.append(k)
        f_obj.append(obj_func.ravel()[0])
        alpha_val.append(alpha)
        x_val.append(X)
        d = {'Iteration': iteration, 'Object function': f_obj, 'alpha': alpha_val}
        df = pd.DataFrame(d)
        k +=1  
    
    if X[n] > 1e-8:
        print("Infeasible problem.")
    else:
        pass
        for i in range(1,n+1):
            print ('X%s: %01f' %(i, X[i-1]))
        print(df)
        print(X[0:n]) 



def mat_parse(file):
    mat_content = scipy.io.loadmat(file)
    mat_struct = mat_content['Problem']
    val = mat_struct[0, 0]
    maxtrixA = csr_matrix(val['A']).todense()
    vectorB = val['b']
    vectorC = val['aux'][0][0][0]
    return maxtrixA, vectorB, vectorC


q,v,t = mat_parse("lp_adlittle.mat")
prob = optimization(q, v, t)

