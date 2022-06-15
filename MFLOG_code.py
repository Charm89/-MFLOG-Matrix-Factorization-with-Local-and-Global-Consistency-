# -*- coding: utf-8 -*-
"""
@author: rforouza
"""

import csv
from numpy import genfromtxt
import numpy as np
import math
from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot as plt
import random 
from IPython.display import display, Markdown
import pandas
from sklearn.metrics import mean_squared_error
from  sklearn.decomposition import NMF
from sklearn.metrics import mean_absolute_error
from math import log
import os
from scipy import ndimage, misc

#Loading data
root_directory = os.getcwd()
##Station distance array 
filename = os.path.join(root_directory, 'data/station_distance.csv')
D_array = genfromtxt(filename, delimiter=',')
##Flow matrices
window_X = list()
for i in range (23):
    filename = os.path.join(root_directory, 'data/adj_past'+str(i)+'.csv')
    window_X.append(genfromtxt(filename, delimiter=','))
window_X_future = list()
for i in range (10):
    filename = os.path.join(root_directory, 'data/adj_future'+str(i)+'.csv')
    window_X_future.append(genfromtxt(filename, delimiter=','))
##Time spans
filename = os.path.join(root_directory, 'data/time_list.txt')
time_list = genfromtxt(filename, delimiter=',').tolist()
filename = os.path.join(root_directory, 'data/future_time_list.txt')
future_time_list = genfromtxt(filename, delimiter=',').tolist()

#Computing Inverse distance array and laplacian 
np.fill_diagonal(D_array,1000 )#remove distance from zero to 300, We pick 300 as it is the minimum distance between stations 
for i in range(len(D_array)):
    for j in range(len(D_array)):
        if D_array[i,j] <= 1000:
            D_array [i,j] = 1000
H = (1000/D_array)**2 #Inversed distance array
L = ndimage.laplace(H) #laplacian

#DAILY PREDICTION WITH LOCAL and GLOBAL CONSISTENCY (MFLOG)#
r = 100 #latent dimensions
##Initializing U, V (W0,W1,...), and s
model = NMF(n_components=r, init='nndsvda')
U = model.fit_transform(window_X[-1])
V = np.transpose(model.components_)
rc = 10^5
lambda_ = 0.000000000005
#DAILY PREDICTION with TFNM model(NO Local and Global Consistency)#
W0 = V/(time_list[0]+1)
W1 = V/(time_list[0]+1)

####lambda_ = 0.000000001
alpha = 0.0000001 #0.000005
beta = 0.0005 #1 #1000,0.005:13.5:
n_nodes = len(D_array) #number of nodes 

T = time_list[-1] #Present time stamp
T_plus = future_time_list [0] #Prediction time stamp
##Initializing U, V (W0,W1,...), and s
s = 1
#Updating U and V (W0,W1,...)
training_error_list = list()

for i in range(1000):
    window_length = len(time_list)

    #Calculating diff J (J = J1+J2+J3) w.r.t U
    diff_J_U = np.zeros((n_nodes, r))
    for j in range(window_length):
        t = time_list[j]#time
        X = window_X[j]
        V = W0 + W1*t #+ W2*t**2
        diff_J_U += -2*X.dot(V)+2*U.dot(np.transpose(V)).dot(V)
    if alpha == 0 and beta == 0:
        diff_J_U += rc*U # General regularization term
    diff_J_U += beta*(L.dot(U)+np.transpose(L).dot(U))


    #Calculating diff J (J = J1+J2+J3) w.r.t W0,W1,...
    diff_J_W0 = np.zeros((n_nodes, r))
    diff_J_W1 = np.zeros((n_nodes, r))
    diff_J_s = 0
    #diff_J_W2 = np.zeros((n_nodes, r))
    for j in range(window_length):
        decay  = (j + window_length)/(2*window_length)
        #decay = 1
        t = time_list[j]#time
        X = window_X[j]
        V = W0 + W1*t #+ W2*t**2
        diff_J_W0 += decay*(-2*np.transpose(X).dot(U)+2*V.dot(np.transpose(U)).dot(U))
        diff_J_W1 += decay*(-2*np.transpose(X).dot(U)+2*V.dot(np.transpose(U)).dot(U))*t
        #diff_J_W2 += (-2*np.transpose(X).dot(U)+2*V.dot(np.transpose(U)).dot(U))*t**2
    if alpha == 0 and beta == 0:
        diff_J_W0 += rc*W0 # General regularization term
        diff_J_W1 += rc*W1 # General regularization term

    V_T = W0 + W1*T #+ W2*T **2
    V_T_plus = W0 + W1 * T_plus #+ W2 * T_plus**2

    diff_J_W0 += decay*(1*(4*alpha*V_T.dot(np.transpose(V_T)).dot(V_T) - 4*alpha*s*V_T_plus.dot(np.transpose(V_T_plus)).dot(V_T)) +\
    1*(- 4*alpha*s*V_T.dot(np.transpose(V_T)).dot(V_T_plus) + 4*alpha* s**2 *V_T_plus.dot(np.transpose(V_T_plus)).dot(V_T_plus)))

    diff_J_W1 += decay*(T*(4*alpha*V_T.dot(np.transpose(V_T)).dot(V_T) - 4*alpha*s*V_T_plus.dot(np.transpose(V_T_plus)).dot(V_T)) +\
    T_plus *(- 4*alpha*s*V_T.dot(np.transpose(V_T)).dot(V_T_plus) + 4*alpha* s**2 *V_T_plus.dot(np.transpose(V_T_plus)).dot(V_T_plus)))

    diff_J_s += decay*(-1*alpha*np.trace(V_T.dot(np.transpose(V_T)).dot(V_T_plus).dot(np.transpose(V_T_plus))) - \
    alpha*np.trace(V_T_plus.dot(np.transpose(V_T_plus)).dot(V_T).dot(np.transpose(V_T))) + \
    2*alpha*s*np.trace(V_T_plus.dot(np.transpose(V_T_plus)).dot(V_T_plus).dot(np.transpose(V_T_plus))))

    #Updating the factors    
    U = U - lambda_*diff_J_U
    W0 = W0 - lambda_*diff_J_W0
    W1 = W1 - lambda_*diff_J_W1
    s = s - lambda_* diff_J_s
    #W2 = W2 - lambda_*diff_J_W2

    #Training error
    X_bar_window = list()
    error_window = list()
    training_error = 0
    for j in range(window_length):
        t = time_list[j]#time
        X = window_X[j]
        V = W0 + W1*t #+W2*t**2
        X_bar = U.dot(np.transpose(V))
        error = np.linalg.norm(X-X_bar,ord='fro')
        X_bar_window.append(X_bar)
        error_window.append(error)
        training_error += error
    print('training error for MFLOG model is: ', training_error)
    training_error_list.append(training_error)
training_error_df = pd.DataFrame(training_error_list, columns ={'error'}) #DataFrame of  errors at each iteration
training_error_df['iteration'] = training_error_df.index
training_error_df['iteration'] = training_error_df['iteration'] +1
plt.figure(figsize = [6,4])
ax = plt.gca()
plt1 = plt.plot(training_error_df.iteration, training_error_df.error, color='green', linewidth = 1, 
          markerfacecolor='green', markersize=2,label='Scaled Trip Count')
#plt.legend()
plt.xlabel('Iteration') 
#naming the y axis 
plt.ylabel('Training Error')
plt.title('Training Error at Each Iteration')

##Forecasting error (with local and global consistency)
RMSE_LG  = list()
t = future_time_list[0]#time
X = window_X_future[0]
V = W0 + W1*t #+ W2*t**2
forecasted_X_LG = U.dot(np.transpose(V))
X_reshaped = X.flatten()
forecasted_X_reshaped_LG = forecasted_X_LG.flatten()
#forecasted_X_reshaped_LG = forecasted_X_reshaped_LG.clip(min=0.5)

X_copy = X.copy()
forecasted_X_LG_copy = forecasted_X_LG.copy()

X = X.clip(min = 0)
forecasted_X_LG = forecasted_X_LG.clip(min = 0)

#print('forecasting error is: ',forecasting_error)

flow_MAE_MFLOG = mean_absolute_error(X.flatten(), forecasted_X_LG.flatten())
#check_in_MAE_MFLOG = mean_absolute_error(sum(X_copy), sum(forecasted_X_LG_copy))
check_in_MAE_MFLOG = mean_absolute_error(sum(X), sum(forecasted_X_LG))

#print(flow_MAE_MFLOG, check_in_MAE_MFLOG)

print ('MAE for Flow Forecasting with MFLOG model is:', flow_MAE_MFLOG)
print ('MAE for Check-in Forecasting with MFLOG model is:', check_in_MAE_MFLOG)


