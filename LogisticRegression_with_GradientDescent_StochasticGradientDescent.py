# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:06:21 2020

@author: User
"""

import numpy as np
import pandas as pd
import random
from random import uniform 
import seaborn as sns
import matplotlib as plt
#############################Generate Random Data###########################################################
def get_data(separable):    # Set to 1 to ensure separable data, 0 otherwise):
    random.seed(9999) 
    d = 2
    n = 40
    x_mat = np.zeros(80).reshape(40,2)  
    y_vec = np.zeros(40).reshape(40,1)
    gamma = 0.1; # Margin parameter
    n_wrong = 4; # In the non-separable case, this is the number of points whose labels are changed
    theta = np.array([1,-1])
    theta0 = 0.2
    t = 0
    while t < n:
        x = np.array([uniform(-0.5,0.5),uniform(-0.5,0.5)])
        if np.dot(x,theta) + theta0 > gamma :
            y = 1 
            t += 1
        elif np.dot(x,theta) + theta0 < -gamma :
            y = -1
            t += 1
        else:
            continue
    #     x_mat[t-1][0],x_mat[t-1][1],x_mat[t-1][2] = x[0], x[1], 1
        x_mat[t-1][0],x_mat[t-1][1] = x[0], x[1]
        y_vec[t-1] = y
    if separable == 0:
        for t in range(n_wrong):
            y_vec[t] = -y_vec[t]
    return x_mat, y_vec
x_mat, y_vec = get_data(1)

##############Plotting####################################3
def get_first(x_mat, y_vec, y_value):
    index = [(i[0]==y_value) for i in y_vec]
    mat = x_mat[index]
    return [i[0]for i in mat]
def get_second(x_mat, y_vec, y_value):
    index = [(i[0]==y_value) for i in y_vec]
    mat = x_mat[index]
    return [i[1]for i in mat]
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
# y = 1 
ax.scatter(get_first(x_mat, y_vec, 1), get_second(x_mat, y_vec, 1), label = 'tab:blue' , c ='tab:blue')
# y = -1 
ax.scatter(get_first(x_mat, y_vec, -1), get_second(x_mat, y_vec, -1), label = 'tab:red', c =  'tab:red')

# ax.legend()
ax.grid(True)
plt.show()

###########Plot Boundary 0.25 and 0.75####

        ############Gradient Descent / Stochastic Gradient Descent #####################
def gd(x_mat, y_vec, initial_w, step_size,N_iter, do_sgd = 0): #% Set to 1 for SGD, 0 for regular GD):
    # % % Implementation of gradient descent%
    theta_lst = [] 
    theta0_lst = []
    if do_sgd == 0:
        theta_next = np.array(initial_w[0])
        theta0_next = initial_w[1]
        for i in range(N_iter):
            lst0 = np.zeros(1)
            lst = np.zeros(2)
            for j in range(len(x_mat)):
                yt = y_vec[j]
                xt = x_mat[j]
                z = -yt*(theta_next.transpose().dot(xt)+theta0_next)
                lst0 += (-yt*likelihood(xt,yt,theta_next,theta0_next))
                lst += (-yt*xt*likelihood(xt,yt,theta_next,theta0_next))
            theta_next = theta_next - step_size *lst/len(x_mat)
            theta0_next = theta0_next - step_size * lst0/len(x_mat)
            theta_lst += [theta_next] 
            theta0_lst += [theta0_next]
    else: #Stochastic
        theta_next = np.array(initial_w[0])
        theta0_next = initial_w[1]
        for i in range(N_iter):
            lst0 = np.zeros(1)
            lst = np.zeros(2)
            j = random.randint(0,39)
            yt = y_vec[j]
            xt = x_mat[j]
            z = -yt*(theta_next.transpose().dot(xt)+theta0_next)
            lst0 += (-yt*likelihood(xt,yt,theta_next,theta0_next))
            lst += (-yt*xt*likelihood(xt,yt,theta_next,theta0_next))
            theta_next = theta_next - step_size *lst/len(x_mat)
            theta0_next = theta0_next - step_size * lst0/len(x_mat)
            theta_lst += [theta_next] 
            theta0_lst += [theta0_next]
    return theta_next, theta0_next, theta_lst, theta0_lst


def likelihood(x,y,theta,theta0): 
    z = -y*(theta.transpose().dot(x)+theta0)
    result = np.exp(z)/(1+np.exp(z))
    return result    
            
    
# def loglikelihood(w, X, y):
#     result = 0
#     for i in range(len(X)):
#         exp_part = np.exp(-y[i] * np.dot(np.transpose(w),X[i]))
#         result += np.log(1+exp_part)
#     return result

theta_next, theta0_next, theta_lst, theta0_lst= gd(x_mat, y_vec,[[0,0],0], 1, 100000, do_sgd = 1)        
theta, theta0

acc = 0
for i in range(len(x_mat)):
    if theta.transpose().dot(x_mat[i])+theta0>0:
        if y_vec[i][0]==1:
            acc+=1
    else:
        if y_vec[i][0]==-1:
            acc+=1
acc/40

def plot_theta(theta_lst, theta0_lst):
    length = len(theta_lst)
    theta0 = pd.Series(theta0_lst).apply(lambda x: x[0]).tolist()
    theta1 , theta2 = pd.Series(theta_lst).apply(lambda x: x[0]).tolist(),pd.Series(theta_lst).apply(lambda x: x[1]).tolist()
    fig, ax = plt.subplots()
    ax.plot(theta0, c = "red", label = "theta0")
    ax.plot(theta1, c = "blue", label = "theta1")
    ax.plot(theta2, c = "green", label = "theta2")
    ax.grid(True)
    ax.legend()
    plt.show()
def plot_boundary(theta, theta0):
    ps = [0.75, 0.5,0.25]
    fig, ax = plt.subplots()
    for p in ps:
        logv = np.log(p/(1-p))
        x = np.array([-0.5,0.5])
        if p == 0.25:
            c = "red" #0.25
        elif p==0.75:
            c = "blue" #0.75
        else:
            c = "purple"
        ax.plot(x , -theta[0]/theta[1]*x + (logv-theta0)/theta[1], c = c)
    # y = 1 
    ax.scatter(get_first(x_mat, y_vec, 1), get_second(x_mat, y_vec, 1), label = 'tab:blue' , c ='tab:blue')
    # y = -1 
    ax.scatter(get_first(x_mat, y_vec, -1), get_second(x_mat, y_vec, -1), label = 'tab:red', c =  'tab:red')
    ax.grid(True)
#     ax.set(xlim=(-0.6, 0.6), ylim=(-0.6, 0.6))
    plt.show()
# Theta Change
# sep, GD
x_mat, y_vec = get_data(1)
theta_next, theta0_next, theta_lst, theta0_lst = gd(x_mat, y_vec,[[0,0],0], 1, 100, do_sgd = 0) 
plot_theta(theta_lst, theta0_lst)
