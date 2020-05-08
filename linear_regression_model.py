#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:45:35 2020

@author: kris
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Step 1

def initialize_parameters(lenw):
    w = np.random.randn(1, lenw)  #Here I am assigning random weights initially and random numbers of weights will be randomly generated through standard normal distribution
    b = 0 #Intialization of BIAS term
    return w, b


#step 2 - This is for forward propagation 

def forward_prop(X,w, b):
    z = np.dot(w, X) + b #Here I am creating matrix multiplication of randomly generated weights and Input variables (X) and also adding BIAS term to it
    return z


 # Step 3 , Cost function calculation 
    
def cost_function(z, y): #Here Z is the predicted value and y is the actual label given in the dataset
    m = y.shape[1] #1 is for number of column
    J= (1/(2*m))*np.sum(np.square(z-y)) #This is our cost function for linear regression, (Actual- Predicted)**2
    return J

# Step4. Now that I have defined the Cost function, I will be defining the backprop in order to optimize the weights and BIAS

def back_prop(X,y, z):
    m = y.shape[1]
    dz= (1/m)*(z-y)
    dw= np.dot(dz,X.T)
    db= np.sum(dz)
    return dw,db


# Step 5 Is our Gradient Descent updates

def gradient_descent_update(w,b,dw,db, learning_rate):
    w = w- learning_rate*dw
    b = b- learning_rate*db
    return w, b


# Step 6 Defining linear regression model

def linear_regression_model(X_train, y_train, X_val, y_val, learning_rate, epochs):
    lenw = X_train.shape[0]
    w,b = initialize_parameters(lenw) #This step is to initialize the parameters
    costs_train = []
    m_train = y_train.shape[1]
    m_val = y_val.shape[1]
    
#Now I will iterate it through for loop and execute step no 2 which is forward_prop, 
#The Step 3 Computation of Cost function, I will pass back_prop to update the weights and bias and find minima point in Gradient descent
    for i in range(1, epochs+1):
        z_train = forward_prop(X_train, w, b)
        cost_train= cost_function(z_train, y_train)
        dw, db = back_prop(X_train, y_train, z_train)
        w, b = gradient_descent_update(w, b, dw, db, learning_rate)

#Now I am gonna store all these training cost into a list, Now Why am I doing it, Because I want to visualize it
        if i%10==0:#It means for every 10 iternation, It is gonna be stored in a list
            costs_train.append(cost_train)
        
        
        
        
    MAE_train = (1/m_train)*np.sum(np.abs(z_train-y_train)) #m_Train is Trainign samples which I will define there at costs_train
        
        
        #Now I will define cost function for validation and also will find MAE for validation
    z_val = forward_prop(X_val, w, b)
    cost_val = cost_function(z_val, y_val)
    MAE_val = (1/m_val)*np.sum(np.abs(z_val-y_val))
        
        #Let's print Cost_Val, train_Val, MAE _val and MAE_train and I want to see these values for every iteration hence using epochs
        
    print('Epochs '+str(i)+ '/' +str(epochs)+': ')
    print('Training_cost '+str(cost_train)+'|'+'Validation_cost '+str(cost_val))
    print('Training_MAE '+str(MAE_train)+'|'+'Validation_MAE '+str(MAE_val))
        










