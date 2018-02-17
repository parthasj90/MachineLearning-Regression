#import the essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

#This class is the main algorithm implementation of linear regression using gradient descent.
#It has 5 functions: init,fit,fit_normal,calculateRMSE and predict
class ModelLinearRegression:
    #initializes three class variables:iterations,threshold and learningrate of the model
    def __init__(self,iterations = 1000,threshold=0.005,learningrate=0.0004):
        self.iterations = iterations
        self.threshold = threshold
        self.learningrate = learningrate

    #fits the provided feature data and output class to the model using gadient descent
    #returns the iterations array and the rmse array for all the interations
    def fit(self,X,y):
        diff = sys.maxsize
        j = 0
        rmse_array = []
        self.w = np.array([0] * len(X[0]))
        rmse_array.append(self.calculateRMSE(self.predict(X), y))
        while diff > self.threshold and j < self.iterations:
            gradient = np.array([0] * len(X[0]))
            for i in range(len(X)):
                gradient = gradient + (self.w.T.dot(X[i]) - y[i]) * X[i]
            old_rmse = self.calculateRMSE(self.predict(X),y)
            self.w = self.w - (self.learningrate * gradient)
            new_rmse = self.calculateRMSE(self.predict(X),y)
            rmse_array.append(new_rmse)
            diff = abs(new_rmse - old_rmse)
            j = j + 1
        return range(j+1),rmse_array

    # fits the provided feature data and output class to the model using normal equations
    def fit_normal(self,X,y):
        XtX = np.linalg.inv(X.T.dot(X))
        XtX_xT = XtX.dot(X.T)
        self.w =  XtX_xT.dot(y)

    #model prediction for the given feature data X
    def predict(self, X):
        result = []
        for i in range(len(X)):
            result.append(self.w.T.dot( X[i]))
        return result

    # Calculate the root mean squared error of the two params y_actual and y_pred
    def calculateRMSE(self,y_actual, y_pred):
        sse = 0
        for i in range(len(y_pred)):
            error = y_pred[i] - y_actual[i]
            sse = sse + (error * error)
        rmse = np.sqrt(sse / len(y_pred))
        return rmse









