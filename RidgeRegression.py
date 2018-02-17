#import the essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

#This class is the main algorithm implementation of ridge regression using gradient descent.
#It has 4 functions: init,fit ,predict and calculateRMSE
class ModelRidgeRegression:
    #initializes three class variables:iterations,threshold and learningrate of the model
    def __init__(self,l):
        self.l = l

    # fits the provided feature data and output class to the model using normal equations
    def fit(self,X,y):
        XtX = np.linalg.inv(X.T.dot(X) + self.l* np.identity(len(X[0])))
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









