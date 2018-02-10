import pandas as pd
import numpy as np
import matplotlib as plt
import sys


class ModelLinearRegression:
    def __init__(self,iterations,threshold,learningrate):
        self.iterations = iterations
        self.threshold = threshold
        self.learningrate = learningrate

    def fit(self,X,y):
        diff = sys.maxsize
        j = 0
        self.w = np.array([0] * len(X[0]))
        while diff > self.threshold and j < self.iterations:
            gradient = np.array([0] * len(X[0]))
            for i in range(len(X)):
                gradient = gradient + (self.w.T.dot(X[i]) - y[i]) * X[i]
            old_rmse = calculateRMSE(self.predict(X),y)
            self.w = self.w - (self.learningrate * gradient)
            new_rmse = calculateRMSE(self.predict(X),y)
            diff = abs(new_rmse - old_rmse)
            j = j + 1

    def predict(self, X):
        result = []
        for i in range(len(X)):
            result.append(self.w.T.dot( X[i]))
        return result


def sk_model(X_train,y_train,X_test,y_test):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)
    y_pred_test = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    rmse_test = calculateRMSE(y_test,y_pred_test)
    rmse_train = calculateRMSE(y_train,y_pred_train)
    print("training rmse",rmse_train)
    print("testing rmse",rmse_test)


def my_model(X_train,y_train,X_test,y_test):
    constant = np.ones((len(X_train), 1))
    X_train = np.hstack((constant, X_train))
    const = np.ones((len(X_test), 1))
    X_test = np.hstack((const, X_test))
    regressor = ModelLinearRegression(1000,0.005,0.0004)
    regressor.fit(X_train,y_train)
    y_pred_test = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    rmse_test = calculateRMSE(y_test,y_pred_test)
    rmse_train = calculateRMSE(y_train,y_pred_train)
    print("training rmse",rmse_train)
    print("testing rmse",rmse_test)
    return rmse_train,rmse_test

def calculateRMSE(y,y_pred):
    sse = 0
    for i in range(len(y_pred)):
        error = y_pred[i] - y[i]
        sse = sse + (error * error)
    rmse = np.sqrt(sse / len(y_pred))
    return rmse


def main():
    #preprocess the dataset
    dataset = pd.read_csv('housing.csv',header=None)
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values

    #kfoldgeneration
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    kf = KFold(n_splits=10)
    rmse_train_all = []
    rmse_test_all = []
    counter = 1
    for train_index, test_index in kf.split(X):
        print("Executing Fold: ",counter)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.fit_transform(X_test)
        print("Executing sklearn model")
        sk_model(X_train,y_train,X_test,y_test)
        print("Executing my model")
        rmse_train,rmse_test = my_model(X_train,y_train,X_test,y_test)
        rmse_train_all.append(rmse_train)
        rmse_test_all.append(rmse_test)
        counter += 1
    print("Average Test RMSE",np.mean(rmse_test_all))
    print("Standard Deviation Test RMSE",np.std(rmse_test_all))


if __name__ == '__main__':
    main()








