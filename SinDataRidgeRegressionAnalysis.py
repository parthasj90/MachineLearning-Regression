#import the essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculateRMSE(y_actual, y_pred):
    sse = 0
    for i in range(len(y_pred)):
        error = y_pred[i] - y_actual[i]
        sse = sse + (error * error)
    rmse = np.sqrt(sse / len(y_pred))
    return rmse

#SKLEARN LINEAR REGRESSION MODEL implementaion
def sk_model(X_train,y_train,X_test,y_test,l):
    from sklearn.linear_model import Ridge
    regressor = Ridge(l)
    regressor.fit(X_train,y_train)
    y_pred_test = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    rmse_test = calculateRMSE(y_test,y_pred_test)
    rmse_train = calculateRMSE(y_train,y_pred_train)
    print("training rmse",rmse_train)
    print("testing rmse",rmse_test)


def my_model_normal(X_train,y_train,X_test,y_test,l):
    from RidgeRegression import ModelRidgeRegression
    regressor = ModelRidgeRegression(l)
    regressor.fit(X_train, y_train)
    y_pred_test = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    rmse_test = regressor.calculateRMSE(y_test,y_pred_test)
    rmse_train = regressor.calculateRMSE(y_train,y_pred_train)
    print("training rmse(normal)",rmse_train)
    print("testing rmse(normal)",rmse_test)
    return rmse_train,rmse_test


def main():
    from sklearn.preprocessing import PolynomialFeatures
    #preprocess the dataset
    #to run anyother dataset just change the filename
    filename_train = 'sinData_Train.csv'
    dataset = pd.read_csv(filename_train,header=None)
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    lambdas = []
    i =0
    lambdas.append(i)
    while i<=10:
        i = i + 0.2
        lambdas.append(i)
    #--------------for power = 5 , change the below variable to 5---------------------
    power = 9
    x = X
    for i in range(2,power + 1):
        add = X ** i
        x = np.append(x, add, axis=1)

    #kfoldgeneration
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    kf = KFold(n_splits=10)
    rmse_train_lambdas = []
    rmse_test_lambdas = []
    for l in lambdas:
        print("Executing Lambda: ", l)
        rmse_train_all_normal = []
        rmse_test_all_normal = []
        counter = 1
        for train_index, test_index in kf.split(x):
            print("Executing Fold: ", counter)
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            #print("Executing sklearn model")
            #sk_model(X_train,y_train,X_test,y_test,l)
            print("Executing my model")
            rmse_train_normal, rmse_test_normal = my_model_normal(X_train, y_train, X_test, y_test,l)
            rmse_train_all_normal.append(rmse_train_normal)
            rmse_test_all_normal.append(rmse_test_normal)
            counter += 1
        rmse_train_lambdas.append(np.mean(rmse_test_all_normal))
        rmse_test_lambdas.append(np.mean(rmse_train_all_normal))

    print("Average Test RMSE with normal equations",np.mean(rmse_train_lambdas))
    print("Standard Deviation Test RMSE with normal equations",np.std(rmse_train_lambdas))
    print("Average Train RMSE with normal equations", np.mean(rmse_test_lambdas))
    print("Standard Deviation Train RMSE with normal equations", np.std(rmse_test_lambdas))
    plt.plot(lambdas,rmse_train_lambdas,'b--',lambdas,rmse_test_lambdas,'r--')
    plt.title('RMSE ACROSS ALL LAMBDAS WITH NORMAL EQUATIONS')
    plt.xlabel('LAMBDAS')
    plt.ylabel('RMSE')
    plt.show()


if __name__ == '__main__':
    main()