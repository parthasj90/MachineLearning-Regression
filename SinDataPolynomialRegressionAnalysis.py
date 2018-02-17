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


def my_model_normal(X_train,y_train,X_test,y_test):
    from LinearRegression import ModelLinearRegression
    constant = np.ones((len(X_train), 1))
    X_train = np.hstack((constant, X_train))
    const = np.ones((len(X_test), 1))
    X_test = np.hstack((const, X_test))
    regressor = ModelLinearRegression()
    regressor.fit_normal(X_train, y_train)
    y_pred_test = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    rmse_test = regressor.calculateRMSE(y_test,y_pred_test)
    rmse_train = regressor.calculateRMSE(y_train,y_pred_train)
    print("training rmse(normal)",rmse_train)
    print("testing rmse(normal)",rmse_test)
    return rmse_train,rmse_test

def main():
    #preprocess the dataset
    #to run anyother dataset just change the filename
    filename_train = 'sinData_Train.csv'
    filename_test = 'sinData_Validation.csv'
    trainingdataset = pd.read_csv(filename_train,header=None)
    X_train = trainingdataset.iloc[:,:-1].values
    y_train = trainingdataset.iloc[:,-1].values
    testingdataset = pd.read_csv(filename_test,header=None)
    X_test = testingdataset.iloc[:,:-1].values
    y_test = testingdataset.iloc[:,-1].values
    powers = range(1,16)
    rmse_train_all_normal = []
    rmse_test_all_normal = []
    x_train = X_train
    x_test = X_test
    for power in powers:
        print("Executing for power: ",power)
        if power != 1:
            add = X_train**power
            x_train = np.append(x_train,add,axis=1)
            add = X_test**power
            x_test = np.append(x_test, add, axis=1)
        #print("Executing sklearn model")
        #sk_model(X_train,y_train,X_test,y_test)
        print("Executing my model")
        rmse_train_normal, rmse_test_normal = my_model_normal(x_train, y_train, x_test, y_test)
        rmse_train_all_normal.append(rmse_train_normal)
        rmse_test_all_normal.append(rmse_test_normal)
    print("Average Test RMSE with normal equations",np.mean(rmse_test_all_normal))
    print("Standard Deviation Test RMSE with normal equations",np.std(rmse_test_all_normal))
    print("Average Train RMSE with normal equations", np.mean(rmse_train_all_normal))
    print("Standard Deviation Train RMSE with normal equations", np.std(rmse_train_all_normal))
    plt.plot(powers,rmse_train_all_normal,'b--',powers,rmse_test_all_normal,'r--')
    plt.title('RMSE ACROSS ALL POWERS WITH NORMAL EQUATIONS')
    plt.xlabel('POWERS')
    plt.ylabel('RMSE')
    plt.show()


if __name__ == '__main__':
    main()