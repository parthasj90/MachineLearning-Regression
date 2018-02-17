#import the essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#SKLEARN LINEAR REGRESSION MODEL implementaion
def sk_model(X_train,y_train,X_test,y_test):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)
    y_pred_test = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    #rmse_test = calculateRMSE(y_test,y_pred_test)
    #mse_train = calculateRMSE(y_train,y_pred_train)
    #print("training rmse",rmse_train)
    #print("testing rmse",rmse_test)

#MY MODEL LINEAR REGRESSION MODEL implentation
def my_model(X_train,y_train,X_test,y_test):
    from LinearRegression import ModelLinearRegression
    constant = np.ones((len(X_train), 1))
    X_train = np.hstack((constant, X_train))
    const = np.ones((len(X_test), 1))
    X_test = np.hstack((const, X_test))
    regressor = ModelLinearRegression(1000,0.0001,0.0007)
    iteration_array,rmse_array = regressor.fit(X_train,y_train)
    #regressor = ModelLinearRegression()
    #regressor.fit_normal(X_train, y_train)
    y_pred_test = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    rmse_test = regressor.calculateRMSE(y_test,y_pred_test)
    rmse_train = regressor.calculateRMSE(y_train,y_pred_train)
    plt.plot(iteration_array,rmse_array)
    plt.title('RMSE for all iterations in a fold')
    plt.xlabel('ITERATIONS')
    plt.ylabel('RMSE')
    plt.show()
    print("training rmse",rmse_train)
    print("testing rmse",rmse_test)
    return rmse_train,rmse_test

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
    filename = 'concreteData.csv'
    dataset = pd.read_csv(filename,header=None)
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values

    #kfoldgeneration
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    kf = KFold(n_splits=10)
    rmse_train_all = []
    rmse_test_all = []
    rmse_train_all_normal = []
    rmse_test_all_normal = []
    counter = 1
    for train_index, test_index in kf.split(X):
        print("Executing Fold: ",counter)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        #print("Executing sklearn model")
        #sk_model(X_train,y_train,X_test,y_test)
        print("Executing my model")
        rmse_train,rmse_test = my_model(X_train,y_train,X_test,y_test)
        rmse_train_normal, rmse_test_normal = my_model_normal(X_train, y_train, X_test, y_test)
        rmse_train_all.append(rmse_train)
        rmse_test_all.append(rmse_test)
        rmse_train_all_normal.append(rmse_train_normal)
        rmse_test_all_normal.append(rmse_test_normal)
        counter += 1
    print("Average Test RMSE with gradient descent",np.mean(rmse_test_all))
    print("Standard Deviation Test RMSE with gradient descent",np.std(rmse_test_all))
    print("Average Train RMSE with gradient descent", np.mean(rmse_train_all))
    print("Standard Deviation Train RMSE with gradient descent", np.std(rmse_train_all))
    print("Average Test RMSE with normal equations",np.mean(rmse_test_all_normal))
    print("Standard Deviation Test RMSE with normal equations",np.std(rmse_test_all_normal))
    print("Average Train RMSE with normal equations", np.mean(rmse_train_all_normal))
    print("Standard Deviation Train RMSE with normal equations", np.std(rmse_train_all_normal))
    folds = range(1,11)
    plt.plot(folds,rmse_train_all,'b--',folds,rmse_test_all,'r--')
    plt.title('RMSE ACROSS ALL FOLDS WITH GRADIENT DESCENT')
    plt.xlabel('FOLDS')
    plt.ylabel('RMSE')
    plt.show()
    plt.plot(folds,rmse_train_all_normal,'b--',folds,rmse_test_all_normal,'r--')
    plt.title('RMSE ACROSS ALL FOLDS WITH NORMAL EQUATIONS')
    plt.xlabel('FOLDS')
    plt.ylabel('RMSE')
    plt.show()


if __name__ == '__main__':
    main()