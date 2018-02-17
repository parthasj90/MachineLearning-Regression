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
    filename = 'yachtData.csv'
    dataset = pd.read_csv(filename,header=None)
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values

    #kfoldgeneration
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    kf = KFold(n_splits=10)

    powers = range(1,8)
    rmse_train_power = []
    rmse_test_power = []
    x = X
    for power in powers:
        print("Executing Power: ", power)
        rmse_train_all_normal = []
        rmse_test_all_normal = []

        if power != 1 :
            add = X**power
            x = np.append(x,add,axis=1)
        print(x[0])
        counter = 1
        for train_index, test_index in kf.split(x):
            print("Executing Fold: ",counter)
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            #print("Executing sklearn model")
            #sk_model(X_train,y_train,X_test,y_test)
            #print(X_train[0])
            #print(X_test[0])
            print("Executing my model")
            rmse_train_normal, rmse_test_normal = my_model_normal(X_train, y_train, X_test, y_test)
            rmse_train_all_normal.append(rmse_train_normal)
            rmse_test_all_normal.append(rmse_test_normal)
            counter += 1
        rmse_test_power.append(np.mean(rmse_test_all_normal))
        rmse_train_power.append(np.mean(rmse_train_all_normal))


    plt.plot(powers,rmse_train_power,'b--',powers,rmse_test_power,'r--')
    plt.title('RMSE ACROSS ALL POWERS WITH NORMAL EQUATIONS')
    plt.xlabel('POWERS')
    plt.ylabel('RMSE')
    plt.show()


if __name__ == '__main__':
    main()