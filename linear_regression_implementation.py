import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LinearRegression:

    def __init__(self):
        self.w = 0
        self.b = 0

    def forward_prop(self, X, w, b):
        y_pred=np.dot(w, X) + b
        return y_pred

    def cost_function(self, y_pred, y_real):
        m = y_real.shape[1]
        MSE = (1/(2*m)) * np.sum(np.square(y_pred-y_real))
        return MSE

    def derivative_result(self, X, y_real, y_pred):
        m = y_real.shape[1]
        der_ypred = (1/m) * (y_pred - y_real)
        der_w = np.dot(der_ypred, X.T)
        der_b = np.sum(der_ypred)
        return der_w, der_b

    def gradient_descent_update(self, w, b, der_w, der_b, learning_rate):
        w = w - learning_rate * der_w
        b = b - learning_rate * der_b
        return w, b

    def run_model(self, X_train, y_real, X_val, y_val, learning_rate, epochs):
        w, b = self.w, self.b
        costs_train = []
        m_train = y_real.shape[1]
        m_val = y_val.shape[1]

        z_val = []
        MSE_val = 0
        for i in range(1, epochs+1):
            y_pred = self.forward_prop(X_train,w,b)
            epoch_cost = self.cost_function(y_pred,y_real)

            dw, db = self.derivative_result(X_train, y_real, y_pred)
            w, b = self.gradient_descent_update(w, b , dw, db, learning_rate)

            costs_train.append(epoch_cost)

        z_val = self.forward_prop(X_val,w,b)
        MSE_val = (1 / m_val) * np.sum(np.abs(z_val - y_val))

        self.print_results(w, b,epoch_cost,MSE_val, costs_train, z_val, y_val, learning_rate)

    def print_results(self, w, b, epoch_cost,MSE_val, costs_train, z_val, y_val, learning_rate):
        #: print actual and predicted values
        print('Actual Val', 'Predicted Val')
        for i in range(0, len(z_val[0])):
            print("{},{}".format(y_val[0][i],z_val[0][i]))

        print('Model Training Data cost is {}'.format(epoch_cost))
        print('Model Testing cost is {}'.format(MSE_val))
        print('Intercept: {}'.format(b))
        print("Coefficient: {}".format(w[0]))

        plt.plot(costs_train)
        plt.xlabel('Iterations')
        plt.ylabel('Training cost')
        plt.title('Learning rate '+str(learning_rate))
        plt.show()


class DataSet:

    @classmethod
    def get_data_set(cls):
        #: Get test and training data from AWS
        X_train = pd.read_csv('https://utd-class.s3.amazonaws.com/qsar_fish_toxicity_training_data.csv', ";")
        X_val = pd.read_csv('https://utd-class.s3.amazonaws.com/qsar_fish_toxicity_testing.csv', ";")

        #: Get exact values for both training and testing
        y_train = X_train['quantitive_response']
        y_val = X_val['quantitive_response']

        #: Get features for both training and testing data
        X_train = X_train[['CICO', 'SM1_Dz', 'GATS1i', 'NdsCH', 'Ndssc', 'MLOGP']]
        X_val = X_val[['CICO', 'SM1_Dz', 'GATS1i', 'NdsCH', 'Ndssc', 'MLOGP']]

        X_train =X_train.T
        y_train = np.array([y_train])
        X_val = X_val.T
        y_val = np.array([y_val])
        return X_train, y_train, X_val, y_val


#: Get the data set
X_train, y_train, X_val, y_val = DataSet.get_data_set()
#: Initialize linear regression and run model
linear_regression = LinearRegression()
linear_regression.run_model(X_train, y_train, X_val, y_val, 0.001, 1000)
