import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    def __init__(self):
        # Parameters
        self.input_size = 11
        self.output_size = 1
        self.hidden_size = 30

        # Weights and bias
        self.bias = 1
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, x):                       # x : input vector
        y = np.dot(x, self.W1) + self.bias      # y = W1 * x + bias
        yo = self.sigmoid(y)                    # yo = sigmoid(y)
        z = np.dot(yo, self.W2).sum()           # z = W2 * yo
        zo = self.sigmoid(z)                    # zo = sigmoid(z)
        return y, yo, z, zo

    def backward(self, x, y, yo, z, zo, t):     # t : target
        dloss_dzo = 2 * (zo - t)                # The function that produces the error is a square function
        dzo_dz = self.sigmoid_derivative(z)
        dloss_dz = dloss_dzo * dzo_dz
        dz_dw2 = yo
        dloss_dw2 = np.dot(dz_dw2.T, dloss_dz)
        dz_dy0 = self.W2
        dloss_dyo = np.dot(dloss_dz, dz_dy0.T)
        dyo_dy = self.sigmoid_derivative(y)
        dy_dw1 = x.T
        dloss_dy = dloss_dyo * dyo_dy
        dloss_w1 = np.dot(dy_dw1, dloss_dy)
        dy_dbias = 1
        dloss_dbias = dloss_dy * dy_dbias
        return dloss_dw2, dloss_w1, dloss_dbias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self, input_vectors, targets, learning_rate, iterations):
        RMSE = []       # Array for plotting figure
        rmse = 0        # RMSE after iterations
        N = len(input_vectors)
        for current_iteration in range(iterations):
            sum_squared_error = 0
            for i in range(N):
                target = targets[i]
                input_vector = np.reshape(input_vectors[i], (1, 11))
                y, yo, z, zo = self.forward(input_vector)
                dloss_dw2, dloss_w1, dloss_dbias = self.backward(input_vector, y, yo, z, zo, target)
                self.W1 = self.W1 - learning_rate * dloss_w1  # update weights W1
                self.W2 = self.W2 - learning_rate * dloss_dw2  # update weights W2
                self.bias = self.bias - learning_rate * np.sum(dloss_dbias)  # update bias
                squared_error = math.pow(zo - target, 2)
                sum_squared_error += squared_error
            rmse += sum_squared_error
            RMSE.append(np.sqrt(sum_squared_error / N))
            print(current_iteration)
        rmse = np.sqrt(rmse / (N * iterations))
        return rmse, RMSE


def split_data():
    data = np.genfromtxt('winequality-red.csv', skip_header=1, delimiter=';', usemask=True)
    x = data[:, :11]
    y = data[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    return x_train, x_test, y_train, y_test


def standardize_data(x, y):
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)
    std_x = np.std(x, axis=0)
    std_y = np.std(y, axis=0)
    standardized_x = (x - mean_x) / std_x
    standardized_y = (y - mean_y) / std_y
    return standardized_x, standardized_y


def main():
    iterations = 1000
    learning_rate = 0.5
    NN = NeuralNetwork()
    x_train, x_test, y_train, y_test = split_data()
    input_vectors, targets = standardize_data(x_train, y_train)
    error, rmse = NN.train(input_vectors, targets, learning_rate, iterations)
    fig = plt.figure()
    plt.plot(rmse)
    fig.suptitle('Learning rate = 0.5', fontsize=20)
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Root mean squared error', fontsize=16)
    plt.show()


if __name__ == '__main__':
    main()
