from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import numpy as np
from urllib.request import urlopen
import csv
import codecs
from sklearn.model_selection import train_test_split


def load_remote_csv(csv_url):
    data_set = list()
    response = urlopen(csv_url)
    csv_file = csv.reader(codecs.iterdecode(response, 'utf-8'))
    for line in csv_file:
        data_set.append(line)
    return data_set

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (float(row[i]) - float(minmax[i][0])) / (float(minmax[i][1]) - float(minmax[i][0]))

def preprocessing(dataset):
    str_column_to_int(dataset, len(dataset[0]) - 1)
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def split_training_testing(dataset):
    train_dataset, test_dataset = train_test_split(dataset)

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(x, A_function):
    if A_function == "sigmoid":
        return sigmoid(x)
    elif A_function == "tanh":
        return tanh(x)
    elif A_function == "reLu":
        return reLu(x)


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def tanh(x):
    y = np.divide(np.sinh(x), np.cosh(x))
    return y


def reLu(x):
   return np.maximum(0,x)


def transfer_derivative(x, A_function):
    if A_function == "sigmoid":
        return sigmoid_derivative(x)
    elif A_function == "tanh":
        return tanh_derivative(x)
    elif A_function == "reLu":
        return reLu_derivative(x)

def sigmoid_derivative(x):
    return x * (1.0 - x)

def tanh_derivative(x):
    return 1 - np.square(x)

def reLu_derivative(x):
    scaling_factor = 0.001
    if x < 0:
        return 0
    else:
        return 0.001

def forward_propagate(network, row, A_function):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            x = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(x, A_function)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def backward_propagate_error(network, expected, A_function):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'], A_function)


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs, A_function):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row, A_function)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected, A_function)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def predict_training(network, row, A_function):
    outputs = forward_propagate(network, row, A_function)
    return outputs.index(max(outputs))

def back_propagation(train, test, l_rate, n_epoch, n_hidden, A_function):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs, A_function)
    predictions = list()
    for row in test:
        prediction = predict_training(network, row, A_function)
        predictions.append(prediction)
    return (predictions)


seed(1)
csv_url = 'https://utd-class.s3.amazonaws.com/neural_network/seeds_dataset.csv'
dataset = load_remote_csv(csv_url)

preprocessing(dataset)

# evaluate algorithm
n_folds = 2
l_rate = 0.3
n_epoch = 1000
n_hidden = 4
A_function = "tanh"
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden, A_function)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


