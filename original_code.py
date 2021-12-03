import numpy as np
import matplotlib.pyplot as plt
import csv


class neuron_fcn(object):
    def output(self, neuron, derivative=False):
        """Dispatch method"""
        # Get the method from 'self'. Default to a lambda.
        name = neuron['activation_function']
        method = getattr(self, str(name), "Invalid_function")
        # Call the method as we return it
        return method(neuron, derivative)

    def Invalid_function(self, *arg):
        print("Error: Invalid activation function")
        return None

    def activation_potential(self, neuron, inputs):
        activation = 0
        if neuron["bias"]:
            inputs = np.append(inputs, 1)
        for i, weight in enumerate(neuron["weights"]):
            activation += weight * inputs[i]
        return activation

    def linear(self, neuron, derivative=False):
        out = 0
        if not derivative:
            out = neuron['activation_potential']
        else:
            out = 1
        return out

    def logistic(self, neuron, derivative=False):
        out = 0
        if not derivative:
            out = 1.0 / (1.0 + np.exp(-neuron['activation_potential']))
        else:
            out = neuron['output'] * (1.0 - neuron['output'])
        return out

    def tanh(self, neuron, derivative=False):
        out = 0
        if not derivative:
            out = (np.exp(neuron['activation_potential']) - np.exp(-neuron['activation_potential'])) / (
                    np.exp(neuron['activation_potential']) + np.exp(-neuron['activation_potential']))
        else:
            out = 1.0 - np.power(neuron['output'], 2)
        return out

    def relu(self, neuron, derivative=False):
        out = 0
        if not derivative:
            out = np.maximum(0, neuron['activation_potential'])
        else:
            if neuron['activation_potential'] >= 0:
                out = 1
        return out


class loss_fcn(object):
    # Error value of neuron
    def loss(self, loss, expected, outputs, derivative):
        """Dispatch method"""
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, str(loss), lambda: "Invalid_function")
        # Call the method as we return it
        return method(expected, outputs, derivative)

    def sum(self, loss, expected, outputs, derivative):
        """Dispatch method"""
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, str(loss), lambda: "Invalid loss function")
        # Call the method as we return it
        error_sum = 0
        for exp, out in zip(expected, outputs):
            error_sum += method(exp, out, derivative)
        return error_sum

    def Invalid_function(self, *arg):
        print("Error: Invalid loss function")
        return None

    # Mean Square Error loss function
    def mse(self, expected, outputs, derivative=False):
        error_value = 0
        if not derivative:
            error_value = 0.5 * (expected - outputs) ** 2
        else:
            error_value = -(expected - outputs)
        return error_value

    # Cross-entropy loss function
    def binary_cross_entropy(self, expected, outputs, derivative=False):
        error_value = 0
        if not derivative:
            error_value = -expected * np.log(outputs) - (1 - expected) * np.log(1 - outputs)
        else:
            error_value = -(expected / outputs - (1 - expected) / (1 - outputs))
        # print(f"output = {outputs}, expected = {expected}, error = {error_value}, derivative = {derivative}")
        return error_value


# Initialize a network
class Neural_network(object):
    def create_network(self, structure):
        self.nnetwork = list()
        for index, layer in enumerate(structure[1:], start=1):
            new_layer = []
            for i in range(layer['units']):
                neuron = {
                    'weights': [np.random.randn() for i in range(structure[index - 1]['units'] + int(layer['bias']))],
                    'bias': layer['bias'],
                    'activation_function': layer['activation_function'],
                    'activation_potential': 0,
                    'delta': [0 for i in range(structure[index - 1]['units'] + int(layer['bias']))],
                    'output': 0}
                new_layer.append(neuron)
            self.nnetwork.append(new_layer)
        return self.nnetwork

    # Forward propagate input to a network output
    def forward_propagate(self, nnetwork, inputs):
        row = list(inputs.copy())
        for layer in nnetwork:
            next_row = []
            for neuron in layer:
                bias_inputs = row.copy()
                if neuron['bias']:
                    bias_inputs.append(1)
                tf = neuron_fcn()
                neuron['activation_potential'] = tf.activation_potential(neuron, row)
                neuron['output'] = tf.output(neuron, derivative=False)
                next_row.append(neuron['output'])
            row = next_row.copy()
        return row

    # Backpropagate error and store it in neuron
    def backward_propagate(self, loss_function, nnetwork, expected):
        for i in reversed(range(len(nnetwork))):
            layer = nnetwork[i]
            errors = list()
            if i != len(nnetwork) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in nnetwork[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    loss = loss_fcn()
                    errors.append(loss.loss(loss_function, expected[j], neuron['output'], derivative=True))
            for j in range(len(layer)):
                tf = neuron_fcn()
                neuron = layer[j]
                neuron['delta'] = errors[j] * tf.output(neuron, derivative=True)

    # Update network weights with error
    def update_weights(self, nnetwork, inputs, l_rate):
        for i in range(len(nnetwork)):
            row = inputs

            if i != 0:
                row = [neuron['output'] for neuron in nnetwork[i - 1]]

            for neuron in nnetwork[i]:
                for j in range(len(row)):
                    neuron['weights'][j] -= l_rate * neuron['delta'] * row[j]
                if neuron['bias']:
                    neuron['weights'][-1] -= l_rate * neuron['delta']

    # Train a network for a fixed number of epochs
    def train(self, nnetwork, x_train, y_train, l_rate=0.01, n_epoch=100, loss_function='mse', verbose=1):
        for epoch in range(n_epoch):
            sum_error = 0
            for iter, (x_row, y_row) in enumerate(zip(x_train, y_train)):
                if not len(np.shape(x_row)):
                    x_row = [x_row]
                if not len(np.shape(y_row)):
                    y_row = [y_row]

                outputs = self.forward_propagate(nnetwork, x_row)

                loss = loss_fcn()
                l = loss.sum(loss_function, y_row, outputs, derivative=False)
                sum_error += l
                if verbose > 1:
                    print(f"iteration = {iter + 1}, output = {outputs}, target = {y_row}, loss = {l:.4f}")

                self.backward_propagate(loss_function, nnetwork, y_row)

                self.update_weights(nnetwork, x_row, l_rate)

            if verbose > 0:
                print('>epoch=%d, loss=%.3f' % (epoch + 1, sum_error))
        return nnetwork

    # Calculate network output
    def predict(self, neuron, inputs):
        y = []
        for input in inputs:
            y.append(self.forward_propagate(neuron, input))
        return y


def generate_regression_data(n, tosave=True, fname="reg_data"):
    # Generate regression dataset
    X = np.linspace(-5, 5, n).reshape(-1, 1)
    y = np.sin(2 * X) + np.cos(X) + 5
    # simulate noise
    data_noise = np.random.normal(0, 0.2, n).reshape(-1, 1)
    # Generate training data
    Y = y + data_noise

    plt.plot(X, y, label="Real process")
    plt.plot(X, Y, 'r--o', label="Training data")
    plt.legend()
    plt.show()

    np.savetxt('X_data.dat', X)
    np.savetxt('Y_data.dat', Y)

    return X.reshape(-1, 1).tolist(), Y.reshape(-1, 1).tolist()


def read_regression_data(fname="reg_data"):
    X = np.loadtxt('X_data.dat')
    Y = np.loadtxt('Y_data.dat')

    plt.plot(X, Y, 'r--o', label="Training data")
    plt.legend()
    plt.grid()
    plt.show()

    return X, Y


def test_regression():
    # Read data
    X, Y = read_regression_data()

    # Create network
    model = Neural_network()
    structure = [{'type': 'input', 'units': 1},
                 {'type': 'dense', 'units': 4, 'activation_function': 'linear', 'bias': True},
                 {'type': 'dense', 'units': 4, 'activation_function': 'logistic', 'bias': True},
                 {'type': 'dense', 'units': 1, 'activation_function': 'linear', 'bias': True}]

    network = model.create_network(structure)
    for layer in network:
        print(layer)

    model.train(network, X, Y, 0.01, 4000, 'mse')
    print(f"G1 = {network}")

    X_test = np.linspace(-7, 7, 100).reshape(-1, 1)
    X_test = np.array(X_test).tolist()

    predicted = model.predict(network, X_test)

    plt.plot(X, Y, 'r--o', label="Training data")
    plt.plot(X_test, predicted, 'b--x', label="Predicted")
    plt.legend()
    plt.grid()
    plt.show()


def generate_classification_data(n=40, tosave=True, fname="class_data"):
    # Class 1 - samples generation
    X1_1 = 2 + 4 * np.random.rand(n, 1)
    X1_2 = 1 + 4 * np.random.rand(n, 1)
    class1 = np.concatenate((X1_1, X1_2), axis=1)
    Y1 = np.ones(n)

    # Class 0 - samples generation
    X0_1 = 4 + 4 * np.random.rand(n, 1)
    X0_2 = 3 + 4 * np.random.rand(n, 1)
    class0 = np.concatenate((X0_1, X0_2), axis=1)
    Y0 = np.zeros(n)

    X = np.concatenate((class1, class0))
    Y = np.concatenate((Y1, Y0))

    idx0 = [i for i, v in enumerate(Y) if v == 0]
    idx1 = [i for i, v in enumerate(Y) if v == 1]

    np.savetxt('X_data.dat', X)
    np.savetxt('Y_data.dat', Y)

    return X, Y, idx0, idx1


def read_classification_data(fname="class_data"):
    X = np.loadtxt('X_data.dat')
    Y = np.loadtxt('Y_data.dat')

    idx0 = [i for i, v in enumerate(Y) if v == 0]
    idx1 = [i for i, v in enumerate(Y) if v == 1]

    return X, Y, idx0, idx1


def test_classification():
    # Read data
    X, Y, idx0, idx1 = read_classification_data()

    # Create network
    model = Neural_network()
    structure = [{'type': 'input', 'units': 2},
                 {'type': 'dense', 'units': 32, 'activation_function': 'relu', 'bias': True},
                 {'type': 'dense', 'units': 1, 'activation_function': 'logistic', 'bias': True}]

    network = model.create_network(structure)
    for layer in network:
        print(layer)

    model.train(network, X, Y, 0.001, 1000, 'binary_cross_entropy', 1)

    y = model.predict(network, X)
    t = 0
    for n, m in zip(y, Y):
        t += 1 - np.abs(np.round(np.array(n)) - np.array(m))
        print(f"pred = {n}, pred_round = {np.round(n)}, true = {m}")

    ACC = t / len(X)
    print(f"Classification accuracy = {ACC * 100}%")

    # Plotting decision regions
    xx, yy = np.meshgrid(np.arange(0, 8, 0.1),
                         np.arange(0, 8, 0.1))

    X_vis = np.c_[xx.ravel(), yy.ravel()]

    h = model.predict(network, X_vis)
    h = np.array(h) >= 0.5
    h = np.reshape(h, (len(xx), len(yy)))

    plt.contourf(xx, yy, h, cmap='jet')
    plt.scatter(X[idx1, 0], X[idx1, 1], marker='^', c="red", edgecolors="white", label="class 1")
    plt.scatter(X[idx0, 0], X[idx0, 1], marker='o', c="blue", edgecolors="white", label="class 0")
    plt.show()


generate_classification_data()
test_classification()

# # generate_regression_data(30)
# test_regression()
