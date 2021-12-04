import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

class Activations():
    '''
    A collection of activation functions written using numpy functions.
    Included activation functions are linear, logistic (sigmoid), relu,
    and leaky relu. All methods are static, with the capability of 
    calculating their derivatives.
    '''

    @staticmethod
    def read_function(name):
        ''''
        returns the method corresponding to the name.
        '''
        return getattr(Activations, str(name), Activations.InvalidFunction)
    
    @staticmethod
    def InvalidFunction(*arg):
        '''
        raises an exception if the activation function requested does not exist.
        '''
        raise Exception("Invalid activation function")
    
    @staticmethod
    def linear(inp, derivative = False):
        '''
        returns image of function f(x) = x
        '''
        out = np.array(inp)
        if derivative:
            out = np.ones(out.shape)
        return out
    
    @staticmethod
    def logistic(inp, derivative = False):
        '''
        returns the sigmoid of the input
        '''
        out = np.array(inp)
        if derivative:
            n_dev = Activations.logistic(out, False)
            out = (1 - n_dev) * n_dev
        else:
            out = 1.0/(1 + np.exp(-1 * inp))
        return out
    
    @staticmethod
    def relu(inp, derivative = False):
        '''
        returns the image of the function f(x) = max(0, x) 
        '''
        out = np.array(inp)
        if derivative:
            out = np.where(out >= 0, 1, 0)
        else:
            out = np.where(out >=0, out, 0)
        return out
    
    @staticmethod
    def leaky_relu(inp, derivative = False, leak = 0.01):
        '''
        returns a modified computation of relu, where negative input yields
        a smaller output magnitude than that of when input is positve.
        the value of leak is a hyperparameter which may be tuned. 
        '''
        out = np.array(inp)
        if derivative:
            out = np.where(out >=0, 1, leak)
        else:
            out = np.where(out >=0, out, leak * out)
        return out

class Losses():
    '''
    A collection of vector friendly loss functions.
    Included loss functions are mean squared error, and binary crosss entropy.
    '''

    @staticmethod
    def read_function(name):
        '''
        returns the method corresponding to the name.
        '''
        return getattr(Losses, str(name), Losses.InvalidFunction)
    
    @staticmethod
    def InvalidFunction(*arg):
        '''
        raises an exception if the activation function requested does not exist.
        '''
        raise Exception("Invalid loss function")
        
    @staticmethod
    def mse(expected, outputs):
        '''
        mean Square Error loss function
        '''
        expected = np.array(expected)
        outputs = np.array(outputs)
        return 0.5 * (expected - outputs) ** 2

    @staticmethod
    def binary_cross_entropy(expected, outputs, derivative=False):
        '''
        cross-entropy loss function. implemented using scikit-learn
        '''
        expected = np.array(expected)
        outputs = np.array(outputs)
        # return -1 * expected * np.log(outputs) - (1 - expected) * np.log(1 - outputs)
        return metrics.log_loss(expected, outputs)

class NeuralNetwork():
    '''
    A collection of methods for building and training a multi-layer perceptron.
    Uses only gradient descent as an optimizer. Has the capability to do SGD, 
    mini-batch gradient descent or full batch gradient descent 
    '''

    # parameters and activations
    net_params = dict()
    activations = list()
    num_layers = 0
    
    def __init__(self, structure):
        '''
        build a structure for the neural network
        '''
        self.create_network(structure)
    
    def create_network(self, structure):
        '''
        creates matrices corresponding to weights. only supports dense/
        fully-connectedlayers. can opt for biases, and choose activation
        functions available in the Activations class.
        '''
        
        # ensure that the first layer is an input layer
        try:
            assert(structure[0]["type"] == "input")
        except:
            raise Exception("The first layer needs to be an input layer")
        
        self.num_layers = len(structure) - 1
        prev_units = int(structure[0]["units"])

        for i, layer in enumerate(structure[1: ]):

            # weight matrix at a layer has (i - 1 x i) shape 
            current_units = layer["units"]
            
            # initialize weight and bias paramters
            self.net_params["w" + str(i + 1)] = np.zeros((prev_units, current_units))
            self.net_params["b" + str(i + 1)] = None
            
            # add a bias parameter only if specified
            if layer["bias"]:
                self.net_params["b" + str(i + 1)] = np.zeros((1, layer["units"]))
            
            # store activation function in attribute and continue loop
            self.activations.append(layer["activation_function"])
            prev_units = current_units
        
        return self
    
    def show_model(self):
        '''
        displays the shape of the weight paramaters at each layer
        '''
        for name, arr in self.net_params.items():
            print("Shape of ", name, ": ", arr.shape)
            
    def initialize_weights(self, leak = 0.01):
        '''
        initializes weights bases on the Kaiming initialization.
        biases are initialized randomnly
        '''
        for i in range(1, self.num_layers + 1):

            # leak only exists in leaky relu
            if self.activations[i - 1] != "leaky_relu":
                leak = 0
            
            # using Kaiming initialization only on weights and not biases
            param = self.net_params["w" + str(i)]
            self.net_params["w" + str(i)] = np.sqrt(2/((1 + leak * leak)\
                                            * param.shape[0])) * np.random.randn(*param.shape)
            
            # using random initialization on bias if it is included
            param = self.net_params["b" + str(i)]
            if param is not None:
                self.net_params["b" + str(i)] = np.random.randn(*param.shape)
            
        return self
    
    def forward_prop(self, inp):
        '''
        performs forward propogation: from input to output
        '''
        # sanity check
        inp = np.array(inp)
        
        # propogate
        for l in range(1, self.num_layers + 1):
            
            bias = self.net_params["b" + str(l)]
            
            if bias is None:
                self.net_params["z" + str(l)] = np.matmul(inp, self.net_params["w" + str(l)])
            else:
                self.net_params["z" + str(l)] = np.matmul(inp, self.net_params["w" + str(l)]) + bias
            
            act = Activations.read_function(self.activations[l - 1])
            self.net_params["a" + str(l)] = act(self.net_params["z" + str(l)])
            inp = self.net_params["a" + str(l)]
        
        return self
    
    def back_prop(self, inp, out, loss_name):
        '''
        performs back propogation, calculating error at each layer
        '''
        # sanity check
        inp = np.array(inp)
        out = np.array(out)
        
        # only two types of losses are considered
        # ll is the string casted version of number of layers
        # for last layer
        ll = str(self.num_layers)
        if loss_name == "mse":
            self.net_params["delta" + ll] = np.multiply(self.net_params["a" + ll] - out,\
                                            Activations.read_function(self.activations[-1])\
                                            (self.net_params["z" + ll], derivative = True))
        elif loss_name == "binary_cross_entropy":
            self.net_params["delta" + ll] = self.net_params["a" + ll] - out
        else:
            raise Exception("Only two types of loss functions are supported: MSE and BCE")
        
        # for every other layer except the last
        for l in reversed(range(1, self.num_layers)):
            self.net_params["delta" + str(l)] = np.multiply(Activations.read_function(self.activations[l - 1])\
                                                (self.net_params["z" + str(l)], derivative = True),\
                                                np.matmul(self.net_params["delta" + str(l + 1)],\
                                                np.transpose(self.net_params["w" + str(l + 1)]),))
        return self
    
    def update_weights(self, inp, l_rate):
        '''
        updates parameters depending on gradient
        '''
        # update for the first layer
        self.net_params["dw1"] = np.matmul(np.transpose(inp), self.net_params["delta1"])
        if self.net_params["b1"] is not None:
            self.net_params["db1"] = np.sum(self.net_params["delta1"], axis = 0, keepdims = True)
        
        # calculate gradient of weights
        for l in range(2, self.num_layers + 1):
            self.net_params["dw" + str(l)] = np.matmul(np.transpose(self.net_params["a" + str(l - 1)]),\
                                             self.net_params["delta" + str(l)])
            if self.net_params["b" + str(l)] is not None:
                self.net_params["db" + str(l)] = np.sum(self.net_params["delta" + str(l)], axis = 0, keepdims = True)
                
        # update weights
        for l in range(1, self.num_layers + 1):
            self.net_params["w" + str(l)] -= l_rate/inp.shape[0] * self.net_params["dw" + str(l)]
            if self.net_params["b" + str(l)] is not None:
                self.net_params["b" + str(l)] -= l_rate/inp.shape[0] * self.net_params["db" + str(l)]
                
        return self
    
    def predict(self, inp):
        '''
        returns an array of output classes given input features.
        output is determined using the current weights of the network
        '''
        self.forward_prop(inp)
        result = np.round(self.net_params["a" + str(self.num_layers)])
        return result
    
    def calc_loss(self, out, loss_func):
        '''
        calculates the cumulative loss
        '''
        return np.sum(Losses.read_function(loss_func)(out, self.net_params["a" + str(self.num_layers)]))
    
    def accuracy(self, inp, out):
        '''
        returns accuracy as a percentage
        '''
        return sum(self.predict(inp) == out)[0]/out.shape[0] * 100
    
    def train(self, X, Y, n_epochs = 500, batch_size = 16, loss_func = "mse", l_rate = 0.001, rand_scale = 0.1, verbose = True):
        '''
        main function to train. can specify number of epochs, batch size, 
        loss function (only those available in the Losses class), learning rate,
        random scaling for data augmentation and verbosity.
        '''
        
        # augmenting data
        X = np.concatenate((X, augment_data(X, rand_scale)))
        Y = np.concatenate((Y, Y))
        
        self.initialize_weights()
        
        num_batches = int(np.ceil(X.shape[0]/batch_size))
        
        for epoch in range(n_epochs):
            
            self.forward_prop(X)
            if verbose:
                print("Loss after ", epoch, " epoch(s): ", self.calc_loss(Y, loss_func),\
                      ", Accuracy: ", self.accuracy(X, Y))
            
            # can be SGD, mini-batch, or full-batch depending on the batch_size parameter
            for i in range(num_batches):
                
                # slice the data according to batch size
                x = X[i * batch_size: (i + 1) * batch_size, :]
                y = Y[i * batch_size: (i + 1) * batch_size, :]
                
                # update parameters after each batch
                self.forward_prop(x)
                self.back_prop(x, y, loss_func)
                self.update_weights(x, l_rate)
        
        # print final loss and accuracy. accuracy is the only available metric
        self.forward_prop(X)
        print("Final loss: ", self.calc_loss(Y, loss_func), "Final accuracy: ", self.accuracy(X, Y))
        
        return self

def gen_train_data(n = 40):
    '''
    generates training data with n number of samples
    '''
    X1_1 = 2 + 4 * np.random.rand(n, 1)
    X1_2 = 1 + 4 * np.random.rand(n, 1)
    class1 = np.concatenate((X1_1, X1_2), axis=1)
    Y1 = np.ones((n, 1))

    X0_1 = 4 + 4 * np.random.rand(n, 1)
    X0_2 = 3 + 4 * np.random.rand(n, 1)
    class0 = np.concatenate((X0_1, X0_2), axis=1)
    Y0 = np.zeros((n, 1))

    X = np.concatenate((class1, class0))
    Y = np.concatenate((Y1, Y0))

    return X, Y

def augment_data(data, rand_scale = 0.1):
    '''
    adds noise to data depending on the scaling factor
    '''
    return data + rand_scale * np.random.randn(*data.shape)


if __name__ == "__main__":

    X, Y = gen_train_data()
    
    # structure of the neural network model
    structure = [{'type': 'input', 'units': 2},
            {'type': 'dense', 'units': 12, 'activation_function': 'leaky_relu', 'bias': True},
            {'type': 'dense', 'units': 12, 'activation_function': 'leaky_relu', 'bias': True},
            {'type': 'dense', 'units': 1, 'activation_function': 'logistic', 'bias': True}]
    
    # build and train model
    nn = NeuralNetwork(structure)
    nn.train(X, Y, n_epochs=1000, batch_size=8, rand_scale = 0.1, l_rate = 0.001, loss_func = "binary_cross_entropy")

    # show decision boundary on a plot
    # creating a mesh of values
    xx, yy = np.meshgrid(np.arange(0, 9, 0.1), np.arange(0, 9, 0.1))
    X_vis = np.c_[xx.ravel(), yy.ravel()]

    # predicting outcomes on the mesh
    h = nn.predict(X_vis)
    h = np.array(h) >= 0.5
    h = np.reshape(h, (len(xx), len(yy)))

    # indices for points
    idx0 = [i for i, v in enumerate(Y) if v == 0]
    idx1 = [i for i, v in enumerate(Y) if v == 1]

    # plot points and contours with colours
    plt.contourf(xx, yy, h, cmap='jet')
    plt.scatter(X[idx1, 0], X[idx1, 1], marker='^', c="red", edgecolors="white", label="class 1")
    plt.scatter(X[idx0, 0], X[idx0, 1], marker='o', c="blue", edgecolors="white", label="class 0")

    plt.show()