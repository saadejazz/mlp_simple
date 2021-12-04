# Neural Network Task 2

## 1.1. Leaky Relu  

Leaky Relu has been added to the ```Activations``` class. All methods of the ```Activations``` class are static.  

```python
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
```

## 2.2. Data Augmentation  

A function to add random noise to the data is created. This function takes a ```rand_scale``` as argument to scale the noise. The data generated after noise is added to the original data. The function is as follows:  

```python
def augment_data(data, rand_scale = 0.1):
    '''
    adds noise to data depending on the scaling factor
    '''
    return data + rand_scale * np.random.randn(*data.shape)
```  

## 3.3. Mini-batch gradient descent  

The code is restructured to a vectorized implementation, and updates are made in batches. The new train function is as follows:  

```python
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
```

## 2. Classification Test  

Generated data is augmented and trained on the structure defined in the task. The result and decision boundaries are then plotted. The main code is as follows:  

```python
X, Y = gen_train_data()
    
# structure of the neural network model
structure = [{'type': 'input', 'units': 2},
        {'type': 'dense', 'units': 12, 'activation_function': 'leaky_relu', 'bias': True},
        {'type': 'dense', 'units': 12, 'activation_function': 'leaky_relu', 'bias': True},
        {'type': 'dense', 'units': 1, 'activation_function': 'logistic', 'bias': True}]

# build and train model
nn = NeuralNetwork(structure)
nn.train(X, Y, n_epochs=1000, batch_size=4, loss_func = "binary_cross_entropy")

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
```



