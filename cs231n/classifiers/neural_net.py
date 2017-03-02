import numpy as np
import matplotlib.pyplot as plt
import math

class Unit(object):
    """
    A unit (variable) in a neural network. Each unit would carry a forward
    pass value and a gradient value.
    """

    def __init__(self, value=0.0, grad=0.0):
        self.value = value
        self.grad = grad

# Create some gates to be used in the net (implementation put on hold...)


class add_gate(object):

    def __init__(self):
        self.x = Unit()
        self.y = Unit()

    def forward(self, x, y):
        self.x = x
        self.y = y
        self.out = Unit(x.value + y.value, 0.0)

    def backward(self):
        self.x.grad += 1 * self.out.grad
        self.y.grad += 1 * self.out.grad


class mul_gate(object):

    def __init__(self):
        self.x = Unit()
        self.y = Unit()

    def forward(self, x, y):
        self.x = x
        self.y = y
        self.out = Unit(x.value * y.value, 0.0)

    def backward(self):
        self.x.grad += self.y.value * self.out.grad
        self.y.grad += self.x.value * self.out.grad


class ReLu_gate(object):

    def __init__(self):
        self.x = Unit()

    def forward(self, x):
        self.x = x
        self.out = Unit(max(0, self.x), 0.0)

    def backward(self):
        pass  # to be continued...


class TwoLayerNet(object):

    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The Fxs of the second fully-connected layer are the Fx for each class.
    """

    def __init__(self, input_size, hidden_size, Fx_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - Fx_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, Fx_size)
        self.params['b2'] = np.zeros(Fx_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return Fx, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix Fx of shape (N, C) where Fx[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """

        # ReLU function:
        def ReLU(x):
            return np.maximum(x, 0)

        # SoftMax Function
        def softMax(z):
            # preserve numerical stability
            z -= np.amax(z, axis=1, keepdims=True)
            return (np.exp(z) / np.exp(z).sum(axis=1, keepdims=True))

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        Fx = None
        #######################################################################
        # TODO: Perform the forward pass, computing the class Fx for the input. #
        # Store the result in the Fx variable, which should be an array of      #
        # shape (N, C).                                                        #
        #######################################################################

        # f(X, W) = W2 * max(0, W1 * X + b1) + b2
        z1 = np.dot(X, W1)   # W1 * X
        z1 = z1 + b1         # W1 * X + b1
        a1 = ReLU(z1)        # max(0, W1 * X + b1)
        z2 = np.dot(a1, W2)  # W2 * max(0, W1 * X + b1)
        a2 = z2 + b2     # W2 * max(0, W1 * X + b1) + b2

        # output
        Fx = a2
        #######################################################################
        #                              END OF YOUR CODE                         #
        #######################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return Fx

        # Compute the loss
        loss = None
        #######################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include #
        # both the data loss and L2 regularization for W1 and W2. Store the result #
        # in the variable loss, which should be a scalar. Use the Softmax          #
        # classifier loss. So that your results match ours, multiply the           #
        # regularization loss by 0.5                                               #
        #######################################################################

        # Convert to normalized probabilities
        Fx = softMax(Fx)
        # Keep only estimated probabilities for correct class
        margin = Fx[np.arange(N), y]

        # Cross Entropy Loss - log probabilities
        data_loss = -np.mean(np.log(margin))
        wght_loss = 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))
        loss = data_loss + wght_loss

        #######################################################################
        #                              END OF YOUR CODE                       #
        #######################################################################

        # Backward pass: compute gradients
        grads = {}
        #######################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #######################################################################

        # Correct Score matrix
        Y = np.zeros_like(Fx)
        Y[np.arange(N), y] = 1

        # Back prop through Fx
        dFx = (Fx.copy() - Y) / N  # M x C

        #####################################################################
        # X1 [M x D]    W1 [D x H]    b1 [M x 1]
        # a1 [M x H]    W2 [H x C]    b2 [H x 1]
        # a2 [M x C]    Fx [M x C]
        # da2 [M x C]   dFx [M x C]
        # da1 [M x H]   dW2 [H x C]   db2 [H x 1]
        # dX1 [M x D]   dW1 [D x H]   db1 [M x 1]
        #####################################################################

        #
        dz2 = dFx  # M x C
        da1 = np.dot(dz2, W2.T)  # M x C * C * H --> M x H
        dW2 = np.dot(a1.T, dz2) + reg * W2  # H x M * M x C
        db2 = np.sum(dz2, axis=0)  # C x M * M x 1

        dz1 = da1 * (a1 > 0)  # Backprop Relu
        dW1 = np.dot(X.T, dz1) + reg * W1  # D x H
        db1 = np.sum(dz1, axis=0)  # M * M x H

        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1

        #######################################################################
        #                              END OF YOUR COD                        #
        #######################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        val_acc = 0

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ###################################################################
            # TODO: Create a random minibatch of training data and labels,
            # storing them in X_batch and y_batch respectively.
            ###################################################################

            batch_indices = np.random.choice(num_train,
                                             batch_size,
                                             replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            ###################################################################
            #                             END OF YOUR CODE                    #
            ###################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ###################################################################
            # TODO: Use the gradients in the grads dictionary to update the
            # parameters of the network (stored in the dictionary self.params)
            # using stochastic gradient descent. You'll need to use the grads
            # stored in the grads dictionary defined above.
            ###################################################################
            for key in self.params:
                self.params[key] -= learning_rate * grads[key]
            ###################################################################
            #                             END OF YOUR CODE                    #
            ###################################################################

            if verbose and it % 100 == 0:
                print(
                    'iteration {:d} / {:d}: loss {:05.2f}'
                    .format(it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning
            # rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict Fx for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        # ReLU function:
        def ReLU(x):
            return np.maximum(x, 0)

        # SoftMax Function
        def softMax(z):
            # preserve numerical stability
            z -= np.amax(z, axis=1, keepdims=True)
            return (np.exp(z) / np.exp(z).sum(axis=1, keepdims=True))

        #######################################################################
        # TODO: Implement this function; it should be VERY simple!            #
        #######################################################################
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        z1 = np.dot(X, W1) + b1  # M x H
        a1 = ReLU(z1)
        L2 = np.dot(a1, W2) + b2  # M x C
        Fx = softMax(L2)

        y_pred = np.argmax(Fx, axis=1)
        #######################################################################
        #                              END OF YOUR CODE                       #
        #######################################################################

        return y_pred
