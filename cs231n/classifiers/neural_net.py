# mine

import numpy as np
import matplotlib.pyplot as plt
import math

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
        # Initialization for ReLU Neurons
        sqrt_layer1 = math.sqrt(2 / input_size)  # as compared to STD
        sqrt_layer2 = math.sqrt(2 / hidden_size)
        self.params['W1'] = np.random.randn(input_size, hidden_size)
        self.params['W1'] *= sqrt_layer1
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = np.random.randn(hidden_size, Fx_size)
        self.params['W2'] *= sqrt_layer2
        self.params['b2'] = np.zeros(Fx_size)

    def loss(self, X, y=None, reg=0.01, p=1.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i]
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

        # Forward Pass with dropout
        # f(X, W) = W2 * max(0, W1 * X + b1) + b2
        z1 = np.dot(X, W1) + b1
        drop1 = np.random.choice([1, 0], size=z1.shape, p=[p, 1 - p]) / p
        a1 = ReLU(z1) * drop1

        z2 = np.dot(a1, W2) + b2
        drop2 = np.random.choice([1, 0], size=z2.shape, p=[p, 1 - p]) / p
        a2 = z2 * drop2

        # output
        Fx = a2

        # If the targets are not given then jump out, we're done
        if y is None:
            return Fx

        # Compute the loss
        loss = None

        # Convert to normalized probabilities
        Fx = softMax(Fx)

        # Keep only estimated probabilities for correct class
        margin = Fx[np.arange(N), y]

        # Cross Entropy Loss - log probabilities
        data_loss = -np.mean(np.log(margin))
        wght_loss = 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))
        loss = data_loss + wght_loss

        # Backward pass: compute gradients
        grads = {}

        # Correct Score matrix
        Y = np.zeros_like(Fx)
        Y[np.arange(N), y] = 1

        #####################################################################
        # X1 [M x D]    W1 [D x H]    b1 [M x 1]
        # a1 [M x H]    W2 [H x C]    b2 [H x 1]
        # a2 [M x C]    Fx [M x C]
        # da2 [M x C]   dFx [M x C]
        # da1 [M x H]   dW2 [H x C]   db2 [H x 1]
        # dX1 [M x D]   dW1 [D x H]   db1 [M x 1]
        #####################################################################
        # Backproogation
        #####################################################################

        dFx = (Fx.copy() - Y) / N # M x C
        da2 = 1 * dFx

        dz2 = drop2 * da2  # M x C
        dW2 = np.dot(a1.T, dz2) # H x M * M x C
        db2 = np.sum(dz2, axis=0)  # C x M * M x 1

        da1 = np.dot(dz2, W2.T)  # M x C * C * H --> M x H
        dz1 = da1 * (a1 > 0) * drop1  # Backprop Relu
        dW1 = np.dot(X.T, dz1)   # D x H
        db1 = np.sum(dz1, axis=0)  # M * M x H

        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1

        # Add regularization terms

        grads['W2'] += reg * W2
        grads['W1'] += reg * W1

        #######################################################################
        #                              END OF YOUR COD                        #
        #######################################################################

        return loss, grads, data_loss, wght_loss

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False, dropout_val=0.5):
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
        W1_ratio = []
        W2_ratio = []

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
            loss, grads, dloss, wloss = self.loss(X_batch,
                                                  y=y_batch,
                                                  reg=reg, p=dropout_val)
            loss_history.append(loss)

            # Track Update vs Weights ratio
            # Calculate parameters
            W1_vals = np.linalg.norm(self.params['W1'].ravel())
            W2_vals = np.linalg.norm(self.params['W2'].ravel())
            W1_grad = np.linalg.norm(learning_rate * grads['W1'].ravel())
            W2_grad = np.linalg.norm(learning_rate * grads['W2'].ravel())

            # Calculate Ratio
            W1_new_ratio = W1_grad / W1_vals
            W2_new_ratio = W2_grad / W2_vals

            # Add ratio to list
            W1_ratio.append(W1_new_ratio)
            W2_ratio.append(W2_new_ratio)

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
                    'iteration {:d} / {:d}: loss {:05.2f} ({:4.2f}|{:4.2f})'
                    .format(it, num_iters, loss, dloss, wloss))

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
            'W1_ratio_history': W1_ratio[31:],
            'W2_ratio_history': W2_ratio[31:]
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
        z2 = np.dot(a1, W2) + b2  # M x C
        a2 = z2
        Fx = softMax(a2)

        y_pred = np.argmax(Fx, axis=1)
        #######################################################################
        #                              END OF YOUR CODE                       #
        #######################################################################

        return y_pred
