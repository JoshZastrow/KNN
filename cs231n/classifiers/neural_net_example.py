import numpy as np
import matplotlib.pyplot as plt
import math, time

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.
  In other words, the network has the following architecture:
  input - fully connected layer - ReLU - fully connected layer - softmax
  The outputs of the second fully-connected layer are the Fx for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
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
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

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
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    Fx = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class Fx for the input. #
    # Store the result in the Fx variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # Values of the hidden layer

    L1 = np.maximum( X.dot(W1) + b1, 0 )
    Fx = L1.dot(W2) + b2

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    if y is None:
      return Fx

    # Creating a output matrix from output vector
    # Fx is the output
    Y = np.zeros(Fx.shape)
    for i,row in enumerate(Y):
        row[y[i]] = 1

    # Compute the loss
    loss = 0.0
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################

    #Exponentiating the Fx , then normalizing the exponentiated value
    exp_Fx = pow(math.e, Fx)
    Fx = (exp_Fx.T / np.sum(exp_Fx, axis = 1)).T

    for i in range(N):
      loss -= np.log( Fx[i][y[i]] )
    loss /= N
    loss += 0.5*reg*np.sum(W1 * W1)
    loss += 0.5*reg*np.sum(W2 * W2)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    # compute the gradient on scores
    dFx = probs
    dFx[range(N),y] -= 1
    dFx /= N

    # W2 and b2
    grads['W2'] = np.dot(a1.T, dFx)
    grads['b2'] = np.sum(dFx, axis=0)

    # next backprop into hidden layer
    dhidden = np.dot(dFx, W2.T)

    # backprop the ReLU non-linearity
    dhidden[a1 <= 0] = 0

    # finally into W,b
    grads['W1'] = np.dot(X.T, dhidden)
    grads['b1'] = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False ,dropout_fraction = 0):
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

    # Dropout on input
    Binom_variables = np.random.choice(
        [0, 1],
        size=X.shape,
        p=[dropout_fraction, 1 - dropout_fraction])

    X = (X * Binom_variables) / (1 - dropout_fraction)

    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    val_acc = 0

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):


        #######################################################################
        # TODO: Create a random minibatch of training data and labels, storing
        # them in X_batch and y_batch respectively.
        #######################################################################
        batch_indicies = np.random.choice(num_train,
                                          batch_size,
                                          replace=False)
        X_batch = X[batch_indicies]
        y_batch = y[batch_indicies]
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

        # Compute loss and gradients using the current minibatch
        loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
        loss_history.append(loss)

        #######################################################################
        # TODO: Use the gradients in the grads dictionary to update the       #
        # parameters of the network (stored in the dictionary self.params)    #
        # using stochastic gradient descent. You'll need to use the gradients #
        # stored in the grads dictionary defined above.                       #
        #######################################################################
      for variable in self.params:
          self.params[variable] -= learning_rate*grads[variable]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        val_acc = (self.predict(X_val) == y_val).mean()
        print 'iteration %d / %d: loss %f, val_acc %f' % (it, num_iters, loss, val_acc)

      # Every epoch, check train and val accuracy and decay learning rate.
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

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    L1 = np.maximum( X.dot(self.params['W1']) + self.params['b1'], 0 )
    Fx = L1.dot(self.params['W2']) + self.params['b2']
    exp_Fx = pow(math.e, Fx)
    Fx   = (exp_Fx.T/np.sum(exp_Fx, axis = 1)).T

    y_pred = np.argmax(Fx, axis = 1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################
    return y_pred

