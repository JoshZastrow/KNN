import numpy as np


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized
    inputs:
    W: K classes x D dimensions of weights (ex: 10 x 3073)
    X: D dimensions x N examples. Data is in columns
    y: 1D array of labels 0....K-1 for K classes

    Output:
    A tuple consistion of:
    loss: loss/cost value of this score
    grad: gradient with respect to the weights, size: K x D
    """

    # Initialize weights, loss, get dataset size
    dW = np.zeros_like(W)
    loss = 0.0
    N = y.shape

    # Perform Softmax
    #################################################################

    # Generate a linear classifier scores
    scores = W.dot(X)

    # Normalize the exponents -- highest value should be zero
    # This is to preserve numerical stablity
    C = -np.amax(scores)

    # Softmax on the scores:
    scores = C * np.exp(scores)

    # Sum the scores for the loss function
    sum_scores = np.sum(scores, axis=0)

    # Get the correct class scores
    y_predictions = scores[y, np.arange(N)]

    # Loss Function
    loss = (y_predictions / sum_scores)  # score / total score
    loss = -np.log(loss)
    loss = np.sum(loss)
    loss += reg * np.sum(scores**2)

    # Gradient Descent
    probablilities = - scores / sum_scores  # K X N
    probablilities[y, np.arange(N)] += 1

    dW = np.dot(probablilities, X.T)  # K X N * N X D = K X D
    dW /= - X.shape[1]
    dW = reg * W

    return loss, dW


def numerical_gradient(f, x):
    """
    Computes gradient numerically
    inputs:
        {f}     : function
        {x}     : single parameter for function
    output:
        {grad}  : gradient
    """

    fx = f(x)
    h = .0001

    grad = np.zeros_like(x)

    # iterate over all elements of x
    it = np.nditer(x, flags=['multi-index'], op_flags=['read-write'])

    while not it.finished:

        ix = it.multi_index  # get index of current iteration
        x_old = x[ix]  # save current value
        x[ix] = x_old + h  # enter the new value
        fxh = f(x)  # rerun the function with the new value in place
        x[ix] = x_old  # reset the x value back to beginning

        grad[ix] = (fxh - fx) / h
        it.iternext

    return grad

# Weights
M = 15
D = 6
K = 3

# Weights
W = np.random.random_integers(1, 5, size=(K, D))
X = np.random.random_integers(1, 6, size=(D, M))
y = np.random.random_integers(0, K - 1, size=(M))
r = 5

softmax_loss_vectorized(W, X, y, r)
