import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs:
    - W: C x D array of weights (ex: 10 x 3073 features)
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    for i in range(X.shape[1]):  # for each example
        # create the linear hypthothesis for each class
        scores = W.dot(X[:, i])
        scores -= np.max(scores)  # prevents numerical instability
        prob = 0.0  # initialize a probability for scores
        loss -= scores[y[i]]  # subtract the correct score from the loss

        for curr_score in scores:  # for each class
            # sum up the scores to get a total probability
            prob += np.exp(curr_score)

        for j in range(W.shape[0]):  # for each class
            # get the probability of this class score being correct
            prob_ji = np.exp(scores[j]) / prob
            # gradient descent step, probability multiplied by Xi
            margin = - prob_ji * X[:, i].T

            if j == y[i]:  # if this class is the correct one
                # penalize for any difference in probability not 100%
                margin = (1 - prob_ji) * X[:, i].T
                # place the gradient in the gradient matrix
                dW[j, :] += -margin

        loss += np.log(prob)

    loss /= X.shape[1]
    dW /= X.shape[1]

    # Regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


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
    # Parameters
    N = X.shape[1]  # Training Examples

    # Generate a linear classifier scores
    scores = W.dot(X)

    # Normalize, so max score is 0 per class
    scores -= np.amax(scores, axis=0, keepdims=True).reshape((1, N))

    # Sum the scores for the loss function, using softmax function
    prob = np.exp(scores) / np.exp(scores).sum(axis=0, keepdims=True)

    # Get the probabilities for the correct class label
    prob_y = prob[y, np.arange(N)]

    # Cross Entropy Loss Function
    loss = np.mean(-np.log(prob_y)) + 0.5 * reg * np.sum(W**2)

    # print('Loss value: ', loss)
    prob[y, np.arange(N)] += 1

    # Gradient
    ind = np.zeros_like(prob)
    ind[y, np.arange(N)] = 1
    
    dW = np.dot((prob - ind), X.T)  # K X N * N X D = K X D
    dW /= -N  # divide by N examples
    dW += reg * W

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


def test_softmax():

    # Parameters
    N = 1  # Training Example size (of 1)
    D = 30  # Dimensions/Features
    K = 5  # Classes

    # Weights
    W = np.random.random(size=(K, D))
    X = np.random.random_integers(1, 255, size=(D, N))
    y = np.random.random_integers(0, K - 1, size=(N))
    r = 5

    softmax_loss_vectorized(W, X, y, r)

# test_softmax()
