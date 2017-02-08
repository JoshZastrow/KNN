import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops)
    Inputs:
    - W: C x D array of weights
    - X: D x M array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0

    for i in range(num_train):

        scores = W.dot(X[:, i])
        # example: y of example 74 is 5(cat) --> y[74] = 5
        # we are getting from the scores list (score of each class)
        # just the score of class = 5 (cat)
        correct_class_score = scores[y[i]]

        for j in range(num_classes):

            # when j = 5 (which is the class label for example 74)
            # we skip this loop
            if j == y[i]:
                continue

            # example: scores[5] - scores[y[74]] + 1
            margin = scores[j] - correct_class_score + 1  # note delta = 1

            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
 
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero    
    '''
    Trains Weights on an SVM Linear Classifier
        Args:
            W: [k, N] matrix - classes by features matrix
            X: [N, M] matrix - features by examples matrix
            Y: [M, 1] matrix - example(k class) by 1 matrix

        Returns:
            loss: loss score (how many wrong-classes had better score)
            dW: gradient of weights
    '''

    # Initialize gradients and get sizes:
    dW = np.zeros_like(W)
    examples = X.shape[1]

    delta = 1  # how different the scores need to be

    # Scores for each class, each example
    hypothesis = np.dot(W, X)  # [K, M] = 10 x 45000

    # Correct score for each example
    correct_scores = hypothesis[y, np.arange(examples)]  # 1 X 45000

    # Score difference between wrong and right classes
    diff = hypothesis - correct_scores + delta  # 10 X 45000

    # Cost function: wrong classes that scored higher than correct class
    cost = np.maximum(np.zeros_like(diff), diff)

    # loss function
    loss = 1 / examples * np.sum(cost) + .5 * reg * np.sum(W**2)

    # Gradient Calculations
    #################################################################

    grad = cost

    # Gradient Calculation--convert wrong scores to 1's
    grad[diff > 0] = 1  # 10 x 45000 (classes x examples)

    # Total each examples tallied score
    diff_total = np.sum(grad, axis=0)  # 1 X 45000

    # Multiply the instances of a wrong score with the total
    grad[y, np.arange(examples)] = - diff_total[np.arange(examples)]

    dW = 1 / examples * np.dot(grad, X.T) + reg * W

    return loss, dW
