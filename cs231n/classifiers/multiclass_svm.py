# Import numpy and random-> shuffle
import numpy as np
from random import shuffle


###################################################################
# TODO (2/6/17):                                                  #
#   - make the loss delta tune-able                               #
#   - make the hinge loss equation tune-able (quadratic or linear)#
###################################################################

# define the naive svm loss function,
# takes in the weights, training data, training labels, lambda value
def linear_svm_naive(W, X, y, reg):
    '''Trains Weights on an SVM Linear Classifier
        Args:
            W: [k, N] matrix - classes by features matrix
            X: [N, M] matrix - features by examples matrix
            Y: [M, 1] matrix - example(k class) by 1 matrix

        Returns:
            loss: loss score (how many wrong-classes had better score)
            dW: gradient of weights
    '''
    # Create an empty gradient matrix (shape of weight matrix
    dW = np.zeros_like(W)

    # Variables for # classes, # examples (k and M), loss to 0
    k_classes = W.shape[0]
    m_examples = X.shape[1]
    loss = 0.0
    learning_rate = 0.5
    # loop through each training example
    for m in range(m_examples):

        # get the class scores--> linear classifier W0 + W1*X1 + W2*X2 ....
        score = np.dot(W, X[:, m])  # yields a [k, 1] matrix of class scores

        # get the correct label score (score of the class it's supposed to be)
        class_score = score[y[m]]

        # loop through each class
        for k in range(k_classes):

            # skip the loop if the class is the same as label (zero margin)
            if k == y[m]:
                continue

            # calculate difference between the score of that class and the
            # score of the correct class ( + 1 for the delta value)
            margin = score[k] - class_score + 1

            # add the margin to the loss total if the difference is greater
            # than zero (as in the linear classifier was more confident -- gave
            # a better score to this class than to the actual class label)
            if margin > 0:
                loss += margin

                # Calculate the gradient of this example's class
                # should be # incorrect classes * x but this is a loop
                dW[k, :] += X[:, m].T

                # Remove the gradient from the correct class
                dW[y[m], :] -= X[:, m].T

    # divide the total loss by the number of training examples (average loss)
    loss = loss / m_examples
    dW = dW / m_examples
    # Add regularization term (1/2 * lambda * Sum of Weights ^ 2)
    loss += 0.5 * reg * np.sum(W**2)

    # return the loss value and the gradient as a tuple
    return loss, dW


def linear_svm_vectorized(W, X, y, reg):
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
