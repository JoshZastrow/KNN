import numpy as np

class KNearestNeighbor:
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Input:
        X - A num_train x dimension array where each row is a training point.
        y - A vector of length num_train, where y[i] is the label for X[i, :]
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Input:
        X - A num_test x dimension array where each row is a test point.
        k - The number of nearest neighbors that vote for predicted label
        num_loops - Determines which method to use to compute distances
                    between training points and test points.

        Output:
        y - A vector of length num_test, where y[i] is the predicted label for the
            test point X[i, :].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the 
        test data.

        Input:
        X - An num_test x dimension array where each row is a test point.

        Output:
        dists - A num_test x num_train array where dists[i, j] is the distance
                between the ith test point and the jth training point.
        """
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        

        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]               #
                #####################################################################
                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j])**2, axis=1))

        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        print('\n\n'
              'Matrix Shapes: \n'
              '\tWhere Y=Test examples, M=Train examples, N=Features\n'
              '\t\t Test Batch (Y, N): {} \n'
              '\t\tTrain Batch (M, N): {}\n'
              '\t\tDists Batch (Y, M): {}\n'.format(X.shape, 
                                                    self.X_train.shape,
                                                    dists.shape))    
        for m in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            #######################################################################
            # dists[i] = np.dot(X[i],self.X_train.T)
            Diffs = np.sum(np.abs(self.X_train - X[m, :]), axis=1)  
            dists[m, :] = Diffs.T
            
            if m % 1000 == 0:
                print('Example ', m)
                print('Distance matrix shape: ', dists.shape)
                print('Training matrix shape: ', self.X_train.shape)
                print('Diffrnce matrix shape: ', Diffs.T.shape,'\n\n')
            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        
        #########################################################################
        # TODO:                                                                  #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # L2 distance: L2(t1, t2) =  SQRT ( SUM : (t1 - t2) ^ 2 )
        # where: (t1 -t2)^2 = (t1^2 - t1 * t2 + t2 ^ 2) 
        
        t1 = X        
        t2 = self.X_train

        
        t1_sqrd = np.sum(t1**2, axis=1).reshape(-1, 1)  # creates a m' x 1
        t2_sqrd = np.sum(t2**2, axis=1)  # creates a m x 1
        t1_t2 = np.dot(t1, t2.T) * 2.0
        
        # Produces a m_test x m_train matrix
        dists = np.sqrt(t1_sqrd - t1_t2 + t2_sqrd)

        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Input:
        dists - A num_test x num_train array where dists[i, j] gives the distance
                between the ith test point and the jth training point.

        Output:
        y - A vector of length num_test where y[i] is the predicted label for the
            ith test point.
        """
        num_test = dists.shape[0]

        # y_pred is an Y X 1 array, with the predicition for each test
        y_pred = np.zeros(num_test)

        for i in range(num_test):

            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # training point, and use self.y_train to find the labels of these      #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################

            # list of most similar examples
            closest_examples = np.argsort(dists[i])[:k]
            
            # look up the labels for the k most similar examples
            closest_label = self.y_train[closest_examples]
            
            # bin together the frequencies of the same example
            # return the most frequent guess
            y_pred[i] = np.argmax(np.bincount(closest_label))
 
        return y_pred

