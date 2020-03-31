from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    for i in range(num_train):
        scores = X[i].dot(W)
        probs = np.exp(scores) / np.sum(np.exp(scores))
        correct_class_prob = probs[y[i]]
        loss -= np.log(correct_class_prob)
        for j in range(num_classes):
            if j == y[i]:
                dW[:,j] += X[i].T * (-1 + probs[j])
            else:
                dW[:,j] += X[i].T * probs[j]
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros(W.shape)
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #############################################################################
    # Implement a vectorized version of the structured softmax loss, storing    #
    # the result in loss.                                                       #
    #############################################################################
    scores = np.dot(X, W)
    probs = np.exp(scores) / np.sum(np.exp(scores), axis = 1).reshape(-1,1)
    correct_class_probs = probs[np.arange(probs.shape[0]),y]
    loss += np.mean(-np.log(correct_class_probs))
    loss += reg * np.sum(W * W)
    
    #############################################################################
    # Implement a vectorized version of the gradient for the structured softmax #
    # loss, storing the result in dW.                                           #
    #############################################################################
    masks = probs
    masks[np.arange(masks.shape[0]),y] += -1
    dW = np.dot(X.T, masks)
    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
