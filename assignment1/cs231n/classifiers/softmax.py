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
  loss = []
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  dW_i = np.zeros_like(W)  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    f = X[i].dot(W)
    f -= np.max(f)
    loss_without_log = 0
    for j in range(len(f)):
        loss_without_log += np.exp(f[j])
    for j in range(len(f)):
        if j==y[i]:
            dW_i[:,j] = (np.exp(f[j])/loss_without_log-1)*X[i]
        else:
            dW_i[:,j] = X[i]*np.exp(f[j])/loss_without_log
    loss.append(-np.log(np.exp(f[y[i]])/loss_without_log))
    dW += dW_i
  loss = np.mean(loss) + reg*np.sum(W*W)
  dW = dW/num_train + reg*2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros_like(W)
  f = X.dot(W)
  f = (f.T - np.max(f, axis=1)).T
  cor_clas_scores = f[np.arange(num_train), y]
  f_exp = np.exp(f)
  cor_clas_scores_exp = np.exp(cor_clas_scores)
  loss = np.mean(-np.log(cor_clas_scores_exp/np.sum(f_exp, axis = 1)))
  divided_by_sum_scores = (f_exp.T/np.sum(f_exp, axis = 1)).T
  divided_by_sum_scores[np.arange(num_train), y] = divided_by_sum_scores[np.arange(num_train), y]-1

  dW = np.dot(X.T, divided_by_sum_scores)/num_train + reg*2*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

