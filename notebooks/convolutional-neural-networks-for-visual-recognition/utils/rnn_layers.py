from __future__ import print_function, division
from builtins import range
import numpy as np
from utils.activation_layers import sigmoid
from utils.activation_layers import softmax


"""
This file defines layer types that are commonly used for recurrent neural networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    next_h = np.tanh(np.dot(prev_h, Wh) + np.dot(x, Wx) + b)
    cache = x, prev_h, Wx, Wh, b, next_h
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, prev_h, Wx, Wh, b, next_h = cache
    # (N, H) = (N, H) * (N, H)
    dz = dnext_h * (1 - next_h**2)
    # (N, D) = (N, H) * (H, D)
    dx = np.dot(dz, Wx.T)
    # (N, H) = (N, H) * (H, H)
    dprev_h = np.dot(dz, Wh.T)
    # (D, H) = (D, N) * (N, H)
    dWx = np.dot(x.T, dz)
    # (H, H) = (H, N) * (N, H)
    dWh = np.dot(prev_h.T, dz)
    # (H,)
    db = np.sum(dz, axis = 0)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, []
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, T, _ = x.shape
    _, H = h0.shape
    
    next_h = h0
    h = np.zeros((N, T, H))
    
    for t in np.arange(T):
        next_h, step_cache = rnn_step_forward(x[:, t, :], next_h, Wx, Wh, b)
        h[:, t, :] = next_h
        cache.append(step_cache)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, T, H = dh.shape
    x_T, _, _, _, _, _ = cache[T-1]
    _, D = x_T.shape
    
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros(H)
    dprev_h_step = np.zeros((N, H))
    
    for t in np.arange(T)[::-1]:
        dx_step, dprev_h_step, dWx_step, dWh_step, db_step = rnn_step_backward(dh[:,t,:] + dprev_h_step, cache[t])
        dh0 = dprev_h_step
        dx[:,t,:] = dx_step
        dWx += dWx_step
        dWh += dWh_step
        db += db_step
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    out = W[x, :]
    cache = x, W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Get the size
    _, H = prev_h.shape
    # compute the intermediate vector
    z = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    # compute the gates
    zf = z[:, :H]
    zi = z[:, H:2*H]
    zg = z[:, 2*H:3*H]
    zo = z[:, 3*H:4*H]
    # compute the gates: f -> forget gate, i -> update gate, g -> cell gate, o -> output gate
    f = sigmoid(zf)
    i = sigmoid(zi)
    g = np.tanh(zg)
    o = sigmoid(zo)
    # compute the next cell state
    next_c = f * prev_c + i * g
    # compute next hidden state
    next_h = o * np.tanh(next_c)
    # Get the cache for the backward pass
    cache = x, prev_h, prev_c, Wx, Wh, b, zf, zi, zg, zo, f, i, g, o, next_c, next_h

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Backward pass
    # unroll cache
    x, prev_h, prev_c, Wx, Wh, b, zf, zi, zg, zo, f, i, g, o, next_c, next_h = cache
    # backprop
    do = np.tanh(next_c) * dnext_h
    dnext_c += o * (1 - np.tanh(next_c)**2) * dnext_h
    # backprop
    df = prev_c * dnext_c
    dprev_c = f * dnext_c
    di = g * dnext_c
    dg = i * dnext_c
    # backprop
    dzo = sigmoid(zo) * (1 - sigmoid(zo)) * do
    dzg = (1 - np.tanh(zg)**2) * dg
    dzi = sigmoid(zi) * (1 - sigmoid(zi)) * di
    dzf = sigmoid(zf) * (1 - sigmoid(zf)) * df
    # backprop
    dz = np.hstack((dzf, dzi, dzg, dzo))
    # backprop
    dx = np.dot(dz, Wx.T)
    dWx = np.dot(x.T, dz)
    dprev_h = np.dot(dz, Wh.T)
    dWh = np.dot(prev_h.T, dz)
    db = np.sum(dz, axis=0)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial hidden state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, []
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # define sizes
    N, T, _ = x.shape
    _, H = h0.shape
    # initialization
    next_h = h0
    next_c = np.zeros_like(next_h)
    h = np.zeros((N, T, H))
    # run the loop on times
    for t in np.arange(T):
        next_h, next_c, step_cache = lstm_step_forward(x[:,t,:], next_h, next_c, Wx, Wh, b)
        h[:,t,:] = next_h
        cache.append(step_cache)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Backprop into the lstm.
    # Dimensions
    N, T, H = dh.shape
    x_T, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = cache[T-1]
    _, D = x_T.shape
    # Initialize dx,dh0,dWx,dWh,db
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H,))
    # On transpose dh
    dprev_h_step = np.zeros((N, H))
    dprev_c_step = np.zeros_like(dprev_h_step)
    
    for t in np.arange(T)[::-1]:
        dx_step, dprev_h_step, dprev_c_step, dWx_step, dWh_step, db_step = lstm_step_backward(dh[:,t,:] + dprev_h_step, dprev_c_step, cache[t])
        dh0 = dprev_h_step
        dx[:,t,:] = dx_step
        dWx += dWx_step
        dWh += dWh_step
        db += db_step
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(h, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of H dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a output vector of
    dimension V.

    Inputs:
    - h: Input data of shape (N, T, H)
    - w: Weights of shape (H, V)
    - b: Biases of shape (V,)

    Returns a tuple of:
    - out: Output data of shape (N, T, V)
    - cache: Values needed for the backward pass
    """
    N, T, H = h.shape
    _, V = w.shape
    out = np.dot(h.reshape(N * T, H), w).reshape(N, T, V) + b
    cache = h, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, V)
    - cache: Values from forward pass

    Returns a tuple of:
    - dh: Gradient of input, of shape (N, T, H)
    - dw: Gradient of weights, of shape (H, V)
    - db: Gradient of biases, of shape (V,)
    """
    h, w, b, out = cache
    N, T, H = h.shape
    _, V = w.shape

    dh = np.dot(dout.reshape(N * T, V), w.T).reshape(N, T, H)
    dw = np.dot(dout.reshape(N * T, V).T, h.reshape(N * T, H)).T
    db = np.sum(dout, axis=(0, 1))

    return dh, dw, db


