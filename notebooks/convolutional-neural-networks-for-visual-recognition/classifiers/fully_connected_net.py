from builtins import range
from builtins import object
import numpy as np

from utils.layers import *
from utils.layer_utils import *


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, input_dim, hidden_dims, num_classes,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - input_dim: An integer giving the size of the input.
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero    #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        layers = []
        
        layers.append(input_dim)
        layers.extend(hidden_dims)
        layers.append(num_classes)
        
        for l in np.arange(self.num_layers):
            self.params["W" + str(l + 1)] = weight_scale * np.random.randn(layers[l], layers[l + 1])
            self.params["b" + str(l + 1)] = np.zeros(layers[l + 1])
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
            for l in np.arange(self.num_layers - 1):
                self.params["gamma" + str(l + 1)] = np.ones(layers[l + 1])
                self.params["beta" + str(l + 1)] = np.zeros(layers[l + 1])
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]
            for l in np.arange(self.num_layers - 1):
                self.params["gamma" + str(l + 1)] = np.ones(layers[l + 1])
                self.params["beta" + str(l + 1)] = np.zeros(layers[l + 1])
            
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        A = X
        caches = []
        for l in np.arange(self.num_layers - 1):
            W = self.params["W" + str(l + 1)]
            b = self.params["b" + str(l + 1)]
            if self.normalization=='batchnorm':
                bn_param = self.bn_params[l]
                gamma = self.params["gamma" + str(l + 1)]
                beta = self.params["beta" + str(l + 1)]
                A, cache = affine_bn_relu_forward(A, W, b, gamma, beta, bn_param)
            elif self.normalization=='layernorm':
                ln_param = {}
                gamma = self.params["gamma" + str(l + 1)]
                beta = self.params["beta" + str(l + 1)]
                A, cache = affine_ln_relu_forward(A, W, b, gamma, beta, {})
            elif self.use_dropout:
                A, cache = affine_relu_dropout_forward(A, W, b, self.dropout_param)
            else:
                A, cache = affine_relu_forward(A, W, b)
            
            caches.append(cache)
        
        W = self.params["W" + str(self.num_layers)]
        b = self.params["b" + str(self.num_layers)]
        A, cache = affine_forward(A, W, b)
        caches.append(cache)
        
        scores = A
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization                #
        #                                                                          #
        # When using batch/layer normalization, you dont need to regularize the    #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        data_loss, dscores = softmax_loss(scores, y)
        
        reg = self.reg
        reg_loss = 0.0
        for l in np.arange(self.num_layers):
            W = self.params["W" + str(l + 1)]
            reg_loss += 0.5 * reg * np.sum(W * W)
        
        loss = data_loss + reg_loss
        
        dA = dscores
        dA, dW, db = affine_backward(dA, caches[self.num_layers - 1])
        grads["W" + str(self.num_layers)] = dW
        grads["b" + str(self.num_layers)] = db
        
        for l in np.arange(self.num_layers - 1)[::-1]:
            if self.use_dropout:
                dA, dW, db = affine_relu_dropout_backward(dA, caches[l])
            elif self.normalization=='batchnorm':
                dA, dW, db, dgamma, dbeta = affine_bn_relu_backward(dA, caches[l])
                grads["gamma" + str(l + 1)] = dgamma
                grads["beta" + str(l + 1)] = dbeta
            elif self.normalization=='layernorm':
                dA, dW, db, dgamma, dbeta = affine_ln_relu_backward(dA, caches[l])
                grads["gamma" + str(l + 1)] = dgamma
                grads["beta" + str(l + 1)] = dbeta
            else:
                dA, dW, db = affine_relu_backward(dA, caches[l])
            
            grads["W" + str(l + 1)] = dW
            grads["b" + str(l + 1)] = db

        for l in np.arange(self.num_layers):
            grads["W" + str(l + 1)] += reg * self.params["W" + str(l + 1)]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
