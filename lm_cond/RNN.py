import numpy as np
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import pickle as p
import copy
sys.path.append("../misc")
sys.path.append("../data")

import pickle
from  tensorflow.contrib.rnn import *
# import tensorflow.contrib.rnn
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
import collections
from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.layers.core import Dense
from LDA import LDA
#
from tensorflow.contrib.layers import embed_sequence


class _Linear(object):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of weight variable.
      dtype: data type for variables.
      build_bias: boolean, whether to build a bias variable.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      kernel_initializer: starting value to initialize the weight.

    Raises:
      ValueError: if inputs_shape is wrong.
    """

    def __init__(self,
                 args,
                 output_size,
                 build_bias,
                 bias_initializer=None,
                 kernel_initializer=None):
        self._build_bias = build_bias
        _BIAS_VARIABLE_NAME = "bias"
        _WEIGHTS_VARIABLE_NAME = "kernel"
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]
            self._is_sequence = False
        else:
            self._is_sequence = True

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            if shape[1].value is None:
                raise ValueError("linear expects shape[1] to be provided for shape %s, "
                                 "but saw %s" % (shape, shape[1]))
            else:
                total_arg_size += shape[1].value

        dtype = [a.dtype for a in args][0]

        scope = vs.get_variable_scope()
        with vs.variable_scope(scope) as outer_scope:
            self._weights = vs.get_variable(
                _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
                dtype=dtype,
                initializer=kernel_initializer)
            if build_bias:
                with vs.variable_scope(outer_scope) as inner_scope:
                    inner_scope.set_partitioner(None)
                    if bias_initializer is None:
                        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
                    self._biases = vs.get_variable(
                        _BIAS_VARIABLE_NAME, [output_size],
                        dtype=dtype,
                        initializer=bias_initializer)

    def __call__(self, args):
        if not self._is_sequence:
            args = [args]

        if len(args) == 1:
            res = math_ops.matmul(args[0], self._weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), self._weights)
        if self._build_bias:
            res = nn_ops.bias_add(res, self._biases)
        return res

class tmGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(tmGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None
        self.topic_size = 10
        self.embed_size = 64
        self.W_b = tf.get_variable("Wb", [self.topic_size, self.embed_size])
        self.W_c = tf.get_variable("Wc", [self.embed_size, self.embed_size])

        self.U_b = tf.get_variable("Ub", [self.topic_size, self._num_units])
        self.U_c = tf.get_variable("Uc", [self.embed_size, self._num_units])
        self.topic = tf.placeholder(dtype=tf.float32, shape=[None,10])


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        inputs = tf.matmul(self.topic, self.W_b) * tf.matmul(inputs, self.W_c)
        state = tf.matmul(self.topic, self.U_b) * tf.matmul(inputs, self.U_c)
        # inputs = tf.layers.dense(inputs, self._num_units_input, activation=None ) * tf.layers.dense(self.topic, self._num_units_input, activation=None)
        # state = tf.layers.dense(state, self._num_units, activation=None ) * tf.layers.dense(self.topic, self._num_units, activation=None)

        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = u * state + (1 - u) * c

        return new_h, new_h

# RNN implementation


class tmLSTMCell(RNNCell):
    """Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, embed_size, topic_size=10, forget_bias=1.0,
                 state_is_tuple=True, activation=None, reuse=None):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.

          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(tmLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._embed_size = embed_size
        self._topic_size = topic_size

        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._linear = None
        self.topic = tf.placeholder(dtype=tf.float32, shape=[None,self._topic_size])
        self.initialize_weights()



    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    def initialize_weights(self):
        # here the weights initialized as necessary for Compositional Language Model

        #additional weights for i, f, o , c
        # i
        with tf.variable_scope("the_lstm_weights", reuse=tf.AUTO_REUSE):
            self.W_ib = tf.get_variable("Wib", [self._topic_size, self._embed_size])
            self.W_ic = tf.get_variable("Wic", [self._embed_size, self._embed_size])
            self.U_ib = tf.get_variable("Uib", [self._topic_size, self._num_units])
            self.U_ic = tf.get_variable("Uic", [self._embed_size, self._num_units])
            # f
            self.W_fb = tf.get_variable("Wfb", [self._topic_size, self._embed_size])
            self.W_fc = tf.get_variable("Wfc", [self._embed_size, self._embed_size])
            self.U_fb = tf.get_variable("Ufb", [self._topic_size, self._num_units])
            self.U_fc = tf.get_variable("Ufc", [self._embed_size, self._num_units])
            # o
            self.W_ob = tf.get_variable("Wob", [self._topic_size, self._embed_size])
            self.W_oc = tf.get_variable("Woc", [self._embed_size, self._embed_size])
            self.U_ob = tf.get_variable("Uob", [self._topic_size, self._num_units])
            self.U_oc = tf.get_variable("Uoc", [self._embed_size, self._num_units])
            # c
            self.W_cb = tf.get_variable("Wcb", [self._topic_size, self._embed_size])
            self.W_cc = tf.get_variable("Wcc", [self._embed_size, self._embed_size])
            self.U_cb = tf.get_variable("Ucb", [self._topic_size, self._num_units])
            self.U_cc = tf.get_variable("Ucc", [self._embed_size, self._num_units])

            # standard weights
            # i
            self.W_ia = tf.get_variable("Wia", [self._embed_size, self._num_units])
            self.U_ia = tf.get_variable("Uia", [self._num_units, self._num_units])
            self.b_i = tf.get_variable(name="Bi", shape=[self._num_units], initializer=tf.zeros_initializer())

            # f
            self.W_fa = tf.get_variable("Wfa", [self._embed_size, self._num_units])
            self.U_fa = tf.get_variable("Ufa", [self._num_units, self._num_units])
            self.b_f = tf.get_variable(name="Bf", shape=[self._num_units], initializer=tf.zeros_initializer())
            # o
            self.W_oa = tf.get_variable("Woa", [self._embed_size, self._num_units])
            self.U_oa = tf.get_variable("Uoa", [self._num_units, self._num_units])
            self.b_o = tf.get_variable(name="Bo", shape=[self._num_units], initializer=tf.zeros_initializer())
            # c
            self.W_ca = tf.get_variable("Wca", [self._embed_size, self._num_units])
            self.U_ca = tf.get_variable("Uca", [self._num_units, self._num_units])
            self.b_c = tf.get_variable(name="Bc", shape=[self._num_units], initializer=tf.zeros_initializer())

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """



        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
        # added

        x_i = tf.matmul(self.topic, self.W_ib) * tf.matmul(inputs, self.W_ic)
        x_f = tf.matmul(self.topic, self.W_fb) * tf.matmul(inputs, self.W_fc)
        x_o = tf.matmul(self.topic, self.W_ob) * tf.matmul(inputs, self.W_oc)
        x_c = tf.matmul(self.topic, self.W_cb) * tf.matmul(inputs, self.W_cc)

        h_i = tf.matmul(self.topic, self.U_ib) * tf.matmul(inputs, self.U_ic)
        h_f = tf.matmul(self.topic, self.U_fb) * tf.matmul(inputs, self.U_fc)
        h_o = tf.matmul(self.topic, self.U_ob) * tf.matmul(inputs, self.U_oc)
        h_c = tf.matmul(self.topic, self.U_cb) * tf.matmul(inputs, self.U_cc)


        i_m = sigmoid( tf.matmul(x_i, self.W_ia ) + tf.matmul( h_i, self.U_ia ) + self.b_i)
        f_m = sigmoid( tf.matmul(x_f, self.W_fa ) + tf.matmul( h_f, self.U_fa ) + self.b_f)
        o_m = sigmoid( tf.matmul(x_o, self.W_oa ) + tf.matmul( h_o, self.U_oa ) + self.b_o)
        c_m_tilde = sigmoid( tf.matmul(x_c, self.W_ca) +tf.matmul( h_c, self.U_ca ) + self.b_c)

        # based on peephole lstm
        new_c = (c_m_tilde * f_m  +  i_m * c_m_tilde)
        new_h = self._activation(new_c) * o_m


        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state

