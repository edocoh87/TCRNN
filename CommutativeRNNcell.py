import numpy as np
import scipy.stats as stats
import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.layers import base as base_layer
from tensorflow.python.util import nest
from commutative_rnn_utils import get_init_arrays, _zero_state_tensors, zero_state

RAND_BOUND = .1

class CommutativeRNNcell(tf.contrib.rnn.BasicRNNCell):
    def __init__(self,
                num_units,
                computation_dim,
                dropout_rate_ph,
                initialization_scheme,
                initial_state,
                weight_config,
                trainable=True,
                activation=None,
                reuse=None,
                name=None,
                dtype=None,
                **kwargs):
        super(CommutativeRNNcell, self).__init__(
            num_units=num_units, reuse=reuse, name=name, dtype=dtype, **kwargs)
        # if context.executing_eagerly() and context.num_gpus() > 0:
        #     logging.warn("%s: Note that this cell is not optimized for performance. "
        #                 "Please use tf.contrib.cudnn_rnn.CudnnRNNTanh for better "
        #                 "performance on GPU.", self)

        # Inputs must be 2-dimensional.
        # self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        self._computation_dim = computation_dim
        if len(computation_dim) == 1:
            self._computation_dim += [self._num_units]
        else:
            assert computation_dim[1] == self._num_units, 'the second layer must have equal size to hidden layer but is {}, and hidden size is {}'.format(computation_dim[1], self._num_units)
        #self._computation_dim = computation_dim + [self._num_units]
        self.dropout_rate_ph = dropout_rate_ph
        self.trainable = trainable
        self.initialization_scheme = initialization_scheme
        self.initial_state = initial_state
        self.weight_config = weight_config
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units
      
    def get_weights(self):
        # print('get weights: ', self.input_depth)
        W, THETA = array_ops.split(self._kernel, tf.convert_to_tensor([self.input_depth.value, self._num_units]), 0)
        W = array_ops.transpose(W)
        THETA = array_ops.transpose(THETA)
        A = array_ops.transpose(self._kernel_out)
        return A, W, THETA

    def get_comm_regularizer(self, epsilon=1e-5):
        def G(U, V, norm_matrix):
            inner_prod_mat = tf.matmul(U, V, transpose_b=True)
            cos_alpha = tf.divide(inner_prod_mat, norm_matrix + epsilon)
            alpha = tf.acos(cos_alpha)
            return (norm_matrix/np.pi) * (tf.sin(alpha) + (np.pi - alpha) * cos_alpha)

        A, W, THETA = self.get_weights()
        Q = tf.matmul(A, A, transpose_a=True)
        U = tf.concat([THETA, W], axis=1)
        V = tf.concat([W, THETA], axis=1)
        norm_per_row = tf.norm(U, axis=1, keepdims=True)
        # norm_per_row = tf.Print(_norm_per_row, [_norm_per_row], "norm_per_row: ", summarize=100)
        # print('norm_per_row: ', norm_per_row)
        
        # norm_matrix_ij = norm(u_i)*norm(u_j)
        norm_matrix = tf.matmul(norm_per_row, norm_per_row, transpose_b=True)
        
        UU = G(U, U, norm_matrix)
        UV = G(U, V, norm_matrix)

        return tf.reduce_sum(2*tf.multiply(Q, UU-UV))

    # @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        # if inputs_shape[-1] is None:
        #     raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
        #         #                         % str(inputs_shape))
        self.input_depth = inputs_shape[-1]        
        network_type = 'dense'
        kernel_init_arr, kernel_out_init_arr = get_init_arrays(
                                                    network_type,
                                                    self.initialization_scheme,
                                                    self.input_depth,
                                                    self._computation_dim,
                                                    self._num_units)
        #kernels_init_arr = [np.random.normal(scale=RAND_BOUND, size=(self._computation_dim[i-1], self._computation_dim[i]))
        kernels_init_arr = [stats.truncnorm.rvs(-2*RAND_BOUND, 2*RAND_BOUND, scale=RAND_BOUND, size=(self._computation_dim[i-1], self._computation_dim[i]))
                                                                                    for i in range(2, len(self._computation_dim))]
        
        # with tf.variable_scope("rnn_weights"):
        self.__kernel = self.add_variable(
            "kernel",
            # shape=[self.input_depth + self._num_units, self._computation_dim])
            trainable=self.trainable,
            shape=[self.input_depth + self._num_units, self._computation_dim[0]],
            initializer=tf.constant_initializer(kernel_init_arr))
            #initializer=tf.initializers.identity(dtype=self.dtype))
        
        # the output weights don't exist in the standard implementation.
        self.__kernel_out = self.add_variable(
            "kernel_out",
            # shape=[self._computation_dim, self._num_units])
            trainable=self.trainable,
            shape=[self._computation_dim[0], self._computation_dim[1]],
            initializer=tf.constant_initializer(kernel_out_init_arr))
            #initializer=tf.initializers.identity(dtype=self.dtype))

        self.__kernels = [self.add_variable(
            "kernel_{}".format(i+1),
            # shape=[self._computation_dim, self._num_units])
            trainable=self.trainable,
            shape=[self._computation_dim[i-1], self._computation_dim[i]],
            initializer=tf.constant_initializer(kernels_init_arr[i-2])) for i in range(2, len(self._computation_dim))]

        # self._bias = self.add_variable(
        #     _BIAS_VARIABLE_NAME,
        #     shape=[self._num_units],
        #     initializer=init_ops.zeros_initializer(dtype=self.dtype))

        def lr_mult(lr_ph):
            @tf.custom_gradient
            def _lr_mult(x):
                def grad(dy):
                    return dy * lr_ph * tf.ones_like(x)
                return x, grad
            return _lr_mult
            
        rnn_lr_ph = tf.get_default_graph().get_tensor_by_name('rnn_lr_ph:0')
        self._kernel = lr_mult(rnn_lr_ph)(self.__kernel)
        
        # self._kernel_out = self.__kernel_out
        self._kernel_out = lr_mult(rnn_lr_ph)(self.__kernel_out)

        # self._kernels = [lr_mult(rnn_lr_ph)(_ker) for _ker in self.__kernels]
        self._kernels = self.__kernels
            # initializer=tf.initializers.identity(dtype=self.dtype))
        
        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        # _state = tf.Print(state, [state], "state: ", summarize=100)
        input_and_state = array_ops.concat([inputs, state], 1)
        input_and_state = tf.nn.dropout(input_and_state, self.dropout_rate_ph)
        gate_inputs = math_ops.matmul(input_and_state, self._kernel)
        
        # gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        gate_outputs = self._activation(gate_inputs)
        
        # this is not in the standard rnn cell and the reason we had to implement a new cell..
        gate_outputs = tf.nn.dropout(gate_outputs, self.dropout_rate_ph)
        output = math_ops.matmul(gate_outputs, self._kernel_out)

        _inputs = output
        for i in range(len(self._kernels)):
            _inputs = tf.nn.dropout(_inputs, self.dropout_rate_ph)
            _gate_inputs = math_ops.matmul(_inputs, self._kernels[i])
            if i < len(self._kernels)-1:
                _inputs = self._activation(_gate_inputs)
                # _inputs = tf.nn.leaky_relu(_gate_inputs)
            else:
                outputs = _gate_inputs
            
        # output = tf.maximum(inputs, state)
        return output, output
        # return _output, _output

    