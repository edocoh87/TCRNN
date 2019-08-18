import numpy as np
import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class CommutativeRNNcell(tf.contrib.rnn.BasicRNNCell):
    def __init__(self,
                num_units,
                computation_dim,
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
        print('get weights: ', self.input_depth)
        W, THETA = array_ops.split(self._kernel, tf.convert_to_tensor([self.input_depth.value, self._num_units]), 0)
        W = array_ops.transpose(W)
        THETA = array_ops.transpose(THETA)
        A = array_ops.transpose(self._kernel_out)
        return A, W, THETA

    def get_comm_regularizer(self, epsilon=1e-3):
        def G(U, V, norm_matrix):
            inner_prod_mat = tf.matmul(U, V, transpose_b=True)
            cos_alpha = tf.divide(inner_prod_mat, norm_matrix + epsilon)
            alpha = tf.acos(cos_alpha)
            return (norm_matrix/np.pi) * (tf.sin(alpha) + (np.pi - alpha) * cos_alpha)

        A, W, THETA = self.get_weights()
        print('A: ', A)
        print('W: ', W)
        print('THETA: ', THETA)
        # A = tf.Print(_A, [_A, W, THETA], "weights:", summarize=100)
        Q = tf.matmul(A, A, transpose_a=True)
        U = tf.concat([THETA, W], axis=1)
        V = tf.concat([W, THETA], axis=1)
        print('U: ', U)
        # the norms of row i of U (they're the same for V)
        norm_per_row = tf.norm(U, axis=1, keepdims=True)
        # norm_per_row = tf.Print(_norm_per_row, [_norm_per_row], "norm_per_row: ", summarize=100)
        print('norm_per_row: ', norm_per_row)
        
        # norm_matrix_ij = norm(u_i)*norm(u_j)
        norm_matrix = tf.matmul(norm_per_row, norm_per_row, transpose_b=True)
        
        print('norm_matrix: ', norm_matrix)

        UU = G(U, U, norm_matrix)
        UV = G(U, V, norm_matrix)

        return tf.reduce_sum(2*tf.multiply(Q, UU-UV))
    # @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        # if inputs_shape[-1] is None:
        #     raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
        #         #                         % str(inputs_shape))
        
        self.input_depth = inputs_shape[-1]
        print('inputs_shape {}'.format(self.input_depth))
        init_arr = np.zeros((self.input_depth + self._num_units, self._computation_dim))
        for i in range(self._num_units):
            init_arr[i, 3*i] = 1
            init_arr[i+self._num_units, 3*i] = -1
            init_arr[i+self._num_units, 3*i+1] = 1
            init_arr[i+self._num_units, 3*i+2] = -1
        self._kernel = self.add_variable(
            "kernel",
            # shape=[self.input_depth + self._num_units, self._computation_dim])
            trainable=False,
            shape=[self.input_depth + self._num_units, self._computation_dim],
            initializer=tf.constant_initializer(init_arr))
            # initializer=tf.initializers.identity(dtype=self.dtype))
        assert self._num_units*3 == self._computation_dim, "max aggregation."
        out_init_arr = np.zeros((self._computation_dim, self._num_units))
        for i in range(self._num_units):
            out_init_arr[3*i, i] = 1
            out_init_arr[3*i+1, i] = 1
            out_init_arr[3*i+2, i] = -1
        # the output weights don't exist in the standard implementation.
        self._kernel_out = self.add_variable(
            "kernel_out",
            # shape=[self._computation_dim, self._num_units])
            trainable=False,
            shape=[self._computation_dim, self._num_units],
            initializer=tf.constant_initializer(out_init_arr))
            # initializer=tf.initializers.identity(dtype=self.dtype))
        # self._bias = self.add_variable(
        #     _BIAS_VARIABLE_NAME,
        #     shape=[self._num_units],
        #     initializer=init_ops.zeros_initializer(dtype=self.dtype))
        

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        # print('inputs: ', inputs)
        # print('state: ', state)
        # print('concatenated: ', array_ops.concat([inputs, state], 1))
        # print('weights: ', self._kernel)
        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._kernel)
        # gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        gate_outputs = self._activation(gate_inputs)
        
        # this is not in the standard rnn cell and the reason we had to implement a new cell..
        output = math_ops.matmul(gate_outputs, self._kernel_out)
        return output, output
