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

RAND_BOUND = 0.3
MINUS_INF = -1e4
def _concat(prefix, suffix, static=False):
  """Concat that enables int, Tensor, or TensorShape values.
  This function takes a size specification, which can be an integer, a
  TensorShape, or a Tensor, and converts it into a concatenated Tensor
  (if static = False) or a list of integers (if static = True).
  Args:
    prefix: The prefix; usually the batch size (and/or time step size).
      (TensorShape, int, or Tensor.)
    suffix: TensorShape, int, or Tensor.
    static: If `True`, return a python list with possibly unknown dimensions.
      Otherwise return a `Tensor`.
  Returns:
    shape: the concatenation of prefix and suffix.
  Raises:
    ValueError: if `suffix` is not a scalar or vector (or TensorShape).
    ValueError: if prefix or suffix was `None` and asked for dynamic
      Tensors out.
  """
  if isinstance(prefix, ops.Tensor):
    p = prefix
    p_static = tensor_util.constant_value(prefix)
    if p.shape.ndims == 0:
      p = array_ops.expand_dims(p, 0)
    elif p.shape.ndims != 1:
      raise ValueError("prefix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % p)
  else:
    p = tensor_shape.as_shape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (
        constant_op.constant(p.as_list(), dtype=dtypes.int32)
        if p.is_fully_defined() else None)
  if isinstance(suffix, ops.Tensor):
    s = suffix
    s_static = tensor_util.constant_value(suffix)
    if s.shape.ndims == 0:
      s = array_ops.expand_dims(s, 0)
    elif s.shape.ndims != 1:
      raise ValueError("suffix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % s)
  else:
    s = tensor_shape.as_shape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (
        constant_op.constant(s.as_list(), dtype=dtypes.int32)
        if s.is_fully_defined() else None)

  if static:
    shape = tensor_shape.as_shape(p_static).concatenate(s_static)
    shape = shape.as_list() if shape.ndims is not None else None
  else:
    if p is None or s is None:
      raise ValueError("Provided a prefix or suffix of None: %s and %s" %
                       (prefix, suffix))
    shape = array_ops.concat((p, s), 0)
  return shape

# def _zero_state_tensors(state_size, batch_size, dtype):
#   """Create tensors of zeros based on state_size, batch_size, and dtype."""
#   def get_state_shape(s):
#     """Combine s with batch_size to get a proper tensor shape."""
#     c = _concat(batch_size, s)
#     # size = array_ops.zeros(c, dtype=dtype)
#     # size = tf.constant(MINUS_INF) * array_ops.ones(c, dtype=dtype)
#     size = tf.random.normal(c, dtype=dtype)
#     if not context.executing_eagerly():
#       c_static = _concat(batch_size, s, static=True)
#       size.set_shape(c_static)
#     return size

#   return nest.map_structure(get_state_shape, state_size)

def angle_between(v1, v2, epsilon=1e-5):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    # print_v1 = tf.Print(v1, [v1])
    # print_v2 = tf.Print(v2, [v2])
    
    v1_u = tf.nn.l2_normalize(v1)
    v2_u = tf.nn.l2_normalize(v2)
    return tf.acos(tf.clip_by_value(tf.matmul(tf.transpose(v1_u), v2_u), -1.0 + epsilon, 1.0 - epsilon))

def get_arccos_integral(v1, v2):
    # 1/pi  * norm(v1)*norm(v2) * (sin(alpha) + (pi-alpha)cos(alpha))
    angle = angle_between(v1, v2)
    return ((tf.norm(v1) * tf.norm(v2)) / np.pi) * (tf.sin(angle) + (np.pi - angle)*tf.cos(angle))

def g(u,v):
    return get_arccos_integral(u,v)

def get_expectation_v2(A, W, THETA):
        h, d = W.get_shape().as_list()
        Q = tf.matmul(tf.transpose(A), A)
        total_expectation = 0
        print('building regularizer')
        for i in range(h):
            print('row {}'.format(i))
            w_i = tf.reshape(W[i,:], [d, 1])
            theta_i = tf.reshape(THETA[i,:], [d, 1])
            u_i = tf.concat([theta_i, w_i], axis=0)
            v_i = tf.concat([w_i, theta_i], axis=0)
            for j in range(i+1):
                w_j = tf.reshape(W[j,:], [d, 1])
                theta_j = tf.reshape(THETA[j,:], [d, 1])
                u_j = tf.concat([theta_j, w_j], axis=0)
                v_j = tf.concat([w_j, theta_j], axis=0)
                if j < i:
                    factor = 2
                else:
                    factor = 1
                total_expectation += factor*2*Q[i,j]*(g(u_i, u_j) - g(u_i, v_j))
        return tf.squeeze(total_expectation)

class CommutativeRNNcell(tf.contrib.rnn.BasicRNNCell):
    def __init__(self,
                num_units,
                computation_dim,
                dropout_rate_ph,
                initialization_scheme,
                initial_state,
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

    # # when using only one network for all the dimensions together.
    # def get_very_sparse_regularizer(self):
    #     # A, W, THETA = self.get_weights()
    #     A = self._kernel_out 
    #     _W, _THETA = array_ops.split(self._kernel, num_or_size_splits=2, axis=0)
    #     W = tf.transpose(_W)
    #     THETA = tf.transpose(_THETA)
    #     # THETA = tf.Print(tmp_THETA, [tmp_THETA, W], 'weight vectors:')
    #     # print(self._kernel)
    #     # print(W, THETA)
        
    #     aa = tf.matmul(A, A, transpose_a=True)
    #     g_w_w = get_arccos_integral(W, W)
    #     g_w_theta = get_arccos_integral(W, THETA)
    #     g_theta_theta = get_arccos_integral(THETA, THETA)
    #     reg_loss = aa*(g_w_w - 2*g_w_theta + g_theta_theta)
    #     return tf.squeeze(reg_loss)

    # def get_sparse_regularizer(self):
    #     # A, W, THETA = self.get_weights()
    #     A = self._kernel_out 
    #     _W, _THETA = array_ops.split(self._kernel, num_or_size_splits=2, axis=0)
    #     W = tf.transpose(_W)
    #     THETA = tf.transpose(_THETA)
    #     # THETA = tf.Print(tmp_THETA, [tmp_THETA, W], 'weight vectors:')
    #     # print(self._kernel)
    #     # print(W, THETA)
        
    #     aa = tf.matmul(A, A, transpose_a=True)
    #     g_w_w = get_arccos_integral(W, W)
    #     g_w_theta = get_arccos_integral(W, THETA)
    #     g_theta_theta = get_arccos_integral(THETA, THETA)
    #     reg_loss = aa*(g_w_w - 2*g_w_theta + g_theta_theta)
    #     return tf.squeeze(reg_loss)

    def _get_comm_regularizer(self, epsilon=1e-3):
        def G(U, V, norm_matrix):
            inner_prod_mat = tf.matmul(U, V, transpose_b=True)
            cos_alpha = tf.divide(inner_prod_mat, norm_matrix + epsilon)
            #cos_alpha = tf.Print(_cos_alpha, [_cos_alpha], 'cos_alpha', summarize=10000)
            alpha = tf.acos(cos_alpha)
            #alpha = tf.Print(_alpha, [cos_alpha, _alpha], 'cos_alpha and alpha', summarize=10000)
            norms = (norm_matrix/np.pi) * (tf.sin(alpha) + (np.pi - alpha) * cos_alpha)
            # norms = tf.Print(_norms, [_norms], 'cos_alpha and alpha', summarize=10000)
            #return (norm_matrix/np.pi) * (tf.sin(alpha) + (np.pi - alpha) * cos_alpha)
            return norms
        
        A, W, THETA = self.get_weights()
        # print('A: ', A)
        # print('W: ', W)
        # print('THETA: ', THETA)
        # A = tf.Print(_A, [_A, W, THETA], "weights:", summarize=100)
        Q = tf.matmul(A, A, transpose_a=True)
        U = tf.concat([THETA, W], axis=1)
        V = tf.concat([W, THETA], axis=1)
        # print('U: ', U)
        # the norms of row i of U (they're the same for V)
        norm_per_row = tf.norm(U, axis=1, keepdims=True)
        # norm_per_row = tf.Print(_norm_per_row, [_norm_per_row], "norm_per_row: ", summarize=100)
        # print('norm_per_row: ', norm_per_row)
        
        # norm_matrix_ij = norm(u_i)*norm(u_j)
        norm_matrix = tf.matmul(norm_per_row, norm_per_row, transpose_b=True)
        # norm_matrix = tf.Print(_norm_matrix, [norm_per_row, _norm_matrix], 'norms ', summarize=10000)
        
        # print('norm_matrix: ', norm_matrix)

        UU = G(U, U, norm_matrix)
        UV = G(U, V, norm_matrix)

        return tf.reduce_sum(2*tf.multiply(Q, UU-UV))

    def _get_comm_regularizer_sparse(self, epsilon=1e-3):
        def G(U, V, norm_matrix):
            inner_prod_mat = tf.matmul(U, V, transpose_b=True)
            cos_alpha = tf.divide(inner_prod_mat, norm_matrix + epsilon)
            #cos_alpha = tf.Print(_cos_alpha, [_cos_alpha], 'cos_alpha', summarize=10000)
            alpha = tf.acos(cos_alpha)
            #alpha = tf.Print(_alpha, [cos_alpha, _alpha], 'cos_alpha and alpha', summarize=10000)
            norms = (norm_matrix/np.pi) * (tf.sin(alpha) + (np.pi - alpha) * cos_alpha)
            # norms = tf.Print(_norms, [_norms], 'cos_alpha and alpha', summarize=10000)
            #return (norm_matrix/np.pi) * (tf.sin(alpha) + (np.pi - alpha) * cos_alpha)
            return norms
        
        total_expectation = 0
        for (kernel, kernel_out) in zip(self._kernel, self._kernel_out):
            W, THETA = array_ops.split(kernel, num_or_size_splits=2 , axis=0)
            W = array_ops.transpose(W)
            THETA = array_ops.transpose(THETA)
            A = array_ops.transpose(kernel_out)
            # A, W, THETA = self.get_weights()
            # print('A: ', A)
            # print('W: ', W)
            # print('THETA: ', THETA)
            # A = tf.Print(_A, [_A, W, THETA], "weights:", summarize=100)
            Q = tf.matmul(A, A, transpose_a=True)
            U = tf.concat([THETA, W], axis=1)
            V = tf.concat([W, THETA], axis=1)
            # print('U: ', U)
            # the norms of row i of U (they're the same for V)
            norm_per_row = tf.norm(U, axis=1, keepdims=True)
            # norm_per_row = tf.Print(_norm_per_row, [_norm_per_row], "norm_per_row: ", summarize=100)
            # print('norm_per_row: ', norm_per_row)
            
            # norm_matrix_ij = norm(u_i)*norm(u_j)
            norm_matrix = tf.matmul(norm_per_row, norm_per_row, transpose_b=True)
            # norm_matrix = tf.Print(_norm_matrix, [norm_per_row, _norm_matrix], 'norms ', summarize=10000)
            
            # print('norm_matrix: ', norm_matrix)

            UU = G(U, U, norm_matrix)
            UV = G(U, V, norm_matrix)
            total_expectation += tf.reduce_sum(2*tf.multiply(Q, UU-UV))

        return total_expectation

    def get_comm_regularizer(self):
        # A = self._kernel_out 
        # W, THETA = array_ops.split(self._kernel, num_or_size_splits=2, axis=0)
        # print(W, THETA)
        # return get_expectation_v2(A, W, THETA)
        # return self.get_very_sparse_regularizer()
        return self._get_comm_regularizer_sparse()

    # @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        # if inputs_shape[-1] is None:
        #     raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
        #         #                         % str(inputs_shape))
        self.input_depth = inputs_shape[-1]
        assert self._computation_dim[0] % self._num_units == 0, 'computation_dim first element must be divisible by hidden layer size.'
        self.neurons_per_cell = int(self._computation_dim[0] / self._num_units)
        # print('inputs_shape {}'.format(self.input_depth))
        if self.initialization_scheme == 'max':
            kernel_init_arr = np.zeros((2, self.neurons_per_cell))
            kernel_out_init_arr = np.zeros((self.neurons_per_cell, 1))
            kernel_init_arr[0,0] = 1.
            kernel_init_arr[1,0] = -1.
            kernel_init_arr[1,1] = 1.
            kernel_init_arr[1,2] = -1.

            kernel_out_init_arr[0,0] = 1.
            kernel_out_init_arr[1,0] = 1.
            kernel_out_init_arr[2,0] = -1.
            # print('initializing transition matrix to max')
            # assert self.input_depth == self._num_units, 'input_depth must be equal to _num_units.'
            # print('num of units: ', self._num_units)
            # print('comp dim: ', self._computation_dim)
            # assert self._computation_dim[0] >= 3*self._num_units, '_computation_dim[0] must be at least x3 larger than _num_units to implement maximum.'
            # assert self._computation_dim[1] >= self._num_units, '_computation_dim[1] must be at least the size of _num_unit to implement maximum.'
            
            # kernel_init_arr = np.zeros((self.input_depth + self._num_units,
            #                                         self._computation_dim[0]))
            # # kernel_init_arr = np.random.uniform(low=-RAND_BOUND, high=RAND_BOUND, size=(self.input_depth + self._num_units, self._computation_dim))
            # # kernel_init_arr = np.random.normal(scale=RAND_BOUND, size=(self.input_depth + self._num_units, self._computation_dim))
            # for i in range(self._num_units):
            #     kernel_init_arr[i, 3*i] = 1
            #     kernel_init_arr[i+self._num_units, 3*i] = -1
            #     kernel_init_arr[i+self._num_units, 3*i+1] = 1
            #     kernel_init_arr[i+self._num_units, 3*i+2] = -1
            
            # kernel_out_init_arr = np.zeros((self._computation_dim[0], self._computation_dim[1]))
            # # kernel_out_init_arr = np.random.uniform(low=-RAND_BOUND, high=RAND_BOUND, size=(self._computation_dim, self._num_units))
            # # kernel_out_init_arr = np.random.normal(scale=RAND_BOUND, size=(self._computation_dim, self._num_units))
            # for i in range(self._num_units):
            #     kernel_out_init_arr[3*i, i] = 1
            #     kernel_out_init_arr[3*i+1, i] = 1
            #     kernel_out_init_arr[3*i+2, i] = -1


        elif self.initialization_scheme == 'sum':
            print('initializing transition matrix to sum')
            assert self.input_depth == self._num_units, 'input_depth must be equal to _num_units.'
            print('num of units: ', self._num_units)
            print('comp dim: ', self._computation_dim)
            assert self._computation_dim[0] >= 2*self._num_units, '_computation_dim[0] must be at least x2 larger than _num_units to implement summation.'
            assert self._computation_dim[1] >= self._num_units, '_computation_dim[1] must be at least the size of _num_unit to implement summation.'
            # kernel_init_arr = np.zeros((self.input_depth + self._num_units,
            #                                        self._computation_dim[0]))
            # kernel_init_arr = np.random.uniform(low=-RAND_BOUND, high=RAND_BOUND, size=(self.input_depth + self._num_units, self._computation_dim))
            kernel_init_arr = np.random.normal(scale=RAND_BOUND, size=(self.input_depth + self._num_units, self._computation_dim[0]))
            for i in range(self._num_units):
                kernel_init_arr[i, 2*i] = 1
                kernel_init_arr[i, 2*i+1] = -1
                kernel_init_arr[i+self._num_units, 2*i] = 1
                kernel_init_arr[i+self._num_units, 2*i+1] = -1

            # kernel_out_init_arr = np.zeros((self._computation_dim[0], self._computation_dim[1]))
            # kernel_out_init_arr = np.random.uniform(low=-RAND_BOUND, high=RAND_BOUND, size=(self._computation_dim, self._num_units))
            kernel_out_init_arr = np.random.normal(scale=RAND_BOUND, size=(self._computation_dim[0], self._computation_dim[1]))
            for i in range(self._num_units):
                kernel_out_init_arr[2*i, i] = 1
                kernel_out_init_arr[2*i+1, i] = -1


        elif self.initialization_scheme == 'rand':
            # xavier initialization:
            # kernel_init_arr = np.random.rand(self.input_depth + self._num_units,
            #    self._computation_dim)*np.sqrt(1.0 / int(self.input_depth + self._num_units + self._computation_dim))
            # kernel_init_arr = stats.truncnorm.rvs(-2*RAND_BOUND, 2*RAND_BOUND, scale=RAND_BOUND, size=(2, self.neurons_per_cell))
            kernel_init_arr = [stats.truncnorm.rvs(-2*RAND_BOUND, 2*RAND_BOUND, scale=RAND_BOUND, size=(2, self.neurons_per_cell)) for i in range(self._num_units)]
            # kernel_init_arr = stats.truncnorm.rvs(-2*RAND_BOUND, 2*RAND_BOUND, scale=RAND_BOUND, size=(self.input_depth + self._num_units, self.neurons_per_cell))
            #kernel_init_arr = np.random.normal(scale=RAND_BOUND, size=(self.input_depth + self._num_units, self._computation_dim[0]))
            #kernel_init_arr = np.random.normal(scale=np.sqrt(1.0 / int(self.input_depth + self._num_units + self._computation_dim)), 
	    #		size=(self.input_depth + self._num_units, self._computation_dim))

            # kernel_out_init_arr = np.random.rand(self._computation_dim,
            #kernel_out_init_arr = np.random.normal(scale=RAND_BOUND, size=(self._computation_dim[0], self._computation_dim[1]))
            # kernel_out_init_arr = stats.truncnorm.rvs(-2*RAND_BOUND, 2*RAND_BOUND, scale=RAND_BOUND, size=(self.neurons_per_cell, self._computation_dim[1]))
            kernel_out_init_arr = [stats.truncnorm.rvs(-2*RAND_BOUND, 2*RAND_BOUND, scale=RAND_BOUND, size=(self.neurons_per_cell, 1)) for i in range(self._num_units)]
            #kernel_out_init_arr = np.random.normal(scale=np.sqrt(1.0 / int(self._num_units + self._computation_dim)),
	    #	 			size=(self._computation_dim, self._num_units))
        
        #kernels_init_arr = [np.random.normal(scale=RAND_BOUND, size=(self._computation_dim[i-1], self._computation_dim[i]))
        kernels_init_arr = [stats.truncnorm.rvs(-2*RAND_BOUND, 2*RAND_BOUND, scale=RAND_BOUND, size=(self._computation_dim[i-1], self._computation_dim[i]))
                                                                                    for i in range(2, len(self._computation_dim))]
        
        
        # # _lr_ph = tf.placeholder(tf.float32, shape=[], name='rnn_lr_ph')
        # _lr_ph = tf.get_default_graph().get_tensor_by_name('rnn_lr_ph:0')    

        # with tf.variable_scope("rnn_weights"):
        self.__kernel = [self.add_variable(
            "kernel_{}".format(i+1),
            # shape=[self.input_depth + self._num_units, self._computation_dim])
            trainable=self.trainable,
            # shape=[self.input_depth + self._num_units, self._computation_dim[0]],
            shape=[2, self.neurons_per_cell],
            # shape=[self.input_depth + self._num_units, self.neurons_per_cell],
            initializer=tf.constant_initializer(kernel_init_arr[i])) for i in range(self._num_units)]
            # initializer=tf.initializers.identity(dtype=self.dtype))
        
        # the output weights don't exist in the standard implementation.
        self.__kernel_out = [self.add_variable(
            "kernel_out_{}".format(i+1),
            # shape=[self._computation_dim, self._num_units])
            trainable=self.trainable,
            # shape=[self._computation_dim[0], self._computation_dim[1]],
            shape=[self.neurons_per_cell, 1],
            # shape=[self.neurons_per_cell, self._computation_dim[1]],
            initializer=tf.constant_initializer(kernel_out_init_arr[i])) for i in range(self._num_units)]


        # self.__kernels = [self.add_variable(
        #     "kernel_{}".format(i+1),
        #     # shape=[self._computation_dim, self._num_units])
        #     trainable=self.trainable,
        #     shape=[self._computation_dim[i-1], self._computation_dim[i]],
        #     initializer=tf.constant_initializer(kernels_init_arr[i-2])) for i in range(2, len(self._computation_dim))]

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
        # self._kernel = lr_mult(rnn_lr_ph)(self.__kernel)
        # self._kernel_out = lr_mult(rnn_lr_ph)(self.__kernel_out)
        self._kernel = [lr_mult(rnn_lr_ph)(curr_ker) for curr_ker in self.__kernel]
        self._kernel_out = [lr_mult(rnn_lr_ph)(curr_ker) for curr_ker in self.__kernel_out]
        
        # MAKE MATRICES BIG AGAIN!
        # list_of_matrices = []
        # for u in range(self._num_units):
        #     helper_matrix = np.zeros((2*self._num_units, 2*self._num_units))
        #     helper_matrix[u,u] = 1
        #     helper_matrix[u+self._num_units,u+self._num_units] = 1
        #     list_of_matrices.append(tf.matmul(tf.constant(helper_matrix, dtype=tf.float32), tmp_kernel))

        # list_of_matrices_out = []
        # for u in range(self._num_units):
        #     helper_matrix = np.zeros((self._num_units, self._num_units))
        #     helper_matrix[u,u] = 1
        #     list_of_matrices_out.append(tf.matmul(tmp_kernel_out, tf.constant(helper_matrix, dtype=tf.float32)))
        #     # helper_matrix = tf.constant()
        # self._kernel = tf.concat(list_of_matrices, axis=1)
        # self._kernel_out = tf.concat(list_of_matrices_out, axis=0)
        # # self._kernels = [lr_mult(rnn_lr_ph)(_ker) for _ker in self.__kernels]
        # self._kernels = self.__kernels
            # initializer=tf.initializers.identity(dtype=self.dtype))
        
        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        # print('inputs: ', inputs)
        # print('state: ', state)
        # exit()
        input_shape = tf.shape(inputs)
        # print('concatenated: ', array_ops.concat([inputs, state], 1))
        # print('weights: ', self._kernel)
        # _state = tf.Print(state, [state], "state: ", summarize=100)

        # split to columns coordinatewise
        inputs_arr = array_ops.split(inputs, self._num_units, 1)
        state_arr = array_ops.split(state, self._num_units, 1)
        # print(inputs_arr)
        # print(state_arr)
        output_arr = []
        for i, pair in enumerate(zip(inputs_arr, state_arr)):
            curr_input_and_state = array_ops.concat(pair, 1) 
            curr_input_and_state = tf.nn.dropout(curr_input_and_state, self.dropout_rate_ph)
            gate_inputs = math_ops.matmul(curr_input_and_state, self._kernel[i])
            gate_outputs = self._activation(gate_inputs)
            curr_output = math_ops.matmul(gate_outputs, self._kernel_out[i])
            output_arr.append(curr_output)
            # print(curr_output)
        
        # print('outputs:')
        # print(output_arr)
        output = tf.concat(output_arr, axis=1)
        # print(output)
        # exit()
        # _inputs = array_ops.reshape(inputs, [-1, 1])
        # _state = array_ops.reshape(state, [-1, 1])
        # _input_and_state = array_ops.concat([_inputs, _state], 1) 
        # _input_and_state = tf.nn.dropout(_input_and_state, self.dropout_rate_ph)
        # gate_inputs = math_ops.matmul(_input_and_state, self._kernel)

        # # input_and_state = array_ops.concat([inputs, state], 1)
        # # input_and_state = tf.nn.dropout(input_and_state, self.dropout_rate_ph)

        # # gate_inputs = math_ops.matmul(input_and_state, self._kernel)
        
        # # gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        # gate_outputs = self._activation(gate_inputs)
        
        # # this is not in the standard rnn cell and the reason we had to implement a new cell..
        # # gate_outputs = tf.nn.dropout(gate_outputs, self.dropout_rate_ph)
        # output = math_ops.matmul(gate_outputs, self._kernel_out)
        # output = tf.reshape(output, input_shape)

        # _inputs = output
        # for i in range(len(self._kernels)):
        #     _inputs = tf.nn.dropout(_inputs, self.dropout_rate_ph)
        #     _gate_inputs = math_ops.matmul(_inputs, self._kernels[i])
        #     if i < len(self._kernels)-1:
        #         #_inputs = self._activation(_gate_inputs)
        #         _inputs = tf.nn.leaky_relu(_gate_inputs)
        #     else:
        #         outputs = _gate_inputs
            
        
        # output = tf.maximum(inputs, state)
        return output, output
        # return _output, _output

    def _zero_state_tensors(self, state_size, batch_size, dtype):
        """Create tensors of zeros based on state_size, batch_size, and dtype."""
        def get_state_shape(s):
            """Combine s with batch_size to get a proper tensor shape."""
            c = _concat(batch_size, s)
            
            if self.initial_state == 'rand':
                size = tf.random_normal(c, stddev=1.0, dtype=dtype)
            elif self.initial_state == 'minus-inf':
                size = tf.constant(MINUS_INF) * array_ops.ones(c, dtype=dtype)
            elif self.initial_state == 'zeros':
                size = array_ops.zeros(c, dtype=dtype)

            if not context.executing_eagerly():
                c_static = _concat(batch_size, s, static=True)
                size.set_shape(c_static)
            return size

        return nest.map_structure(get_state_shape, state_size)

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          If `state_size` is an int or TensorShape, then the return value is a
          `N-D` tensor of shape `[batch_size, state_size]` filled with zeros.
          If `state_size` is a nested list or tuple, then the return value is
          a nested list or tuple (of the same structure) of `2-D` tensors with
          the shapes `[batch_size, s]` for each s in `state_size`.
        """
        # Try to use the last cached zero_state. This is done to avoid recreating
        # zeros, especially when eager execution is enabled.
        state_size = self.state_size
        # is_eager = context.executing_eagerly()
        # if is_eager and _hasattr(self, "_last_zero_state"):
        #     (last_state_size, last_batch_size, last_dtype,
        #         last_output) = getattr(self, "_last_zero_state")
        #     if (last_batch_size == batch_size and last_dtype == dtype and last_state_size == state_size):
        #         return last_output
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            output = self._zero_state_tensors(state_size, batch_size, dtype)
        # if is_eager:
        #     self._last_zero_state = (state_size, batch_size, dtype, output)
        return output
