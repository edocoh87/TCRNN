import numpy as np
import scipy.stats as stats
RAND_BOUND = .01

def get_init_arrays(
            netowrk_type,
            initialization_scheme,
            input_depth,
            computation_dim,
            num_units):
    if netowrk_type == 'dense':
        if initialization_scheme == 'max':
            assert input_depth == num_units, 'input_depth must be equal to _num_units.'
            assert computation_dim[0] >= 3*num_units, '_computation_dim[0] must be at least x3 larger than _num_units to implement maximum.'
            assert computation_dim[1] >= num_units, '_computation_dim[1] must be at least the size of _num_unit to implement maximum.'
            
            kernel_init_arr = np.zeros((input_depth + num_units, computation_dim[0]))
            for i in range(num_units):
                kernel_init_arr[i, 3*i] = 1
                kernel_init_arr[i+num_units, 3*i] = -1
                kernel_init_arr[i+num_units, 3*i+1] = 1
                kernel_init_arr[i+num_units, 3*i+2] = -1
            
            kernel_out_init_arr = np.zeros((computation_dim[0], computation_dim[1]))
            for i in range(num_units):
                kernel_out_init_arr[3*i, i] = 1
                kernel_out_init_arr[3*i+1, i] = 1
                kernel_out_init_arr[3*i+2, i] = -1

        elif initialization_scheme == 'sum':
            print('initializing transition matrix to sum')
            assert input_depth == num_units, 'input_depth must be equal to _num_units.'
            assert computation_dim[0] >= 2*num_units, '_computation_dim[0] must be at least x2 larger than _num_units to implement summation.'
            assert computation_dim[1] >= num_units, '_computation_dim[1] must be at least the size of _num_unit to implement summation.'

            kernel_init_arr = np.random.normal(scale=RAND_BOUND, size=(input_depth + num_units, computation_dim[0]))
            for i in range(num_units):
                kernel_init_arr[i, 2*i] = 1
                kernel_init_arr[i, 2*i+1] = -1
                kernel_init_arr[i+num_units, 2*i] = 1
                kernel_init_arr[i+num_units, 2*i+1] = -1

            kernel_out_init_arr = np.random.normal(scale=RAND_BOUND, size=(computation_dim[0], computation_dim[1]))
            for i in range(num_units):
                kernel_out_init_arr[2*i, i] = 1
                kernel_out_init_arr[2*i+1, i] = -1


        elif initialization_scheme == 'rand':
            kernel_init_arr = stats.truncnorm.rvs(-2*RAND_BOUND, 2*RAND_BOUND, scale=RAND_BOUND, size=(input_depth + num_units, computation_dim[0]))
            kernel_out_init_arr = stats.truncnorm.rvs(-2*RAND_BOUND, 2*RAND_BOUND, scale=RAND_BOUND, size=(computation_dim[0], computation_dim[1]))
            
        else:
            raise ValueError("unkonwn initialization scheme.")

        return kernel_init_arr, kernel_out_init_arr

    if netowrk_type == 'sparse':
        # do stuff...
        raise ValueError('not implemented yet.')
        # return kernel_init_arr, kernel_out_init_arr, kernels_init_arr



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

def _zero_state_tensors(self, state_size, batch_size, dtype):
    MINUS_INF = -1e4
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
