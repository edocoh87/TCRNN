import numpy as np
import tensorflow as tf
from functools import partial

from CommutativeRNNcell import CommutativeRNNcell
# from SparseCommutativeRNNcell import CommutativeRNNcell
import commutative_regularizer

def create_lr_fn(schedule):
    def lr_fn(step):
        for s in schedule:
            if step <= s[0]:
                return s[1]
        raise ValueError("Learning rate schedular is't defined for step {}".format(step))
    
    return lr_fn

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / np.sqrt(sum(shape)))
    #return tf.truncated_normal(shape=shape, stddev=1. / np.sqrt(sum(shape)))

linear_activation = lambda x: x

def create_input_fn(arch, dropout_rate_ph, activation=tf.nn.tanh, reduce_max=False):
    weights = dict(
                [('w{}'.format(i), tf.Variable(glorot_init([arch[i+1], arch[i]]))) for i in range(len(arch)-1)]
    )
    biases = dict(
                #[('b{}'.format(i), tf.Variable(glorot_init([arch[i+1]]))) for i in range(len(arch)-1)]
                [('b{}'.format(i), tf.Variable(tf.zeros([arch[i+1]]))) for i in range(len(arch)-1)]
    )
    #input_dropout_rate_ph = tf.placeholder_with_default(dropout, shape=(), name='input_dropout_rate_ph')
    def _input_fn(x):
        set_size = None
        try:
            _btch, set_size, _input_dim = x.get_shape().as_list()
        except:
            raise ValueError('expected input is 3 dimensional: [batch, max_seq_len, input_dim].')

        for i in range(len(arch)-1):
            if reduce_max:
                xm = tf.reduce_max(x, axis=1, keepdims=True)
                x -= xm
            x = tf.reshape(x, [-1, arch[i]])
            #if dropout > 0:
            x = tf.nn.dropout(x, dropout_rate_ph)
            x = tf.matmul(x, weights['w{}'.format(i)], transpose_b=True) + biases['b{}'.format(i)]
            x = tf.reshape(x, [-1, set_size, arch[i+1]])
            x = activation(x)
        return x

    return _input_fn

def create_output_fn(arch, dropout_rate_ph, activation=tf.nn.tanh, disable_last_layer_activation=False):
    weights = dict(
                [('w{}'.format(i), tf.Variable(glorot_init([arch[i+1], arch[i]]))) for i in range(len(arch)-1)]
    )
    biases = dict(
                #[('b{}'.format(i), tf.Variable(glorot_init([arch[i+1]]))) for i in range(len(arch)-1)]
                [('b{}'.format(i), tf.Variable(tf.zeros([arch[i+1]]))) for i in range(len(arch)-1)]
    )
    #output_dropout_rate_ph = tf.placeholder_with_default(dropout, shape=(), name='output_dropout_rate_ph')
    def _output_fn(x):
        for i in range(len(arch)-1):
            #if dropout > 0:
            x = tf.nn.dropout(x, dropout_rate_ph)
            x = tf.matmul(x, weights['w{}'.format(i)], transpose_b=True) + biases['b{}'.format(i)]
            # wheather the last layer is linear or not.
            curr_acivation = linear_activation if \
                                (i == len(arch)-2 and disable_last_layer_activation) else activation
            x = curr_acivation(x)
        return x

    return _output_fn


def deepset_model(x, seqlen, input_model_fn, output_model_fn, seq_max_len, input_dim):
    # stack the sequence and batch into one axis to create a matrix [batch_size*n_steps, input_dim]
    pre_shp = tf.shape(x)
    x = tf.reshape(x, [-1, input_dim])
    x = input_model_fn(x)
    post_shp = tf.shape(x)
    mask = tf.sequence_mask(seqlen, maxlen=seq_max_len, dtype=tf.float32)
    mask = tf.reshape(mask, [-1, 1])

    # mask out the irrelevant indices
    x = mask*x
    x = tf.reshape(x, [pre_shp[0], pre_shp[1], post_shp[-1]])

    # aggregate each sequence
    x_pooled = tf.reduce_sum(x, axis=1)
    # perform the post aggregation function
    outputs = output_model_fn(x_pooled)
    return outputs



class CommRNN(object):
    def __init__(self,
                 n_hidden_dim,
                 n_computation_dim,
                 dropout_rate_ph,
                 initial_state,
                 initialization_scheme,
                 trainable,
                 activation,
                 input_model_fn,
                 output_model_fn):
        
        self.n_hidden_dim = n_hidden_dim
        self.n_computation_dim = n_computation_dim
        self.dropout_rate_ph = dropout_rate_ph
        self.initial_state = initial_state
        self.initialization_scheme = initialization_scheme 
        self.trainable = trainable
        self.activation = activation
        self.input_model_fn = input_model_fn
        self.output_model_fn = output_model_fn
        self.rnn_cell = CommutativeRNNcell(
                                num_units = self.n_hidden_dim,
                                computation_dim = self.n_computation_dim,
                                dropout_rate_ph = self.dropout_rate_ph,
                                initial_state = self.initial_state,
                                initialization_scheme = self.initialization_scheme,
                                trainable = self.trainable,
                                activation = self.activation)

    def build_rnn(self, x, seq_max_len, seqlen=None):
        # def dynamicRNN(x, seqlen, cell, input_model_fn, output_model_fn, seq_max_len, n_hidden, initial_state=None):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # print(x) 
        x_transformed = self.input_model_fn(x)
        # print(x_transformed)
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        #x = tf.unstack(x, seq_max_len, 1)
        x = tf.unstack(x_transformed, seq_max_len, 1)
        # # x_transformed = [tf.matmul(_x, tf.transpose(weights['in'])) for _x in x]
        #x_transformed = [self.input_model_fn(_x) for _x in x]
        # pre_shp = tf.shape(x)
        # x_transformed = self.input_model_fn(x)
        # post_shp = tf.shape(x)
        # x_transformed = tf.reshape(x_transformed, [pre_shp[0], pre_shp[1], post_shp[-1]])
        # print(x_transformed)
        # exit()

        # x_transformed = [tf.matmul(_x, weights['in']) for _x in x]
        # Get rnn cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        # outputs, states = tf.nn.dynamic_rnn(
        outputs, states = tf.contrib.rnn.static_rnn(
                                          self.rnn_cell,
                                          #x[1:],
                                          x,
                                          dtype=tf.float32,
                                          sequence_length=seqlen,
                                          initial_state=None)
                                          #initial_state=x[0])
        if seqlen is not None:
            # When performing dynamic calculation, we must retrieve the last
            # dynamically computed output, i.e., if a sequence length is 10, we need
            # to retrieve the 10th output.
            # However TensorFlow doesn't support advanced indexing yet, so we build
            # a custom op that for each sample in batch size, get its length and
            # get the corresponding relevant output.

            # 'outputs' is a list of output at every timestep, we pack them in a Tensor
            # and change back dimension to [batch_size, n_step, n_input]
            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])

            # Hack to build the indexing and retrieve the right output.
            batch_size = tf.shape(outputs)[0]
            # Start indices for each sample
            index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
            # Indexing
            outputs = tf.gather(tf.reshape(outputs, [-1, self.n_hidden_dim]), index)
            # Linear activation, using outputs computed above
            # return outputs, tf.reduce_sum(outputs, axis=-1, keepdims=True)
            # return outputs, tf.matmul(outputs, weights['out']) #+ biases['out']
        else:
            # if seqlen is not provided, the length is fixed and there is no need
            # for dynamic calculations, the sequence is always of length seq_max_len
            outputs = outputs[-1]

        outputs = self.output_model_fn(outputs)
        return outputs
        
    def build(self, x, seq_max_len, seqlen=None):
        return self.build_rnn(x, seq_max_len, seqlen)

    def build_reg(self):
        self.comm_reg = self.rnn_cell.get_comm_regularizer()
        return self.comm_reg



class DeepSet(object):
    def __init__(self,
                 input_dim,
                 input_model_fn,
                 output_model_fn,
                 aggregation_mode='sum'):
        
        self.input_dim = input_dim
        self.input_model_fn = input_model_fn
        self.output_model_fn = output_model_fn
        
        self.aggregation_mode = aggregation_mode
        if self.aggregation_mode == 'sum':
            self.agg_fn = tf.reduce_sum
        elif self.aggregation_mode == 'max':
            self.agg_fn = tf.reduce_max


    def build(self, x, seq_max_len, seqlen=None):
        # stack the sequence and batch into one axis to create a matrix [batch_size*n_steps, input_dim]
        pre_shp = tf.shape(x)
        # x = tf.reshape(x, [-1, self.input_dim])
        x = self.input_model_fn(x)
        post_shp = tf.shape(x)
        x = tf.reshape(x, [-1, post_shp[-1]])
                
        if seqlen is not None:
            mask = tf.sequence_mask(seqlen, maxlen=seq_max_len, dtype=tf.float32)
            mask = tf.reshape(mask, [-1, 1])

            # mask out the irrelevant indices
            x = mask*x

        x = tf.reshape(x, [pre_shp[0], pre_shp[1], post_shp[-1]])
        # aggregate each sequence
        x_pooled = self.agg_fn(x, axis=1)
        # perform the post aggregation function
        outputs = self.output_model_fn(x_pooled)
        return outputs
