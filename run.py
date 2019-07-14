import tensorflow as tf
import numpy as np
import argparse

from MNISTImageGenerator import MNISTImageGenerator
from DigitSequenceGenerator import DigitSequenceGenerator
from PointCloudGenerator import PointCloudGenerator
from SanDiskGenerator import SanDiskGenerator

import models

######################
# Required Arguments
######################
parser = argparse.ArgumentParser(description='Run set experiments.')
parser.add_argument('--model', required=True, type=str, choices=['CommRNN', 'DeepSet'],
                                help="model to use for experiment.")
parser.add_argument('--display_step', type=int, default=200)
parser.add_argument('--training_steps', required=True, type=int)
parser.add_argument('--batch_size', required=True, type=int)
parser.add_argument('--n_hidden_dim', required=True, type=int)
parser.add_argument('--experiment', required=True, type=str, choices=
                    ['pnt-cld', 'img-max', 'img-sum', 'dgt-max', 'dgt-sum', 'dgt-prty', 'san-disk'])

######################
# Optional Arguments
######################
parser.add_argument('--n_computation_dim', type=int, default=0)
parser.add_argument('--input_model_arch', type=str, default='[]')
parser.add_argument('--output_model_arch', type=str, default='[]')
parser.add_argument('--reg_coef', type=float, default=1e-1)
parser.add_argument('--lr_schedule', type=str, default="[(np.inf, 1e-3)]")
parser.add_argument('--aggregation_mode', type=str, choices=['sum', 'max'], default='sum')

args = parser.parse_args()

tf.reset_default_graph()


######################
# Parameters
######################
ARCHITECTURE = args.model
display_step = args.display_step
training_steps = args.training_steps
batch_size = args.batch_size
n_hidden_dim = args.n_hidden_dim


if args.experiment == 'pnt-cld':
    DataGenerator = PointCloudGenerator
    n_input_dim = 3
    n_output_dim = 40
    # change the down sample to perform other experiments.
    data_params = {'down_sample': 100} 
    seq_max_len = 100
    use_seqlen = False
    eval_on_varying_seq = False

elif args.experiment in ['img-max', 'img-sum']:
    DataGenerator = MNISTImageGenerator
    n_input_dim = 784
    n_output_dim = 1
    # mode is either max or sum.
    mode = args.experiment.split('-')[-1] # take the relevant mode.
    data_params = {'n_samples': 100000,
                   'max_seq_len': 10,
                   'min_seq_len': 1,
                   'mode': mode}
    seq_max_len = 10
    use_seqlen = True
    eval_on_varying_seq = True
    test_sequences = np.arange(5, 55, 5)
    num_test_examples = 500

elif args.experiment in ['dgt-max', 'dgt-sum', 'dgt-prty']:
    DataGenerator = DigitSequenceGenerator
    n_input_dim = 1
    n_output_dim = 2 if args.experiment == 'dgt-prty' else 1
    mode = args.experiment.split('-')[-1] # take the relevant mode.
    data_params = {'n_samples': 100000,
                   'max_seq_len': 10,
                   'min_seq_len': 1, # for parity this  has to be 1.
                   'mode': mode}
    seq_max_len = 10
    use_seqlen = True
    eval_on_varying_seq = True
    test_sequences = np.arange(10, 110, 10)
    num_test_examples = 500

elif args.experiment == 'san-disk':
    DataGenerator = SanDiskGenerator
    n_input_dim = 11
    n_output_dim = 2
    data_params = {} 
    seq_max_len = 50
    use_seqlen = True
    eval_on_varying_seq = False

n_computation_dim = args.n_computation_dim
if n_computation_dim == 0:
    n_computation_dim = n_hidden_dim

input_model_arch = [n_input_dim] + eval(args.input_model_arch) + [n_hidden_dim]
output_model_arch = [n_hidden_dim] + eval(args.output_model_arch) + [n_output_dim]

expct_comm_reg_weight = args.reg_coef
learning_rate_fn = models.create_lr_fn(eval(args.lr_schedule))


######################
# Common
######################
trainset = DataGenerator(**data_params)
testset = DataGenerator(**data_params, train=False)

# tf Graph input
x = tf.placeholder(tf.float32, [None, seq_max_len, n_input_dim])
y = tf.placeholder(tf.float32, [None, n_output_dim])
# A placeholder for indicating each sequence length
if use_seqlen:
    seqlen = tf.placeholder(tf.int32, [None])
else:
    seqlen = None
lr = tf.placeholder(tf.float32, [])

input_model_fn = models.create_model_fn(arch=input_model_arch, activation=tf.nn.tanh, reduce_max=False)
output_model_fn = models.create_model_fn(arch=output_model_arch, activation=tf.nn.tanh, disable_last_layer_activation=True, dropout=False)

if ARCHITECTURE == 'CommRNN':
    model = models.CommRNN(
                n_hidden_dim=n_hidden_dim,
                n_computation_dim=n_computation_dim,
                activation=tf.nn.relu,
                input_model_fn=input_model_fn,
                output_model_fn=output_model_fn)

    pred = model.build(x, seq_max_len, seqlen)
    commutative_regularization_term = model.build_reg()
elif ARCHITECTURE == 'DeepSet':
    model = models.DeepSet(
                input_dim=n_input_dim,
                input_model_fn=input_model_fn,
                output_model_fn=output_model_fn,
                aggregation_mode=args.aggregation_mode)
    pred = model.build(x, seq_max_len, seqlen)
    commutative_regularization_term = None
else:
    raise Exception("Unkown model, please select 'CommRNN' or 'DeepSet'.")


if n_output_dim == 1:
    loss = tf.losses.mean_squared_error(labels=y, predictions=pred)
    correct_pred = tf.equal(tf.round(pred), y)
else:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
    correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
# cost = loss + comm_reg_weight*empricial_regularization_loss + expct_comm_reg_weight*commutative_regularization_term
cost = loss if commutative_regularization_term is None else \
                loss + expct_comm_reg_weight*commutative_regularization_term
                    
# cost = loss + expct_comm_reg_weight*commutative_regularization_term
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
# optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

summary_ops = [cost, accuracy]
if not commutative_regularization_term is None:
    summary_ops += [commutative_regularization_term]

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    training_results = []
    for step in range(1, training_steps + 1):
        # batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        if use_seqlen:
            batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
            curr_feed_dict = {x: batch_x,
                              y: batch_y,
                              seqlen: batch_seqlen,
                              lr: learning_rate_fn(step)}
        else:
            batch_x, batch_y = trainset.next(batch_size)
            curr_feed_dict = {x: batch_x,
                              y: batch_y,
                              lr: learning_rate_fn(step)}

        
        # Run optimization op (backprop)
        _ = sess.run([train_op], feed_dict=curr_feed_dict)
        if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss

            _summary_ops =  sess.run(summary_ops, feed_dict=curr_feed_dict)
            # _pred, _y = sess.run([tf.round(pred), y], feed_dict=curr_feed_dict)
            # print('DEBUG pred: {}'.format(_pred))
            # print('DEBUG y: {}'.format(_y))
            summary_print = "Step " + str(step) + ", Minibatch Loss=" + \
                            "{:.6f}".format(_summary_ops[0]) + ", Training Accuracy=" + \
                            "{:.5f}".format(_summary_ops[1]) + \
                            ", learning rate={}".format(learning_rate_fn(step))

            if not commutative_regularization_term is None:
                summary_print += ", Regularization Loss=" + "{:.6f}".format(_summary_ops[2])
                            

            print(summary_print)
            # for i in range(len(_pred)):
            #   print('pred {}, target {}, sequence length {}'.format(_pred[i], batch_y[i], batch_seqlen[i]))

    print("Optimization Finished!")

    # Calculate accuracy
    if use_seqlen:
        test_data, test_label, test_seqlen = testset.next()
        test_feed_dict = {x: test_data, y: test_label, seqlen: test_seqlen}
    else:
        test_data, test_label = testset.next()
        test_feed_dict = {x: test_data, y: test_label}
    # test_seqlen = testset.seqlen
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict=test_feed_dict))


    if eval_on_varying_seq:
        for curr_seq_len in test_sequences:

            test_x_ph = tf.placeholder("float", [None, curr_seq_len, n_input_dim])
            test_y_ph = tf.placeholder("float", [None, n_output_dim])
            test_seqlen_ph = tf.placeholder(tf.int32, [None])

            test_pred = model.build(test_x_ph, curr_seq_len, test_seqlen_ph)
            data_params['n_samples'] = num_test_examples
            data_params['max_seq_len'] = curr_seq_len

            test_correct_pred = tf.equal(tf.round(test_pred), test_y_ph)
            test_accuracy = tf.reduce_mean(tf.cast(test_correct_pred, tf.float32))
            curr_testset = DataGenerator(**data_params, train=False)
            test_x, test_y, test_seqlen = curr_testset.next()
            curr_feed_dict = {test_x_ph: test_x,
                              test_y_ph: test_y,
                              test_seqlen_ph: test_seqlen}
            _test_accuracy = sess.run(test_accuracy, feed_dict = curr_feed_dict)
            print("Testing Accuracy for length {}: {}".format(curr_seq_len, _test_accuracy))
