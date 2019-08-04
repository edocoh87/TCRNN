import tensorflow as tf
import numpy as np
import argparse
from pdb import set_trace as trace
from glob import glob
import multiprocessing

from MNISTImageGenerator import MNISTImageGenerator
from DigitSequenceGenerator import DigitSequenceGenerator
from PointCloudGenerator import PointCloudGenerator
from SanDiskGenerator import SanDiskGenerator

import models
from utils import *
from SDFeedDictGenerator import FeedDictGenerator
from create_queue import *

######################
# Required Arguments
######################
parser = argparse.ArgumentParser(description='Run set experiments.')
parser.add_argument('--model', required=True, type=str, choices=['CommRNN', 'DeepSet'],
                                help="model to use for experiment.")
parser.add_argument('--training_steps', required=True, type=int)
parser.add_argument('--batch_size', required=True, type=int)
parser.add_argument('--n_hidden_dim', required=True, type=int)
parser.add_argument('--experiment', required=True, type=str, choices=
                    ['pnt-cld', 'img-max', 'img-sum', 'dgt-max', 'dgt-sum', 'dgt-prty', 'san-disk'])

######################
# Optional Arguments
######################
parser.add_argument('--display_step', type=int, default=200)
parser.add_argument('--val_display_step', type=int, default=1000)
parser.add_argument('--exp_name', type=str, default='unnamed_exp')
parser.add_argument('--n_computation_dim', type=int, default=0)
parser.add_argument('--input_model_arch', type=str, default='[]')
parser.add_argument('--output_model_arch', type=str, default='[]')
parser.add_argument('--reg_coef', type=float, default=1e-1)
parser.add_argument('--lr_schedule', type=str, default="[(np.inf, 1e-3)]")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--num_val_samples", type=int, default=10000)
parser.add_argument("--take_last_k_cycles", type=int, default=-1)
parser.add_argument("--oversample_pos", action="store_true")
parser.add_argument("--ignore_string_loc", action="store_true")
parser.add_argument("--concat_all_cycles", action="store_true")
parser.add_argument("--fpr", type=float, default=0.09)
parser.add_argument("--num_test_samples", type=int, default=10000)
parser.add_argument("--dropout_rate", type=float, default=0)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--neg_weights", type=float, default=1.0)

args = parser.parse_args()
if args.debug:
    args.num_workers = 1
tf.reset_default_graph()


######################
# Parameters
######################
ARCHITECTURE = args.model
display_step = args.display_step
val_display_step = args.val_display_step
training_steps = args.training_steps
batch_size = args.batch_size
n_hidden_dim = args.n_hidden_dim

pos_files_path = lambda dataset: glob('/specific/netapp5_2/gamir/achiya/Sandisk/new_data/PC3/fails/{0}/phase*.csv'
                                      .format(dataset))

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
    FILES_PATH = '/specific/netapp5_2/gamir/achiya/Sandisk/new_data/PC3/split/'
    n_features = 11
    train_files = glob(FILES_PATH + 'train/DUT*/*.csv')
    val_files = glob(FILES_PATH + 'val/DUT*/*.csv')
    test_files = glob(FILES_PATH + 'test/DUT*/*.csv')

    train_batches_queue, train_workers = create_queue_and_workers(args.debug, train_files, batch_size, n_features,
                                                                  pos_files_path('train'), args.oversample_pos,
                                                                  args.ignore_string_loc, args.take_last_k_cycles,
                                                                  args.num_workers, pos_replacement=True)

    val_batches_queue, val_workers = create_queue_and_workers(args.debug, val_files, batch_size, n_features,
                                                              pos_files_path('val'), args.oversample_pos,
                                                              args.ignore_string_loc, args.take_last_k_cycles,
                                                              args.num_workers, pos_replacement=False)

    DataGenerator = SanDiskGenerator
    n_input_dim = 11
    if not args.ignore_string_loc:
        n_input_dim += 11893
    n_output_dim = 2
    data_params = {
        'debug_mode': args.debug,
        'take_last_k_cycles': args.take_last_k_cycles
    }
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
# tf Graph input
if args.take_last_k_cycles == -1:
    x = tf.placeholder(tf.float32, [None, seq_max_len, n_input_dim])
else:
    x = tf.placeholder(tf.float32, [None, args.take_last_k_cycles, n_input_dim])
y = tf.placeholder(tf.float32, [None, n_output_dim])
lr = tf.placeholder(tf.float32, [])
is_train = tf.placeholder(tf.bool, [])
# A placeholder for indicating each sequence length
seqlen = None
if use_seqlen:
    seqlen = tf.placeholder(tf.int32, [None])

placeholders = (x, y, lr, is_train, seqlen)

input_model_fn = models.create_model_fn(arch=input_model_arch, dropout_rate=args.dropout_rate, activation=tf.nn.tanh)
output_model_fn = models.create_model_fn(arch=output_model_arch, dropout_rate=args.dropout_rate, activation=tf.nn.tanh,
                                         disable_last_layer_activation=True)

if ARCHITECTURE == 'CommRNN':
    model = models.CommRNN(
                n_hidden_dim=n_hidden_dim,
                n_computation_dim=n_computation_dim,
                activation=tf.nn.relu,
                input_model_fn=input_model_fn,
                output_model_fn=output_model_fn)

    pred = model.build(x, seq_max_len if args.take_last_k_cycles == -1 else args.take_last_k_cycles, is_train, seqlen)
    commutative_regularization_term = model.build_reg()
elif ARCHITECTURE == 'DeepSet':
    model = models.DeepSet(
                input_dim=n_input_dim,
                input_model_fn=input_model_fn,
                output_model_fn=output_model_fn,
                aggregation_mode='sum')
    pred = model.build(x, seq_max_len if args.take_last_k_cycles == -1 else args.take_last_k_cycles, is_train, seqlen)
    commutative_regularization_term = None
else:
    raise Exception("Unkown model, please select 'CommRNN' or 'DeepSet'.")

with_reg_loss = commutative_regularization_term is not None

if n_output_dim == 1:
    loss = tf.losses.mean_squared_error(labels=y, predictions=pred)
    correct_pred = tf.equal(tf.round(pred), y)
else:
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=pred,
                                                                   pos_weight=1.0 / args.neg_weights))
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
saver = tf.train.Saver()

TrainGenerator = FeedDictGenerator(train_batches_queue, placeholders, use_seqlen, learning_rate_fn, 'Train')
ValGenerator = FeedDictGenerator(val_batches_queue, placeholders, use_seqlen, learning_rate_fn, 'Val')

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    training_results = []
    for step in range(1, training_steps + 1):
        curr_feed_dict = TrainGenerator.create_feed_dict()
        # Run optimization op (backprop)
        _ = sess.run([train_op], feed_dict=curr_feed_dict)
        if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss
            print_stats_from_feeddict(sess, placeholders, (summary_ops, ), curr_feed_dict, with_reg_loss,
                                      curr_feed_dict[y], step, args.fpr)
        if step % val_display_step == 0:
            print_stats_from_generator(sess, (summary_ops, pred), with_reg_loss, ValGenerator, 10000, step, args.fpr,
                                       dset='val', num_workers=args.num_workers)
            val_batches_queue, val_workers = \
                create_queue_and_workers(args.debug, val_files, batch_size, n_features, pos_files_path('val'),
                                         args.oversample_pos, args.ignore_string_loc, args.take_last_k_cycles,
                                         pos_replacement=False, prev_workers=val_workers, num_workers=args.num_workers)
            ValGenerator.set_queue(val_batches_queue)

    print("Optimization Finished!")
    save_path = 'models/{}'.format(args.exp_name)
    os.makedirs(save_path, exist_ok=True)
    ckpt_save_path = saver.save(sess, os.path.join(save_path, 'model.ckpt'))
    print("Saved model to file: {}".format(ckpt_save_path))

    test_batches_queue, final_val_workers = \
        create_queue_and_workers(args.debug, test_files, batch_size, n_features, pos_files_path('test'),
                                 args.oversample_pos, args.ignore_string_loc, args.take_last_k_cycles,
                                 pos_replacement=False, prev_workers=train_workers + val_workers,
                                 num_workers=args.num_workers)

    TestGenerator = FeedDictGenerator(test_batches_queue, placeholders, use_seqlen, lambda x: 0, 'FinalVal')
    print_stats_from_generator(sess, (summary_ops, pred), with_reg_loss, TestGenerator, args.num_test_samples, 0,
                               fpr=args.fpr, dset='test', num_workers=args.num_workers, plot_roc=True,
                               roc_save_path=save_path)

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
