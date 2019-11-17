import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import json

from MNISTImageGenerator import MNISTImageGenerator
from DigitSequenceGenerator import DigitSequenceGenerator
from AnomalySequenceGenerator import AnomalySequenceGenerator
from PointCloudGenerator import PointCloudGenerator
from SanDiskGenerator import SanDiskGenerator
# from CelebAGenerator import CelebAGenerator

import models

#np.random.seed(0)

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
                    ['pnt-cld', 'img-max', 'img-sum', 'dgt-max', 'dgt-sum', 'dgt-prty', 'dgt-anmly', 'san-disk', 'celeba'])

######################
# Optional Arguments
######################
parser.add_argument('--num_of_runs', type=int, default=1)
parser.add_argument('--display_step', type=int, default=200)
parser.add_argument('--n_computation_dim', type=str, default=None)
#parser.add_argument('--initialize_to_max', action='store_true')
parser.add_argument('--initialization_scheme', type=str, choices=['max', 'sum', 'rand'], default='rand')
parser.add_argument('--weight_config', type=str, choices=['dense', 'sparse', 'shared'], default='dense')
parser.add_argument('--initial_state', type=str, choices=['rand', 'minus-inf', 'zeros'], default='rand')
parser.add_argument('--non_trainable_rnn', action='store_false')
parser.add_argument('--save_model_to_path', type=str, default=None)
parser.add_argument('--restore_from_path', type=str, default=None)
# parser.add_argument('--trainable_rnn', type=bool, default=True)
parser.add_argument('--input_model_arch', type=str, default='[]')
parser.add_argument('--output_model_arch', type=str, default='[]')
parser.add_argument('--reg_coef', type=float, default=1e-1)
parser.add_argument('--l1_coef', type=float, default=0.0)
parser.add_argument('--l2_coef', type=float, default=0.0)
parser.add_argument('--lr_schedule', type=str, default="[(np.inf, (1e-3, 1.0))]")
parser.add_argument('--aggregation_mode', type=str, choices=['sum', 'max'], default='sum',
            help="the aggregation mode to use (relevant only for DeepSet architecture.")
parser.add_argument('--log_dir', type=str, default='logs/')
parser.add_argument('--input_dropout_rate', type=float, default=0.0,
            help="the dropout rate to use in the input model (the default value of 0 will result in no dropout).")
parser.add_argument('--output_dropout_rate', type=float, default=0.0,
            help="the dropout rate to use in the output model (the default value of 0 will result in no dropout).")
parser.add_argument('--rnn_dropout_rate', type=float, default=0.0,
            help="the dropout rate to use in the rnn model (the default value of 0 will result in no dropout).")

args = parser.parse_args()

def run(print_log):

    now = datetime.datetime.now()
    dir_name = now.strftime("%Y_%m_%d-%H:%M:%S")

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
        n_output_dim = 2 if args.experiment in ['dgt-prty'] else 1
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

    elif args.experiment == 'dgt-anmly':
        DataGenerator = AnomalySequenceGenerator
        n_input_dim = 1
        n_output_dim = 2
        data_params = {'n_samples': 100000,
                       'max_seq_len': 40,
                       'min_seq_len': 10,
                       'mode': 'anmly'}
        seq_max_len = 40
        use_seqlen = True
        eval_on_varying_seq = True
        test_sequences = np.arange(30, 70, 10)
        num_test_examples = 500

    elif args.experiment == 'san-disk':
        DataGenerator = SanDiskGenerator
        n_input_dim = 11
        n_output_dim = 2
        data_params = {} 
        seq_max_len = 50
        use_seqlen = True
        eval_on_varying_seq = False

    elif args.experiment == 'celeba':
        DataGenerator = CelebAGenerator
        n_input_dim = 218*178
        n_output_dim = 7
        data_params = {}
        seq_max_len = 7
        use_seqlen = False
        eval_on_varying_seq = False

    #eval_on_varying_seq = False

    if args.n_computation_dim is None:
        n_computation_dim = [n_hidden_dim]
    else:
        n_computation_dim = eval(args.n_computation_dim)


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
    rnn_lr_ph = tf.placeholder(tf.float32, shape=[], name='rnn_lr_ph')

    input_dropout_rate_ph = tf.placeholder_with_default(1-args.input_dropout_rate, shape=(), name='input_dropout_rate_ph')
    output_dropout_rate_ph = tf.placeholder_with_default(1-args.output_dropout_rate, shape=(), name='output_dropout_rate_ph')
    rnn_dropout_rate_ph = tf.placeholder_with_default(1-args.rnn_dropout_rate, shape=(), name='rnn_dropout_rate_ph')

    input_model_fn = models.create_input_fn(arch=input_model_arch, activation=tf.nn.tanh, reduce_max=True,
                                    dropout_rate_ph=input_dropout_rate_ph)
    output_model_fn = models.create_output_fn(arch=output_model_arch, activation=tf.nn.tanh,
                                    disable_last_layer_activation=True, dropout_rate_ph=output_dropout_rate_ph)

    if ARCHITECTURE == 'CommRNN':
        model = models.CommRNN(
                    n_hidden_dim=n_hidden_dim,
                    n_computation_dim=n_computation_dim,
                    dropout_rate_ph=rnn_dropout_rate_ph,
                    trainable=args.non_trainable_rnn,
                    initialization_scheme=args.initialization_scheme,
                    weight_config=args.weight_config,
                    initial_state=args.initial_state,
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
                        

    # set L1 regularization on RNN weights.
    rnn_weights = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="rnn/commutative_rn_ncell")
    if (args.l1_coef + args.l2_coef > 0):
        norm_regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=args.l1_coef, scale_l2=args.l2_coef)
        norm_regularization_penalty = tf.contrib.layers.apply_regularization(norm_regularizer, rnn_weights)
        cost += norm_regularization_penalty
    else:
        norm_regularization_penalty = None

    # cost = loss + expct_comm_reg_weight*commutative_regularization_term
    # train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-3)
    #optimizer = tf.contrib.opt.AdamWOptimizer(
    #                                weight_decay=1e-7,
    #                                epsilon=1e-3,
    #                                learning_rate=lr)
                                #).minimize(cost)
    gradients, variables = zip(*optimizer.compute_gradients(cost))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables))

    # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

    # Evaluate model

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    val_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Cost', cost)
    tf.summary.scalar('Train Accuracy', accuracy)
    tf.summary.scalar('Validation Accuracy', val_accuracy)
    tf.summary.scalar('Learning Rate', lr)

    summary_ops = [cost, accuracy]
    if commutative_regularization_term is not None:
        summary_ops += [commutative_regularization_term]
        tf.summary.scalar('regularization loss', commutative_regularization_term)

    if norm_regularization_penalty is not None:
        summary_ops += [norm_regularization_penalty]

    merged = tf.summary.merge_all()
    summary_ops = summary_ops + [merged]
    # making directory to save tensorboard files.
    log_dir_path = os.path.join(args.log_dir, dir_name)

    if os.path.exists(log_dir_path):
        log_dir_path += '_2'
    os.makedirs(log_dir_path)

    #os.mkdir(log_dir_path)
    with open(os.path.join(log_dir_path, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:
        tf.set_random_seed(0)
        sess.run(init)
        if args.restore_from_path is not None:
            saver.restore(sess, args.restore_from_path)
            print_log("Model restored from: '{}'".format(args.restore_from_path))
        train_writer = tf.summary.FileWriter(log_dir_path, sess.graph)
        # Run the initializer
        training_results = []
        for step in range(1, training_steps + 1):
            # batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
            if use_seqlen:
                batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
                curr_feed_dict = { seqlen: batch_seqlen }
            else:
                batch_x, batch_y = trainset.next(batch_size)
                curr_feed_dict = { }
            
            
            _lr, _rnn_lr_mlt = learning_rate_fn(step)
            curr_feed_dict.update({ x: batch_x,
                                    y: batch_y,
                                    lr: _lr,
                                    rnn_lr_ph: _rnn_lr_mlt, })
                                    # rnn_lr_ph: 0.0, })
            
            
            
            # Run optimization op (backprop)
            _ = sess.run([train_op], feed_dict=curr_feed_dict)
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss

                _summary_ops =  sess.run(summary_ops, feed_dict=curr_feed_dict)
                # _pred, _y = sess.run([tf.round(pred), y], feed_dict=curr_feed_dict)
                # print('DEBUG pred: {}'.format(_pred))
                # print('DEBUG y: {}'.format(_y))
                train_writer.add_summary(_summary_ops[-1], step)
                summary_print = "Step " + str(step) + ", Minibatch Loss=" + \
                                "{:.6f}".format(_summary_ops[0]) + ", Training Accuracy=" + \
                                "{:.5f}".format(_summary_ops[1]) + \
                                ", lr={}".format((_lr, _rnn_lr_mlt))

                if commutative_regularization_term is not None:
                    summary_print += ", Regularization Loss=" + "{:.6f}".format(_summary_ops[2])
                
                if norm_regularization_penalty is not None:
                    summary_print += ", Norm Loss=" + "{:.6f}".format(_summary_ops[3])
                
                val_data = trainset.get_validation()
                if val_data is not None:
                    val_acc = sess.run(val_accuracy, feed_dict = {x: val_data[0],
                                                                  y: val_data[1],
                                                                  input_dropout_rate_ph: 1.0,
                                                                  output_dropout_rate_ph: 1.0,
                                                                  rnn_dropout_rate_ph: 1.0,})
                    summary_print += ", Validation Accuracy={:.5f}".format(val_acc)

                    pred1 = sess.run(pred, feed_dict = {x: val_data[0],
                                                        y: val_data[1],
                                                        input_dropout_rate_ph: 1.0,
                                                        output_dropout_rate_ph: 1.0,
                                                        rnn_dropout_rate_ph: 1.0,})
                    shuffled_data = val_data[0]
                    for i in range(len(shuffled_data)):
                        curr_perm = np.random.permutation(len(shuffled_data[i]))
                        shuffled_data[i] = shuffled_data[i][curr_perm]

                    shuffled_pred = sess.run(pred, feed_dict = {x: shuffled_data,
                                                                y: val_data[1],
                                                                input_dropout_rate_ph: 1.0,
                                                                output_dropout_rate_ph: 1.0,
                                                                rnn_dropout_rate_ph: 1.0,})

                    mse = ((pred1 - shuffled_pred)**2).mean()
                    summary_print += ", MSE on shuffled val data={:.5f}".format(mse)
                    #summary_print += ", Validation Accuracy=" + "{:.5f}".format(val_acc)
                print_log(summary_print)
                # for i in range(len(_pred)):
                #   print('pred {}, target {}, sequence length {}'.format(_pred[i], batch_y[i], batch_seqlen[i]))

        train_writer.close()
        print_log("Optimization Finished!")
        
        if args.save_model_to_path is not None:
            save_path = saver.save(sess, args.save_model_to_path)
            print_log("Model saved in path: {}".format(save_path))

        # Calculate accuracy
        if use_seqlen:
            test_data, test_label, test_seqlen = testset.next()
            test_feed_dict = {x: test_data,
                              y: test_label,
                              seqlen: test_seqlen}
        else:
            test_data, test_label = testset.next()
            test_feed_dict = {x: test_data, y: test_label}
        # test_seqlen = testset.seqlen
        test_feed_dict.update({ input_dropout_rate_ph: 1.0,
                                output_dropout_rate_ph: 1.0,
                                rnn_dropout_rate_ph: 1.0 })
        _accuracy = sess.run(accuracy, feed_dict=test_feed_dict)
        print_log("Testing Accuracy: {}".format(_accuracy))


        if eval_on_varying_seq:
            return_acc = {}
            for curr_seq_len in test_sequences:

                test_x_ph = tf.placeholder("float", [None, curr_seq_len, n_input_dim])
                test_y_ph = tf.placeholder("float", [None, n_output_dim])
                test_seqlen_ph = tf.placeholder(tf.int32, [None])

                test_pred = model.build(test_x_ph, curr_seq_len, test_seqlen_ph)
                data_params['n_samples'] = num_test_examples
                data_params['max_seq_len'] = curr_seq_len

                #test_correct_pred = tf.equal(tf.round(test_pred), test_y_ph)
                if n_output_dim == 1:
                    test_correct_pred = tf.equal(tf.round(test_pred), test_y_ph)
                else:
                    test_correct_pred = tf.equal(tf.argmax(test_pred, axis=1), tf.argmax(test_y_ph, axis=1))
                
                test_accuracy = tf.reduce_mean(tf.cast(test_correct_pred, tf.float32))
                curr_testset = DataGenerator(**data_params, train=False)
                test_x, test_y, test_seqlen = curr_testset.next()
                curr_feed_dict = {test_x_ph: test_x,
                                  test_y_ph: test_y,
                                  seqlen: test_seqlen,
                                  test_seqlen_ph: test_seqlen}
                _test_accuracy = sess.run(test_accuracy, feed_dict = curr_feed_dict)
                return_acc['acc_{}'.format(curr_seq_len)] = _test_accuracy
                print_log("Testing Accuracy for length {}: {}".format(curr_seq_len, _test_accuracy))
            return return_acc

if __name__ == '__main__':
    def print_func(s):
        blank = ' '*200
        print(blank, end='\r', flush=True)
        print(s, end='\r', flush=True)


    listify = lambda x: x if isinstance(x, list) else [x]
    def merge_dicts(d1, d2):
        if d1 == None:
            return d2
        assert d1.keys() == d2.keys(), 'dictionaries should have the same set of keys.'
        d3 = {}
        for k in d1.keys():
            d3[k] = listify(d1[k]) + listify(d2[k])
        return d3

    def print_results(d):
        for k in d.keys():
            var = np.var(d[k])
            mean = np.mean(d[k])
            print('{}: mean {}, variance {}'.format(k, mean, var))

    avg_acc_dict = None
    for r in range(args.num_of_runs):
        curr_acc_dict = run(print_func)
        avg_acc_dict = merge_dicts(avg_acc_dict, curr_acc_dict)
    print_results(avg_acc_dict)