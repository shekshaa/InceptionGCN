from __future__ import division
from __future__ import print_function

import time

from utils import *
from models import GCN, MLP, ResGCN
from visualize import *
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import csv

# Set random seed
seed = 123
np.random.seed(seed)
# tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('adj_type', 'age', 'Adjacency matrix creation')

# 'cora', 'citeseer', 'pubmed', 'tadpole' # Please don't work with citation networks!!
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')

# 'gcn(re-parametrization trick)', 'gcn_cheby(simple_gcn)', 'dense', 'res_gcn_cheby(our model)'
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 110, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 50, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 25, 'Number of units in hidden layer 3.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 25, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_bool('featureless', False, 'featureless')


# Loading data
if FLAGS.dataset == 'tadpole':
    # features is in sparse format
    # node weights used for weighted loss
    adj, features, all_labels, one_hot_labels, node_weights, dense_features = load_tadpole_data(FLAGS.adj_type)

else:
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, all_labels, one_hot_labels = load_citation_data(FLAGS.dataset)
    adj, features, all_labels, one_hot_labels = load_citation_data(FLAGS.dataset)
    dense_features = features
    node_weights = np.ones((dense_features.shape[0],))
    # Some preprocessing
    features = preprocess_features(features)


# creating placeholders and support based on number of supports fed to network
def create_support_placeholder(num_supports):
    # num_supports = locality_upper_bound + 1
    support = chebyshev_polynomials(adj, num_supports - 1)
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32, name='support_{}'.format(i)) for i in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, one_hot_labels.shape[1])),
        'labels_mask': tf.placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    return support, placeholders


# Table experiment for simple gcn on different localities
def table_experiment(locality_upper_bound):
    """locality_upper_bound: up to what degree looking for degree size of each layer"""
    with open('Average_accuracy.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['K1', 'K2', 'train_avg_acc', 'val_avg_acc', 'test_avg_acc', 'test_avg_auc'])
    support, placeholders = create_support_placeholder(locality_upper_bound + 1)
    for l1 in range(1, locality_upper_bound + 1):
        for l2 in range(1, locality_upper_bound + 1):
            train_k_fold('gcn_cheby', support, placeholders, l1, l2)


def train_k_fold(model_name, support, placeholders, locality1=1, locality2=2, locality_sizes=None):
    """model_name: name of model (using option defined for FLAGS.model in top
       locality1 & locality2: values of k for 2 GC blocks of gcn_cheby(simple gcn model)
       locality_sizes: locality sizes included in each GC block for res_gcn_cheby(our proposed model)
    """
    # # Define supports and number of them
    # if model_name == 'res_gcn_cheby':
    #     num_supports = np.max(locality_sizes) + 1
    #     support = chebyshev_polynomials(adj, num_supports - 1)
    # elif model_name == 'gcn_cheby':
    #
    # elif model_name == 'gcn':
    #     num_supports = 1
    #     support = [preprocess_adj(adj)]
    # else:
    #     num_supports = 1
    #
    # print(num_supports)
    # Define placeholders


    # Create model
    logging = False
    if model_name == 'res_gcn_cheby':
        is_pool = True  # options are pooling and concatenation
        is_skip_connection = True  # having skip connection or not
        model = ResGCN(placeholders, input_dim=features[2][1], logging=logging, locality_sizes=locality_sizes,
                       is_pool=is_pool, is_skip_connection=is_skip_connection)

    elif model_name == 'gcn':
        model = GCN(placeholders, input_dim=features[2][1], logging=logging)

    elif model_name == 'gcn_cheby':
        is_skip_connection = True  # having skip connection or not
        locality = [locality1, locality2]  # locality sizes of different blocks
        model = GCN(placeholders, input_dim=features[2][1], logging=logging, is_simple=True,
                    is_skip_connection=is_skip_connection, locality=locality)

    elif model_name == 'dense':
        model = MLP(placeholders, input_dim=features[2][1], logging=logging)

    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy, merged_summary], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

    num_nodes = dense_features.shape[0]
    num_folds = 10
    fold_size = int(num_nodes / num_folds)

    # list of results including accuracy, auc, confusion matrix
    train_accuracy = []
    val_accuracy = []
    test_accuracy = []
    test_confusion_matrices = []
    test_auc = []

    # index of fold for validation set and test set
    val_part = 0
    test_part = 1

    # shape of features
    print('whole features shape: ', dense_features.shape)

    # Num_folds cross validation
    for fold in range(num_folds):
        print('Starting fold {}'.format(fold + 1))

        # rotating folds of val and test
        val_part = (val_part + 1) % 10
        test_part = (test_part + 1) % 10

        # defining train, val and test mask
        train_mask = np.ones((num_nodes,), dtype=np.bool)
        val_mask = np.zeros((num_nodes,), dtype=np.bool)
        test_mask = np.zeros((num_nodes,), dtype=np.bool)
        train_mask[val_part * fold_size: min((val_part + 1) * fold_size, num_nodes)] = 0
        train_mask[test_part * fold_size: min((test_part + 1) * fold_size, num_nodes)] = 0
        val_mask[val_part * fold_size: min((val_part + 1) * fold_size, num_nodes)] = 1
        test_mask[test_part * fold_size: min((test_part + 1) * fold_size, num_nodes)] = 1

        # defining train, val and test labels
        y_train = np.zeros(one_hot_labels.shape)
        y_val = np.zeros(one_hot_labels.shape)
        y_test = np.zeros(one_hot_labels.shape)
        y_train[train_mask, :] = one_hot_labels[train_mask, :]
        y_val[val_mask, :] = one_hot_labels[val_mask, :]
        y_test[test_mask, :] = one_hot_labels[test_mask, :]

        # number of samples in each set
        print('# of training samples: ', np.sum(train_mask))
        print('# of validation samples: ', np.sum(val_mask))
        print('# of testing samples: ', np.sum(test_mask))

        tmp_labels = [item + 1 for item in all_labels]
        train_labels = train_mask * tmp_labels
        val_labels = val_mask * tmp_labels
        test_labels = test_mask * tmp_labels

        # distribution of train, val and test set over classes
        train_class = [train_labels.tolist().count(i) for i in range(1, 4)]
        print('train class distribution:', train_class)
        val_class = [val_labels.tolist().count(i) for i in range(1, 4)]
        print('val class distribution:', val_class)
        test_class = [test_labels.tolist().count(i) for i in range(1, 4)]
        print('test class distribution:', test_class)

        # saving initial boolean masks for later use
        init_train_mask = train_mask
        init_val_mask = val_mask
        init_test_mask = test_mask

        # changing mask for having weighted loss
        train_mask = node_weights * train_mask
        val_mask = node_weights * val_mask
        test_mask = node_weights * test_mask

        # Initialize session
        sess = tf.Session()

        # Session with GPU
        # config = tf.ConfigProto(device_count = {'GPU': 1})
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # loss and accuracy scalar curves
        tf.summary.scalar(name='{}_{}_loss_fold_{}'.format(locality1, locality2, fold + 1), tensor=model.loss)
        tf.summary.scalar(name='{}_{}_accuracy_fold_{}'.format(locality1, locality2, fold + 1), tensor=model.accuracy)
        merged_summary = tf.summary.merge_all()

        # defining train, test and val writers in /tmp/model_name/ path
        train_writer = tf.summary.FileWriter(logdir='/tmp/' + model_name +
                                                    '_{}_{}/train_fold_{}/'.format(locality1, locality2, fold + 1))
        test_writer = tf.summary.FileWriter(logdir='/tmp/' + model_name +
                                                   '_{}_{}/test_fold_{}/'.format(locality1, locality2, fold + 1))
        val_writer = tf.summary.FileWriter(logdir='/tmp/' + model_name +
                                                  '_{}_{}/val_fold_{}/'.format(locality1, locality2, fold + 1))

        # Train model
        cost_val = []
        train_results = []
        for epoch in range(FLAGS.epochs):
            t = time.time()

            # Construct feed dictionary for training
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            train_results = sess.run([model.opt_op, model.loss, model.accuracy, merged_summary], feed_dict=feed_dict)
            train_writer.add_summary(train_results[-1], epoch)

            # Evaluation on val set
            val_cost, val_acc, val_summary, duration = evaluate(features, support, y_val, val_mask, placeholders)
            cost_val.append(val_cost)
            val_writer.add_summary(val_summary, epoch)

            # Evaluation on test set
            test_cost, test_acc, test_summary, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
            test_writer.add_summary(test_summary, epoch)

            # Print results of train, val and test
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_results[1]),
                  "train_acc=", "{:.5f}".format(train_results[2]), "val_loss=", "{:.5f}".format(val_cost),
                  "val_acc=", "{:.5f}".format(val_acc), "time=", "{:.5f}".format(time.time() - t))
            print("Test set results:", "test_loss=", "{:.5f}".format(test_cost),
                  "test_accuracy=", "{:.5f}".format(test_acc))

            # Check val loss for early stopping
            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                print("Early stopping on epoch {}...".format(epoch + 1))
                break

        print("Optimization Finished!")

        # Collecting final results of train, test & val
        train_accuracy.append(train_results[2])
        val_accuracy.append(val_acc)
        test_accuracy.append(test_acc)

        # Visualizing layers' embedding
        # if FLAGS.model == 'res_gcn_cheby':
        #     path = '/tmp/' + FLAGS.model + '/layers/'
        #     layer_writer = tf.summary.FileWriter(logdir=path)
        #     write_meta_data_labels(all_labels, path)
        #     visualize_node_embeddings_resgcn(features, support, placeholders, sess, model, layer_writer, is_pool, path,
        #                                      len(locality_sizes))
        #     layer_writer.close()

        # Confusion matrix on test set
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
        feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
        model_outputs = sess.run(model.outputs, feed_dict=feed_dict)
        prediction = np.argmax(model_outputs, axis=1)[init_test_mask]
        confusion_mat = confusion_matrix(y_true=np.asarray(all_labels)[init_test_mask], y_pred=prediction,
                                         labels=[0, 1, 2])
        test_confusion_matrices.append(confusion_mat)
        print('Confusion matrix of test set:')
        print(confusion_mat)

        # Roc auc score on test set
        auc = roc_auc_score(y_true=one_hot_labels[init_test_mask, :], y_score=model_outputs[init_test_mask, :])
        test_auc.append(auc)
        print('Test auc:', auc)
        print('--------')

        # Closing writers
        train_writer.close()
        test_writer.close()
        val_writer.close()
        sess.close()

    if model_name == 'gcn_cheby':
        print('Results of k1={} k2={}'.format(locality1, locality2))

    elif model_name == 'gcn':
        print('Results of re-parametrization model')

    elif model_name == 'res_gcn_cheby':
        print('Results of res_gcn with localities of: ', locality_sizes)

    else:
        print('Results of 4 layer dense neural network')

    print('Accuracy on {} folds'.format(num_folds))
    print('train:', train_accuracy)
    print('val', val_accuracy)
    print('test', test_accuracy)
    print()

    train_avg_acc = np.mean(train_accuracy)
    val_avg_acc = np.mean(val_accuracy)
    test_avg_acc = np.mean(test_accuracy)
    print('Average accuracy on {} folds'.format(num_folds))
    print('train_avg: ', train_avg_acc, '±', np.std(train_accuracy))
    print('val_avg: ', val_avg_acc, '±', np.std(val_accuracy))
    print('test_avg: ', test_avg_acc, '±', np.std(test_accuracy))
    print()

    print('Test auc on {} folds'.format(num_folds))
    print(test_auc)
    print()

    test_avg_auc = np.mean(test_auc)
    print('Average test auc on {} folds'.format(num_folds))
    print(test_avg_auc, '±', np.std(test_auc))

    if model_name == 'gcn_cheby':
        with open('Average_accuracy.csv', mode='a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([locality1, locality2, train_avg_acc, val_avg_acc, test_avg_acc, test_avg_auc])


locality_sizes = [2, 5]
num_supports = np.max(locality_sizes) + 1
support, placeholders = create_support_placeholder(num_supports)
train_k_fold('res_gcn_cheby', support, placeholders, locality_sizes=locality_sizes)
