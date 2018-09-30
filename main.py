from train import *

flags = tf.app.flags
FLAGS = flags.FLAGS


# possible values for hyper-parameters
learning_rates = [0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005]
dropout = [0., .1, .2, .3]
weight_decay = [5e-4, 1e-4, 5e-3, 1e-3]
early_stopping = [20, 25, 30]
locality_upper_bound = 6
adj, features, all_labels, one_hot_labels, node_weights, dense_features, num_class = load_data()
support, placeholders = create_support_placeholder(FLAGS.model, locality_upper_bound + 1, adj, features, one_hot_labels)

# looking for best hyper-parameters for just two models (simple GCN and InceptionGCN)
for lr in learning_rates:
    FLAGS.learning_rate = lr
    for dr in dropout:
        FLAGS.dropout = dr
        for wd in weight_decay:
            FLAGS.weight_decay = wd
            for es in early_stopping:
                FLAGS.early_stopping = es
                for l1 in range(1, locality_upper_bound + 1):
                    for l2 in range(1, l1):
                        if FLAGS.model == 'gcn_cheby':
                            _, val_acc, _ = train_k_fold('gcn_cheby', support, placeholders, features, all_labels,
                                                         one_hot_labels, node_weights, dense_features, num_class,
                                                         locality1=l1, locality2=l2)
                        elif FLAGS.model == 'res_gcn_cheby':
                            locality_sizes = [l2, l1]
                            _, val_acc, _ = train_k_fold('res_gcn_cheby', support, placeholders, features, all_labels,
                                                         one_hot_labels, node_weights, dense_features, num_class,
                                                         locality_sizes=locality_sizes)
