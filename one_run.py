from train import *

is_pool = FLAGS.is_pool
is_skip_connection = FLAGS.is_skip_connection

adj, features, all_labels, one_hot_labels, node_weights, dense_features, num_class = load_data()  # load data

if FLAGS.model == 'gcn_cheby':
    # simple gcn example
    locality1 = 5
    locality2 = 2
    num_supports = max(locality1, locality2) + 1
    support, placeholders = create_support_placeholder(FLAGS.model, num_supports, adj, features, one_hot_labels)
    train_acc, val_acc, test_acc = train_k_fold(FLAGS.model, support, placeholders, features, all_labels,
                                                one_hot_labels, node_weights, dense_features, num_class,
                                                is_skip_connection=is_skip_connection,
                                                locality1=locality1, locality2=locality2)
elif FLAGS.model == 'res_gcn_cheby':
    # ResGCN example
    locality_sizes = [2, 5]
    num_supports = np.max(locality_sizes) + 1
    support, placeholders = create_support_placeholder(FLAGS.model, num_supports, adj, features, one_hot_labels)
    train_acc, val_acc, test_acc = train_k_fold(FLAGS.model, support, placeholders, features, all_labels,
                                                one_hot_labels, node_weights, dense_features, num_class,
                                                is_pool=is_pool, is_skip_connection=is_skip_connection,
                                                locality_sizes=locality_sizes)
else:
    # gcn or dense example
    num_supports = 1
    support, placeholders = create_support_placeholder(FLAGS.model, num_supports, adj, features, one_hot_labels)
    train_acc, val_acc, test_acc = train_k_fold(FLAGS.model, support, placeholders, features, all_labels,
                                                one_hot_labels, node_weights, dense_features, num_class)

avg_std_log(train_acc, val_acc, test_acc)
