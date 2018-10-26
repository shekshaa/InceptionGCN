from train import *

# Note: set learning rate = 0.05 for case of dense network otherwise 0.01

if FLAGS.model == 'gcn_cheby':
    # simple gcn example
    locality1 = 5
    locality2 = 2
    num_supports = max(locality1, locality2) + 1
    support, placeholders = create_support_placeholder(FLAGS.model, num_supports, adj_dict[FLAGS.adj_type])
    train_acc, val_acc, test_acc = train_k_fold(FLAGS.model, support, placeholders,
                                                is_skip_connection=FLAGS.is_skip_connection,
                                                locality1=locality1, locality2=locality2)
elif FLAGS.model == 'res_gcn_cheby':
    # ResGCN example
    locality_sizes = [2, 5]
    num_supports = np.max(locality_sizes) + 1
    support, placeholders = create_support_placeholder(FLAGS.model, num_supports, adj_dict[FLAGS.adj_type])
    train_acc, val_acc, test_acc = train_k_fold(FLAGS.model, support, placeholders,
                                                is_pool=FLAGS.is_pool, is_skip_connection=FLAGS.is_skip_connection,
                                                locality_sizes=locality_sizes)
else:
    # gcn or dense example
    num_supports = 1
    support, placeholders = create_support_placeholder(FLAGS.model, num_supports, adj_dict[FLAGS.adj_type])
    train_acc, val_acc, test_acc = train_k_fold(FLAGS.model, support, placeholders)

avg_std_log(train_acc, val_acc, test_acc)
