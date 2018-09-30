from train import *

# Table experiment for simple gcn on different localities
is_skip_connection = FLAGS.is_skip_connection
locality_upper_bound = 6
adj, features, all_labels, one_hot_labels, node_weights, dense_features, num_class = load_data()
support, placeholders = create_support_placeholder('gcn_cheby', locality_upper_bound + 1, adj, features,
                                                   one_hot_labels)

# Open csv file to write average results of different locality settings
with open('Average_accuracy.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    # write header of file
    writer.writerow(['K1', 'K2', 'train_avg_acc', 'val_avg_acc', 'test_avg_acc', 'train_std_acc', 'val_std_acc',
                     'test_std_acc'])
    for l1 in range(1, locality_upper_bound + 1):
        for l2 in range(1, locality_upper_bound + 1):
            train_accuracy, val_accuracy, test_accuracy = train_k_fold('gcn_cheby', support, placeholders, features,
                                                                       all_labels, one_hot_labels, node_weights,
                                                                       dense_features, num_class,
                                                                       is_skip_connection=is_skip_connection,
                                                                       locality1=l1, locality2=l2)

            train_avg, train_std, val_avg, val_std, test_avg, test_std = avg_std_log(train_accuracy, val_accuracy,
                                                                                     test_accuracy)
            writer.writerow([l1, l2, train_avg, val_avg, test_avg, train_std, val_std, test_std])
