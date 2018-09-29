from train import *

# Table experiment for simple gcn on different localities
locality_upper_bound = 6
adj, features, all_labels, one_hot_labels, node_weights, dense_features, num_class = load_data()
support, placeholders = create_support_placeholder('gcn_cheby', locality_upper_bound + 1, adj, features,
                                                   one_hot_labels)

with open('Average_accuracy.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['K1', 'K2', 'train_avg_acc', 'val_avg_acc', 'test_avg_acc', 'train_std_acc', 'val_std_acc',
                     'test_std_acc'])
    for l1 in range(1, locality_upper_bound + 1):
        for l2 in range(1, locality_upper_bound + 1):
            train_accuracy, val_accuracy, test_accuracy = train_k_fold('gcn_cheby', support, placeholders, features,
                                                                       all_labels, one_hot_labels, node_weights,
                                                                       dense_features, num_class, l1, l2)
            train_avg_acc = np.mean(train_accuracy)
            val_avg_acc = np.mean(val_accuracy)
            test_avg_acc = np.mean(test_accuracy)

            train_std_acc = np.std(train_accuracy)
            val_std_acc = np.std(val_accuracy)
            test_std_acc = np.std(test_accuracy)

            print('Average accuracies:')
            print('train_avg: ', train_avg_acc, '±', train_std_acc)
            print('val_avg: ', val_avg_acc, '±', val_std_acc)
            print('test_avg: ', test_avg_acc, '±', test_std_acc)
            print()
            print()
            writer.writerow([l1, l2, train_avg_acc, val_avg_acc, test_avg_acc, train_std_acc, val_std_acc, test_std_acc])
