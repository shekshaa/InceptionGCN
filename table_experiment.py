from train import *


def table_experiment(adj_type, learning_rate, is_skip_connection):
    # Table experiment for simple gcn on different localities
    adj = adj_dict[adj_type]
    locality_upper_bound = 6
    support, placeholders = create_support_placeholder('gcn_cheby', locality_upper_bound + 1, adj)
    FLAGS.learning_rate = learning_rate
    best_train = 0
    best_k1 = 0
    best_k2 = 0
    train_avg_table = np.zeros((locality_upper_bound, locality_upper_bound))
    test_avg_table = np.zeros((locality_upper_bound, locality_upper_bound))
    val_avg_table = np.zeros((locality_upper_bound, locality_upper_bound))
    train_std_table = np.zeros((locality_upper_bound, locality_upper_bound))
    test_std_table = np.zeros((locality_upper_bound, locality_upper_bound))
    val_std_table = np.zeros((locality_upper_bound, locality_upper_bound))
    for l1 in range(1, locality_upper_bound + 1):
        for l2 in range(1, locality_upper_bound + 1):
            train_accuracy, val_accuracy, test_accuracy = train_k_fold('gcn_cheby', support, placeholders,
                                                                       is_skip_connection=is_skip_connection,
                                                                       locality1=l1, locality2=l2)

            train_avg, train_std, val_avg, val_std, test_avg, test_std = avg_std_log(train_accuracy, val_accuracy,
                                                                                     test_accuracy)
            train_avg_table[l1 - 1, l2 - 1] = train_avg
            train_std_table[l1 - 1, l2 - 1] = train_std
            val_avg_table[l1 - 1, l2 - 1] = val_avg
            val_std_table[l1 - 1, l2 - 1] = val_std
            test_avg_table[l1 - 1, l2 - 1] = test_avg
            test_std_table[l1 - 1, l2 - 1] = test_std

            if best_train < train_avg:
                best_train = train_avg
                best_k1 = l1
                best_k2 = l2

    file_writer(train_avg_table, train_std_table, 'train', adj_type, is_skip_connection, locality_upper_bound)
    file_writer(val_avg_table, val_std_table, 'val', adj_type, is_skip_connection, locality_upper_bound)
    file_writer(test_avg_table, test_std_table, 'test', adj_type, is_skip_connection, locality_upper_bound)

    return best_train, best_k1, best_k2


def file_writer(avg_table, std_table, table_type, adj_type, is_skip_connection, locality_upper_bound):
    # Open csv file to write average results of different locality settings
    with open('Acc_avg_std_{}_skip_{}.csv'.format(adj_type, is_skip_connection), mode='a') as csv_file:
        writer = csv.writer(csv_file)
        # write header of file
        header = [table_type]
        newline = ['']
        for i in range(1, locality_upper_bound + 1):
            header.append(str(i))
            newline.append('')
        writer.writerow(header)
        for i in range(1, locality_upper_bound + 1):
            row = [str(i)]
            for j in range(locality_upper_bound):
                row.append('{:.2f} ± {:.2f}'.format(avg_table[i - 1, j] * 100, std_table[i - 1, j] * 100))
            writer.writerow(row)
        writer.writerow(newline)
        writer.writerow(newline)
