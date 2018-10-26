from table_experiment import *


skip_option = [True, False]
pool_option = [True, False]
general_learning_rate = 0.01
learning_rate_flag = True
learning_rates_dict = {'age': 0.01, 'gender': 0.01, 'fdg': 0.01, 'apoe': 0.01, 'mixed': 0.01}
adj_types = ['age', 'gender', 'fdg', 'apoe', 'mixed']

for skip in skip_option:
    for typ in adj_types:
        best_train, best_k1, best_k2 = table_experiment(typ, general_learning_rate if learning_rate_flag
                                                        else learning_rates_dict[typ], skip)
        tf.reset_default_graph()
        locality_sizes = [best_k1, best_k2]
        num_supports = np.max(locality_sizes) + 1
        # print(num_supports)
        support, placeholders = create_support_placeholder('res_gcn_cheby', num_supports, adj_dict[typ])

        results = []
        for pool in pool_option:
            train_acc, val_acc, test_acc = train_k_fold('res_gcn_cheby', support, placeholders,
                                                        is_pool=pool, is_skip_connection=skip,
                                                        locality_sizes=locality_sizes)
            train_avg, train_std, val_avg, val_std, test_avg, test_std = avg_std_log(train_acc, val_acc, test_acc)
            results.append([train_avg, train_std, val_avg, val_std, test_avg, test_std])

        with open('Acc_avg_std_{}_skip_{}.csv'.format(typ, skip), mode='a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['', 'train', 'val', 'test'])
            for j in range(2):
                row = ['concat' if j == 0 else 'pool']
                for i in range(0, 6, 2):
                    row.append('{:.2f} ± {:.2f}'.format(results[j][i] * 100, results[j][i + 1] * 100))
                writer.writerow(row)

        tf.reset_default_graph()
