from train import *

# Table experiment for simple gcn on different localities
locality_upper_bound = 7
adj, features, all_labels, one_hot_labels, node_weights, dense_features, num_class = load_data()
"""locality_upper_bound: up to what degree looking for degree size of each layer"""
with open('Average_accuracy.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['K1', 'K2', 'train_avg_acc', 'val_avg_acc', 'test_avg_acc', 'test_avg_auc'])

support, placeholders = create_support_placeholder('gcn_cheby', locality_upper_bound + 1, adj, features,
                                                   one_hot_labels)
for l1 in range(1, locality_upper_bound + 1):
    for l2 in range(1, locality_upper_bound + 1):
        train_k_fold('gcn_cheby', support, placeholders, l1, l2)
