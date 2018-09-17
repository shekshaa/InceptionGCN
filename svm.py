from utils import *
import tensorflow as tf
from sklearn.metrics import roc_auc_score, confusion_matrix

# Set random seed
seed = 123
np.random.seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS

adj_type = 'age'
adj, features, all_labels, one_hot_labels, node_weights, dense_features = load_tadpole_data(adj_type)

num_nodes = dense_features.shape[0]
num_folds = 10
train_proportion = 0.8
train_size = int(num_nodes * train_proportion)
fold_size = int(train_size / num_folds)

svm_train_acc = []
svm_val_acc = []
svm_test_acc = []
test_confusion_matrices = []
val_part = 0
test_part = 1

print('whole features shape: ', dense_features.shape)

for k in range(num_folds):
    print('Starting fold {}'.format(k + 1))
    val_part = (val_part + 1) % 10
    test_part = (test_part + 1) % 10

    train_mask = np.ones((num_nodes,), dtype=np.bool)
    val_mask = np.zeros((num_nodes,), dtype=np.bool)
    test_mask = np.zeros((num_nodes,), dtype=np.bool)

    train_mask[val_part * fold_size: min((val_part + 1) * fold_size, num_nodes)] = 0
    train_mask[test_part * fold_size: min((test_part + 1) * fold_size, num_nodes)] = 0
    val_mask[val_part * fold_size: min((val_part + 1) * fold_size, num_nodes)] = 1
    test_mask[test_part * fold_size: min((test_part + 1) * fold_size, num_nodes)] = 1

    y_train = np.zeros(one_hot_labels.shape)
    y_val = np.zeros(one_hot_labels.shape)
    y_test = np.zeros(one_hot_labels.shape)

    y_train[train_mask, :] = one_hot_labels[train_mask, :]
    y_val[val_mask, :] = one_hot_labels[val_mask, :]
    y_test[test_mask, :] = one_hot_labels[test_mask, :]

    print('# of training samples: ', np.sum(train_mask))
    print('# of validation samples: ', np.sum(val_mask))
    print('# of testing samples: ', np.sum(test_mask))

    tmp_labels = [item + 1 for item in all_labels]
    train_labels = train_mask * tmp_labels
    val_labels = val_mask * tmp_labels
    test_labels = test_mask * tmp_labels

    train_class = [train_labels.tolist().count(i) for i in range(1, 4)]
    print('train class distribution:', train_class)
    val_class = [val_labels.tolist().count(i) for i in range(1, 4)]
    print('val class distribution:', val_class)
    test_class = [test_labels.tolist().count(i) for i in range(1, 4)]
    print('test class distribution:', test_class)
    # SVM
    # print(tmp.shape)
    train_features = dense_features[train_mask, :]
    train_labels = np.asarray(all_labels)[train_mask]
    val_features = dense_features[val_mask, :]
    val_labels = np.asarray(all_labels)[val_mask]
    test_features = dense_features[test_mask, :]
    test_labels = np.asarray(all_labels)[test_mask]

    svc2 = svm.SVC(kernel='linear').fit(train_features, train_labels)
    train_pred = svc2.predict(train_features)
    val_pred = svc2.predict(val_features)
    test_pred = svc2.predict(test_features)
    svm_train_acc.append(np.mean(np.equal(train_pred, train_labels)))
    svm_val_acc.append(np.mean(np.equal(val_pred, val_labels)))
    svm_test_acc.append(np.mean(np.equal(test_pred, test_labels)))
    print('train acc:', svm_train_acc[-1])
    print('val acc:', svm_val_acc[-1])
    print('test acc:', svm_test_acc[-1])

    confusion_mat = confusion_matrix(y_true=np.asarray(all_labels)[test_mask], y_pred=test_pred, labels=[0, 1, 2])
    test_confusion_matrices.append(confusion_mat)
    print('Confusion matrix of test set:')
    print(confusion_mat)

    test_score = svc2.predict_proba(test_features)
    svm_test_auc = roc_auc_score(y_true=one_hot_labels[test_mask, :], y_score=test_score)
    print('-------')

print('train_avg: ', np.mean(svm_train_acc))
print('val_avg: ', np.mean(svm_val_acc))
print('test_avg: ', np.mean(svm_test_acc))
