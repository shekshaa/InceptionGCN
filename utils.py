import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.spatial import distance
import sys
import csv
from sklearn import svm
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def one_hot_to_class(one_hot_labels):
    num_rows, num_class = one_hot_labels.shape
    class_labels = np.zeros(shape=(num_rows,))
    for i in range(num_rows):
        class_labels[i] = np.argmax(one_hot_labels[i]) + 1
    return class_labels


def load_citation_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("citation_datasets/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("citation_datasets/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    class_labels = one_hot_to_class(labels)
    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, class_labels
    return adj, features, class_labels, labels


def load_mit_data(adj_type):
    num_nodes = 84
    with open('MIT_dataset/calls.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        call_adj = np.zeros((num_nodes, num_nodes))
        i = 0
        for row in rows:
            call_adj[i, :] = [float(item) for item in row]
            i += 1
    with open('MIT_dataset/politics.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        politics_adj = np.zeros((num_nodes, num_nodes))
        i = 0
        for row in rows:
            politics_adj[i, :] = [float(item) for item in row]
            i += 1
    with open('MIT_dataset/subject_organization.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        subject_organization_adj = np.zeros((num_nodes, num_nodes))
        i = 0
        for row in rows:
            subject_organization_adj[i, :] = [float(item) for item in row]
            i += 1
    with open('MIT_dataset/cluster_labels_norm.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        labels = np.zeros((num_nodes,), dtype=np.int32)
        i = 0
        for row in rows:
            labels[i] = float(row[0])
            i += 1

    features = sparse_to_tuple(sp.coo_matrix(np.eye(num_nodes)))

    if adj_type == 'calls':
        adj = call_adj
    elif adj_type == 'politics':
        adj = politics_adj
    elif adj_type == 'subject':
        adj = subject_organization_adj
    else:
        raise NotImplementedError

    idx = np.arange(num_nodes)
    np.random.shuffle(idx)
    train_proportion = 0.6
    val_proportion = 0.2
    train_idx = idx[:int(train_proportion * num_nodes)]
    val_idx = idx[int(train_proportion * num_nodes): int((train_proportion + val_proportion) * num_nodes)]
    test_idx = idx[int((train_proportion + val_proportion) * num_nodes):]

    train_mask = sample_mask(train_idx, num_nodes)
    val_mask = sample_mask(val_idx, num_nodes)
    test_mask = sample_mask(test_idx, num_nodes)

    num_labels = np.max(labels)
    one_hot_labels = np.zeros((num_nodes, int(num_labels + 1)))
    one_hot_labels[np.arange(num_nodes), labels] = 1

    train_label = np.zeros(one_hot_labels.shape)
    val_label = np.zeros(one_hot_labels.shape)
    test_label = np.zeros(one_hot_labels.shape)
    train_label[train_mask, :] = one_hot_labels[train_mask, :]
    val_label[val_mask, :] = one_hot_labels[val_mask, :]
    test_label[test_mask, :] = one_hot_labels[test_mask, :]

    return adj, features, train_label, val_label, test_label, train_mask, val_mask, test_mask, labels


def load_tadpole_data(adj_type):
    with open('tadpole_dataset/tadpole_2.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        ages = []
        gender = []
        fdg = []
        features = []
        labels = []
        cnt = 0
        age_col_num = 0
        gender_col_num = 0
        fdg_col_num = 0
        label_col_num = 0
        for row in rows:
            if cnt != 0:
                row_features = row[fdg_col_num + 1:]
                if row_features.count('') == 0:
                    ages.append(float(row[age_col_num]))
                    gender.append(row[gender_col_num])
                    fdg.append(float(row[fdg_col_num]))
                    labels.append(int(row[label_col_num]) - 1)
                    features.append([float(item) for item in row_features])
                    cnt += 1
            else:
                age_col_num = row.index('AGE')
                gender_col_num = row.index('PTGENDER')
                fdg_col_num = row.index('FDG')
                label_col_num = row.index('DXCHANGE')
                cnt += 1

        num_nodes = len(labels)

        age_threshold = 2
        age_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.abs(ages[i] - ages[j]) <= age_threshold:
                    age_affinity[i, j] = age_affinity[j, i] = 1

        gender_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if gender[i] == gender[j]:
                    gender_affinity[i, j] = gender_affinity[j, i] = 1

        reshaped_fdg = np.reshape(np.asarray(fdg), newshape=[-1, 1])
        svc = svm.SVC(kernel='linear').fit(reshaped_fdg, labels)
        prediction = svc.predict(reshaped_fdg)
        fdg_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if prediction[i] == prediction[j]:
                    fdg_affinity[i, j] = fdg_affinity[j, i] = 1

        features = np.asarray(features)
        column_sum = np.array(features.sum(0))
        r_inv = np.power(column_sum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        features = features.dot(r_mat_inv)

        dist = distance.pdist(features, metric='euclidean')
        dist = distance.squareform(dist)
        sigma = np.mean(dist)
        w = np.exp(- dist ** 2 / (2 * sigma ** 2))
        age_affinity *= w
        gender_affinity *= w
        fdg_affinity *= w

        if adj_type == 'mixed':
            adj = (age_affinity + gender_affinity + fdg_affinity) / 3
        elif adj_type == 'age':
            adj = age_affinity
        elif adj_type == 'gender':
            adj = gender_affinity
        elif adj_type == 'fdg':
            adj = fdg_affinity
        else:
            raise NotImplementedError

        c_1 = [i for i in range(num_nodes) if labels[i] == 0]
        c_2 = [i for i in range(num_nodes) if labels[i] == 1]
        c_3 = [i for i in range(num_nodes) if labels[i] == 2]

        # balanced
        # c_1_num = 100
        # c_2_num = 120
        # c_3_num = 80

        # imbalanced
        c_1_num = len(c_1)
        c_2_num = len(c_2)
        c_3_num = len(c_3)
        num_nodes = c_1_num + c_2_num + c_3_num
        np.random.shuffle(c_1)
        np.random.shuffle(c_2)
        np.random.shuffle(c_3)
        selection_c_1 = c_1[:c_1_num]
        selection_c_2 = c_2[:c_2_num]
        selection_c_3 = c_3[:c_3_num]
        idx = np.concatenate((selection_c_1, selection_c_2, selection_c_3), axis=0)
        node_weights = np.zeros((num_nodes,))
        node_weights[selection_c_1] = 1 - c_1_num / float(num_nodes)
        node_weights[selection_c_2] = 1 - c_2_num / float(num_nodes)
        node_weights[selection_c_3] = 1 - c_3_num / float(num_nodes)
        np.random.shuffle(idx)
        features = features[idx, :]
        labels = [labels[item] for item in idx]
        adj = adj[idx, :]
        adj = adj[:, idx]
        node_weights = node_weights[idx]

        print(idx)
        # plot features
        # pca = PCA(n_components=5)
        # pca.fit_transform(features)
        # transformed = pca.transform(features)
        # print(pca.explained_variance_)
        # print(pca.components_)
        # print(pca.mean_)
        # components = [0, 3]
        # plt.scatter(transformed[:, components[0]], transformed[:, components[1]], c=labels,
        #             cmap=plt.cm.get_cmap('spectral', 3))
        # plt.xlabel('component 1')
        # plt.ylabel('component 2')
        # plt.colorbar()
        # plt.show()

        # train_proportion = 0.8
        # val_proportion = 0.1

        # train_mask = np.zeros((num_nodes,), dtype=np.bool)
        # val_mask = np.zeros((num_nodes,), dtype=np.bool)
        # test_mask = np.zeros((num_nodes,), dtype=np.bool)
        # train_mask[:int(train_proportion * num_nodes)] = 1
        # val_mask[int(train_proportion * num_nodes): int((train_proportion + val_proportion) * num_nodes)] = 1
        # test_mask[int((train_proportion + val_proportion) * num_nodes):] = 1

        num_labels = 3
        one_hot_labels = np.zeros((num_nodes, num_labels))
        one_hot_labels[np.arange(num_nodes), labels] = 1

        # train_label = np.zeros(one_hot_labels.shape)
        # val_label = np.zeros(one_hot_labels.shape)
        # test_label = np.zeros(one_hot_labels.shape)
        # train_label[train_mask, :] = one_hot_labels[train_mask,:]
        # val_label[val_mask, :] = one_hot_labels[val_mask, :]
        # test_label[test_mask, :] = one_hot_labels[test_mask, :]

        # train_mask = node_weights * train_mask
        # val_mask = node_weights * val_mask
        # test_mask = node_weights * test_mask
        # SVM performance
        # train_features = features[train_idx, :]
        # train_labels = [labels[i] for i in train_idx]
        # test_features = features[test_idx, :]
        # test_labels = [labels[i] for i in test_idx]
        # svc2 = svm.SVC(kernel='linear').fit(train_features, train_labels)
        # train_pred = svc2.predict(train_features)
        # test_pred = svc2.predict(test_features)
        # print('test acc:', np.mean(np.equal(test_pred, test_labels)))
        # print('train acc:', np.mean(np.equal(train_pred, train_labels)))
        sparse_features = sparse_to_tuple(sp.coo_matrix(features))

        # return adj, features, train_label, val_label, test_label, train_mask, val_mask, test_mask, labels
        return adj, sparse_features, labels, one_hot_labels, node_weights, features


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
