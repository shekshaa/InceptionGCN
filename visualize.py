from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
from models import Dense
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

flags = tf.app.flags
FLAGS = flags.FLAGS


def write_meta_data_labels(all_labels, path_name):
    with open(path_name + 'meta_data_labels.csv', 'w') as csv_file:
        for label in all_labels:
            csv_file.write(str(label))
            csv_file.write('\n')


def add_config(sess, config, node_embedding, path):
    sess.run(node_embedding[-1].initializer)
    embedding = config.embeddings.add()
    embedding.tensor_name = node_embedding[-1].name
    embedding.metadata_path = path + 'meta_data_labels.csv'


def get_activations(features, support, placeholders, sess, model):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    activations = [[layer.outputs, layer.pooled_outputs, layer.total_output] for layer in model.layers
                   if not isinstance(layer, Dense)]
    activations = sess.run(activations, feed_dict=feed_dict)
    return activations


def euclidean_distance(f1, f2):
    diff = f1 - f2
    return np.sqrt(np.dot(diff, diff))


def affinity_visualize(adj, dense_features, all_labels, num_sample, num_classes):
    graph = nx.Graph()
    num_nodes = dense_features.shape[0]
    c = []
    for j in range(num_classes):
        c.append([i for i in range(num_nodes) if all_labels[i] == j])
        c[-1] = c[-1][:num_sample]
    idx = np.concatenate(c, axis=0)
    dense_features = dense_features[idx, :]
    all_labels = [all_labels[item] for item in idx]
    adj = adj[idx, :]
    adj = adj[:, idx]
    num_nodes = len(idx)
    graph.add_nodes_from(np.arange(num_nodes))
    cnt = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] != 0:
                cnt += 1
                graph.add_edge(i, j, weight=euclidean_distance(dense_features[i, :], dense_features[j, :]))

    node_colors = []
    colors = ['r', 'g', 'b']
    for i in range(num_nodes):
        node_colors.append(colors[all_labels[i]])
    nx.draw_networkx(graph, nx.spring_layout(graph, weight='weight', iterations=50, scale=1000), node_size=5, width=0.1,
                     node_color=node_colors, with_labels=False)
    plt.show()


def features_embedding_visualize(features_activations, all_labels, title):
    transformed = TSNE(n_components=2).fit_transform(features_activations)
    colors = ['r', 'g', 'b']
    node_colors = []
    for i in range(features_activations.shape[0]):
        node_colors.append(colors[all_labels[i]])
    plt.scatter(transformed[:, 0], transformed[:, 1], c=node_colors)
    plt.title(title)
    plt.show()


def visualize_node_embeddings_resgcn(features, all_labels, support, placeholders, sess, model, is_pool, num_GCNs):
    activations = get_activations(features, support, placeholders, sess, model)
    num_layers = len(activations)
    for i in range(num_layers):
        for j in range(num_GCNs):
            features_embedding_visualize(activations[i][0][j], all_labels, 'layer_{}'.format(i) + '_GCN_{}'.format(j))
        if is_pool:
            features_embedding_visualize(activations[i][1], all_labels, 'layer_{}_pooled'.format(i))
        features_embedding_visualize(activations[i][2], all_labels, 'layer_{}_final'.format(i))
    # config = projector.ProjectorConfig()
    # node_embedding = []
    # diffs = []
    # for i in range(num_layers):
    #     # diff_layer = []
    #     for j in range(num_GCNs):
    #         node_embedding.append(tf.Variable(activations[i][0][j],
    #                                           name='layer_{}'.format(i) + '_GCN_{}'.format(j)))
    #         # diff_layer.append(np.mean(np.equal(activations[i][0][j], activations[i][1])))
    #         add_config(sess, config, node_embedding, path)
    #     if is_pool:
    #         node_embedding.append(tf.Variable(activations[i][1], name='layer_{}_pooled'.format(i)))
    #         add_config(sess, config, node_embedding, path)
    #
    #     node_embedding.append(tf.Variable(activations[i][2], name='layer_{}_final'.format(i)))
    #     add_config(sess, config, node_embedding, path)
    #     # diffs.append(diff_layer)
    #
    # # print(diffs)
    # saver_embed = tf.train.Saver(node_embedding)
    # saver_embed.save(sess, path + 'embedding_layers', 1)
    # projector.visualize_embeddings(writer, config)
    # for i in range(num_layers):
    #     if isinstance(activations[i], tf.SparseTensorValue):
    #         activations[i] = sparse_to_dense([activations[i].indices, activations[i].values, activations[i].dense_shape])
    #
    # config = projector.ProjectorConfig()
    # node_embeddings = []
    # for i in range(1, num_layers):
    #     node_embeddings.append(tf.Variable(activations[i], name='embedding_layer_{}'.format(i)))
    #     sess.run(node_embeddings[-1].initializer)
    #     embedding = config.embeddings.add()
    #     embedding.tensor_name = node_embeddings[-1].name
    #     embedding.metadata_path = '/tmp/gcn/meta_data_labels.csv'
    #
    # saver_embed = tf.train.Saver(node_embeddings)
    # saver_embed.save(sess, '/tmp/gcn/embedding_layers', 1)
    # projector.visualize_embeddings(writer, config)
