from tensorflow.contrib.tensorboard.plugins import projector
import os
import tensorflow as tf
from models import Dense
import numpy as np

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
    print(embedding.metadata_path)


def visualize_node_embeddings_resgcn(features, support, placeholders, sess, model, writer, is_pool, path, num_GCNs):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    activations = [[layer.outputs, layer.pooled_outputs, layer.total_output] for layer in model.layers
                   if not isinstance(layer, Dense)]
    activations = sess.run(activations, feed_dict=feed_dict)
    num_layers = len(activations)
    config = projector.ProjectorConfig()
    node_embedding = []
    diffs = []
    for i in range(num_layers):
        diff_layer = []
        for j in range(num_GCNs):
            node_embedding.append(tf.Variable(activations[i][0][j],
                                              name='layer_{}'.format(i) + '_GCN_{}'.format(j)))
            diff_layer.append(np.mean(np.equal(activations[i][0][j], activations[i][1])))
            add_config(sess, config, node_embedding, path)
        if is_pool:
            node_embedding.append(tf.Variable(activations[i][1], name='layer_{}_pooled'.format(i)))
            add_config(sess, config, node_embedding, path)

        node_embedding.append(tf.Variable(activations[i][2], name='layer_{}_final'.format(i)))
        add_config(sess, config, node_embedding, path)
        diffs.append(diff_layer)

    print(diffs)
    saver_embed = tf.train.Saver(node_embedding)
    saver_embed.save(sess, path + 'embedding_layers', 1)
    projector.visualize_embeddings(writer, config)

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
