from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            # print(hidden.shape)
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        tf.summary.scalar(name='loss', tensor=self.loss)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        tf.summary.scalar(name='accuracy', tensor=self.accuracy)

    def _build(self):
        # 4 layer dense neural network
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=FLAGS.hidden2,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 logging=self.logging))

        # self.layers.append(Dense(input_dim=FLAGS.hidden2,
        #                          output_dim=FLAGS.hidden3,
        #                          placeholders=self.placeholders,
        #                          act=tf.nn.relu,
        #                          dropout=True,
        #                          logging=self.logging))
        #
        # self.layers.append(Dense(input_dim=FLAGS.hidden3,
        #                          output_dim=self.output_dim,
        #                          placeholders=self.placeholders,
        #                          act=lambda x: x,
        #                          dropout=True,
        #                          logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, is_simple=False, is_skip_connection=False, locality=None, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.is_simple = is_simple
        self.is_skip_connection = is_skip_connection
        self.locality = locality
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        # 4 layer gcn model using re-parametrization trick
        if not self.is_simple:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                sparse_inputs=True,
                                                logging=self.logging,))

            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden2,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                logging=self.logging))

        # 2 layer simple gcn model
        else:
            if self.is_skip_connection:
                self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                    output_dim=FLAGS.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.relu,
                                                    dropout=True,
                                                    sparse_inputs=True,
                                                    logging=self.logging,
                                                    is_simple=True,
                                                    locality=self.locality[0],
                                                    is_skip_connection=True))

                l2_input_dim = self.input_dim + FLAGS.hidden1

                self.layers.append(GraphConvolution(input_dim=l2_input_dim,
                                                    output_dim=self.output_dim,
                                                    placeholders=self.placeholders,
                                                    act=lambda x: x,
                                                    dropout=True,
                                                    logging=self.logging,
                                                    is_simple=True,
                                                    is_skip_connection=False,
                                                    locality=self.locality[1]))

            else:
                self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                    output_dim=FLAGS.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.relu,
                                                    dropout=True,
                                                    sparse_inputs=True,
                                                    logging=self.logging,
                                                    is_simple=True,
                                                    locality=self.locality[0]))

                self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                    output_dim=self.output_dim,
                                                    placeholders=self.placeholders,
                                                    act=lambda x: x,
                                                    dropout=True,
                                                    logging=self.logging,
                                                    is_simple=True,
                                                    locality=self.locality[1]))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class ResGCN(Model):
    def __init__(self, placeholders, input_dim, locality_sizes, is_pool=False, is_skip_connection=True, **kwargs):
        super(ResGCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.locality_sizes = locality_sizes
        self.is_pool = is_pool
        self.is_skip_connection = is_skip_connection
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        tf.summary.scalar(name='loss', tensor=self.loss)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        tf.summary.scalar(name='accuracy', tensor=self.accuracy)

    def _build(self):
        # convolutional layer 1
        self.layers.append(ResGraphConvolution(input_dim=self.input_dim,
                                               output_dim=FLAGS.hidden1,
                                               locality_sizes=self.locality_sizes,
                                               placeholders=self.placeholders,
                                               act=tf.nn.relu,
                                               dropout=True,
                                               sparse_inputs=True,
                                               logging=self.logging,
                                               is_pool=self.is_pool,
                                               is_skip_connection=self.is_skip_connection))

        # changing input dim and output dim of layer 1 in different cases of having skip connections and pooling types
        if self.is_skip_connection:
            if not self.is_pool:
                l2_input_size = len(self.locality_sizes) * FLAGS.hidden1 + self.input_dim
                l2_output_size = len(self.locality_sizes) * FLAGS.hidden2 + l2_input_size
            else:
                l2_input_size = FLAGS.hidden1 + self.input_dim
                l2_output_size = FLAGS.hidden2 + l2_input_size
        else:
            if not self.is_pool:
                l2_input_size = len(self.locality_sizes) * FLAGS.hidden1
                l2_output_size = len(self.locality_sizes) * FLAGS.hidden2
            else:
                l2_input_size = FLAGS.hidden1
                l2_output_size = FLAGS.hidden2

        # convolutional layer 2
        self.layers.append(ResGraphConvolution(input_dim=l2_input_size,
                                               output_dim=FLAGS.hidden2,
                                               locality_sizes=self.locality_sizes,
                                               placeholders=self.placeholders,
                                               act=lambda x: x,
                                               dropout=True,
                                               sparse_inputs=False,
                                               logging=self.logging,
                                               is_pool=self.is_pool,
                                               is_skip_connection=self.is_skip_connection))

        # changing input dim and output dim of layer 2 in different cases of having skip connections and pooling types
        # if self.is_skip_connection:
        #     if not self.is_pool:
        #         l3_input_size = len(self.locality_sizes) * FLAGS.hidden2 + l2_input_size
        #         l3_output_size = len(self.locality_sizes) * FLAGS.hidden3 + l3_input_size
        #     else:
        #         l3_input_size = FLAGS.hidden2 + l2_input_size
        #         l3_output_size = FLAGS.hidden3 + l3_input_size
        # else:
        #     if not self.is_pool:
        #         l3_input_size = len(self.locality_sizes) * FLAGS.hidden2
        #         l3_output_size = len(self.locality_sizes) * FLAGS.hidden3
        #     else:
        #         l3_input_size = FLAGS.hidden2
        #         l3_output_size = FLAGS.hidden3

        # convolutional layer 3
        # self.layers.append(ResGraphConvolution(input_dim=l3_input_size,
        #                                        output_dim=FLAGS.hidden3,
        #                                        locality_sizes=self.locality_sizes,
        #                                        placeholders=self.placeholders,
        #                                        act=lambda x: x,
        #                                        dropout=True,
        #                                        sparse_inputs=False,
        #                                        logging=self.logging,
        #                                        is_pool=self.is_pool,
        #                                        is_skip_connection=self.is_skip_connection))

        # last dense layer for predicting classes
        self.layers.append(Dense(input_dim=l2_output_size, output_dim=self.output_dim, placeholders=self.placeholders,
                                 dropout=False,
                                 sparse_inputs=False, act=lambda x: x, bias=True, featureless=False))

    def predict(self):
        return tf.nn.softmax(self.outputs)
