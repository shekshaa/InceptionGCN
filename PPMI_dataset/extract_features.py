import matplotlib.pyplot as plt
import hickle as hkl
from PPMI_dataset.autoencoder_3d import *
from PPMI_dataset.load_data import load_ids_labels
import os
import csv

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('depth', 144, 'Depth of images')
flags.DEFINE_integer('height', 240, 'Height of images')
flags.DEFINE_integer('width', 256, 'Width of images')
flags.DEFINE_float('learning_rate', 100, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')


def extract_features(path):
    # loading labels and ids of patients
    ids, labels = load_ids_labels(path)
    print(ids, labels)

    # variable initializations
    num_instance = ids.shape[0]
    num_classes = 2
    depth = FLAGS.depth
    height = FLAGS.height
    width = FLAGS.width

    # Start session
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()

    # Placeholder
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, depth, height, width, 1])
    one_hot_labels = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

    # Model
    filters = [4, 8]
    strides = [(2, 2, 2), (2, 2, 2)]

    # for saving hidden unit activation of images
    hidden_units = np.zeros((num_instance, FLAGS.hidden))
    model_output = convolutional_autoencoder_3d(image_placeholder, num_convolutions=1, filters=filters, strides=strides,
                                                num_hidden_units=FLAGS.hidden, num_classes=num_classes)
    model_loss = calculate_loss(model_output['x_'], one_hot_labels)
    model_accuracy = calculate_accuracy(model_output['x_'], one_hot_labels)
    model_opt = optimization(model_loss, learning_rate=FLAGS.learning_rate)

    # Initializer
    sess.run(tf.global_variables_initializer())

    # Loss & Acc list
    loss_list = np.zeros((FLAGS.epochs, num_instance))
    accuracy_list = np.zeros((FLAGS.epochs, num_instance))

    for epoch in range(FLAGS.epochs):
        print('Starting Epoch ', epoch + 1)
        for i in range(num_instance):
            full_path = os.path.join(path, str(ids[i]) + '.hkl')
            input_image = hkl.load(full_path)
            input_image = np.expand_dims(input_image, 3)
            input_image = np.expand_dims(input_image, 0)
            label = [[2 - labels[i], labels[i] - 1]]
            if epoch + 1 == FLAGS.epochs:
                _, out, loss, acc = sess.run([model_opt, model_output['hidden_units'], model_loss, model_accuracy],
                                             feed_dict={image_placeholder: input_image, one_hot_labels: label})
                hidden_units[i, :] = out[0]
                print(hidden_units[i, :])
            else:
                _, loss, acc = sess.run([model_opt, model_loss, model_accuracy],
                                        feed_dict={image_placeholder: input_image, one_hot_labels: label})
            loss_list[epoch, i] = loss
            accuracy_list[epoch, i] = acc
            print('After instance {}: Loss={},  Acc={}'.format(i + 1, loss, acc))
        print('Average loss: {}'.format(np.mean(loss_list[epoch])))
        print('Average accuracy: {}'.format(np.mean(accuracy_list[epoch])))

    return loss_list, hidden_units

path = './data/'
losses, features = extract_features(path)
with open('features.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    for i in range(features.shape[0]):
        writer.writerow(features[i, :])

plt.plot(losses)
plt.show()
