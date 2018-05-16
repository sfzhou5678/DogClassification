import os
import random

import numpy as np
import tensorflow as tf

from Configs import *
from data_reader import load_image
from common_tool import pickle_load, pickle_dump

from model.TransferModels import TransferModel


def get_bottleneck(sess, image_data, image_data_tensor, bottleneck_tensor):
  bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)

  return bottleneck_values


def prepare_cache(train_data_folder, cache_folder,
                  pretrained_meta, pretrained_ckpt,
                  n_classes, tag):
  """

  :param train_data_folder:
  :return:
  """
  g = tf.Graph()
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(graph=g, config=sess_config) as sess:
    saver = tf.train.import_meta_graph(pretrained_meta)
    saver.restore(sess, pretrained_ckpt)

    graph = tf.get_default_graph()
    bottleneck_tensor = graph.get_tensor_by_name("avg_pool:0")
    jpeg_data_tensor = graph.get_tensor_by_name("images:0")

    print('preparing %s cache(will take a few minutes):' % tag)
    for i in range(n_classes):
      cur_folder_path = os.path.join(train_data_folder, str(i))
      if not os.path.exists(os.path.join(cache_folder, str(i))):
        os.makedirs(os.path.join(cache_folder, str(i)))

      for file in os.listdir(cur_folder_path):
        if file == 'aug':
          aug_folder_path = os.path.join(cur_folder_path, 'aug')
          aug_cache_folder_path = os.path.join(cache_folder, str(i), 'aug')
          if not os.path.exists(aug_cache_folder_path):
            os.makedirs(aug_cache_folder_path)

          for aug_file in os.listdir(aug_folder_path):
            bottleneck_path = os.path.join(aug_cache_folder_path, aug_file) + '.pkl'
            if not os.path.exists(bottleneck_path):
              image_path = os.path.join(aug_folder_path, aug_file)
              image_data = load_image(image_path)
              bottleneck_values = get_bottleneck(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

              pickle_dump(bottleneck_values, bottleneck_path)
          continue

        bottleneck_path = os.path.join(cache_folder, str(i), file) + '.pkl'
        if not os.path.exists(bottleneck_path):
          image_path = os.path.join(cur_folder_path, file)
          image_data = load_image(image_path)
          bottleneck_values = get_bottleneck(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

          pickle_dump(bottleneck_values, bottleneck_path)
      if i % 10 == 0:
        print(round((i / n_classes) * 100, 0), '%')
    print('100 %')


cache_file_dict = {}


def get_data_batch(batch_size, n_classes, cache_folder, use_aug=True):
  bottlenecks = []
  labels = []
  global cache_file_dict
  for _ in range(batch_size):
    category = random.randrange(n_classes)
    if (cache_folder, category) not in cache_file_dict:
      files = os.listdir(os.path.join(cache_folder, str(category)))
      for i in range(len(files)):
        if files[i] == 'aug':
          files.remove('aug')
          if use_aug:
            aug_files = os.listdir(os.path.join(cache_folder, str(category), 'aug'))
            aug_files = [os.path.join('aug', aug_file) for aug_file in aug_files]
            files += aug_files
          break
      cache_file_dict[(cache_folder, category)] = files
    else:
      files = cache_file_dict[(cache_folder, category)]

    file = random.sample(files, 1)[0]
    bottleneck = pickle_load(os.path.join(cache_folder, str(category), file))

    ground_truth = np.zeros(n_classes, dtype=np.float32)
    ground_truth[category] = 1.0

    bottlenecks.append(bottleneck)
    labels.append(ground_truth)
  return bottlenecks, labels


def train(config, train_cache_folder, valid_cache_folder):
  g_new_layers = tf.Graph()
  with g_new_layers.as_default():
    bottleneck_input = tf.placeholder(tf.float32, [None, config.bottleneck_tensor_size],
                                      name='bottleneck_input')
    labels = tf.placeholder(tf.float32, [None, config.n_classes], name='labels')

    # define a FC layer as the classifier,
    # it takes as input the extracted features by pre-trained model(bottleneck_input)
    # we only train this layer
    with tf.name_scope('fc-layer'):
      weights = tf.Variable(tf.truncated_normal([config.bottleneck_tensor_size, config.n_classes], stddev=0.001))
      biases = tf.Variable(tf.zeros([config.n_classes]))
      logits = tf.matmul(bottleneck_input, weights) + biases

      final_tensor = tf.nn.softmax(logits)

    # loss function & train op
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy)

    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
      config.learning_rate,
      global_step,
      100,
      0.985
    )
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # calculate accuracy
    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      top5_correction_predcition = tf.nn.in_top_k(predictions=final_tensor,
                                                  targets=tf.argmax(labels, 1),
                                                  k=5)
      top5_accuracy = tf.reduce_mean(tf.cast(top5_correction_predcition, tf.float32))

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(graph=g_new_layers, config=sess_config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # saver = tf.train.Saver()

    for step in range(1, 10000 + 1):
      train_bottlenecks, train_labels = get_data_batch(config.batch_size, config.n_classes, train_cache_folder,
                                                       use_aug=False)
      # sess.run(train_model.train_op,
      #          feed_dict={train_model.bottleneck_input: train_bottlenecks,
      #                     train_model.labels: train_labels})

      _, train_loss, train_acc, train_top5_acc = sess.run(
        [train_op, loss, accuracy, top5_accuracy],
        feed_dict={bottleneck_input: train_bottlenecks,
                   labels: train_labels})

      if step % 100 == 0:
        print('train:', train_loss, train_acc, train_top5_acc)
        valid_bottlenecks, valid_labels = get_data_batch(config.batch_size * 10, config.n_classes, valid_cache_folder)
        valid_loss, valid_acc, valid_top5_acc = sess.run([loss, accuracy, top5_accuracy],
                                                         feed_dict={bottleneck_input: valid_bottlenecks,
                                                                    labels: valid_labels})
        print('valid:', valid_loss, valid_acc, valid_top5_acc)


def train_resnet(layer):
  # TODO: 移出去
  data_folder = r'D:\DeeplearningData\Dog identification'
  label_map_path = os.path.join(data_folder, 'label_name.txt')

  train_data_folder = os.path.join(data_folder, 'train')
  train_label_path = os.path.join(data_folder, 'label_train.txt')

  valid_data_folder = os.path.join(data_folder, 'test1')
  valid_label_path = os.path.join(data_folder, 'label_val.txt')

  new_label_map_path = os.path.join(data_folder, 'new_label_map.txt')

  config = ResNetConfig(train_data_folder, valid_data_folder,
                        layer=layer, batch_size=128)

  train_cache_folder = os.path.join(data_folder, config.model_type, 'train-cache')
  if not os.path.exists(train_cache_folder):
    os.makedirs(train_cache_folder)

  valid_cache_folder = os.path.join(data_folder, config.model_type, 'valid-cache')
  if not os.path.exists(valid_cache_folder):
    os.makedirs(valid_cache_folder)

  pretrained_ckpt = 'pre-trained/tensorflow-resnet-pretrained/ResNet-L%d.ckpt' % config.layer
  pretrained_meta = 'pre-trained/tensorflow-resnet-pretrained/ResNet-L%d.meta' % config.layer

  # config.n_classes = 10  # fixme:
  prepare_cache(train_data_folder, train_cache_folder,
                pretrained_meta, pretrained_ckpt, config.n_classes, tag='train')

  prepare_cache(valid_data_folder, valid_cache_folder,
                pretrained_meta, pretrained_ckpt, config.n_classes, tag='valid')

  # train
  train(config, train_cache_folder, valid_cache_folder)


if __name__ == '__main__':
  train_resnet(layer=50)
