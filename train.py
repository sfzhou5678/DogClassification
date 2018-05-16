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
    print('100%')


cache_file_dict = {}


def get_data_batch(batch_size, n_classes, cache_folder):
  bottlenecks = []
  labels = []
  global cache_file_dict
  for _ in range(batch_size):
    category = random.randrange(n_classes)
    if (cache_folder, category) not in cache_file_dict:
      files = os.listdir(os.path.join(cache_folder, str(category)))
      for i in range(len(files)):
        if files[i] == 'aug':
          aug_files = os.listdir(os.path.join(cache_folder, str(category), 'aug'))
          aug_files = [os.path.join('aug', aug_file) for aug_file in aug_files]
          files.remove('aug')
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
  initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
  with tf.name_scope('Train'):
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      train_model = TransferModel(config=config, is_train=True)

  with tf.name_scope('Valid'):
    with tf.variable_scope("Model", reuse=True):
      valid_model = TransferModel(config=config, is_train=False)

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # saver = tf.train.Saver()

    for step in range(1, 10000 + 1):
      train_bottlenecks, train_labels = get_data_batch(config.batch_size, config.n_classes, train_cache_folder)
      # sess.run(train_model.train_op,
      #          feed_dict={train_model.bottleneck_input: train_bottlenecks,
      #                     train_model.labels: train_labels})

      _, train_loss, train_acc, train_top5_acc = sess.run(
        [train_model.train_op, train_model.loss, train_model.acc, train_model.top5_acc],
        feed_dict={train_model.bottleneck_input: train_bottlenecks,
                   train_model.labels: train_labels})

      if step % 100 == 0:
        print('train:', train_loss, train_acc, train_top5_acc)
        #   # fixme: 考虑全拿出来？
        valid_bottlenecks, valid_labels = get_data_batch(config.batch_size * 10, config.n_classes, valid_cache_folder)
        valid_loss, valid_acc, valid_top5_acc = sess.run([valid_model.loss, valid_model.acc, valid_model.top5_acc],
                                                         feed_dict={valid_model.bottleneck_input: valid_bottlenecks,
                                                                    valid_model.labels: valid_labels})
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

  config.n_classes = 10  # fixme:
  # prepare_cache(train_data_folder, train_cache_folder,
  #               pretrained_meta, pretrained_ckpt, config.n_classes, tag='train')
  #
  # prepare_cache(valid_data_folder, valid_cache_folder,
  #               pretrained_meta, pretrained_ckpt, config.n_classes, tag='valid')

  # train
  train(config, train_cache_folder, valid_cache_folder)


if __name__ == '__main__':
  train_resnet(layer=50)
