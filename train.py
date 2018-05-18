import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import nets

from Configs import *
from data_reader import load_image
from common_tool import pickle_load, pickle_dump

from resnet_v2 import resnet_arg_scope, resnet_v2_50, resnet_v2_101, resnet_v2_152
from inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2

from model.TransferModels import TransferFCModel

slim = tf.contrib.slim


def get_resnet_param(layer):
  arg_scope = resnet_arg_scope()
  if layer == 50:
    pretrained_model = resnet_v2_50
  elif layer == 101:
    pretrained_model = resnet_v2_101
  elif layer == 152:
    pretrained_model = resnet_v2_152
  else:
    raise Exception('error model %d' % layer)
  ckpt_path = 'pre-trained/tensorflow-resnet-pretrained/resnet_v2_%d.ckpt' % layer

  return arg_scope, pretrained_model, ckpt_path


def get_inception_param(type):
  if type == 'inception_resnet_v2':
    arg_scope = inception_resnet_v2_arg_scope()
    pretrained_model = inception_resnet_v2
    ckpt_path = 'pre-trained/tensorflow-inception-pretrained/inception_resnet_v2_2016_08_30.ckpt'
  else:
    raise Exception('error model %s' % type)

  return arg_scope, pretrained_model, ckpt_path


def get_resnet_pretrained_net(arg_scope, model, ckpt_path):
  preprocessed_inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], 'images')
  with slim.arg_scope(arg_scope):
    net, end_points = model(preprocessed_inputs, is_training=False)
  net = tf.squeeze(net, axis=[1, 2])
  bottleneck_tensor = net
  init_fn = slim.assign_from_checkpoint_fn(
    ckpt_path,
    slim.get_variables_to_restore(),
    ignore_missing_vars=False)
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess = tf.Session(config=sess_config)
  init_fn(sess)
  return bottleneck_tensor, preprocessed_inputs, sess


def get_inception_pretrained_net(arg_scope, model, ckpt_path):
  preprocessed_inputs = tf.placeholder(tf.float32, [None, 299, 299, 3], 'images')
  with slim.arg_scope(arg_scope):
    _, end_points = model(preprocessed_inputs, is_training=False)
    net = end_points['PreLogitsFlatten']
    # net = tf.squeeze(net, axis=[1, 2])
    bottleneck_tensor = net

    init_fn = slim.assign_from_checkpoint_fn(
      ckpt_path,
      slim.get_variables_to_restore(),
      ignore_missing_vars=False)

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess = tf.Session(config=sess_config)
  init_fn(sess)
  return bottleneck_tensor, preprocessed_inputs, sess


def get_bottleneck(sess, image_data, image_data_tensor, bottleneck_tensor):
  bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
  # bottleneck_values = np.squeeze(bottleneck_values)

  return bottleneck_values


def prepare_cache(train_data_folder, cache_folder,
                  sess, preprocessed_inputs, bottleneck_tensor,
                  n_classes, img_size,
                  use_aug, tag, batch_size=128):
  """

  :param train_data_folder:
  :return:
  """

  calc_time = 0
  save_time = 0

  img_paths = []
  cache_save_paths = []
  for category in range(n_classes):
    cur_folder_path = os.path.join(train_data_folder, str(category))
    if not os.path.exists(os.path.join(cache_folder, str(category))):
      os.makedirs(os.path.join(cache_folder, str(category)))

    for file in os.listdir(cur_folder_path):
      if file == 'aug':
        if use_aug:
          aug_folder_path = os.path.join(cur_folder_path, 'aug')
          aug_cache_folder_path = os.path.join(cache_folder, str(category), 'aug')
          if not os.path.exists(aug_cache_folder_path):
            os.makedirs(aug_cache_folder_path)

          for aug_file in os.listdir(aug_folder_path):
            bottleneck_path = os.path.join(aug_cache_folder_path, aug_file) + '.pkl'
            if not os.path.exists(bottleneck_path):
              image_path = os.path.join(aug_folder_path, aug_file)
              img_paths.append(image_path)
              cache_save_paths.append(bottleneck_path)
        continue

      bottleneck_path = os.path.join(cache_folder, str(category), file) + '.pkl'
      if not os.path.exists(bottleneck_path):
        image_path = os.path.join(cur_folder_path, file)
        img_paths.append(image_path)
        cache_save_paths.append(bottleneck_path)

  if len(img_paths) > 0:
    print('preparing %s cache(will take a few minutes):' % tag)
    times = (len(img_paths) - 1) // batch_size + 1
    for i in range(times):
      image_datas = [load_image(img_path, size=img_size)[0] for img_path in
                     img_paths[i * batch_size:(i + 1) * batch_size]]
      caches = get_bottleneck(sess, image_datas, preprocessed_inputs, bottleneck_tensor)

      for cache, save_path in zip(caches, cache_save_paths[i * batch_size:(i + 1) * batch_size]):
        pickle_dump(cache, save_path)

      if i % (times // 10 + 1) == 0:
        print(round(i / times * 100, 1), '%')
    print('100 %')


cache_file_dict = {}


def get_data_batch(batch_size, n_classes, cache_folder, target_ids=None, use_aug=True, mode='balanced'):
  """

  :param batch_size:
  :param n_classes:
  :param cache_folder:
  :param target_ids:
  :param use_aug:
  :param mode: mode=['random','balanced]
  :return:
  """
  bottlenecks = []
  labels = []
  weights = []
  global cache_file_dict
  if mode == 'random':
    if len(cache_file_dict) == 0:
      cache_file_dict = []
      for category in range(n_classes):
        files = os.listdir(os.path.join(cache_folder, str(category)))
        for file in files:
          if file == 'aug':
            aug_files = os.listdir(os.path.join(cache_folder, str(category), 'aug'))
            aug_files = [os.path.join('aug', aug_file) for aug_file in aug_files]
            for aug_file in aug_files:
              cache_file_dict.append((category, aug_file))
          else:
            cache_file_dict.append((category, file))

    datas = random.sample(cache_file_dict, batch_size)
    for category, file_path in datas:
      bottleneck = pickle_load(os.path.join(cache_folder, str(category), file_path))

      ground_truth = np.zeros(n_classes, dtype=np.float32)
      ground_truth[category] = 1.0

      bottlenecks.append(bottleneck)
      labels.append(ground_truth)
      if target_ids is None:
        weights.append(1.0)
      else:
        if category in target_ids:
          weights.append(1.0)
        else:
          weights.append(0.0001)
  elif mode == 'balanced':
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
      if target_ids is None:
        weights.append(1.0)
      else:
        if category in target_ids:
          weights.append(1.0)
        else:
          weights.append(0.0001)
  return bottlenecks, labels, weights


def train(config, train_cache_folder, valid_cache_folder,
          use_aug,
          ckpt_path, target_ids):
  initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
  with tf.name_scope('Train'):
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      train_model = TransferFCModel(config, is_training=True)

  with tf.name_scope('Valid'):
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
      valid_model = TransferFCModel(config, is_training=False)

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    valid_cache_file_dict = {}
    for step in range(1, 10000 + 1):
      train_bottlenecks, train_labels, train_weights = get_data_batch(config.batch_size, config.n_classes,
                                                                      train_cache_folder,
                                                                      target_ids=target_ids,
                                                                      use_aug=use_aug,
                                                                      mode='balanced')  #
      _, train_loss, train_acc = sess.run([train_model.train_op, train_model.loss, train_model.accuracy],
                                          feed_dict={train_model.bottleneck_input: train_bottlenecks,
                                                     train_model.labels: train_labels})

      if step % 100 == 0:
        print('train:', train_loss, train_acc)
        total_acc = 0
        total_files = 0

        valid_category_acc = {}
        cache_folder = valid_cache_folder
        for category in target_ids:
          valid_bottlenecks = []
          valid_labels = []

          if category not in valid_cache_file_dict:
            files = os.listdir(os.path.join(cache_folder, str(category)))
            valid_cache_file_dict[category] = files
          else:
            files = valid_cache_file_dict[category]

          for file in files:
            bottleneck = pickle_load(os.path.join(cache_folder, str(category), file))

            ground_truth = np.zeros(config.n_classes, dtype=np.float32)
            ground_truth[category] = 1.0

            valid_bottlenecks.append(bottleneck)
            valid_labels.append(ground_truth)

          valid_acc = sess.run(valid_model.accuracy,
                               feed_dict={valid_model.bottleneck_input: valid_bottlenecks,
                                          valid_model.labels: valid_labels})
          total_acc += valid_acc * len(files)
          total_files += len(files)
          valid_category_acc[category] = valid_acc
        print('valid:', total_acc / total_files)
        print(sorted(valid_category_acc.items(), key=lambda d: d[1], reverse=True))

        # save model
        saver.save(sess, ckpt_path)


def train_resnet(data_folder, layer, use_aug):
  train_data_folder = os.path.join(data_folder, 'train')
  valid_data_folder = os.path.join(data_folder, 'test1')

  config = ResNetConfig(layer=layer, batch_size=128, n_classes=100)
  train_cache_folder = os.path.join(data_folder, config.model_type, 'train-cache')
  if not os.path.exists(train_cache_folder):
    os.makedirs(train_cache_folder)

  valid_cache_folder = os.path.join(data_folder, config.model_type, 'valid-cache')
  if not os.path.exists(valid_cache_folder):
    os.makedirs(valid_cache_folder)

  # config.n_classes = 100  # fixme:
  arg_scope, pretrained_model, ckpt_path = get_resnet_param(layer)
  bottleneck_tensor, preprocessed_inputs, sess = get_resnet_pretrained_net(arg_scope, pretrained_model, ckpt_path)

  prepare_cache(train_data_folder, train_cache_folder,
                sess, preprocessed_inputs, bottleneck_tensor, config.n_classes, img_size=224,
                use_aug=use_aug, tag='train', batch_size=64)

  prepare_cache(valid_data_folder, valid_cache_folder,
                sess, preprocessed_inputs, bottleneck_tensor, config.n_classes, img_size=224,
                use_aug=use_aug, tag='valid', batch_size=64)

  # train
  ckpt_folder = os.path.join(data_folder, 'model-ckpt', config.model_type)
  if not os.path.exists(ckpt_folder):
    os.makedirs(ckpt_folder)

  left = 0
  right = 100
  # .% (left, right)
  train(config, train_cache_folder, valid_cache_folder,
        use_aug=use_aug,
        ckpt_path=os.path.join(ckpt_folder, 'model.ckpt'), target_ids=range(left, right))


def train_inception(data_folder, model_type, use_aug):
  train_data_folder = os.path.join(data_folder, 'train')
  valid_data_folder = os.path.join(data_folder, 'test1')

  config = InceptionResNetConfig(model_type=model_type, n_classes=100, batch_size=128)
  # config = ResNetConfig(train_data_folder, valid_data_folder, batch_size=128)

  train_cache_folder = os.path.join(data_folder, config.model_type, 'train-cache')
  if not os.path.exists(train_cache_folder):
    os.makedirs(train_cache_folder)

  valid_cache_folder = os.path.join(data_folder, config.model_type, 'valid-cache')
  if not os.path.exists(valid_cache_folder):
    os.makedirs(valid_cache_folder)

  # config.n_classes = 5  # fixme:
  arg_scope, pretrained_model, ckpt_path = get_inception_param(model_type)
  bottleneck_tensor, preprocessed_inputs, sess = get_inception_pretrained_net(arg_scope, pretrained_model, ckpt_path)

  prepare_cache(train_data_folder, train_cache_folder,
                sess, preprocessed_inputs, bottleneck_tensor, config.n_classes, img_size=299,
                use_aug=use_aug, tag='train', batch_size=64)

  prepare_cache(valid_data_folder, valid_cache_folder,
                sess, preprocessed_inputs, bottleneck_tensor, config.n_classes, img_size=299,
                use_aug=use_aug, tag='valid', batch_size=64)

  # train
  ckpt_folder = os.path.join(data_folder, 'model-ckpt', config.model_type)
  if not os.path.exists(ckpt_folder):
    os.makedirs(ckpt_folder)

  left = 0
  right = 100
  # .% (left, right)
  train(config, train_cache_folder, valid_cache_folder,
        use_aug=use_aug,
        ckpt_path=os.path.join(ckpt_folder, 'model.ckpt'), target_ids=range(left, right))


if __name__ == '__main__':
  data_folder = r'D:\DeeplearningData\Dog identification'

  train_resnet(data_folder, layer=101, use_aug=False)
  # train_inception(data_folder,model_type='inception_resnet_v2', use_aug=False)
