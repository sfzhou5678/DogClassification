import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import nets

from Configs import *
from data_reader import load_image
from common_tool import pickle_load, pickle_dump

from resnet_v2 import resnet_arg_scope, resnet_v2_50

slim = tf.contrib.slim


def get_bottleneck(sess, image_data, image_data_tensor, bottleneck_tensor):
  bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)

  return bottleneck_values


def prepare_cache(train_data_folder, cache_folder,
                  sess, preprocessed_inputs, bottleneck_tensor,
                  n_classes, tag, batch_size=128):
  """

  :param train_data_folder:
  :return:
  """

  calc_time = 0
  save_time = 0
  print('preparing %s cache(will take a few minutes):' % tag)
  for i in range(n_classes):
    cur_folder_path = os.path.join(train_data_folder, str(i))
    if not os.path.exists(os.path.join(cache_folder, str(i))):
      os.makedirs(os.path.join(cache_folder, str(i)))

    count = 0
    for file in os.listdir(cur_folder_path):
      if file == 'aug':
        # fixme: continue
        continue
        aug_folder_path = os.path.join(cur_folder_path, 'aug')
        aug_cache_folder_path = os.path.join(cache_folder, str(i), 'aug')
        if not os.path.exists(aug_cache_folder_path):
          os.makedirs(aug_cache_folder_path)

        for aug_file in os.listdir(aug_folder_path):
          bottleneck_path = os.path.join(aug_cache_folder_path, aug_file) + '.pkl'
          if not os.path.exists(bottleneck_path):
            image_path = os.path.join(aug_folder_path, aug_file)
            image_data = load_image(image_path)

            bottleneck_values = get_bottleneck(sess, image_data, preprocessed_inputs, bottleneck_tensor)
            # bottleneck_values = sess.run(bottlenecks, {preprocessed_inputs: image_data})

            pickle_dump(bottleneck_values, bottleneck_path)
        continue

      bottleneck_path = os.path.join(cache_folder, str(i), file) + '.pkl'
      if not os.path.exists(bottleneck_path):
        image_path = os.path.join(cur_folder_path, file)
        image_data = load_image(image_path)

        count += 1
        time0 = time.time()
        bottleneck_values = get_bottleneck(sess, image_data, preprocessed_inputs, bottleneck_tensor)
        calc_time += time.time() - time0
        time0 = time.time()
        pickle_dump(bottleneck_values, bottleneck_path)
        save_time += time.time() - time0
        # if count % 10 == 0:
        #   print(count, 'calc time:', calc_time, 'save time:', save_time)
    if i % 10 == 0:
      print(round((i / n_classes) * 100, 0), '%')
  print('100 %')


cache_file_dict = {}


def get_data_batch(batch_size, n_classes, cache_folder, target_ids=None, use_aug=True):
  bottlenecks = []
  labels = []
  weights = []
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
    if target_ids is None:
      weights.append(1.0)
    else:
      if category in target_ids:
        weights.append(1.0)
      else:
        weights.append(0.0001)
  return bottlenecks, labels, weights


def train(config, train_cache_folder, valid_cache_folder, ckpt_path, target_ids):
  use_focal_loss = False

  transfer_graph = tf.Graph()
  with transfer_graph.as_default():
    bottleneck_input = tf.placeholder(tf.float32, [None, config.bottleneck_tensor_size],
                                      name='bottleneck_input')
    labels = tf.placeholder(tf.float32, [None, config.n_classes], name='labels')
    data_weights = tf.placeholder(tf.float32, [None], name='data_weights')

    # define a FC layer as the classifier,
    # it takes as input the extracted features by pre-trained model(bottleneck_input)
    # we only train this layer
    with tf.name_scope('fc-layer'):
      fc_w = tf.Variable(tf.truncated_normal([config.bottleneck_tensor_size, config.n_classes], stddev=0.001))
      fc_b = tf.Variable(tf.zeros([config.n_classes]))
      logits = tf.matmul(bottleneck_input, fc_w) + fc_b

      final_tensor = tf.nn.softmax(logits)

    # loss function & train op
    if use_focal_loss:
      """
      # common loss =-log(p)
      # focal loss = -(1-p)^2.0 * log(p)
      """
      ones = tf.reduce_sum(labels, axis=-1)  # labels are one-hot encoding
      focal_weights = ones - tf.reduce_sum(final_tensor * labels, axis=-1)
      focal_weights = tf.pow(focal_weights, 5.0)

      # * focal_weights
      loss = tf.reduce_mean(tf.reduce_sum(labels * -tf.log(final_tensor), axis=-1) * focal_weights * data_weights)

    else:
      loss = tf.reduce_mean(tf.reduce_sum(labels * -tf.log(final_tensor), axis=-1) * data_weights)

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
  with tf.Session(graph=transfer_graph, config=sess_config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    for step in range(1, 10000 + 1):
      train_bottlenecks, train_labels, train_weights = get_data_batch(config.batch_size, config.n_classes,
                                                                      train_cache_folder,
                                                                      target_ids=target_ids,
                                                                      use_aug=False)
      _, train_loss, train_acc, train_top5_acc = sess.run(
        [train_op, loss, accuracy, top5_accuracy],
        feed_dict={bottleneck_input: train_bottlenecks,
                   labels: train_labels,
                   data_weights: train_weights})

      if step % 100 == 0:
        print('train:', train_loss, train_acc, train_top5_acc)

        # total_loss = 0
        total_acc = 0
        total_files = 0

        valid_category_acc = {}
        cache_folder = valid_cache_folder
        for category in target_ids:
          valid_bottlenecks = []
          valid_labels = []

          if (cache_folder, category) not in cache_file_dict:
            files = os.listdir(os.path.join(cache_folder, str(category)))
            cache_file_dict[(cache_folder, category)] = files
          else:
            files = cache_file_dict[(cache_folder, category)]

          for file in files:
            bottleneck = pickle_load(os.path.join(cache_folder, str(category), file))

            ground_truth = np.zeros(config.n_classes, dtype=np.float32)
            ground_truth[category] = 1.0

            valid_bottlenecks.append(bottleneck)
            valid_labels.append(ground_truth)

          valid_acc = sess.run(accuracy,
                               feed_dict={bottleneck_input: valid_bottlenecks,
                                          labels: valid_labels})
          # total_loss += valid_loss * len(files)
          total_acc += valid_acc * len(files)
          total_files += len(files)
          valid_category_acc[category] = valid_acc
        print('valid:', total_acc / total_files)
        print(sorted(valid_category_acc.items(), key=lambda d: d[1], reverse=True))

        # save model
        saver.save(sess, ckpt_path)


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
  # TODO: speed up
  arg_scope = resnet_arg_scope()
  pretrained_model = resnet_v2_50
  ckpt_path = r'pre-trained/tensorflow-resnet-pretrained/resnet_v2_50.ckpt'
  bottleneck_tensor, preprocessed_inputs, sess = get_pretrained_net(arg_scope, pretrained_model, ckpt_path)

  prepare_cache(train_data_folder, train_cache_folder,
                sess, preprocessed_inputs, bottleneck_tensor, config.n_classes, tag='train', batch_size=128)

  prepare_cache(valid_data_folder, valid_cache_folder,
                sess, preprocessed_inputs, bottleneck_tensor, config.n_classes, tag='valid', batch_size=128)

  # train
  ckpt_folder = os.path.join(data_folder, 'model-ckpt', config.model_type)
  if not os.path.exists(ckpt_folder):
    os.makedirs(ckpt_folder)

  left = 0
  right = 10
  # .% (left, right)
  train(config, train_cache_folder, valid_cache_folder,
        ckpt_path=os.path.join(ckpt_folder, 'model.ckpt'),
        target_ids=range(left, right))


def get_pretrained_net(arg_scope, model, ckpt_path):
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


if __name__ == '__main__':
  train_resnet(layer=152)
