import os
import random
import heapq

import numpy as np
import tensorflow as tf

from Configs import *
from data_reader import load_image
from common_tool import pickle_load, pickle_dump

from model.TransferModels import TransferFCModel


def cal_acc(info_list, n_classes):
  valid_category_acc = {}

  for config, idx in zip(info_list, range(len(info_list))):
    ckpt_cache_list = info_list[config]
    tf.reset_default_graph()
    with tf.name_scope('Test%d' % idx):
      with tf.variable_scope("Model", reuse=None):
        test_model = TransferFCModel(config, is_training=False)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      saver = tf.train.Saver()

      for ckpt_path, cache_folder in ckpt_cache_list:
        saver.restore(sess, ckpt_path)

        for category in range(config.n_classes):
          valid_bottlenecks = []
          valid_labels = []

          files = os.listdir(os.path.join(cache_folder, str(category)))
          for file in files:
            bottleneck = pickle_load(os.path.join(cache_folder, str(category), file))

            valid_bottlenecks.append(bottleneck)
            valid_labels.append(category)

          pred_prob = sess.run(test_model.softmax_logits,
                               feed_dict={test_model.bottleneck_input: valid_bottlenecks})

          # # region voting
          # # 下面这个for是暴力的投票机制，每个pred只投给概率最大的k个结果 （测试结果表明k=5能取得比不用略好的效果，k=1的效果不如不用）
          # k = 5
          # for prob, i in zip(pred_prob, range(len(pred_prob))):
          #   index = heapq.nlargest(k, range(len(prob)), prob.take)
          #   zero = np.zeros_like(prob)
          #   for idx in index:
          #     zero[idx] = prob[idx]
          #   pred_prob[i] = zero
          # # endregion

          if category not in valid_category_acc:
            valid_category_acc[category] = []
          valid_category_acc[category].append(np.array(pred_prob))

  total_acc = 0
  total_files = 0
  for category in range(n_classes):
    pred_prob = np.mean(valid_category_acc[category], axis=0)

    pred_category = np.argmax(pred_prob, axis=-1)
    gt_category = category
    valid_acc = 0
    for pred in pred_category:
      if pred == gt_category:
        valid_acc += 1
    valid_acc /= len(pred_category)

    total_acc += valid_acc * len(files)
    total_files += len(files)
    valid_category_acc[category] = valid_acc
  print('valid:', total_acc / total_files)
  print(sorted(valid_category_acc.items(), key=lambda d: d[1], reverse=True))


def get_info(model, model_param, use_aug, data_mode):
  if model == 'Inception':
    model_ckpt = os.path.join(data_folder, 'model-ckpt', '%s' % model_param,
                              'best_model%s%s.ckpt' % (('-aug' if use_aug else '-noaug', '-' + data_mode)))
    cache_folder = os.path.join(data_folder, '%s' % model_param, 'valid-cache')
    return (model_ckpt, cache_folder)
  elif model == 'ResNet':
    model_ckpt = os.path.join(data_folder, 'model-ckpt', 'ResNet-%d' % model_param,
                              'best_model%s%s.ckpt' % (('-aug' if use_aug else '-noaug', '-' + data_mode)))
    cache_folder = os.path.join(data_folder, 'ResNet-%d' % model_param, 'valid-cache')
    return (model_ckpt, cache_folder)
  else:
    raise Exception('error')


if __name__ == '__main__':
  data_folder = r'Your data folder'

  ## 手动的test方法
  # 在info list中选择要测试的模型，每个config下可以输入多个待测试的模型
  # 最终的预测结果会是所有模型的平均
  info_list = {
    ## ResNet:
    # ResNetConfig(101, n_classes=100, batch_size=128): [
    #   get_info('ResNet', model_param=101, use_aug=True, data_mode='random'),
    # ],
    ResNetConfig(152, n_classes=100, batch_size=128): [
      get_info('ResNet', model_param=152, use_aug=True, data_mode='balanced'),
      get_info('ResNet', model_param=152, use_aug=True, data_mode='random'),
      # get_info('ResNet', model_param=152, use_aug=False, data_mode='balanced'),
      # get_info('ResNet', model_param=152, use_aug=False, data_mode='random'),
    ],

    ## Inception:
    InceptionResNetConfig('inception_resnet_v2', n_classes=100, batch_size=128): [
      get_info('Inception', model_param='inception_resnet_v2', use_aug=True, data_mode='balanced'),
      get_info('Inception', model_param='inception_resnet_v2', use_aug=True, data_mode='random'),
      # get_info('Inception', model_param='inception_resnet_v2', use_aug=False, data_mode='balanced'),
      # get_info('Inception', model_param='inception_resnet_v2', use_aug=False, data_mode='random'),

    ]
  }

  cal_acc(info_list, n_classes=100)

  # ## 暴力的2模型集成测试方法：
  # resnet_info_list = [
  #   get_info('ResNet', model_param=152, use_aug=True, data_mode='balanced'),
  #   get_info('ResNet', model_param=152, use_aug=True, data_mode='random'),
  #   get_info('ResNet', model_param=152, use_aug=False, data_mode='balanced'),
  #   get_info('ResNet', model_param=152, use_aug=False, data_mode='random')]
  #
  # inception_resnet_info_list = [
  #   get_info('Inception', model_param='inception_resnet_v2', use_aug=True, data_mode='balanced'),
  #   get_info('Inception', model_param='inception_resnet_v2', use_aug=True, data_mode='random'),
  #   get_info('Inception', model_param='inception_resnet_v2', use_aug=False, data_mode='balanced'),
  #   get_info('Inception', model_param='inception_resnet_v2', use_aug=False, data_mode='random'),
  # ]
  #
  # for i in range(len(resnet_info_list)):
  #   for j in range(len(inception_resnet_info_list)):
  #     info_list = {
  #       ResNetConfig(152, n_classes=100, batch_size=128): [
  #         resnet_info_list[i]
  #       ],
  #       InceptionResNetConfig('inception_resnet_v2', n_classes=100, batch_size=128): [
  #         inception_resnet_info_list[j]
  #       ]
  #     }
  #     print(i, j)
  #     cal_acc(info_list, n_classes=100)
