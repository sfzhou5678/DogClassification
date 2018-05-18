import os
import random

import numpy as np
import tensorflow as tf

from Configs import *
from data_reader import load_image
from common_tool import pickle_load, pickle_dump


def cal_acc(config, ckpt_path_list):
  transfer_graph = tf.Graph()
  with transfer_graph.as_default():
    bottleneck_input = tf.placeholder(tf.float32, [None, config.bottleneck_tensor_size],
                                      name='bottleneck_input')

    # define a FC layer as the classifier,
    # it takes as input the extracted features by pre-trained model(bottleneck_input)
    # we only train this layer
    with tf.name_scope('fc-layer'):
      fc_w = tf.Variable(tf.truncated_normal([config.bottleneck_tensor_size, config.n_classes], stddev=0.001))
      fc_b = tf.Variable(tf.zeros([config.n_classes]))
      logits = tf.matmul(bottleneck_input, fc_w) + fc_b

      final_tensor = tf.nn.softmax(logits)

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(graph=transfer_graph, config=sess_config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    valid_category_acc = {}
    for ckpt_path, cache_folder in ckpt_path_list:
      saver.restore(sess, ckpt_path)

      for category in range(config.n_classes):
        valid_bottlenecks = []
        valid_labels = []

        files = os.listdir(os.path.join(cache_folder, str(category)))
        for file in files:
          bottleneck = pickle_load(os.path.join(cache_folder, str(category), file))

          valid_bottlenecks.append(bottleneck)
          valid_labels.append(category)

        pred_prob = sess.run(final_tensor,
                             feed_dict={bottleneck_input: valid_bottlenecks})
        if category not in valid_category_acc:
          valid_category_acc[category] = []
        valid_category_acc[category].append(np.array(pred_prob))

    total_acc = 0
    total_files = 0
    for category in range(config.n_classes):
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


if __name__ == '__main__':
  data_folder = r'D:\DeeplearningData\Dog identification'
  label_map_path = os.path.join(data_folder, 'label_name.txt')

  train_data_folder = os.path.join(data_folder, 'train')
  train_label_path = os.path.join(data_folder, 'label_train.txt')

  valid_data_folder = os.path.join(data_folder, 'test1')
  valid_label_path = os.path.join(data_folder, 'label_val.txt')

  new_label_map_path = os.path.join(data_folder, 'new_label_map.txt')

  layer = 152
  config = ResNetConfig(train_data_folder, valid_data_folder,
                        layer=layer, batch_size=128)

  ckpt_path_list = [
    (os.path.join(data_folder, 'model-ckpt', 'ResNet-50', 'model-[0,100].ckpt'),
     os.path.join(data_folder, 'ResNet-50-CPU', 'valid-cache')),
    (os.path.join(data_folder, 'model-ckpt', 'ResNet-50', 'model.ckpt'),
     os.path.join(data_folder, 'ResNet-50', 'valid-cache')),
    (os.path.join(data_folder, 'model-ckpt', 'ResNet-101', 'model.ckpt'),
     os.path.join(data_folder, 'ResNet-101', 'valid-cache')),
    (os.path.join(data_folder, 'model-ckpt', 'ResNet-152', 'model.ckpt'),
     os.path.join(data_folder, 'ResNet-152', 'valid-cache')),


    # os.path.join(data_folder, 'model-ckpt', 'ResNet-152', 'model.ckpt'),
    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[0,100]-random1.ckpt'),
    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[0,100]-random2.ckpt'),

    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[0,50].ckpt'),
    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[50,100].ckpt'),

    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[0,10].ckpt'),
    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[10,20].ckpt'),
    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[20,30].ckpt'),
    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[30,40].ckpt'),
    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[40,50].ckpt'),
    #
    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[50,60].ckpt'),
    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[60,70].ckpt'),
    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[70,80].ckpt'),
    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[80,90].ckpt'),
    # os.path.join(data_folder, 'model-ckpt', config.model_type, 'model-[90,100].ckpt'),
  ]
  cal_acc(config, ckpt_path_list)
