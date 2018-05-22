import os
import random
import time
import threading

import numpy as np
import tensorflow as tf

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

  return bottleneck_values


def load_all_images(img_paths, loaded_images, size, batch_size):
  """
  单独开一个线程提前读取所有图片并存储到loaded_images中
  :param img_paths:
  :param loaded_images:
  :param size:
  :return:
  """
  global idx_lock
  global append_lock
  global global_idx
  global none_idx

  t = len(img_paths)
  while len(loaded_images) < t:
    idx_lock.acquire()
    cur_idx = global_idx
    global_idx += 1
    idx_lock.release()
    if cur_idx >= t:
      break

    # 给缓冲池大小加上限：提前加载的文件数不能大于30 * batch_size
    while cur_idx - none_idx > 30 * batch_size:
      pass

    path = img_paths[cur_idx]
    data = load_image(path, size=size)

    while len(loaded_images) != cur_idx:
      # 一个简单的spin lock
      pass
    append_lock.acquire()
    loaded_images.append(data)
    append_lock.release()


def save_caches(caches, save_paths):
  """
  单独开一个线程保存cache实现加速
  :param caches:
  :param save_paths:
  :return:
  """
  for cache, save_path in zip(caches, save_paths):
    pickle_dump(cache, save_path)


# 用于cache中多线程load image
idx_lock = threading.Lock()
append_lock = threading.Lock()
global_idx = 0
none_idx = 0


def prepare_cache(train_data_folder, cache_folder,
                  sess, preprocessed_inputs, bottleneck_tensor,
                  n_classes, img_size,
                  use_aug, tag, batch_size=64, thread_num=5):
  """
  :param train_data_folder:
  :return:
  """

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
    time_load_img = 0
    time_calc_cache = 0
    time_save = 0

    # region ... 加速用的img缓冲池，开多线程把所有img读取到缓冲池loaded_images中
    global idx_lock
    global append_lock
    global global_idx
    global none_idx  # 记录被标记为none的数据的index

    global_idx = 0
    none_idx = 0

    loaded_images = []
    thread_list = []
    for i in range(thread_num):
      thread_load = threading.Thread(target=load_all_images, args=(img_paths, loaded_images, img_size, batch_size))
      thread_list.append(thread_load)
      thread_load.start()
    # endregion

    for i in range(times):
      time0 = time.time()
      while len(loaded_images) <= (i + 1) * batch_size and len(loaded_images) < len(img_paths):
        pass
      image_datas = loaded_images[i * batch_size:(i + 1) * batch_size]
      # 下面这个for清除内存中的img， 防止爆内存
      for idx in range(i * batch_size, (i + 1) * batch_size):
        if idx < len(loaded_images):
          loaded_images[idx] = None
      none_idx = (i + 1) * batch_size
      time_load_img += time.time() - time0

      time1 = time.time()
      caches = get_bottleneck(sess, image_datas, preprocessed_inputs, bottleneck_tensor)
      time_calc_cache += time.time() - time1

      time2 = time.time()
      thread_save = threading.Thread(target=save_caches,
                                     args=(caches, cache_save_paths[i * batch_size:(i + 1) * batch_size]))
      thread_save.start()
      time_save += time.time() - time2

      if i % (times // 10 + 1) == 0:
        print(round(i / times * 100, 1), '%')
    for thread_load in thread_list:
      thread_load.join()
    print('100 %')
    print('time load img:', time_load_img)
    print('time calc cache:', time_calc_cache)
    print('time save:', time_save)


cache_file_dict = {}


def get_data_batch(batch_size, n_classes, cache_folder, use_aug=True, mode='random'):
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
            files.remove('aug')
            if use_aug:
              aug_files = os.listdir(os.path.join(cache_folder, str(category), 'aug'))
              aug_files = [os.path.join('aug', aug_file) for aug_file in aug_files]
              files += aug_files
            break
        for file in files:
          try:
            cache_file_dict.append((category, pickle_load(os.path.join(cache_folder, str(category), file))))
          except:
            os.remove(os.path.join(cache_folder, str(category), file))

    datas = random.sample(cache_file_dict, batch_size)
    for category, bottleneck in datas:
      ground_truth = np.zeros(n_classes, dtype=np.float32)
      ground_truth[category] = 1.0

      bottlenecks.append(bottleneck)
      labels.append(ground_truth)
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
        datas = []
        for file in files:
          try:
            datas.append(pickle_load(os.path.join(cache_folder, str(category), file)))
          except:
            os.remove(os.path.join(cache_folder, str(category), file))
        cache_file_dict[(cache_folder, category)] = datas
      else:
        datas = cache_file_dict[(cache_folder, category)]

      bottleneck = random.sample(datas, 1)[0]

      ground_truth = np.zeros(n_classes, dtype=np.float32)
      ground_truth[category] = 1.0

      bottlenecks.append(bottleneck)
      labels.append(ground_truth)
  return bottlenecks, labels


def train(config, train_cache_folder, valid_cache_folder,
          use_aug, data_mode,
          best_ckpt_path, ckpt_path):
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

    best_acc = 0
    b_count = 0
    valid_cache_file_dict = {}
    wf = open(ckpt_path[:ckpt_path.rindex('.')] + '.txt', 'a')

    for step in range(1, 2000000 + 1):
      train_bottlenecks, train_labels = get_data_batch(config.batch_size, config.n_classes,
                                                       train_cache_folder,
                                                       use_aug=use_aug,
                                                       mode=data_mode)  #
      _, train_loss, train_acc = sess.run([train_model.train_op, train_model.loss, train_model.accuracy],
                                          feed_dict={train_model.bottleneck_input: train_bottlenecks,
                                                     train_model.labels: train_labels})

      if step % 500 == 0:
        print('train:', train_loss, train_acc)
        total_acc = 0
        total_files = 0

        valid_category_acc = {}
        cache_folder = valid_cache_folder
        for category in range(config.n_classes):
          valid_bottlenecks = []
          valid_labels = []

          if category not in valid_cache_file_dict:
            files = os.listdir(os.path.join(cache_folder, str(category)))
            files = [pickle_load(os.path.join(cache_folder, str(category), path)) for path in files]
            valid_cache_file_dict[category] = files
          else:
            files = valid_cache_file_dict[category]

          for bottleneck in files:
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
        valid_acc = total_acc / total_files
        print(step, 'valid:', valid_acc)
        wf.write(
          '[%d %s] train: %.4f %.4f\t valid : %.4f\n' % (step, time.strftime('%m%d-%H%M', time.localtime(time.time())),
                                                         train_loss, train_acc, valid_acc))
        wf.flush()
        saver.save(sess, ckpt_path)
        if valid_acc > best_acc:
          saver.save(sess, best_ckpt_path)
          best_acc = valid_acc
          b_count = 0
          print(sorted(valid_category_acc.items(), key=lambda d: d[1], reverse=True))
        else:
          b_count += 1
          if b_count >= 15:
            break
    wf.close()


def train_resnet(data_folder, n_classes, layer, use_aug, data_mode):
  train_data_folder = os.path.join(data_folder, 'train')
  valid_data_folder = os.path.join(data_folder, 'test1')

  config = ResNetConfig(layer=layer, batch_size=128, n_classes=n_classes)
  train_cache_folder = os.path.join(data_folder, config.model_type, 'train-cache')
  if not os.path.exists(train_cache_folder):
    os.makedirs(train_cache_folder)

  valid_cache_folder = os.path.join(data_folder, config.model_type, 'valid-cache')
  if not os.path.exists(valid_cache_folder):
    os.makedirs(valid_cache_folder)

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

  train(config, train_cache_folder, valid_cache_folder,
        use_aug=use_aug, data_mode=data_mode,
        ckpt_path=os.path.join(ckpt_folder, 'model%s%s.ckpt' % ('-aug' if use_aug else '-noaug', '-' + data_mode)),
        best_ckpt_path=os.path.join(ckpt_folder,
                                    'best_model%s%s.ckpt' % ('-aug' if use_aug else '-noaug', '-' + data_mode)))


def train_inception(data_folder, n_classes, model_type, use_aug, data_mode):
  train_data_folder = os.path.join(data_folder, 'train')
  valid_data_folder = os.path.join(data_folder, 'test1')

  config = InceptionResNetConfig(model_type=model_type, n_classes=n_classes, batch_size=128)

  train_cache_folder = os.path.join(data_folder, config.model_type, 'train-cache')
  if not os.path.exists(train_cache_folder):
    os.makedirs(train_cache_folder)

  valid_cache_folder = os.path.join(data_folder, config.model_type, 'valid-cache')
  if not os.path.exists(valid_cache_folder):
    os.makedirs(valid_cache_folder)

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

  train(config, train_cache_folder, valid_cache_folder,
        use_aug=use_aug, data_mode=data_mode,
        ckpt_path=os.path.join(ckpt_folder, 'model%s%s.ckpt' % ('-aug' if use_aug else '-noaug', '-' + data_mode)),
        best_ckpt_path=os.path.join(ckpt_folder,
                                    'best_model%s%s.ckpt' % ('-aug' if use_aug else '-noaug', '-' + data_mode)))


if __name__ == '__main__':
  data_folder = r'Your data folder'

  # train_resnet(data_folder,  n_classes=100,layer=152, use_aug=False, data_mode='random')
  train_inception(data_folder, n_classes=100, model_type='inception_resnet_v2', use_aug=False, data_mode='random')
