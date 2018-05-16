import os
import random
import shutil
import numpy as np
from PIL import Image, ImageEnhance


def map_label(train_label_path, new_label_map_path):
  label_map = {}
  with open(train_label_path, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
      img_name, label, url = line.strip().split()
      if label not in label_map:
        label_map[label] = len(label_map)

  with open(new_label_map_path, 'w', encoding='utf-8') as wf:
    for raw_label in label_map:
      wf.write('%s %d\n' % (raw_label, label_map[raw_label]))

  return label_map


def move_data(data_folder, label_path, label_map, tag):
  print('moving %s data:' % tag)
  with open(label_path, encoding='utf-8') as f:
    lines = f.readlines()
    count = 0
    for line in lines:
      img_name, label, _ = line.strip().split()
      new_label = label_map[label]
      new_label_folder = os.path.join(data_folder, str(new_label))
      if not os.path.exists(new_label_folder):
        os.mkdir(new_label_folder)
      try:
        shutil.move(os.path.join(data_folder, img_name + '.jpg'),
                    os.path.join(new_label_folder, img_name + '.jpg'))
      except:
        pass
      count += 1
      if count % (len(lines) // 10) == 0:
        print(round(count / len(lines), 2) * 100, '%')


def save_img(img_array, file_path):
  j = Image.fromarray(img_array)
  j.save(file_path)


def data_augmentation(train_data_folder, n_classes,
                      function,
                      balanced_data=False):
  """

  function: transformation function
  balanced_data:
  :return:
  """
  print('data augmentation:')
  for category in range(n_classes):
    cur_folder = os.path.join(train_data_folder, str(category))
    aug_save_folder = os.path.join(cur_folder, 'aug')
    if not os.path.exists(aug_save_folder):
      os.makedirs(aug_save_folder)

    for file in os.listdir(cur_folder):
      if file == 'aug':
        continue

      filename = file[:file.index('.')]

      img = Image.open(os.path.join(cur_folder, file), mode="r")

      # flip left right
      img.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(aug_save_folder, filename + '_aug_flip.jpg'))

      img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT).save(
        os.path.join(aug_save_folder, filename + '_aug_transpose.jpg'))

      # saturation
      random_factor = np.random.randint(0, 31) / 10.
      ImageEnhance.Color(img).enhance(random_factor).save(
        os.path.join(aug_save_folder, filename + '_aug_saturation.jpg'))

      # brightness
      random_factor = np.random.randint(10, 21) / 10.
      ImageEnhance.Brightness(img).enhance(random_factor).save(
        os.path.join(aug_save_folder, filename + '_aug_brightness.jpg'))

      # contrast
      random_factor = np.random.randint(10, 21) / 10.
      ImageEnhance.Contrast(img).enhance(random_factor).save(
        os.path.join(aug_save_folder, filename + '_aug_contrast.jpg'))

      # sharpness
      random_factor = np.random.randint(0, 31) / 10.
      ImageEnhance.Sharpness(img).enhance(random_factor).save(
        os.path.join(aug_save_folder, filename + '_aug_sharp.jpg'))

    if category % 10 == 0:
      print(round(category / n_classes * 100, 1), '%')
  print('100 %')


data_folder = r'D:\DeeplearningData\Dog identification'
label_map_path = os.path.join(data_folder, 'label_name.txt')

train_data_folder = os.path.join(data_folder, 'train')
train_label_path = os.path.join(data_folder, 'label_train.txt')

valid_data_folder = os.path.join(data_folder, 'test1')
valid_label_path = os.path.join(data_folder, 'label_val.txt')

new_label_map_path = os.path.join(data_folder, 'new_label_map.txt')

if __name__ == '__main__':
  # label_map = map_label(train_label_path, new_label_map_path)
  # move_data(train_data_folder, train_label_path, label_map, tag='train')
  # move_data(valid_data_folder, valid_label_path, label_map, tag='valid')

  data_augmentation(train_data_folder, n_classes=100, function=None, balanced_data=False)
