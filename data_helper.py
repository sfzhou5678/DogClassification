import os
import shutil


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


def move_train_data(train_data_folder, train_label_path, label_map):
  print('moving train data:')
  with open(train_label_path, encoding='utf-8') as f:
    lines = f.readlines()
    count = 0
    for line in lines:
      img_name, label, _ = line.strip().split()
      new_label = label_map[label]
      new_label_folder = os.path.join(train_data_folder, str(new_label))
      if not os.path.exists(new_label_folder):
        os.mkdir(new_label_folder)
      try:
        shutil.move(os.path.join(train_data_folder, img_name + '.jpg'),
                    os.path.join(new_label_folder, img_name + '.jpg'))
      except:
        pass
      count += 1
      if count % (len(lines) // 10) == 0:
        print(round(count / len(lines), 2) * 100, '%')


def data_augmentation(train_data_folder,
                      function,
                      balanced_data=False):
  """
  function: transformation function
  balanced_data: 
  :return:
  """
  pass


data_folder = r'D:\DeeplearningData\Dog identification'
label_map_path = os.path.join(data_folder, 'label_name.txt')

train_data_folder = os.path.join(data_folder, 'train')
train_label_path = os.path.join(data_folder, 'label_train.txt')

valid_data_folder = os.path.join(data_folder, 'test1')
valid_label_path = os.path.join(data_folder, 'label_val.txt')

new_label_map_path = os.path.join(data_folder, 'new_label_map.txt')

if __name__ == '__main__':
  label_map = map_label(train_label_path, new_label_map_path)
  move_train_data(train_data_folder, train_label_path, label_map)
