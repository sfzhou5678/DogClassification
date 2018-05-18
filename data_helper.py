import os
import random
import shutil
import numpy as np
from PIL import Image, ImageEnhance

import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)


def heavy_aug(img, times=16):
  images = np.array(
    [img for _ in range(times)],
    dtype=np.uint8
  )

  # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
  # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
  # image.
  sometimes = lambda aug: iaa.Sometimes(0.5, aug)

  # Define our sequence of augmentation steps that will be applied to every image.
  seq = iaa.Sequential(
    [
      #
      # Apply the following augmenters to most images.
      #
      iaa.Fliplr(0.5),  # horizontally flip 50% of all images
      iaa.Flipud(0.2),  # vertically flip 20% of all images

      # crop some of the images by 0-10% of their height/width
      sometimes(iaa.Crop(percent=(0, 0.1))),

      # Apply affine transformations to some of the images
      # - scale to 80-120% of image height/width (each axis independently)
      # - translate by -20 to +20 relative to height/width (per axis)
      # - rotate by -45 to +45 degrees
      # - shear by -16 to +16 degrees
      # - order: use nearest neighbour or bilinear interpolation (fast)
      # - mode: use any available mode to fill newly created pixels
      #         see API or scikit-image for which modes are available
      # - cval: if the mode is constant, then use a random brightness
      #         for the newly created pixels (e.g. sometimes black,
      #         sometimes white)
      sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-16, 16),
        order=[0, 1],
        cval=(0, 255),
        mode=ia.ALL
      )),

      #
      # Execute 0 to 5 of the following (less important) augmenters per
      # image. Don't execute all of them, as that would often be way too
      # strong.
      #
      iaa.SomeOf((0, 5),
                 [
                   # Convert some images into their superpixel representation,
                   # sample between 20 and 200 superpixels per image, but do
                   # not replace all superpixels with their average, only
                   # some of them (p_replace).
                   sometimes(
                     iaa.Superpixels(
                       p_replace=(0, 1.0),
                       n_segments=(20, 200)
                     )
                   ),

                   # Blur each image with varying strength using
                   # gaussian blur (sigma between 0 and 3.0),
                   # average/uniform blur (kernel size between 2x2 and 7x7)
                   # median blur (kernel size between 3x3 and 11x11).
                   iaa.OneOf([
                     iaa.GaussianBlur((0, 3.0)),
                     iaa.AverageBlur(k=(2, 7)),
                     iaa.MedianBlur(k=(3, 11)),
                   ]),

                   # Sharpen each image, overlay the result with the original
                   # image using an alpha between 0 (no sharpening) and 1
                   # (full sharpening effect).
                   iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                   # Same as sharpen, but for an embossing effect.
                   iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                   # Search in some images either for all edges or for
                   # directed edges. These edges are then marked in a black
                   # and white image and overlayed with the original image
                   # using an alpha of 0 to 0.7.
                   sometimes(iaa.OneOf([
                     iaa.EdgeDetect(alpha=(0, 0.7)),
                     iaa.DirectedEdgeDetect(
                       alpha=(0, 0.7), direction=(0.0, 1.0)
                     ),
                   ])),

                   # Add gaussian noise to some images.
                   # In 50% of these cases, the noise is randomly sampled per
                   # channel and pixel.
                   # In the other 50% of all cases it is sampled once per
                   # pixel (i.e. brightness change).
                   iaa.AdditiveGaussianNoise(
                     loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                   ),

                   # Either drop randomly 1 to 10% of all pixels (i.e. set
                   # them to black) or drop them on an image with 2-5% percent
                   # of the original size, leading to large dropped
                   # rectangles.
                   iaa.OneOf([
                     iaa.Dropout((0.01, 0.1), per_channel=0.5),
                     iaa.CoarseDropout(
                       (0.03, 0.15), size_percent=(0.02, 0.05),
                       per_channel=0.2
                     ),
                   ]),

                   # Invert each image's chanell with 5% probability.
                   # This sets each pixel value v to 255-v.
                   iaa.Invert(0.05, per_channel=True),  # invert color channels

                   # Add a value of -10 to 10 to each pixel.
                   iaa.Add((-10, 10), per_channel=0.5),

                   # Change brightness of images (50-150% of original value).
                   iaa.Multiply((0.5, 1.5), per_channel=0.5),

                   # Improve or worsen the contrast of images.
                   iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                   # Convert each image to grayscale and then overlay the
                   # result with the original with random alpha. I.e. remove
                   # colors with varying strengths.
                   iaa.Grayscale(alpha=(0.0, 1.0)),

                   # In some images move pixels locally around (with random
                   # strengths).
                   sometimes(
                     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                   ),

                   # In some images distort local areas with varying strength.
                   sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                 ],
                 # do all of the above augmentations in random order
                 random_order=True
                 )
    ],
    # do all of the above augmentations in random order
    random_order=True
  )

  images_aug = seq.augment_images(images)
  return images_aug


def simple_aug(img, times=16):
  images = np.array(
    [img for _ in range(times)],
    dtype=np.uint8
  )

  seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Crop(percent=(0, 0.1)),  # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
                  iaa.GaussianBlur(sigma=(0, 0.5))
                  ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
      scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
      translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
      rotate=(-25, 25),
      shear=(-8, 8)
    )
  ], random_order=True)  # apply augmenters in random order

  images_aug = seq.augment_images(images)

  return images_aug


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


def data_augmentation(train_data_folder, size,
                      n_classes, sample_num=16):
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
      if size is not None:
        img = img.resize(size)
      img = np.array(img)

      simple_image_aug = simple_aug(img, sample_num)
      heavy_image_aug = heavy_aug(img, sample_num)

      for img, i in zip(simple_image_aug, range(len(simple_image_aug))):
        Image.fromarray(img).save(os.path.join(aug_save_folder, filename + '_simple_aug-%d.jpg' % i))

      for img, i in zip(heavy_image_aug, range(len(heavy_image_aug))):
        Image.fromarray(img).save(os.path.join(aug_save_folder, filename + '_heavy_aug-%d.jpg' % i))

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

  data_augmentation(train_data_folder, size=(224, 224), n_classes=100, sample_num=4)
