import os
import skimage.io
import skimage.transform


def load_image(path, size=224, batch_size=1):
  img = skimage.io.imread(path)
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
  resized_img = skimage.transform.resize(crop_img, (size, size))

  return resized_img.reshape((batch_size, size, size, 3))
