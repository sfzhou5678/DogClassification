import time


class ResNetConfig():
  def __init__(self, layer, n_classes, batch_size):
    self.layer = layer

    self.model_type = 'ResNet-%d' % self.layer

    self.init_scale = 0.05
    self.learning_rate = 1e-2
    self.batch_size = batch_size
    self.bottleneck_tensor_size = 2048
    self.n_classes = n_classes

    self.extra_inf = '-' + time.strftime('[%m%d-%H%M]', time.localtime(time.time()))


class InceptionResNetConfig():
  def __init__(self, model_type, n_classes, batch_size):
    self.model_type = model_type

    self.init_scale = 0.05
    self.learning_rate = 1e-2
    self.batch_size = batch_size
    self.bottleneck_tensor_size = 1536
    self.n_classes = n_classes

    self.extra_inf = '-' + time.strftime('[%m%d-%H%M]', time.localtime(time.time()))
