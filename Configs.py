import time


class ResNetConfig():
  def __init__(self, train_data_folder, valid_data_folder,
               layer, batch_size):
    self.layer = layer

    self.model_type = 'ResNet-%d' % self.layer
    self.train_data_folder = train_data_folder
    self.valid_data_folder = valid_data_folder
    self.both_training = True

    self.init_scale = 0.05
    self.learning_rate = 1e-2
    self.steps = 12001
    self.batch_size = batch_size
    self.bottleneck_tensor_size = 2048
    self.n_classes = 100

    self.extra_inf = '-' + time.strftime('[%m%d-%H%M]', time.localtime(time.time()))
