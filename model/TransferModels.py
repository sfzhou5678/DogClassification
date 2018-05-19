import tensorflow as tf

slim = tf.contrib.slim


class TransferFCModel(object):
  def __init__(self, config, is_training):
    self.bottleneck_input = tf.placeholder(tf.float32, [None, config.bottleneck_tensor_size],
                                           name='bottleneck_input')
    self.labels = tf.placeholder(tf.float32, [None, config.n_classes], name='labels')

    # define FC layers as the classifier,
    # it takes as input the extracted features by pre-trained models(bottleneck_input)
    # we only train these layers
    with tf.name_scope('fc-layer'):
      net = slim.fully_connected(self.bottleneck_input, 4096)
      net = slim.dropout(net, 0.35, is_training=is_training)

      net = slim.fully_connected(net, 2048)
      net = slim.dropout(net, 0.5, is_training=is_training)

      logits = slim.fully_connected(net, config.n_classes, activation_fn=None, scope='output_layer')
      self.softmax_logits = tf.nn.softmax(logits)

    # loss function & train op
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
    self.loss = tf.reduce_mean(cross_entropy)

    if is_training:
      global_step = tf.contrib.framework.get_or_create_global_step()
      learning_rate = tf.train.exponential_decay(
        config.learning_rate,
        global_step,
        5000,
        0.98
      )
      self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

    # calculate accuracy
    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(self.labels, -1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
