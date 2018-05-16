import tensorflow as tf


class TransferModel(object):
  def __init__(self, config, is_train):
    self.bottleneck_input = tf.placeholder(tf.float32, [None, config.bottleneck_tensor_size],
                                           name='bottleneck_input')
    self.labels = tf.placeholder(tf.float32, [None, config.n_classes], name='labels')

    # define a FC layer as the classifier,
    # it takes as input the extracted features by pre-trained model(bottleneck_input)
    # we only train this layer
    with tf.name_scope('fc-layer'):
      weights = tf.Variable(tf.truncated_normal([config.bottleneck_tensor_size, config.n_classes], stddev=0.001))
      biases = tf.Variable(tf.zeros([config.n_classes]))
      logits = tf.matmul(self.bottleneck_input, weights) + biases
      softmax_logits = tf.nn.softmax(logits)

    # loss function & train op
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
    self.loss = tf.reduce_mean(cross_entropy)

    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
      config.learning_rate,
      global_step,
      100,
      0.985
    )
    self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

    # calculate accuracy
    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(softmax_logits, 1), tf.argmax(self.labels, -1))
      self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      top5_correction_predcition = tf.nn.in_top_k(predictions=softmax_logits,
                                                  targets=tf.argmax(self.labels, -1), k=5)
      self.top5_acc = tf.reduce_mean(tf.cast(top5_correction_predcition, tf.float32))
