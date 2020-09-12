import argparse
import sys
import os
import ast

from tensorflow.python import keras
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import numpy as np

class Empty:
  pass

FLAGS = Empty()


def batch_norm(inputs,is_train,is_conv_out=True,decay=0.999):
    scale=tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_train:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, 0.001)



def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      learning_rate = 1e-4
      training_iters = 200
      batch_size = 50
      display_step = 5
      n_classes = 2
      n_fc1 = 4096
      n_fc2 = 2048

      # 构建模型
      x = tf.placeholder(tf.float32, [None, 32, 32, 3])
      y = tf.placeholder(tf.float32, [None, n_classes])

      W_conv = {
          'conv1': tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.0001)),
          'conv2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01)),
          'conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01)),
          'conv4': tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01)),
          'conv5': tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01)),
          'fc1': tf.Variable(tf.truncated_normal([6 * 6 * 256, n_fc1], stddev=0.1)),
          'fc2': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1)),
          'fc3': tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=0.1))
      }
      b_conv = {
          'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[96])),
          'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
          'conv3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
          'conv4': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
          'conv5': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
          'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc1])),
          'fc2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc2])),
          'fc3': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes]))
      }

      x_image = tf.reshape(x, [-1, 32, 32, 3])

      # 卷积层 1
      conv1 = tf.nn.conv2d(x_image, W_conv['conv1'], strides=[1, 4, 4, 1], padding='VALID')
      conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
      conv1 = batch_norm(conv1, True)
      conv1 = tf.nn.relu(conv1)
      # 池化层 1
      pool1 = tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
      norm1 = tf.nn.lrn(pool1, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

      # 卷积层 2
      conv2 = tf.nn.conv2d(pool1, W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
      conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
      conv2 = batch_norm(conv2, True)
      conv2 = tf.nn.relu(conv2)
      # 池化层 2
      pool2 = tf.nn.avg_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

      # 卷积层3
      conv3 = tf.nn.conv2d(pool2, W_conv['conv3'], strides=[1, 1, 1, 1], padding='SAME')
      conv3 = tf.nn.bias_add(conv3, b_conv['conv3'])
      conv3 = batch_norm(conv3, True)
      conv3 = tf.nn.relu(conv3)

      # 卷积层4
      conv4 = tf.nn.conv2d(conv3, W_conv['conv4'], strides=[1, 1, 1, 1], padding='SAME')
      conv4 = tf.nn.bias_add(conv4, b_conv['conv4'])
      conv4 = batch_norm(conv4, True)
      conv4 = tf.nn.relu(conv4)

      # 卷积层5
      conv5 = tf.nn.conv2d(conv4, W_conv['conv5'], strides=[1, 1, 1, 1], padding='SAME')
      conv5 = tf.nn.bias_add(conv5, b_conv['conv5'])
      conv5 = batch_norm(conv5, True)
      conv5 = tf.nn.relu(conv5)

      # 池化层5
      pool5 = tf.nn.avg_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
      reshape = tf.reshape(pool5, [-1, 6 * 6 * 256])
      fc1 = tf.add(tf.matmul(reshape, W_conv['fc1']), b_conv['fc1'])
      fc1 = batch_norm(fc1, True, False)
      fc1 = tf.nn.relu(fc1)

      # 全连接层 2
      fc2 = tf.add(tf.matmul(fc1, W_conv['fc2']), b_conv['fc2'])
      fc2 = batch_norm(fc2, True, False)
      fc2 = tf.nn.relu(fc2)
      fc3 = tf.add(tf.matmul(fc2, W_conv['fc3']), b_conv['fc3'])

      # 定义损失
      global_step = tf.train.get_or_create_global_step()
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=fc3))
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
      # 评估模型
      # correct_pred = tf.equal(tf.argmax(fc3,1),tf.argmax(y,1))
      # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

      # init = tf.global_variables_initializer()

    def onehot(labels):
      '''one-hot 编码'''
      n_sample = len(labels)
      n_class = max(labels) + 1
      onehot_labels = np.zeros((n_sample, n_class))
      onehot_labels[np.arange(n_sample), labels] = 1
      return onehot_labels

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=FLAGS.global_steps)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           config=tf.ConfigProto(
                                               device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
                                           ),
                                           hooks=hooks) as mon_sess:

      while not mon_sess.should_stop():
        batch_xs, batch_ys = train_images[:100], train_labels[:100]
        _, step = mon_sess.run([optimizer, global_step], feed_dict={x: batch_xs, y: batch_ys})

        sys.stderr.write('global_step: '+str(step))
        sys.stderr.write('\n')


if __name__ == "__main__":
  TF_CONFIG = ast.literal_eval(os.environ["TF_CONFIG"])
  FLAGS.job_name = TF_CONFIG["task"]["type"]
  FLAGS.task_index = TF_CONFIG["task"]["index"]
  FLAGS.ps_hosts = ",".join(TF_CONFIG["cluster"]["ps"])
  FLAGS.worker_hosts = ",".join(TF_CONFIG["cluster"]["worker"])
  FLAGS.global_steps = int(os.environ["global_steps"]) if "global_steps" in os.environ else 100000
  tf.app.run(main=main, argv=[sys.argv[0]])
