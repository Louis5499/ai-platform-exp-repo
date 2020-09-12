# import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python import keras

import os, json

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def input_fn(mode, input_context=None):
  # datasets, info = tfds.load(name='mnist',
  #                               with_info=True,
  #                               as_supervised=True)
  # mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
  #                  datasets['test'])

  # def scale(image, label):
  #   image = tf.cast(image, tf.float32)
  #   image /= 255
  #   return image, label

  # if input_context:
  #   mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
  #                                       input_context.input_pipeline_id)
  # return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

  (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
  CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  validation_images, validation_labels = train_images[:5000], train_labels[:5000]
  train_images, train_labels = train_images[5000:], train_labels[5000:]

  train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
  test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
  validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

  def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label

  train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
  test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
  validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
  print("Training data size:", train_ds_size)
  print("Test data size:", test_ds_size)
  print("Validation data size:", validation_ds_size)

  train_ds = (train_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))

  return train_ds


LEARNING_RATE = 1e-4
def model_fn(features, labels, mode):
  model = tf.keras.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
  ])
  logits = model(features, training=False)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {'logits': logits}
    return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

  optimizer = tf.compat.v1.train.GradientDescentOptimizer(
      learning_rate=LEARNING_RATE)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits)
  loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=optimizer.minimize(
          loss, tf.compat.v1.train.get_or_create_global_step()))

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

config = tf.estimator.RunConfig(train_distribute=strategy)

classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='/tmp/multiworker', config=config)
tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
)