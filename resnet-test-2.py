import numpy as np
import time
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.datasets import cifar10
# import pydot
# from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.initializers import glorot_uniform
# import scipy.misc
# from matplotlib.pyplot import imshow
# %matplotlib inline
from tensorflow.python.keras import backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
import os
import json

def identity_block(X, f, filters,stage, block):
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  
  F1, F2, F3 = filters
  
  X_shortcut = X
  
  X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
  X = Activation('relu')(X)
  
  X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
  X = Activation('relu')(X)
  
  X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base+'2c', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = bn_name_base+'2c')(X)
  
  X = Add()([X, X_shortcut])
  X = Activation('relu')(X)
  
  return X
 
def convolutional_block(X, f, filters, stage, block, s = 2):
  conv_name_base = 'res' + str(stage) + block +'_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  
  F1, F2, F3 = filters
  
  X_shortcut = X
  
  X = Conv2D(F1, (1, 1), strides = (s, s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
  X = Activation('relu')(X)
  
  X = Conv2D(F2, (f, f), strides = (1, 1), name = conv_name_base + '2b', padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
  X = Activation('relu')(X)
  
  X = Conv2D(F3, (1, 1), strides = (1, 1), name = conv_name_base + '2c', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
  
  X_shortcut = Conv2D(F3, (1,1), strides=(s, s), name=conv_name_base+'1', padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
  X_shortcut = BatchNormalization(axis = 3, name=bn_name_base+'1')(X_shortcut)
  
  X = Add()([X, X_shortcut])
  X = Activation('relu')(X)
  
  return X
 
def ResNet50(input_shape = (32, 32, 3), classes = 10):
  X_input = Input(input_shape)
  
  X = ZeroPadding2D((3, 3))(X_input)
  
  X = Conv2D(64, (7, 7), strides=(2, 2), name = 'conv1', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((3, 3), strides=(2,2))(X)
  
  X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block ='a', s=1)
  X = identity_block(X, 3, [64, 64, 256], stage = 2, block = 'b')
  X = identity_block(X, 3, [64, 64, 256], stage = 2, block = 'c')
  
  X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
  X = identity_block(X, f=3, filters=[128, 128, 512], stage = 3, block = 'b')
  X = identity_block(X, f=3, filters=[128, 128, 512], stage = 3, block = 'c')
  X = identity_block(X, f=3, filters=[128, 128, 512], stage = 3, block = 'd')
  
  X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
  X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
  X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
  X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
  X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
  X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='f')
 
  X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
  X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
  X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='c')
  
  X = Flatten()(X)
  X = Dense(classes, activation='softmax', name='fc'+str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
  
  model = Model(inputs = X_input, outputs = X, name='ResNet50')
  
  return model
 
def input_fn():
  X = np.random.random((1, 32, 32, 3))
  Y = np.random.random((1, 10))
  dataset = tf.data.Dataset.from_tensor_slices((X, Y))
  dataset = dataset.repeat(10)
  dataset = dataset.batch(128)
  return dataset
 
tf_config = os.environ.get('TF_CONFIG', '{}')
tf.logging.info("TF_CONFIG %s", tf_config)
tf_config_json = json.loads(tf_config)
cluster = tf_config_json.get('cluster')
job_name = tf_config_json.get('task', {}).get('type')
task_index = tf_config_json.get('task', {}).get('index')

tf_config_json["task"] = {"index": 0, "type": "chief"}
cluster["chief"] = [cluster["worker"][0]]
tf_config_json["cluster"] = cluster

os.environ["TF_CONFIG"] = json.dumps(tf_config_json)
print("cluster=%s job_name=%s task_index=%s", cluster, job_name, task_index)

model = ResNet50(input_shape = (32, 32, 3), classes = 10)
 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
(X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = cifar10.load_data()
 
X_train = X_train_orig/255.
X_test = X_test_orig/255.
 
Y_train =  keras.utils.to_categorical(Y_train_orig,10)
Y_test =  keras.utils.to_categorical(Y_test_orig,10)
 
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
 
keras_estimator = tf.keras.estimator.model_to_estimator(keras_model = model)
 
train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={'input_1': X_train.astype(np.float32)},
  y=Y_train.astype(np.float32),
  num_epochs = 1,
  batch_size = 128,
  shuffle=True)
 
time_start = time.time()
keras_estimator.train(input_fn=train_input_fn, steps=390)
time_end = time.time()
print (time_end - time_start)
 
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={model.input_names[0]: X_test.astype(np.float32)},
    y=Y_test.astype(np.float32),
    num_epochs=1,
    shuffle=True)
 
eva =keras_estimator .evaluate(input_fn=test_input_fn)
print(eva)
