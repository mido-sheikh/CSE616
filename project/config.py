import tensorflow as tf


x = tf.placeholder(tf.float32, (None, 32, 32, 1))   # x is a placeholder for a batch of input images.
y = tf.placeholder(tf.int32, (None))                # y is a placeholder for a batch of output labels.

keep_prob = tf.placeholder(tf.float32)       # For fully-connected layers
keep_prob_conv = tf.placeholder(tf.float32)  # For convolutional layers

EPOCHS = 30
BATCH_SIZE = 64
DIR = 'model'
