import tensorflow as tf
from tensorflow.keras import layers


class EMA(tf.keras.layers.Layer):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = layers.Softmax(axis=-1)
        self.gn = layers.LayerNormalization(axis=-1)
        self.conv1x1 = layers.Conv2D(channels // self.groups, kernel_size=1, strides=1, padding='valid')
        self.conv3x3 = layers.Conv2D(channels // self.groups, kernel_size=3, strides=1, padding='same')

    def call(self, x):
        # print(x.shape)
        # Get the shape dynamically
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = tf.shape(x)[3]
        group_x = tf.reshape(x, (b * self.groups, h, w, -1))
        # Pooling along height and width separately
        x_h = tf.reduce_mean(group_x, axis=2, keepdims=True)
        x_w = tf.reduce_mean(group_x, axis=1, keepdims=True)
        x_w = tf.transpose(x_w, perm=[0, 2, 1, 3])
        hw=tf.concat([x_h, x_w], axis=1)
        hw = self.conv1x1(hw)
        x_h, x_w = tf.split(hw, num_or_size_splits=[h, w], axis=1)
        # print(x_h.shape)
        # print(x_w.shape)
        x=group_x*tf.tanh(x_h) * tf.tanh(tf.transpose(x_w, perm=[0, 2, 1, 3]))
        x1 = self.gn(x)
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(tf.reshape(tf.reduce_mean(x1, axis=[1, 2], keepdims=True), (b * self.groups, 1, -1)))
        x12 = tf.reshape(x2, (b * self.groups,  c // self.groups,-1))
        x21 = self.softmax(tf.reshape(tf.reduce_mean(x2, axis=[1, 2], keepdims=True), (b * self.groups, 1, -1)))
        x22 = tf.reshape(x1, (b * self.groups,  c // self.groups,-1))
        weights = tf.reshape(tf.matmul(x11, x12) + tf.matmul(x21, x22), (b * self.groups,  h, w,1))
        return tf.reshape(group_x * tf.tanh(weights), (b, h, w, c))
