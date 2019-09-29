import tensorflow as tf
from ops import *

class Generator():

    def __init__(self, sess):
        self.input_shape = (128, 128, 3)
        self.residual_blocks = 6
        self.sess = sess

        with tf.variable_scope('generator'):
            self.input, self.output = self.build_generator()
            self.network_params = tf.trainable_variables()


    """
    Create a generator network using the hyperparameter values defined below
    """

    def build_generator(self):
        # placeholder and then feed dictionary

        input = tf.compat.v1.placeholder(tf.float32, shape=self.input_shape)

        # First Convolution block
        x = Conv2D(input, filters=32, kernel_size=7, strides=1, padding="same")
        x = InstanceNormalization(x)
        x = tf.compat.v1.nn.relu(x)

        # 2nd Convolution block
        x = Conv2D(x, filters=64, kernel_size=3, strides=2, padding="same")
        x = InstanceNormalization(x)
        x = tf.compat.v1.nn.relu(x)

        # 3rd Convolution block
        x = Conv2D(x, filters=128, kernel_size=3, strides=2, padding="same")
        x = InstanceNormalization(x)
        x = tf.compat.v1.nn.relu(x)

        # Residual blocks
        for _ in range(self.residual_blocks):
            x = residual_block(x)

        # Upsampling blocks

        # 1st Upsampling block
        x = Conv2DTranspose(x, filters=64, kernel_size=3, strides=2, padding='same', use_bias=False)
        x = InstanceNormalization(x)
        x = tf.compat.v1.nn.relu(x)

        # 2nd Upsampling block
        x = Conv2DTranspose(x, filters=32, kernel_size=3, strides=2, padding='same', use_bias=False)
        x = InstanceNormalization(x)
        x = tf.compat.v1.nn.relu(x)

        # Last Convolution layer
        x = Conv2D(x, filters=3, kernel_size=7, strides=1, padding="same")
        output = tf.compat.v1.nn.tanh(x)

        # model = Model(inputs=[input_layer], outputs=[output])
        return input, output

    def residual_block(x):
        """
        Residual block conv2d(image, options.df_dim, name='d_h0_conv') lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        """
        res = Conv2D(x, filters=128, kernel_size=3, strides=1, padding="same")
        res = BatchNormalization(res, axis=3, momentum=0.9, epsilon=1e-5)
        res = tf.compat.v1.nn.relu(res)

        res = Conv2D(res, filters=128, kernel_size=3, strides=1, padding="same")
        res = BatchNormalization(res, axis=3, momentum=0.9, epsilon=1e-5)

        return Add(res, x)

    def predict(self, inputs):
        return self.sess.run([self.output], feed_dict={
            self.input: inputs,
        })


class Discriminator():
    """
    Create a discriminator network using the hyperparameter values defined below
    """

    def __init__(self, sess):
        self.input_shape = (128, 128, 3)
        self.hidden_layers = 3
        self.sess = sess

        with tf.variable_scope('discriminator'):
            self.input, self.output = self.build_discriminator()
            self.network_params = tf.trainable_variables()


    def build_discriminator(self):

        input = tf.compat.v1.placeholder(tf.float32, shape=self.input_shape)

        x = ZeroPadding2D(input, padding=(1, 1))

        # 1st Convolutional block
        x = Conv2D(x, filters=64, kernel_size=4, strides=2, padding="valid")
        x = LeakyReLU(x, alpha=0.2)

        x = ZeroPadding2D(x, padding=[1, 1])

        # 3 Hidden Convolution blocks
        for i in range(1, hidden_layers + 1):
            x = Conv2D(x, filters=2 ** i * 64, kernel_size=4, strides=2, padding="valid")
            x = InstanceNormalization(x)
            x = LeakyReLU(x, alpha=0.2)

            x = ZeroPadding2D(x, padding=(1, 1))

        # Last Convolution layer
        pre_output = Conv2D(x, filters=1, kernel_size=4, strides=1, padding="valid")
        output = tf.nn.sigmoid(pre_output)

        return input, output

    def predict(self, inputs):
        return self.sess.run([self.output], feed_dict={
            self.input: inputs,
        })
