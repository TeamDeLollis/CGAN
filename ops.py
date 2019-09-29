import tensorflow as tf


def Conv2D(input, filters, kernel_size, strides, padding):
    # return tf.nn.conv2d(input, filter=filters, strides=strides, padding=padding, use_cudnn_on_gpu=True)
    # check that there is no relu !! use_cudnn_on_gpu=True
    return tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            activation=None)


def Conv2DTranspose(input, filters, kernel_size, strides, padding, use_bias):
    tf.contrib.layers.conv2d_transpose(input, filters, kernel_size, stride=strides,
                                       padding=padding, activation_fn=None, biases_initializer=use_bias)


def BatchNormalization(input, axis, momentum, epsilon):
    return tf.layers.batch_normalization(input, axis=axis, momentum=momentum, epsilon=epsilon)


def Add(a, b):
    return tf.compat.v1.add([a, b])


def LeakyReLU(x, alpha):
    return tf.compat.v1.nn.leaky_relu(x, alpha=alpha)

"""

----------  NOT WORKING -------------

def InstanceNormalization(x):
    # x = InstanceNormalization(axis=1)(x)
    #tf.contrib.layers.instance_norm(x)
    with tf.variable_scope("instance_norm", reuse=tf.AUTO_REUSE):
        depth = x.get_shape()[3]
        #scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(x, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x-mean)*inv
        return scale*normalized + offset
"""

def ZeroPadding2D(x, padding):
    # paddings = tf.constant([[1, 1, ], [2, 2]]) #CHECK DIMENSION BEFORE ENTERING, SHOULD BE (NB=1,3,npix,npix)
    # 'constant_values' is 0.
    # rank of 't' is 2.
    # tf.pad(t, paddings, "CONSTANT")

    # --> dovrebbe essere
    if padding == [1, 1]:
        padd = tf.constant([[0, 0],[1, 1], [1, 1], [0, 0]])
        return tf.pad(x, padd, "CONSTANT")
