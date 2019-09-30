import tensorflow as tf


def Conv2D(input, filters, kernel_size, strides, padding):
    return tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            activation=None)


def Conv2DTranspose(input, filters, kernel_size, strides, padding, use_bias):
    return tf.contrib.layers.conv2d_transpose(input, filters, kernel_size=kernel_size, stride=strides,
                                       padding=padding, activation_fn=None)

def BatchNormalization(input, axis, momentum, epsilon):
    return tf.layers.batch_normalization(input, axis=axis, momentum=momentum, epsilon=epsilon)

def Add(a, b):
    return tf.compat.v1.add(a, b)


def LeakyReLU(x, alpha):
    return tf.compat.v1.nn.leaky_relu(x, alpha=alpha)

def InstanceNormalization(x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (x - mean) * inv
    return normalized


def ZeroPadding2D(x, padding):
    if padding == [1, 1]:
        padd = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        return tf.pad(x, padd, "CONSTANT")
