import tensorflow as tf
import numpy as np


def get_radius_angle(h, w):
    result = np.zeros(shape=(h, w, 2), dtype=np.float32)
    x_c = w / 2
    y_c = h / 2
    r_max = np.sqrt(x_c ** 2 + y_c ** 2)
    x = np.arange(w, dtype=np.float32) + np.zeros(shape=(h, w))
    y = np.arange(h, dtype=np.float32) + np.zeros(shape=(w, h))
    y = np.transpose(y, axes=(1, 0))
    r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
    cos_theta = (x - x_c) / r
    theta = np.arccos(cos_theta)
    theta[int(y_c), int(x_c)] = 0 # FIX division on 0
    theta[int(y_c)+1:, :] = 2 * np.pi - theta[int(y_c)+1:, :]
    r /= r_max
    result[:, :, 0] = theta
    result[:, :, 1] = r
    return result


def get_rad_vec(img_shape, crop_size, w, h):
    # creation of radius and angle stacked tensor
    x_c = tf.cast(img_shape[1], tf.float32) / 2
    y_c = tf.cast(img_shape[0], tf.float32) / 2
    r_max = tf.sqrt(x_c ** 2 + y_c ** 2)

    x = tf.cast(tf.range(crop_size), dtype=tf.float32)
    x = tf.reshape(x, (1, crop_size)) + tf.zeros(shape=(crop_size, crop_size))
    y = tf.cast(tf.range(crop_size), dtype=tf.float32)
    y = tf.reshape(y, (crop_size, 1)) + tf.zeros(shape=(crop_size, crop_size))

    x = tf.cast(w, tf.float32) + x
    y = tf.cast(h, tf.float32) + y
    r = tf.sqrt((x - x_c) ** 2 + (y - y_c) ** 2) + 1e-10
    cos_theta = (x - x_c) / r
    theta = tf.math.acos(cos_theta)
    theta = tf.where(tf.math.is_nan(theta), 0.0, theta)
    # elwise if true -> x else -> y
    theta = tf.where(y <= y_c, theta, 2 * np.pi - theta)
    r /= r_max

    return tf.stack([theta, r], axis=2)


def AIN(input_tensor, gamma, beta, kernel_size=3):
    mean, variance = tf.nn.moments(input_tensor, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + 1e-5)
    normalized = (input_tensor - mean) * inv
    return normalized * gamma + beta


def AIN_ResBlock(input_tensor, gamma, beta, filters, kernel_size=3):
    x = AIN(input_tensor, gamma, beta, kernel_size)
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = AIN(x, gamma, beta, kernel_size)
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    return x + input_tensor


def AIN_ResBlock_1x1(input_tensor, gamma, beta, filters, kernel_size=3):
    x = AIN(input_tensor, gamma, beta, kernel_size)
    x1 = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='same')(x)
    x1 = tf.keras.layers.LeakyReLU(alpha=0.02)(x1)
    # skip 1x1
    x2 = tf.keras.layers.Conv2D(filters, 1, strides=1, padding='same')(x)
    x2 = tf.keras.layers.LeakyReLU(alpha=0.02)(x2)
    return x1 + x2


def AIN_ResNet(input_tensor, adaptive_tensor, kernel_size=3):
    adapt = tf.keras.layers.Conv2D(
        64, kernel_size, strides=1, padding='same')(adaptive_tensor)
    adapt = tf.keras.layers.LeakyReLU(alpha=0.02)(adapt)

    gamma_64 = tf.keras.layers.Conv2D(
        64, kernel_size, strides=1, padding='same')(adapt)
    gamma_64 = tf.keras.layers.LeakyReLU(alpha=0.02)(gamma_64)
    beta_64 = tf.keras.layers.Conv2D(
        64, kernel_size, strides=1, padding='same')(adapt)
    beta_64 = tf.keras.layers.LeakyReLU(alpha=0.02)(beta_64)
    
    x = tf.keras.layers.Conv2D(
        64, kernel_size, strides=1, padding='same')(input_tensor)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = AIN_ResBlock(x, gamma_64, beta_64, filters=64, kernel_size=kernel_size)
    last_64 = AIN_ResBlock(x, gamma_64, beta_64, filters=64, kernel_size=kernel_size)

    gamma_128 = tf.keras.layers.Conv2D(
        128, kernel_size, strides=2, padding='same')(gamma_64)
    gamma_128 = tf.keras.layers.LeakyReLU(alpha=0.02)(gamma_128)
    beta_128 = tf.keras.layers.Conv2D(
        128, kernel_size, strides=2, padding='same')(beta_64)
    beta_128 = tf.keras.layers.LeakyReLU(alpha=0.02)(beta_128)

    x = tf.keras.layers.Conv2D(
        128, kernel_size, strides=2, padding='same')(last_64)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = AIN_ResBlock(x, gamma_128, beta_128, filters=128, kernel_size=kernel_size)
    last_128 = AIN_ResBlock(x, gamma_128, beta_128, filters=128, kernel_size=kernel_size)

    gamma_256 = tf.keras.layers.Conv2D(
        256, kernel_size, strides=2, padding='same')(gamma_128)
    gamma_256 = tf.keras.layers.LeakyReLU(alpha=0.02)(gamma_256)
    beta_256 = tf.keras.layers.Conv2D(
        256, kernel_size, strides=2, padding='same')(beta_128)
    beta_256 = tf.keras.layers.LeakyReLU(alpha=0.02)(beta_256)

    x = tf.keras.layers.Conv2D(
        256, kernel_size, strides=2, padding='same')(last_128)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = AIN_ResBlock(x, gamma_256, beta_256, filters=256, kernel_size=kernel_size)
    # x = AIN_ResBlock(x, gamma_256, beta_256, filters=256, kernel_size=kernel_size)
    x = AIN_ResBlock(x, gamma_256, beta_256, filters=256, kernel_size=kernel_size)

    x = tf.keras.layers.Conv2DTranspose(
        128, kernel_size, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x += last_128
    x = AIN_ResBlock(x, gamma_128, beta_128, filters=128, kernel_size=kernel_size)
    x = AIN_ResBlock(x, gamma_128, beta_128, filters=128, kernel_size=kernel_size)

    x = tf.keras.layers.Conv2DTranspose(
        64, kernel_size, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x += last_64
    x = AIN_ResBlock(x, gamma_64, beta_64, filters=64, kernel_size=kernel_size)
    x = AIN_ResBlock(x, gamma_64, beta_64, filters=64, kernel_size=kernel_size)

    out = tf.keras.layers.Conv2D(3, 1, strides=1, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs=(input_tensor, adaptive_tensor), outputs=out)


def AIN_ResNet_v1(input_tensor, adaptive_tensor, kernel_size=3):
    gamma_4 = tf.keras.layers.Conv2D(
        4, kernel_size, strides=1, padding='same')(adaptive_tensor)
    gamma_4 = tf.keras.layers.LeakyReLU(alpha=0.02)(gamma_4)
    beta_4 = tf.keras.layers.Conv2D(
        4, kernel_size, strides=1, padding='same')(adaptive_tensor)
    beta_4 = tf.keras.layers.LeakyReLU(alpha=0.02)(beta_4)

    gamma_16 = tf.keras.layers.Conv2D(
        16, kernel_size, strides=2, padding='same')(gamma_4)
    gamma_16 = tf.keras.layers.LeakyReLU(alpha=0.02)(gamma_16)
    beta_16 = tf.keras.layers.Conv2D(
        16, kernel_size, strides=2, padding='same')(beta_4)
    beta_16 = tf.keras.layers.LeakyReLU(alpha=0.02)(beta_16)

    gamma_64 = tf.keras.layers.Conv2D(
        64, kernel_size, strides=2, padding='same')(gamma_16)
    gamma_64 = tf.keras.layers.LeakyReLU(alpha=0.02)(gamma_64)
    beta_64 = tf.keras.layers.Conv2D(
        64, kernel_size, strides=2, padding='same')(beta_16)
    beta_64 = tf.keras.layers.LeakyReLU(alpha=0.02)(beta_64)

    gamma_256 = tf.keras.layers.Conv2D(
        256, kernel_size, strides=2, padding='same')(gamma_64)
    gamma_256 = tf.keras.layers.LeakyReLU(alpha=0.02)(gamma_256)
    beta_256 = tf.keras.layers.Conv2D(
        256, kernel_size, strides=2, padding='same')(beta_64)
    beta_256 = tf.keras.layers.LeakyReLU(alpha=0.02)(beta_256)

    x = tf.keras.layers.Conv2D(4, 1, strides=1, padding='same')(input_tensor)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)

    x2 = tf.nn.space_to_depth(x, 2)
    x4 = tf.nn.space_to_depth(x2, 2)
    x8 = tf.nn.space_to_depth(x4, 2)

    x8 = AIN_ResBlock_1x1(
        x8, gamma_256, beta_256, filters=256, kernel_size=kernel_size)
    x4 = AIN_ResBlock_1x1(
        x4, gamma_64, beta_64, filters=64, kernel_size=kernel_size)
    x2 = AIN_ResBlock_1x1(
        x2, gamma_16, beta_16, filters=16, kernel_size=kernel_size)
    x = AIN_ResBlock_1x1(
        x, gamma_4, beta_4, filters=4, kernel_size=kernel_size)

    up4 = tf.nn.depth_to_space(x8, 2)
    up4 = up4 + x4
    up4 = AIN_ResBlock_1x1(
        up4, gamma_64, beta_64, filters=64, kernel_size=kernel_size)

    up2 = tf.nn.depth_to_space(up4, 2)
    up2 = up2 + x2
    up2 = AIN_ResBlock_1x1(
        up2, gamma_16, beta_16, filters=16, kernel_size=kernel_size)

    up = tf.nn.depth_to_space(up2, 2)
    up = up + x
    up = AIN_ResBlock_1x1(
        up, gamma_4, beta_4, filters=16, kernel_size=kernel_size)

    x = tf.keras.layers.Conv2D(
        32, 1, strides=1, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)

    out = tf.keras.layers.Conv2D(3, 1, strides=1, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs=(input_tensor, adaptive_tensor), outputs=out)
