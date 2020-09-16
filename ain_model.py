import tensorflow as tf
from tensorflow.keras.layers import Conv2D, PReLU, AvgPool2D, UpSampling2D


def AIN(inputs, noise_map, filters, kernel_size=3, down=1):
    # input normalization
    mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + 1e-5)
    normalized = (inputs - mean) * inv

    if down > 1:
        noise_map = AvgPool2D(pool_size=down, padding='same')(noise_map)

    tmp = Conv2D(filters, kernel_size, strides=1,
                 padding='same', activation='relu')(noise_map)
    gamma = Conv2D(filters, kernel_size, strides=1, padding='same')(tmp)
    beta = Conv2D(filters, kernel_size, strides=1, padding='same')(tmp)
    
    return normalized * (1 + gamma) + beta


def AIN_ResBlock(inputs, noise_map, filters, kernel_size=3, down=1):
    x = AIN(inputs, noise_map, filters, kernel_size, down)
    x = PReLU()(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = AIN(inputs, noise_map, filters, kernel_size, down)
    x = PReLU()(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    return x + inputs


def NoiseEstimator(inputs):
    # https://arxiv.org/pdf/2002.11244v2.pdf
    x = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = AvgPool2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(3, 3, padding='same', activation='relu')(x)
    y = UpSampling2D(size=(8, 8), interpolation='bilinear')(x)
    y = Conv2D(3, 3, padding='same', activation='relu')(y)
    y = Conv2D(3, 3, padding='same', activation='relu')(y)
    return x, y


def AIN_ResUNet(inputs, noise_map, kernel_size=3):
    conv1 = Conv2D(64, kernel_size, padding='same', activation='relu')(inputs)
    conv1 = AIN_ResBlock(conv1, noise_map, 64, kernel_size=kernel_size)
    conv1 = AIN_ResBlock(conv1, noise_map, 64, kernel_size=kernel_size)

    down1 = tf.nn.space_to_depth(conv1, 2)
    conv2 = Conv2D(128, kernel_size, padding='same', activation=PReLU())(down1)
    conv2 = AIN_ResBlock(conv2, noise_map, 128, kernel_size=kernel_size, down=2)
    conv2 = AIN_ResBlock(conv2, noise_map, 128, kernel_size=kernel_size, down=2)

    down2 = tf.nn.space_to_depth(conv2, 2)
    conv3 = Conv2D(256, kernel_size, padding='same', activation=PReLU())(down2)
    conv3 = AIN_ResBlock(conv3, noise_map, 256, kernel_size=kernel_size, down=4)
    conv3 = AIN_ResBlock(conv3, noise_map, 256, kernel_size=kernel_size, down=4)

    down3 = tf.nn.space_to_depth(conv3, 2)
    conv4 = Conv2D(512, kernel_size, padding='same', activation=PReLU())(down3)
    conv4 = AIN_ResBlock(conv4, noise_map, 512, kernel_size=kernel_size, down=8)
    conv4 = AIN_ResBlock(conv4, noise_map, 512, kernel_size=kernel_size, down=8)

    up3 = tf.nn.depth_to_space(conv4, 2)
    up3 = tf.concat([up3, conv3], axis=3)
    conv5 = Conv2D(256, kernel_size, padding='same', activation=PReLU())(up3)
    conv5 = AIN_ResBlock(conv5, noise_map, 256, kernel_size=kernel_size, down=4)
    conv5 = AIN_ResBlock(conv5, noise_map, 256, kernel_size=kernel_size, down=4)

    up2 = tf.nn.depth_to_space(conv5, 2)
    up2 = tf.concat([up2, conv2], axis=3)
    conv6 = Conv2D(128, kernel_size, padding='same', activation=PReLU())(up2)
    conv6 = AIN_ResBlock(conv6, noise_map, 128, kernel_size=kernel_size, down=2)
    conv6 = AIN_ResBlock(conv6, noise_map, 128, kernel_size=kernel_size, down=2)

    up1 = tf.nn.depth_to_space(conv6, 2)
    up1 = tf.concat([up1, conv1], axis=3)
    conv7 = Conv2D(64, kernel_size, padding='same', activation=PReLU())(up1)
    conv7 = AIN_ResBlock(conv7, noise_map, 64, kernel_size=kernel_size)
    conv7 = AIN_ResBlock(conv7, noise_map, 64, kernel_size=kernel_size)

    out = tf.keras.layers.Conv2D(3, 1, padding='same', activation='sigmoid')(conv7)
    return out


def AINDNet(inputs):
    down_noise_map, noise_map = NoiseEstimator(inputs)
    upsample_noise_map = UpSampling2D(
        size=(8, 8), interpolation='bilinear')(down_noise_map)
    noise_map = 0.8 * upsample_noise_map + 0.2 * noise_map
    out = AIN_ResUNet(inputs, noise_map) + inputs # train with and without
    return tf.keras.Model(inputs=inputs, outputs=(noise_map, out))
