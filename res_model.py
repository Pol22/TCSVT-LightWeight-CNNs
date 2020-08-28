import tensorflow as tf


def ResBlock(input_tensor, filters, kernel_size=3):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='same')(input_tensor)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    return x + input_tensor


def ResBlock_1x1(input_tensor, filters, kernel_size=3):
    x1 = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='same')(input_tensor)
    x1 = tf.keras.layers.LeakyReLU(alpha=0.02)(x1)
    # skip 1x1
    x2 = tf.keras.layers.Conv2D(
        filters, 1, strides=1, padding='same')(input_tensor)
    x2 = tf.keras.layers.LeakyReLU(alpha=0.02)(x2)
    return x1 + x2


def ResBlock_reflect(inputs, filters, kernel_size=3, bn=True, skip=False):
    paddings = (kernel_size // 2, kernel_size // 2)
    paddings = ((0, 0), paddings, paddings, (0, 0))
    x1 = tf.pad(inputs, paddings, mode='REFLECT')(inputs)
    x1 = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='VALID')(x1)
    if bn:
        x1 = tf.keras.layers.BatchNormalization()(x1)

    x1 = tf.keras.layers.PReLU()(x1)
    x1 = tf.pad(inputs, paddings, mode='REFLECT')(x1)
    x1 = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='VALID')(x1)
    if bn:
        x1 = tf.keras.layers.BatchNormalization()(x1)

    if skip:
        x1 = tf.keras.layers.PReLU()(x1)
        return x1 + inputs
    else:
        x2 = tf.keras.layers.Conv2D(
            filters, 1, strides=1, padding='SAME')(inputs)
        return tf.keras.layers.PReLU()(x1 + x2)


def ResNet(input_tensor, kernel_size=3):
    x = tf.keras.layers.Conv2D(
        64, kernel_size, strides=1, padding='same')(input_tensor)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = ResBlock(x, filters=64, kernel_size=kernel_size)
    last_64 = ResBlock(x, filters=64, kernel_size=kernel_size)

    x = tf.keras.layers.Conv2D(
        128, kernel_size, strides=2, padding='same')(last_64)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = ResBlock(x, filters=128, kernel_size=kernel_size)
    last_128 = ResBlock(x, filters=128, kernel_size=kernel_size)

    x = tf.keras.layers.Conv2D(
        256, kernel_size, strides=2, padding='same')(last_128)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = ResBlock(x, filters=256, kernel_size=kernel_size)
    x = ResBlock(x, filters=256, kernel_size=kernel_size)
    x = ResBlock(x, filters=256, kernel_size=kernel_size)

    # x = tf.keras.layers.Conv2DTranspose(
    #     128, kernel_size, strides=2, padding='same')(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = tf.keras.layers.Conv2D(
        128, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    # x = tf.nn.depth_to_space(x, 2)
    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x += last_128
    x = ResBlock(x, filters=128, kernel_size=kernel_size)
    x = ResBlock(x, filters=128, kernel_size=kernel_size)

    # x = tf.keras.layers.Conv2DTranspose(
    #     64, kernel_size, strides=2, padding='same')(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = tf.keras.layers.Conv2D(
        64, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    # x = tf.nn.depth_to_space(x, 2)
    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x += last_64
    x = ResBlock(x, filters=64, kernel_size=kernel_size)
    x = ResBlock(x, filters=64, kernel_size=kernel_size)

    out = tf.keras.layers.Conv2D(3, 1, strides=1, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs=input_tensor, outputs=out)


def ResNet_v1(input_tensor, kernel_size=3):
    x2 = tf.nn.space_to_depth(input_tensor, 2)
    x4 = tf.nn.space_to_depth(x2, 2)
    x8 = tf.nn.space_to_depth(x4, 2)

    x8 = ResBlock_1x1(x8, filters=256, kernel_size=kernel_size)
    x4 = ResBlock_1x1(x4, filters=128, kernel_size=kernel_size)
    x2 = ResBlock_1x1(x2, filters=64, kernel_size=kernel_size)
    x = ResBlock_1x1(input_tensor, filters=32, kernel_size=kernel_size)

    up4 = tf.nn.depth_to_space(x8, 2)
    up4 = tf.concat([up4, x4], axis=3)
    up4 = ResBlock_1x1(up4, filters=128, kernel_size=kernel_size)

    up2 = tf.nn.depth_to_space(up4, 2)
    up2 = tf.concat([up2, x2], axis=3)
    up2 = ResBlock_1x1(up2, filters=64, kernel_size=kernel_size)

    up = tf.nn.depth_to_space(up2, 2)
    up = tf.concat([up, x], axis=3)
    up = ResBlock_1x1(up, filters=32, kernel_size=kernel_size)
    x = ResBlock_1x1(up, filters=32, kernel_size=kernel_size)

    out = tf.keras.layers.Conv2D(3, 1, strides=1, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs=input_tensor, outputs=out)


def ResNet_v2(input_tensor, kernel_size=3):
    x2 = tf.nn.space_to_depth(input_tensor, 2)
    x4 = tf.nn.space_to_depth(x2, 2)
    x8 = tf.nn.space_to_depth(x4, 2)

    x8 = ResBlock_reflect(x8, 256, kernel_size=kernel_size)
    x8 = ResBlock_reflect(x8, 256, kernel_size=kernel_size, skip=True)
    x4 = ResBlock_reflect(x4, 128, kernel_size=kernel_size)
    x4 = ResBlock_reflect(x4, 128, kernel_size=kernel_size, skip=True)
    x2 = ResBlock_reflect(x2, 64, kernel_size=kernel_size)
    x2 = ResBlock_reflect(x2, 64, kernel_size=kernel_size, skip=True)
    x = ResBlock_reflect(input_tensor, filters=32, kernel_size=kernel_size)

    up4 = tf.nn.depth_to_space(x8, 2)
    up4 = tf.concat([up4, x4], axis=3)
    up4 = ResBlock_reflect(up4, 128, kernel_size=kernel_size)
    up4 = ResBlock_reflect(up4, 128, kernel_size=kernel_size, skip=True)

    up2 = tf.nn.depth_to_space(up4, 2)
    up2 = tf.concat([up2, x2], axis=3)
    up2 = ResBlock_reflect(up2, 64, kernel_size=kernel_size)
    up2 = ResBlock_reflect(up2, 64, kernel_size=kernel_size, skip=True)

    up = tf.nn.depth_to_space(up2, 2)
    up = tf.concat([up, x], axis=3)
    up = ResBlock_reflect(up, 32, kernel_size=kernel_size)
    x = ResBlock_reflect(up, 32, kernel_size=kernel_size, skip=True)

    out = tf.keras.layers.Conv2D(3, 1, strides=1, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs=input_tensor, outputs=out)
