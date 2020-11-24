import tensorflow as tf


def ResBlock(inputs, filters, kernel_size=3, data_format='channels_last',
             shared_axes=(1, 2)):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, padding='same', data_format=data_format)(inputs)
    x = tf.keras.layers.PReLU(shared_axes=shared_axes)(x)

    x = tf.keras.layers.Conv2D(
        filters, kernel_size, padding='same', data_format=data_format)(x)
    x = tf.keras.layers.PReLU(shared_axes=shared_axes)(x)
    return x + inputs


def ResBlock_1_3_1(inputs, filters, kernel_size=3):
    x = tf.keras.layers.Conv2D(
        filters, 1, strides=1, padding='same')(inputs)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    x = tf.keras.layers.Conv2D(
        filters, 1, strides=1, padding='same')(x)
    
    x = x + inputs
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    return x


def ResBlock_1x1(input_tensor, filters, kernel_size=3):
    x1 = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='same')(input_tensor)
    x1 = tf.keras.layers.PReLU(shared_axes=[1, 2])(x1)
    # skip 1x1
    x2 = tf.keras.layers.Conv2D(
        filters, 1, strides=1, padding='same')(input_tensor)
    x2 = tf.keras.layers.PReLU(shared_axes=[1, 2])(x2)
    return x1 + x2


def ResBlock_reflect(inputs, filters, kernel_size=3, bn=False, skip=False):
    paddings = (kernel_size // 2, kernel_size // 2)
    paddings = ((0, 0), paddings, paddings, (0, 0))
    x1 = tf.pad(inputs, paddings, mode='REFLECT')
    x1 = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='VALID')(x1)
    if bn:
        x1 = tf.keras.layers.BatchNormalization()(x1)

    x1 = tf.keras.layers.PReLU(shared_axes=[1, 2])(x1)
    x1 = tf.pad(x1, paddings, mode='REFLECT')
    x1 = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='VALID')(x1)
    if bn:
        x1 = tf.keras.layers.BatchNormalization()(x1)

    if skip:
        x1 = tf.keras.layers.PReLU(shared_axes=[1, 2])(x1)
        return x1 + inputs
    else:
        x2 = tf.keras.layers.Conv2D(
            filters, 1, strides=1, padding='SAME')(inputs)
        return tf.keras.layers.PReLU(shared_axes=[1, 2])(x1 + x2)


def PixelShuffleUpSampling(inputs, filters, scale=2):
    x = tf.nn.depth_to_space(inputs, scale)
    x = tf.keras.layers.Conv2D(filters, 1, padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    return x


def PixelShuffleDownSampling(inputs, filters, scale=2):
    x = tf.nn.space_to_depth(inputs, scale)
    x = tf.keras.layers.Conv2D(filters, 1, padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    return x


def ResUNet_shuffle(inputs, kernel_size=3):
    x = tf.keras.layers.Conv2D(
        32, kernel_size, strides=1, padding='same')(inputs)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    x = ResBlock(x, filters=32, kernel_size=kernel_size)
    last_32 = ResBlock(x, filters=32, kernel_size=kernel_size)

    x = PixelShuffleDownSampling(last_32, 64)
    x = ResBlock(x, filters=64, kernel_size=kernel_size)
    last_64 = ResBlock(x, filters=64, kernel_size=kernel_size)

    x = PixelShuffleDownSampling(last_64, 128)
    x = ResBlock(x, filters=128, kernel_size=kernel_size)
    last_128 = ResBlock(x, filters=128, kernel_size=kernel_size)

    x = PixelShuffleDownSampling(last_128, 256)
    x = ResBlock(x, filters=256, kernel_size=kernel_size)
    x = ResBlock(x, filters=256, kernel_size=kernel_size)

    x = PixelShuffleUpSampling(x, 128)
    x += last_128
    x = ResBlock(x, filters=128, kernel_size=kernel_size)
    x = ResBlock(x, filters=128, kernel_size=kernel_size)

    x = PixelShuffleUpSampling(x, 64)
    x += last_64
    x = ResBlock(x, filters=64, kernel_size=kernel_size)
    x = ResBlock(x, filters=64, kernel_size=kernel_size)

    x = PixelShuffleUpSampling(x, 32)
    x += last_32
    x = ResBlock(x, filters=32, kernel_size=kernel_size)
    x = ResBlock(x, filters=32, kernel_size=kernel_size)

    x = tf.keras.layers.Conv2D(
        32, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    out = tf.keras.layers.Conv2D(
        3, 1, strides=1, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=out)


def ResNet(inputs, kernel_size=3, data_format='channels_last'):
    if data_format == 'channels_last':
        shared_axes = (1, 2)
    elif data_format == 'channels_first':
        shared_axes = (2, 3)
    else:
        raise Exception('Unknown data format')

    x = tf.keras.layers.Conv2D(
        32, kernel_size, padding='same', data_format=data_format)(inputs)
    x = tf.keras.layers.PReLU(shared_axes=shared_axes)(x)
    x = ResBlock(x, 32, kernel_size, data_format, shared_axes)
    last_32 = ResBlock(x, 32, kernel_size, data_format, shared_axes)

    x = tf.keras.layers.Conv2D(64, kernel_size, strides=2, padding='same',
                               data_format=data_format)(last_32)
    x = tf.keras.layers.PReLU(shared_axes=shared_axes)(x)
    x = ResBlock(x, 64, kernel_size, data_format, shared_axes)
    last_64 = ResBlock(x, 64, kernel_size, data_format, shared_axes)

    x = tf.keras.layers.Conv2D(128, kernel_size, strides=2, padding='same',
                               data_format=data_format)(last_64)
    x = tf.keras.layers.PReLU(shared_axes=shared_axes)(x)
    x = ResBlock(x, 128, kernel_size, data_format, shared_axes)
    last_128 = ResBlock(x, 128, kernel_size, data_format, shared_axes)

    x = tf.keras.layers.Conv2D(256, kernel_size, strides=2, padding='same',
                               data_format=data_format)(last_128)
    x = tf.keras.layers.PReLU(shared_axes=shared_axes)(x)

    x = ResBlock(x, 256, kernel_size, data_format, shared_axes)
    x = ResBlock(x, 256, kernel_size, data_format, shared_axes)
    x = ResBlock(x, 256, kernel_size, data_format, shared_axes)
    x = ResBlock(x, 256, kernel_size, data_format, shared_axes)

    x = tf.keras.layers.UpSampling2D(
        interpolation='bilinear', data_format=data_format)(x)
    # x = tf.concat([x, last_128], axis=3)
    x = tf.keras.layers.Conv2D(
        128, kernel_size, padding='same', data_format=data_format)(x)
    x = tf.keras.layers.PReLU(shared_axes=shared_axes)(x)
    x += last_128
    x = ResBlock(x, 128, kernel_size, data_format, shared_axes)
    x = ResBlock(x, 128, kernel_size, data_format, shared_axes)

    x = tf.keras.layers.UpSampling2D(
        interpolation='bilinear', data_format=data_format)(x)
    # x = tf.concat([x, last_64], axis=3)
    x = tf.keras.layers.Conv2D(
        64, kernel_size, padding='same', data_format=data_format)(x)
    x = tf.keras.layers.PReLU(shared_axes=shared_axes)(x)
    x += last_64
    x = ResBlock(x, 64, kernel_size, data_format, shared_axes)
    x = ResBlock(x, 64, kernel_size, data_format, shared_axes)

    x = tf.keras.layers.UpSampling2D(
        interpolation='bilinear', data_format=data_format)(x)
    # x = tf.concat([x, last_32], axis=3)
    x = tf.keras.layers.Conv2D(
        32, kernel_size, padding='same', data_format=data_format)(x)
    x = tf.keras.layers.PReLU(shared_axes=shared_axes)(x)
    x += last_32

    x = ResBlock(x, 32, kernel_size, data_format, shared_axes)
    x = ResBlock(x, 32, kernel_size, data_format, shared_axes)

    x = tf.keras.layers.Conv2D(
        32, kernel_size, padding='same', data_format=data_format)(x)
    x = tf.keras.layers.PReLU(shared_axes=shared_axes)(x)
    out = tf.keras.layers.Conv2D(
        3, 1, padding='same', activation='sigmoid', data_format=data_format)(x)
    return tf.keras.Model(inputs=inputs, outputs=out)


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

    out = tf.keras.layers.Conv2D(
        3, 1, strides=1, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs=input_tensor, outputs=out)


def ResNet_v2(input_tensor, kernel_size=3):
    x2 = tf.nn.space_to_depth(input_tensor, 2)
    x4 = tf.nn.space_to_depth(x2, 2)
    x8 = tf.nn.space_to_depth(x4, 2)
    x16 = tf.nn.space_to_depth(x8, 2)

    x16 = ResBlock_reflect(x16, 512, kernel_size=kernel_size)
    x16 = ResBlock_reflect(x16, 512, kernel_size=kernel_size, skip=True)
    x8 = ResBlock_reflect(x8, 256, kernel_size=kernel_size)
    x8 = ResBlock_reflect(x8, 256, kernel_size=kernel_size, skip=True)
    x4 = ResBlock_reflect(x4, 128, kernel_size=kernel_size)
    x4 = ResBlock_reflect(x4, 128, kernel_size=kernel_size, skip=True)
    x2 = ResBlock_reflect(x2, 64, kernel_size=kernel_size)
    x2 = ResBlock_reflect(x2, 64, kernel_size=kernel_size, skip=True)
    x = ResBlock_reflect(input_tensor, filters=32, kernel_size=kernel_size)

    up8 = tf.nn.depth_to_space(x16, 2)
    up8 = tf.concat([up8, x8], axis=3)
    up8 = ResBlock_reflect(up8, 256, kernel_size=kernel_size)
    up8 = ResBlock_reflect(up8, 256, kernel_size=kernel_size, skip=True)

    up4 = tf.nn.depth_to_space(up8, 2)
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

    out = tf.keras.layers.Conv2D(
        3, 1, strides=1, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs=input_tensor, outputs=out)
