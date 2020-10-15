import tensorflow as tf


def ResBlock(input_tensor, filters, kernel_size=3):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='same')(input_tensor)
    # x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='same')(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
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


def ResNet(input_tensor, kernel_size=3):
    x = tf.keras.layers.Conv2D(
        32, kernel_size, strides=1, padding='same')(input_tensor)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = ResBlock(x, filters=32, kernel_size=kernel_size)
    last_32 = ResBlock(x, filters=32, kernel_size=kernel_size)

    x = tf.keras.layers.Conv2D(
        64, kernel_size, strides=2, padding='same')(last_32)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = ResBlock(x, filters=64, kernel_size=kernel_size)
    last_64 = ResBlock(x, filters=64, kernel_size=kernel_size)

    x = tf.keras.layers.Conv2D(
        128, kernel_size, strides=2, padding='same')(last_64)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = ResBlock(x, filters=128, kernel_size=kernel_size)
    last_128 = ResBlock(x, filters=128, kernel_size=kernel_size)

    x = tf.keras.layers.Conv2D(
        256, kernel_size, strides=2, padding='same')(last_128)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = ResBlock(x, filters=256, kernel_size=kernel_size)
    x = ResBlock(x, filters=256, kernel_size=kernel_size)
    x = ResBlock(x, filters=256, kernel_size=kernel_size)
    x = ResBlock(x, filters=256, kernel_size=kernel_size)

    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x = tf.concat([x, last_128], axis=3)
    x = tf.keras.layers.Conv2D(
        128, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    # x += last_128
    x = ResBlock(x, filters=128, kernel_size=kernel_size)
    x = ResBlock(x, filters=128, kernel_size=kernel_size)

    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x = tf.concat([x, last_64], axis=3)
    x = tf.keras.layers.Conv2D(
        64, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    # x += last_64
    x = ResBlock(x, filters=64, kernel_size=kernel_size)
    x = ResBlock(x, filters=64, kernel_size=kernel_size)

    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x = tf.concat([x, last_32], axis=3)
    x = tf.keras.layers.Conv2D(
        32, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    # x += last_32
    x = ResBlock(x, filters=32, kernel_size=kernel_size)
    x = ResBlock(x, filters=32, kernel_size=kernel_size)

    x = tf.keras.layers.Conv2D(
        32, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    out = tf.keras.layers.Conv2D(
        3, 1, strides=1, padding='same', activation='sigmoid')(x)
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


def PixelAttention(inputs, features):
    x = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=1,
        padding='same',
        activation='sigmoid'
    )(inputs)
    x = tf.keras.layers.Multiply()([inputs, x])
    return x


def PixelAttentionConv(inputs, features, kernel_size=3):
    x = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=1,
        padding='same',
        activation='sigmoid'
    )(inputs)

    y = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=kernel_size,
        padding='same',
        use_bias=False
    )(inputs)

    out = tf.keras.layers.Multiply()([y, x])
    out = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=kernel_size,
        padding='same',
        use_bias=False
    )(out)
    return out


def SCPA(inputs, features, reduction=2):
    '''
        Self-Calibrated Pixel Attention

        Inspired by https://github.com/zhaohengyuan1/PAN
    '''
    group_width = features // reduction
    residual = inputs

    out_a = tf.keras.layers.Conv2D(
        features=group_width,
        kernel_size=1,
        padding='same',
        use_bias=False
    )(inputs)

    out_b = tf.keras.layers.Conv2D(
        features=group_width,
        kernel_size=1,
        padding='same',
        use_bias=False
    )(inputs)

    out_a = tf.keras.layers.LeakyReLU(alpha=0.2)(out_a)
    out_b = tf.keras.layers.LeakyReLU(alpha=0.2)(out_b)

    out_a = tf.keras.layers.Conv2D(
        features=group_width,
        kernel_size=3,
        padding='same',
        use_bias=False
    )(out_a)
    out_b = PixelAttentionConv(out_b, group_width)

    out_a = tf.keras.layers.LeakyReLU(alpha=0.2)(out_a)
    out_b = tf.keras.layers.LeakyReLU(alpha=0.2)(out_b)

    out = tf.concat([out_a, out_b], axis=3)
    out = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=1,
        padding='same',
        use_bias=False
    )(out)
    out += residual
    return out


def UPA(inputs, features, size=2):
    x = tf.keras.layers.UpSampling2D(
        size=(size, size),
        interpolation='nearest'
    )(inputs)
    x = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=3,
        padding='same'
    )(x)
    x = PixelAttention(x, features)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=3,
        padding='same'
    )(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x


def PAN(inputs, kernel_size=3, nb=16, features=64):
    '''
        Pixel Attention Network

        Inspired by https://arxiv.org/pdf/2010.01073v1.pdf
    '''
    x = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=3,
        padding='same'
    )(inputs)
    residual = x

    for _ in range(nb):
        x = SCPA(x, features)

    x = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=3,
        padding='same'
    )(x)

    x = x + residual

    x = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=3,
        padding='same'
    )(x)
    x = PixelAttention(x, features)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=3,
        padding='same'
    )(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=3,
        padding='same'
    )(x)
    x = PixelAttention(x, features)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(
        features=features,
        kernel_size=3,
        padding='same'
    )(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    # last conv
    x = tf.keras.layers.Conv2D(
        features=3,
        kernel_size=3,
        padding='same'
    )(x)

    x = x + inputs
    return x


def PAN_UNet(inputs, kernel_size=3):
    # TODO
    pass