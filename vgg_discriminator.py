import tensorflow as tf


def get_VGG_discriminator(inputs):
    x = inputs * 255.0
    x = tf.keras.applications.vgg16.preprocess_input(x)
    vgg = tf.keras.applications.VGG16(include_top=False)
    vgg.trainable = False
    x = vgg(x, training=False)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    out = x
    return tf.keras.Model(inputs, out)
