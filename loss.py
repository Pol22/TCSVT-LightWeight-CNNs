import tensorflow as tf


class MSAsymm(tf.keras.losses.Loss):
    '''
        Inspired by
        https://github.com/IDKiro/CBDNet-tensorflow
    '''
    def __init__(self):
        super(MSAsymm, self).__init__()
        self.low_pass_filter = tf.keras.layers.AveragePooling2D(
            pool_size=4,
            strides=1,
            padding='same')

    def call(self, y_true, y_pred):
        inputs = y_pred[0, ...]
        est_noise = y_pred[1, ...]
        output = y_pred[2, ...]

        high_freq_inputs = inputs - self.low_pass_filter(inputs)
        high_freq_true = y_true - self.low_pass_filter(y_true)
        high_freq_noise = high_freq_true - high_freq_true

        loss = tf.keras.losses.mean_squared_error(y_true, output)
        # more penalty should be imposed to their MSE when 
        # estimated noise < true noise
        loss += 0.5 * tf.reduce_mean(
            tf.multiply(
                tf.abs(0.3 - tf.nn.relu(high_freq_noise - est_noise)), 
                tf.square(est_noise - high_freq_noise)))
        # total variation regularizer to constrain 
        # the smoothness of estimated noise
        loss += 0.05 * tf.reduce_mean(
            tf.square(tf.image.image_gradients(est_noise)))
        return loss
