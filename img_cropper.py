import tensorflow as tf


def get_sr_crop(hr_size=256, scale=2):
    if hr_size % scale != 0:
        raise Exception('Incompatible high resolution and scale')

    lr_size = hr_size // scale

    def crop(inputs, res):
        h, w, _ = tf.shape(inputs)
        h_lr = tf.random.uniform(shape=[], maxval=h - lr_size, dtype=tf.int32)
        w_lr = tf.random.uniform(shape=[], maxval=w - lr_size, dtype=tf.int32)
        inputs = inputs[h_lr:h_lr + lr_size, w_lr:w_lr + lr_size, :]
        h_hr = h_lr * scale
        w_hr = w_lr * scale
        res = res[h_hr:h_hr + hr_size, w_hr:w_hr + hr_size, :]
        return inputs, res

    return crop


# import numpy as np
# from PIL import Image

# img1 = Image.open('combined_cover.png')
# img1 = np.asarray(img1, dtype=np.float32) / 255.0
# shape = img1.shape
# print(shape)
# h, w, c = shape
# img2 = tf.image.resize(img1, (h // 2, w // 2))
# # Image.fromarray(np.uint8(img2)).save('img.png')
# crop = get_sr_crop()
# inputs, res = crop(img2, img1)
# Image.fromarray(np.uint8(255.0*inputs)).save('inputs.png')
# Image.fromarray(np.uint8(255.0*res)).save('res.png')
