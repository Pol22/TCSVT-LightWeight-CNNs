import os
import tensorflow as tf
import numpy as np

from random import sample
from utils import parse_dummy_bayer

try:
    from tensorflow.data.experimental import AUTOTUNE
except:
    AUTOTUNE = -1  # Compatibility with older TF versions


class BaseData:
    def _load_data(self):
        pass

    def _get_crop_func(self, ds, params = ('none', 256)):
        return ds

    def _get_transform_func(self, ds, params = []):
        return ds

    def __init__(self, inputs_path, results_path, valid_part=0.2, crop_div=8, dummy_bayer=None):
        self.inputs_path = inputs_path
        self.results_path = results_path
        self.valid_part = valid_part
        self.crop_div = crop_div
        self.dummy_bayer = dummy_bayer

        self.train_inputs = list()
        self.train_results = list()
        self.valid_inputs = list()
        self.valid_results = list()

        self._load_data()

    def get_dataset(self, data_type='train', batch_size=16, repeat_count=1, crop_params=('none', 256), transform_params=[]):
        if data_type == 'train':
            ds = tf.data.Dataset.zip((self.train_inputs, self.train_results))
        elif data_type == 'valid':
            ds = tf.data.Dataset.zip((self.valid_inputs, self.valid_results))
        elif data_type == 'all':
            ds = tf.data.Dataset.zip((self.train_inputs.concatenate(self.valid_inputs),
                                      self.train_results.concatenate(self.valid_results)))
        else:
            raise Exception(f'Not valid dataset type: {data_type}')

        ds = ds.repeat(repeat_count)
        ds = ds.map(normalize, num_parallel_calls=AUTOTUNE)
        ds = self._get_crop_func(ds, crop_params)
        ds = self._get_transform_func(ds, transform_params)
        ds = ds.batch(batch_size)

        return ds

    @staticmethod
    def images_dataset(image_files):
        if len(image_files) == 0:
            return None

        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file, num_parallel_calls=AUTOTUNE)
        ds = ds.map(
            lambda x: tf.image.decode_png(x, channels=3),
            num_parallel_calls=AUTOTUNE)

        return ds


class Data(BaseData):

    def __init__(self, inputs_path, results_path, valid_part=0.2, crop_div=8, dummy_bayer=None):
        BaseData.__init__(self, inputs_path, results_path, valid_part, crop_div, dummy_bayer=dummy_bayer)

    def _load_data(self, random_sampling = False):
        inputs = os.listdir(self.inputs_path)
        inputs = filter(lambda x: '.png' in x, inputs)
        results = os.listdir(self.results_path)
        results = filter(lambda x: '.png' in x, results)

        inputs = sorted(inputs)
        results = sorted(results)

        inputs_path = list(map(lambda x: os.path.join(self.inputs_path, x), inputs))
        results_path = list(map(lambda x: os.path.join(self.results_path, x), results))

        assert(len(inputs_path) == len(results_path))
        self.numel = len(inputs_path)

        valid_size = int(self.valid_part * len(inputs_path))
        valid_indexes = range(valid_size) if not random_sampling else sample(range(len(inputs_path)), valid_size)

        for ind in range(len(inputs_path)):
            if ind in valid_indexes:
                self.valid_inputs.append(inputs_path[ind])
                self.valid_results.append(results_path[ind])
            else:
                self.train_inputs.append(inputs_path[ind])
                self.train_results.append(results_path[ind])

        self.train_inputs = self.images_dataset(self.train_inputs)
        self.train_results = self.images_dataset(self.train_results)
        self.valid_inputs = self.images_dataset(self.valid_inputs)
        self.valid_results = self.images_dataset(self.valid_results)

    def _get_crop_func(self, ds, params = ('none', 256)):
        if len(params) == 0:
            return ds

        crop_type = params[0]
        crop_size = params[1]

        if crop_type == 'none':
            return ds
        elif crop_type == 'pad':
            ds = ds.map(lambda x, y: input_pad(x, y, self.crop_div), num_parallel_calls=AUTOTUNE)
        elif crop_type == 'center':
            ds = ds.map(lambda x, y: center_crop(x, y, crop_size), num_parallel_calls=AUTOTUNE)
        elif crop_type == 'random':
            ds = ds.map(lambda x, y: random_crop(x, y, crop_size), num_parallel_calls=AUTOTUNE)
        else:
            raise Exception(f'Unsupported crop type: {crop_size}')

        return ds

    def _get_transform_func(self, ds, params = []):
        if len(params) == 0:
            return ds
        elif params[0] == 'fliprot':
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        else:
            raise Exception(f'Unsupported transform type: {params[0]}')

        return ds


class PSFPosData(Data):
    def __init__(self, inputs_path, results_path, valid_part=0.2, crop_div=8):
        BaseData.__init__(self, inputs_path, results_path, valid_part, crop_div)

    def _get_crop_func(self, ds, params = ('none', 256)):
        if len(params) == 0:
            return ds

        crop_type = params[0]
        crop_size = params[1]

        if crop_type == 'none':
            return ds
        elif crop_type == 'pad':
            ds = ds.map(lambda x, y: input_pad_rad(x, y, self.crop_div), num_parallel_calls=AUTOTUNE)
        elif crop_type == 'center':
            ds = ds.map(lambda x, y: center_crop_rad(x, y, crop_size), num_parallel_calls=AUTOTUNE)
        elif crop_type == 'random':
            ds = ds.map(lambda x, y: random_crop_rad(x, y, crop_size), num_parallel_calls=AUTOTUNE)
        else:
            raise Exception(f'Unsupported crop type: {crop_size}')

        return ds

    def _get_transform_func(self, ds, params = []):
        return ds


class MosaicData(Data):
    def __init__(self, inputs_path, results_path, valid_part=0.2, crop_div=8, **kwargs):
        self.dummy_bayer = kwargs.get('dummy_bayer')
        self._is_packed = 'p' in self.dummy_bayer
        self.seed = kwargs.get('seed')
        self.deterministic = None if self.seed is None else True

        super(MosaicData, self).__init__(inputs_path, results_path, valid_part, crop_div, dummy_bayer=self.dummy_bayer)

    def get_dataset(self, data_type='train', batch_size=16, repeat_count=1, crop_params=('none', 256), transform_params=[]):
        ds = self._zip_dataset(data_type)
        ds = ds.repeat(repeat_count)
        ds = ds.map(normalize, num_parallel_calls=AUTOTUNE, deterministic=self.deterministic)
        ds = self._apply_transform_func(ds, transform_params)  # important - firstly transform, then mosaic
        ds = self._apply_crop_func(ds, crop_params)
        if self.dummy_bayer is not None:
            ds = ds.map(apply_dummy_bayer_wrapper(self.dummy_bayer), num_parallel_calls=AUTOTUNE, deterministic=self.deterministic)
        ds = ds.batch(batch_size)
        return ds

    def _apply_crop_func(self, ds, params = ('none', 256)):
        # should be stay here for case of crops that non consistent with usual augmentations
        crop_type, crop_size = params[0], params[1]
        crop_size = 2 * crop_size if self._is_packed else crop_size

        if len(params) == 0 or crop_type == 'none':
            return ds
        elif crop_type == 'pad':
            ds = ds.map(lambda x, y: input_pad(x, y, self.crop_div), num_parallel_calls=AUTOTUNE, deterministic=self.deterministic)
        elif crop_type == 'center':
            ds = ds.map(lambda x, y: center_crop(x, y, crop_size), num_parallel_calls=AUTOTUNE, deterministic=self.deterministic)
        elif crop_type == 'random':
            ds = ds.map(lambda x, y: random_crop(x, y, crop_size), num_parallel_calls=AUTOTUNE, deterministic=self.deterministic)
        else:
            raise Exception(f'Unsupported crop type: {crop_size}')
        return ds

    def _apply_transform_func(self, ds, params = []):
        # should be stay here for case of transformations that non consistent with usual augmentations
        if len(params) == 0:
            return ds
        elif params[0] == 'fliprot':
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE, deterministic=self.deterministic)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE, deterministic=self.deterministic)
        else:
            raise Exception(f'Unsupported transform type: {params[0]}')

        return ds

    def _zip_dataset(self, data_type):
        if data_type == 'train':
            ds = tf.data.Dataset.zip((self.train_inputs, self.train_results))
        elif data_type == 'valid':
            ds = tf.data.Dataset.zip((self.valid_inputs, self.valid_results))
        elif data_type == 'all':
            ds = tf.data.Dataset.zip((self.train_inputs.concatenate(self.valid_inputs),
                                      self.train_results.concatenate(self.valid_results)))
        else:
            raise Exception(f'Not valid dataset type: {data_type}')
        return ds


# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------

def normalize(inputs, res):
    inputs = tf.cast(inputs, tf.float32)
    res = tf.cast(res, tf.float32)
    inputs = inputs / 255.0
    res = res / 255.0
    return inputs, res


def random_crop(inputs, res, crop_size):
    img_shape = tf.shape(inputs)[:2]

    w = tf.random.uniform(shape=(), maxval=img_shape[1] - crop_size + 1, dtype=tf.int32)
    h = tf.random.uniform(shape=(), maxval=img_shape[0] - crop_size + 1, dtype=tf.int32)

    inputs_cropped = inputs[h:h + crop_size, w:w + crop_size, :]
    res_cropped = res[h:h + crop_size, w:w + crop_size, :]

    return inputs_cropped, res_cropped


def random_crop_bayer(inputs, res, crop_size):
    img_shape = tf.shape(inputs)[:2]

    w = tf.random.uniform(shape=(), maxval=(img_shape[1] - crop_size) // 2, dtype=tf.int32)
    h = tf.random.uniform(shape=(), maxval=(img_shape[0] - crop_size) // 2, dtype=tf.int32)

    inputs_cropped = inputs[2*h:2*h + crop_size, 2*w:2*w + crop_size, :]
    res_cropped = res[2*h:2*h + crop_size, 2*w:2*w + crop_size, :]

    return inputs_cropped, res_cropped


def input_pad(inputs, res, pad = 8):
    img_shape = tf.shape(inputs)[:2]

    h = img_shape[0] - img_shape[0] % pad
    w = img_shape[1] - img_shape[1] % pad

    inputs_cropped = inputs[:h, :w, :]
    res_cropped = res[:h, :w, :]

    return inputs_cropped, res_cropped


def center_crop(inputs, res, crop_size):
    img_shape = tf.shape(inputs)[:2]

    w = (img_shape[1] - crop_size) // 2 + 1
    h = (img_shape[0] - crop_size) // 2 + 1

    inputs_cropped = inputs[h:h + crop_size, w:w + crop_size, :]
    res_cropped = res[h:h + crop_size, w:w + crop_size, :]

    return inputs_cropped, res_cropped


def random_flip(inputs, res):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (inputs, res),
                   lambda: (tf.image.flip_left_right(inputs),
                            tf.image.flip_left_right(res)))


def random_rotate(inputs, res):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(inputs, rn), tf.image.rot90(res, rn)


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


def random_crop_rad(inputs, res, crop_size):
    img_shape = tf.shape(inputs)

    w = tf.random.uniform(shape=(), maxval=img_shape[1] - crop_size + 1, dtype=tf.int32)
    h = tf.random.uniform(shape=(), maxval=img_shape[0] - crop_size + 1, dtype=tf.int32)

    inputs_cropped = inputs[h:h + crop_size, w:w + crop_size, :]
    res_cropped = res[h:h + crop_size, w:w + crop_size, :]
    rad_theta = get_rad_vec(img_shape, crop_size, w, h)

    return (inputs_cropped, rad_theta), res_cropped


def center_crop_rad(inputs, res, crop_size):
    img_shape = tf.shape(inputs)[:2]

    w = img_shape[1] // 2 - crop_size + 1
    h = img_shape[0] // 2 - crop_size + 1

    inputs_cropped = inputs[h:h + crop_size, w:w + crop_size, :]
    res_cropped = res[h:h + crop_size, w:w + crop_size, :]
    rad_theta = get_rad_vec(img_shape, crop_size, w, h)

    return (inputs_cropped, rad_theta), res_cropped


def input_pad_rad(inputs, res = None, pad = 8):
    img_shape = tf.shape(inputs)

    w = img_shape[1] - img_shape[1] % pad
    h = img_shape[0] - img_shape[0] % pad

    inputs_cropped = inputs[:h, :w, :]
    if res is not None:
        res_cropped = res[:h, :w, :]
    else:
        res_cropped = None

    # creation of radius and angle stacked tensor
    x_c = tf.cast(w, tf.float32) / 2
    y_c = tf.cast(h, tf.float32) / 2
    r_max = tf.sqrt(x_c ** 2 + y_c ** 2)

    x = tf.cast(tf.range(w), dtype=tf.float32)
    x = tf.reshape(x, (1, w)) + tf.zeros(shape=(h, w))
    y = tf.cast(tf.range(h), dtype=tf.float32)
    y = tf.reshape(y, (h, 1)) + tf.zeros(shape=(h, w))
    r = tf.sqrt((x - x_c) ** 2 + (y - y_c) ** 2) + 1e-10
    cos_theta = (x - x_c) / r
    theta = tf.math.acos(cos_theta)
    theta = tf.where(tf.math.is_nan(theta), 0.0, theta)
    # elwise if true -> x else -> y
    theta = tf.where(y <= y_c, theta, 2*np.pi - theta)
    r /= r_max
    rad_theta = tf.stack([theta, r], axis=2)

    return (inputs_cropped, rad_theta), res_cropped


def apply_dummy_bayer_wrapper(dummy_bayer):
    r, g1, g2, b = parse_dummy_bayer(dummy_bayer)
    mp = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
    if 'p' in dummy_bayer:
        def apply_dummy_bayer(inputs, res):
            # inputs is HW3 with RGB format
            # pack to 4 channels
            inputs = tf.stack([inputs[mp[r][0]::2,  mp[r][1]::2, 0],
                               inputs[mp[g1][0]::2, mp[g1][1]::2, 1],
                               inputs[mp[g2][0]::2, mp[g2][1]::2, 1],
                               inputs[mp[b][0]::2,  mp[b][1]::2, 2]], axis=2)
            return inputs, res
    else:
        def apply_dummy_bayer(inputs, res):
            # generate raw
            mask = tf.zeros((2, 2, 3), dtype=inputs.dtype)
            mask = tf.tensor_scatter_nd_update(mask, 
                                               tf.constant([[[mp[r][0], mp[r][1], 0]], 
                                                             [[mp[g1][0], mp[g1][1], 1]],
                                                             [[mp[g2][0], mp[g2][1], 1]],
                                                             [[mp[b][0], mp[b][1], 2]]]), 
                                               tf.constant([[1], [1], [1], [1]], dtype=inputs.dtype))
            mask = tf.tile(mask, [tf.shape(inputs)[0] // 2, tf.shape(inputs)[1] // 2, 1])
            inputs = tf.math.multiply(inputs, mask)
            inputs = tf.reduce_max(inputs, axis=2, keepdims=True)
            # res = tf.slice(res, [0,0,0], [tf.shape(res)[0] - tf.shape(res)[0] % 2, tf.shape(res)[1] - tf.shape(res)[1] % 2, 3])
            return inputs, res
    return apply_dummy_bayer


# -----------------------------------------------------------
#  Utilities
# -----------------------------------------------------------

def get_data(args):
    if args.dummy_bayer is not None:
        return lambda inputs_path, results_path, valid_part=0.2, crop_div=8: \
            MosaicData(inputs_path, results_path, valid_part, crop_div, args.dummy_bayer)
    else: 
        return Data