import os
import sys
import cv2 as cv
import numpy as np
import subprocess
import tensorflow as tf

from datetime import datetime
from glob import glob

from dataset import Data, PSFPosData, input_pad_rad
from losses import MSE
from metrics import LPIPS, PSNR, SSIM, CPBD_MAE, CPBD_PRED, CPBD_TRUE
from utils import ProgressBar, dump_yaml


class Trainer:
    def __init__(self, model, dataset, args):
        self.model = model
        self.dataset = dataset
        self.args = args

    def train(self):
        dataset = self.dataset(self.args.data, self.args.gt, self.args.val_size, crop_div=8)
        out_dir = self._create_out_dir()

        train_dataset = dataset.get_dataset(
            data_type='train',
            batch_size=self.args.batch,
            repeat_count=None,
            crop_params=('random', self.args.crop))

        valid_dataset = dataset.get_dataset(
            data_type='valid',
            batch_size=1,
            repeat_count=None,
            crop_params=('center', self.args.crop))

        optimizer = tf.keras.optimizers.Adam(self.args.lr, beta_1=.5)
        self.model.compile(loss=MSE(), optimizer=optimizer, metrics=[PSNR(), SSIM()])

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_PSNR', patience=10, mode='max')
        csv_log = tf.keras.callbacks.CSVLogger(os.path.join(out_dir, f'train_{self.args.model}.log'))

        checkpoints_path = os.path.join(out_dir, self.args.model)
        saver = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoints_path + '_{epoch:03d}-{val_PSNR:.2f}.h5',
            monitor='val_PSNR')

        callbacks = [csv_log, saver]
        if not self.args.no_early_stop:
            callbacks.append(early_stop)

        self.model.fit(
            train_dataset,
            epochs=self.args.epochs,
            callbacks=callbacks,
            validation_data=valid_dataset,
            verbose=1,
            validation_steps=1,
            steps_per_epoch=dataset.numel * self.args.repeat // self.args.batch)

    def evaluate(self):
        dataset = self.dataset(self.args.data, self.args.gt, self.args.val_size)

        if self.args.val_size == 1. or self.args.val_size == 0.:
            dataset_length = dataset.numel 
            data_type = 'all'
        else:
            dataset_length = int(dataset.numel * self.args.val_size)
            data_type = 'valid'

        eval_dataset = dataset.get_dataset(
            data_type=data_type,
            batch_size=1,
            repeat_count=1,
            crop_params=('pad', self.args.crop))

        prog_bar = ProgressBar(dataset_length, title='Evaluation')
        # TODO set metrics from config
        metrics = [PSNR(), SSIM()]

        if self.args.cpbd_mae:
            metrics.append(CPBD_MAE())
        if self.args.cpbd_mae:
            metrics.append(CPBD_PRED())
        if self.args.cpbd_mae:
            metrics.append(CPBD_TRUE())
        if self.args.lpips:
            metrics.append(LPIPS())

        for inputs, result in eval_dataset:
            predict = self.model.predict(inputs)
            update_str = ''
            for metric in metrics:
                metric.update_state(result.numpy(), predict)
                update_str += f'{metric.name}={metric.result():.4f} - '
            prog_bar.update(update_str)

    def _predict_single(self, input):
        return self.model.predict(input)

    def predict(self):
        files = sorted(glob(self.args.data))
        gt = sorted(glob(self.args.gt))

        if len(files):
            mkdirne(self.args.predict)

        use_gt = True
        if len(gt) != len(files):
            use_gt = False
            print(f'Files in {self.args.gt} does not correspond to input files. Ignoring groundtruth.')

        for i, file in enumerate(files):
            print('Processing', file)

            im = cv.imread(file)[:, :, (2, 1, 0)]
            img_shape = im.shape[:2]

            h = img_shape[0] - img_shape[0] % 8
            w = img_shape[1] - img_shape[1] % 8

            im = im[:h, :w, :]
            input = im.astype(np.float32)[np.newaxis, ...] / 255
            out = self._predict_single(input)

            out = np.clip(out.squeeze() * 255, 0, 255).astype(np.uint8)
            file = os.path.join(self.args.predict, os.path.split(file)[1])

            cv.imwrite(file, out[:, :, (2, 1, 0)])
            cv.imwrite(file + '_orig.png', im[:, :, (2, 1, 0)])

            if use_gt:
                im = cv.imread(gt[i])[:h, :w, :]
                cv.imwrite(file + '_gt.png', im)

    def load_checkpoint(self, checkpoint):
        if not os.path.exists(checkpoint):
            raise Exception(f'No checkpoint found: {checkpoint}')

        try:  # Full models
            self.model = tf.keras.models.load_model(checkpoint, compile = False)
        except:  # Weights-only models
            print("Loading error:", sys.exc_info()[0])
            print("Trying to load weights only:")
            self.model.load_weights(checkpoint)

    def _create_out_dir(self):
        mkdirne(self.args.save_dir)
        timestamp = datetime.now().strftime('%y%m%d_%H%M')
        tag = self.args.tag

        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE).stdout.decode('utf-8')[:6]
        if commit == '':
            commit = 'nohead'

        out_dir = os.path.join(self.args.save_dir, '_'.join((timestamp, commit, self.args.model, tag)))
        mkdirne(out_dir)

        cfg_file = os.path.join(out_dir, 'config.yml')
        dump_yaml(self.args, cfg_file)

        return out_dir


class PSFPosTrainer(Trainer):
    def _predict_single(self, input):
        inputs, _ = input_pad_rad(input.squeeze())
        img, rad = inputs
        img = img[np.newaxis, ...]
        rad = rad[np.newaxis, ...]
        return self.model.predict((img, rad))


def mkdirne(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)