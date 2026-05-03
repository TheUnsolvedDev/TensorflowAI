import pandas as pd
import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import os

from config import *

AUTOTUNE = tf.data.AUTOTUNE


class CelebADataset:
    def __init__(self, train_size=0.8):
        self.train_size = train_size
        self.dataset_path = DATASET_PATH + 'celeba-dataset/'
        self.image_dir = os.path.join(self.dataset_path, 'img_align_celeba')

        self.attr_path = os.path.join(
            self.dataset_path, 'list_attr_celeba.csv')
        df = pd.read_csv(self.attr_path)
        self.image_names = df.iloc[:, 0].values
        self.labels = df.iloc[:, 1:].values
        self.labels = (self.labels + 1) // 2
        self.image_paths = np.array([
            os.path.join(self.image_dir, name)
            for name in self.image_names
        ])
        assert len(self.image_paths) == len(self.labels)
        self.channels = 3

    def prepare_dataset(self):
        idx = np.arange(len(self.image_paths))
        np.random.shuffle(idx)
        image_paths = self.image_paths[idx]
        labels = self.labels[idx]
        # labels = labels[:, 0]
        split = int(len(image_paths) * self.train_size)
        train_images = image_paths[:split]
        train_labels = labels[:split]
        test_images = image_paths[split:]
        test_labels = labels[split:]
        return train_images, train_labels, test_images, test_labels


class Dataset:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist.load_data()
        self.cifar10 = tf.keras.datasets.cifar10.load_data()
        self.fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
        self.cifar100 = tf.keras.datasets.cifar100.load_data()

        self.data_types = [
            'cifar10',
            'fashion_mnist',
            'mnist',
            'cifar100',
            'celeba',
        ]

        self.batch_size = BATCH_SIZE
        self.img_shape = IMAGE_SIZE

    def process_images(self, image, labels, decode=False, type='cifar10'):
        if decode:
            image = tf.io.read_file(image)
            image = tf.image.decode_jpeg(image, channels=3)

        if type in ['cifar10', 'cifar100', 'mnist', 'fashion_mnist']:
            image = tf.image.resize(image, self.img_shape)
        else:
            image = tf.image.resize(image, (IMAGE_SIZE[0]*2, IMAGE_SIZE[1]*2))

        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5

        labels = tf.cast(labels, tf.int32)
        return image, labels

    def build_dataset(self, data, decode=False, type='cifar10', shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices(data)

        if shuffle:
            ds = ds.shuffle(10000)

        ds = ds.map(
            lambda x, y: self.process_images(x, y, decode=decode, type=type),
            num_parallel_calls=AUTOTUNE
        )

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(AUTOTUNE)

        return ds

    def load_data(self, type='mnist'):

        if type == 'mnist':
            self.channels = 1
            (train_images, train_labels), (test_images, test_labels) = self.mnist
            images = np.concat([train_images, test_images])
            labels = np.concat([train_labels, test_labels])
            images = images.reshape(-1, 28, 28, 1)
            labels = tf.one_hot(labels, depth=10)
            train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
            train_ds = train_ds.map(lambda x, y: self.process_images(
                x, y, decode=False, type=type), num_parallel_calls=AUTOTUNE)
            train_ds = train_ds.shuffle(10000).batch(
                self.batch_size).prefetch(AUTOTUNE)

        elif type == 'cifar10':
            self.channels = 3
            (train_images, train_labels), (test_images, test_labels) = self.cifar10
            images = np.concatenate([train_images, test_images])
            labels = np.concatenate([train_labels, test_labels])
            labels = tf.squeeze(labels, axis=-1)
            labels = tf.one_hot(labels, depth=10)
            train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
            train_ds = train_ds.map(lambda x, y: self.process_images(
                x, y, decode=False, type=type), num_parallel_calls=AUTOTUNE)
            train_ds = train_ds.shuffle(10000).batch(
                self.batch_size).prefetch(AUTOTUNE)

        elif type == 'fashion_mnist':
            self.channels = 1
            (train_images, train_labels), (test_images,
                                           test_labels) = self.fashion_mnist
            images = np.concatenate([train_images, test_images])
            labels = np.concatenate([train_labels, test_labels])
            images = images.reshape(-1, 28, 28, 1)
            labels = tf.one_hot(labels, depth=10)
            train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
            train_ds = train_ds.map(lambda x, y: self.process_images(
                x, y, decode=False, type=type), num_parallel_calls=AUTOTUNE)
            train_ds = train_ds.shuffle(10000).batch(
                self.batch_size).prefetch(AUTOTUNE)

        elif type == 'cifar100':
            self.channels = 3
            (train_images, train_labels), (test_images, test_labels) = self.cifar100
            images = np.concatenate([train_images, test_images])
            labels = np.concatenate([train_labels, test_labels])
            labels = tf.squeeze(labels, axis=-1)
            labels = tf.one_hot(labels, depth=100)
            train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
            train_ds = train_ds.map(lambda x, y: self.process_images(
                x, y, decode=False, type=type), num_parallel_calls=AUTOTUNE)
            train_ds = train_ds.shuffle(10000).batch(
                self.batch_size).prefetch(AUTOTUNE)

        elif type == 'celeba':
            self.channels = 3
            dataset = CelebADataset()
            train_images, train_labels, test_images, test_labels = dataset.prepare_dataset()
            images = np.concatenate([train_images, test_images])
            labels = np.concatenate([train_labels, test_labels])
            # labels = tf.one_hot(labels, depth=40)
            train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
            train_ds = train_ds.map(lambda x, y: self.process_images(
                x, y, decode=True, type=type), num_parallel_calls=AUTOTUNE)
            train_ds = train_ds.shuffle(10000).batch(
                self.batch_size).prefetch(AUTOTUNE)

        else:
            raise ValueError(f"Unknown dataset type: {type}")

        return train_ds, self.channels


if __name__ == "__main__":
    dataset = Dataset()

    for type in dataset.data_types:
        print(f"\nTesting: {type}")

        train_ds, channels = dataset.load_data(type)

        for image, label in train_ds.take(1):
            print("Shape:", image.shape)
            print("Label:", label.shape)
            # print(label)
            break
