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

        self.image_locations = [
            os.path.join(self.image_dir, x)
            for x in os.listdir(self.image_dir)
        ]

        self.channels = 3

    def prepare_dataset(self):
        image_paths = np.array(self.image_locations)
        np.random.shuffle(image_paths)

        split = int(len(image_paths) * self.train_size)
        train_data = image_paths[:split]
        test_data = image_paths[split:]

        return train_data, test_data


class AnimeFacesDataset:
    def __init__(self, train_size=0.8):
        self.train_size = train_size
        self.image_dir = DATASET_PATH + 'anime_face_images/'

        self.image_locations = [
            os.path.join(self.image_dir, x)
            for x in os.listdir(self.image_dir)
        ]

        self.channels = 3

    def prepare_dataset(self):
        image_paths = np.array(self.image_locations)
        np.random.shuffle(image_paths)

        split = int(len(image_paths) * self.train_size)
        train_data = image_paths[:split]
        test_data = image_paths[split:]

        return train_data, test_data

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
            'anime_faces'
        ]

        self.batch_size = BATCH_SIZE
        self.img_shape = IMAGE_SIZE

    def process_images(self, image, decode=False, type='cifar10'):
        if decode:
            image = tf.io.read_file(image)
            image = tf.image.decode_jpeg(image, channels=3)

        if type in ['cifar10', 'cifar100', 'mnist', 'fashion_mnist']:
            image = tf.image.resize(image, self.img_shape)
        else:
            image = tf.image.resize(image, (IMAGE_SIZE[0]*2, IMAGE_SIZE[1]*2))

        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5

        return image

    def build_dataset(self, data, decode=False, type='cifar10', shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices(data)

        if shuffle:
            ds = ds.shuffle(10000)

        ds = ds.map(
            lambda x: self.process_images(x, decode=decode, type=type),
            num_parallel_calls=AUTOTUNE
        )

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(AUTOTUNE)

        return ds

    def load_data(self, type='mnist'):

        if type == 'mnist':
            self.channels = 1
            (train_images, _), (test_images, _) = self.mnist

            train_images = train_images.reshape(-1, 28, 28, 1)
            test_images = test_images.reshape(-1, 28, 28, 1)

            train_ds = self.build_dataset(train_images, decode=False, type=type)
            test_ds = self.build_dataset(test_images, decode=False, type=type, shuffle=False)

        elif type == 'cifar10':
            self.channels = 3
            (train_images, _), (test_images, _) = self.cifar10

            train_ds = self.build_dataset(train_images, decode=False, type=type)
            test_ds = self.build_dataset(test_images, decode=False, type=type, shuffle=False)

        elif type == 'fashion_mnist':
            self.channels = 1
            (train_images, _), (test_images, _) = self.fashion_mnist

            train_images = train_images.reshape(-1, 28, 28, 1)
            test_images = test_images.reshape(-1, 28, 28, 1)

            train_ds = self.build_dataset(train_images, decode=False, type=type)
            test_ds = self.build_dataset(test_images, decode=False, type=type, shuffle=False)

        elif type == 'cifar100':
            self.channels = 3
            (train_images, _), (test_images, _) = self.cifar100

            train_ds = self.build_dataset(train_images, decode=False, type=type)
            test_ds = self.build_dataset(test_images, decode=False, type=type, shuffle=False)

        elif type == 'celeba':
            self.channels = 3
            dataset = CelebADataset()
            train_data, test_data = dataset.prepare_dataset()

            train_ds = self.build_dataset(train_data, decode=True, type=type)
            test_ds = self.build_dataset(test_data, decode=True, type=type, shuffle=False)

        elif type == 'anime_faces':
            self.channels = 3
            dataset = AnimeFacesDataset()
            train_data, test_data = dataset.prepare_dataset()

            train_ds = self.build_dataset(train_data, decode=True, type=type)
            test_ds = self.build_dataset(test_data, decode=True, type=type, shuffle=False)

        else:
            raise ValueError(f"Unknown dataset type: {type}")

        return train_ds, test_ds, self.channels


if __name__ == "__main__":
    dataset = Dataset()

    for type in dataset.data_types:
        print(f"\nTesting: {type}")

        train_ds, test_ds, channels = dataset.load_data(type)

        for image in train_ds.take(1):
            print("Shape:", image.shape)
            break