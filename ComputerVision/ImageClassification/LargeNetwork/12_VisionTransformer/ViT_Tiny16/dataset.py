"""Folder-local, deterministic eight-dataset loader. Images remain in [0, 255]."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import DATASET_PATH, INPUT_SIZE, BATCH_SIZE

DATASETS = ('mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'skin_cancer',
            'cassava_leaf_disease', 'chest_xray', 'crop_disease')
SPLIT_VERSION = 1


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def grouped_split(records, seed):
    """Keep lesion identities and byte-identical images together, stratified by class."""
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, r in enumerate(records):
        for key in [('group', r['group']), ('sha256', r['sha256'])]:
            if key in seen:
                j = seen[key]
                if records[j]['label'] != r['label']:
                    raise ValueError('Conflicting labels within a lesion or duplicate image')
                parent[find(i)] = find(j)
            else:
                seen[key] = i
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault((r['label'], find(i)), []).append(i)
    rng = np.random.default_rng(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for label in sorted({r['label'] for r in records}):
        buckets = [v for (c, _), v in groups.items() if c == label]
        if len(buckets) < 3:
            raise ValueError(f'Class {label} needs at least three independent groups')
        rng.shuffle(buckets)
        # Allocate whole groups, approximately 80/10/10 by image count.
        target = sum(map(len, buckets)) * .1
        for split in ('test', 'val'):
            while len(buckets) > (1 if split == 'val' else 2):
                splits[split].extend(buckets.pop())
                if sum(records[i]['label'] == label for i in splits[split]) >= target:
                    break
        splits['train'].extend(i for bucket in buckets for i in bucket)
    return {k: sorted(v) for k, v in splits.items()}


class Dataset:
    def __init__(self, dataset_root=None, input_size=None, batch_size=None, seed=42,
                 manifest_dir=None, augment=True):
        self.root = Path(dataset_root or DATASET_PATH).expanduser().resolve()
        self.img_shape = list(input_size or INPUT_SIZE[:2])[:2]
        self.batch_size = batch_size or BATCH_SIZE
        self.seed = seed
        self.manifest_dir = Path(manifest_dir) if manifest_dir else Path('artifacts/splits')
        self.augment = augment
        self.manifest = None
        self.arrays = None

    def _records(self, name):
        records = []
        if name == 'chest_xray':
            base = self.root / 'chest_x_ray'
            classes = sorted(p.name for p in (base / 'train').iterdir() if p.is_dir())
            for split in ('train', 'val', 'test'):
                for folder in sorted((base / split).iterdir()):
                    if not folder.is_dir():
                        continue
                    if folder.name not in classes:
                        raise ValueError(f'Unknown class directory: {folder}')
                    for p in sorted(folder.rglob('*')):
                        if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                            records.append(dict(path=str(p), label=classes.index(folder.name),
                                                group=str(p), split=split))
        else:
            base = self.root / name
            csv, image_dir, id_col, label_col = {
                'skin_cancer': ('HAM10000_metadata.csv', 'skin_cancer_images', 'image_id', 'dx'),
                'cassava_leaf_disease': ('merged.csv', 'train', 'image_id', 'label'),
                'crop_disease': ('train.csv', 'train_images', 'image_id', 'label'),
            }[name]
            frame = pd.read_csv(base / csv, dtype=str)
            if id_col not in frame or label_col not in frame:
                raise ValueError(f'{base / csv}: requires {id_col} and {label_col}')
            if frame[[id_col, label_col]].isna().any().any():
                raise ValueError('Missing image identifiers or labels')
            classes = sorted(frame[label_col].unique().tolist())
            for _, row in frame.iterrows():
                image_id = row[id_col]
                if name == 'skin_cancer' and not Path(image_id).suffix:
                    image_id += '.jpg'
                p = (base / image_dir / image_id).resolve()
                if not p.is_relative_to((base / image_dir).resolve()):
                    raise ValueError(f'Image path escapes dataset root: {image_id}')
                group = row.get('lesion_id', image_id)
                if pd.isna(group):
                    raise ValueError('Missing lesion identity')
                records.append(dict(path=str(p), label=classes.index(row[label_col]), group=group))
        if not records or len(classes) < 2:
            raise ValueError(f'{name}: empty data or fewer than two classes')
        # CSV duplicates must not overweight a sample.
        unique = {}
        for r in records:
            if r['path'] in unique and unique[r['path']] != r:
                raise ValueError(f'Conflicting metadata: {r["path"]}')
            unique[r['path']] = r
        return list(unique.values()), classes

    def prepare(self, name):
        if name not in DATASETS:
            raise ValueError(name)
        self.name = name
        path = self.manifest_dir / f'{name}.json'
        old = json.loads(path.read_text()) if path.exists() else None
        if name in DATASETS[:4]:
            loader = getattr(tf.keras.datasets, name)
            (x, y), (xt, yt) = loader.load_data()
            y, yt = y.reshape(-1), yt.reshape(-1)
            if x.ndim == 3:
                x, xt = x[..., None], xt[..., None]
            self.arrays = ((x, y), (xt, yt))
            fingerprint = hashlib.sha256()
            for a in (x, y, xt, yt):
                fingerprint.update(a.tobytes())
            source = fingerprint.hexdigest()
            train, val = train_test_split(np.arange(len(y)), test_size=.1,
                                         stratify=y, random_state=self.seed)
            manifest = dict(version=SPLIT_VERSION, dataset=name, seed=self.seed,
                            source=source, classes=[str(i) for i in sorted(np.unique(y))],
                            channels=int(x.shape[-1]), splits=dict(train=train.tolist(),
                            val=val.tolist(), test=list(range(len(yt)))), exclusions=[])
        else:
            records, classes = self._records(name)
            inventory = []
            for r in records:
                p = Path(r['path'])
                if not p.is_file():
                    raise FileNotFoundError(f'Missing dataset image: {p}')
                stat = p.stat()
                inventory.append((r, stat.st_size, stat.st_mtime_ns))
            source = digest(inventory)
            if old and old.get('source') == source and old.get('seed') == self.seed and old.get('version') == SPLIT_VERSION:
                self.manifest = old
                return old
            valid, exclusions = [], []
            for r in records:
                try:
                    with Image.open(r['path']) as im:
                        im.verify()
                    # Validate the decoder actually used during training too.
                    tf.io.decode_image(tf.io.read_file(r['path']), channels=3, expand_animations=False).numpy()
                    r['sha256'] = hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
                    valid.append(r)
                except (OSError, ValueError, tf.errors.OpError) as exc:
                    exclusions.append(dict(path=r['path'], reason=str(exc)))
            if name == 'chest_xray':
                splits = {s: [i for i, r in enumerate(valid) if r['split'] == s]
                          for s in ('train', 'val', 'test')}
                seen = {}
                for r in valid:
                    prev = seen.setdefault(r['sha256'], (r['split'], r['label']))
                    if prev != (r['split'], r['label']):
                        raise ValueError(f'Duplicate image crosses official splits or labels: {r["path"]}')
            else:
                splits = grouped_split(valid, self.seed)
            for split, ids in splits.items():
                if not ids or {valid[i]['label'] for i in ids} != set(range(len(classes))):
                    raise ValueError(f'{name}/{split}: empty split or missing classes')
            manifest = dict(version=SPLIT_VERSION, dataset=name, seed=self.seed, source=source,
                            classes=classes, channels=3, records=valid, splits=splits, exclusions=exclusions)
        manifest['fingerprint'] = digest(manifest)
        if old and old != manifest:
            raise ValueError(f'Dataset/split changed: use a new run ID instead of overwriting {path}')
        atomic_json(path, manifest)
        self.manifest = manifest
        return manifest

    def load_data(self, type='mnist'):
        m = self.prepare(type)
        self.num_classes, self.channels = len(m['classes']), m['channels']
        augment = tf.keras.Sequential([
            *([] if type in ('mnist', 'fashion_mnist', 'chest_xray') else [tf.keras.layers.RandomFlip('horizontal', seed=self.seed)]),
            tf.keras.layers.RandomTranslation(.05, .05, fill_mode='reflect', seed=self.seed + 1),
            tf.keras.layers.RandomZoom(.1, seed=self.seed + 2),
        ])
        pipelines = []
        for split, ids in m['splits'].items():
            if self.arrays is not None:
                images, labels = self.arrays[1 if split == 'test' else 0]
                images, labels = images[ids], labels[ids]
                ds = tf.data.Dataset.from_tensor_slices((images, labels))
                decode = False
            else:
                records = [m['records'][i] for i in ids]
                ds = tf.data.Dataset.from_tensor_slices(([r['path'] for r in records], [r['label'] for r in records]))
                decode = True
            if split == 'train':
                ds = ds.shuffle(min(len(ids), 10000), seed=self.seed, reshuffle_each_iteration=True)
            def process(image, label, decode=decode):
                if decode:
                    image = tf.io.decode_image(tf.io.read_file(image), channels=self.channels, expand_animations=False)
                    image.set_shape([None, None, self.channels])
                image = tf.image.resize(tf.cast(image, tf.float32), self.img_shape)
                return image, tf.one_hot(tf.cast(label, tf.int32), self.num_classes)
            ds = ds.map(process, num_parallel_calls=4).batch(self.batch_size)
            if split == 'train' and self.augment:
                ds = ds.map(lambda x, y: (tf.clip_by_value(augment(x, training=True), 0., 255.), y), num_parallel_calls=1)
            options = tf.data.Options()
            options.threading.private_threadpool_size = 4
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            ds = ds.with_options(options).prefetch(1)
            pipelines.append((split, ds))
        data = dict(pipelines)
        return data['train'], data['val'], data['test'], self.num_classes, self.channels
