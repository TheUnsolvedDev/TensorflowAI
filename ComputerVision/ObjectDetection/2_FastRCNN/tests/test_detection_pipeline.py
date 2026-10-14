"""Focused checks for Fast R-CNN proposal targets and box transforms."""

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset import FastRCNNDataset
from train import infer_image
from utils import decode_box, encode_box, selective_search_available


class DetectionPipelineTests(unittest.TestCase):
    def make_dataset(self, augment):
        dataset = FastRCNNDataset.__new__(FastRCNNDataset)
        dataset.augment = augment
        dataset.rois_per_image = 64
        dataset.positive_fraction = 0.25
        dataset.positive_iou_threshold = 0.5
        dataset.negative_iou_threshold = 0.1
        return dataset

    def test_box_encoding_round_trip(self):
        proposal = np.array([10, 20, 80, 100], np.float32)
        target = np.array([15, 12, 95, 110], np.float32)
        np.testing.assert_allclose(
            decode_box(proposal, encode_box(proposal, target)), target, atol=1e-5)

    def test_sampler_handles_too_few_positives(self):
        dataset = self.make_dataset(augment=True)
        proposals = tf.constant([
            [20., 20., 80., 80.],
            [18., 18., 78., 78.],
            [0., 0., 50., 50.],
            [100., 100., 160., 160.],
        ])
        boxes = tf.constant([[20., 20., 80., 80.]])
        _, labels, _, weights = dataset._sample_targets(
            proposals, tf.constant(4), boxes, tf.constant([3], tf.int32),
            tf.constant(1), tf.constant(200.), tf.constant(200.))
        self.assertEqual(labels.shape, (64,))
        self.assertLessEqual(int(tf.reduce_sum(weights)), 16)
        self.assertTrue(np.all(labels.numpy()[weights.numpy() == 0] == 0))

    def test_sampler_traces_in_graph_mode(self):
        dataset = self.make_dataset(augment=True)

        @tf.function
        def sample():
            return dataset._sample_targets(
                tf.constant([[20., 20., 80., 80.], [100., 100., 160., 160.]]),
                tf.constant(2), tf.constant([[20., 20., 80., 80.]]),
                tf.constant([3], tf.int32), tf.constant(1),
                tf.constant(200.), tf.constant(200.))

        rois, labels, targets, weights = sample()
        self.assertEqual(rois.shape, (64, 4))
        self.assertEqual(labels.shape, (64,))
        self.assertEqual(targets.shape, (64, 4))
        self.assertLessEqual(int(tf.reduce_sum(weights)), 16)

    def test_validation_does_not_append_ground_truth(self):
        dataset = self.make_dataset(augment=False)
        proposals = tf.constant([[0., 0., 30., 30.], [100., 100., 150., 150.]])
        boxes = tf.constant([[40., 40., 80., 80.]])
        _, _, _, weights = dataset._sample_targets(
            proposals, tf.constant(2), boxes, tf.constant([2], tf.int32),
            tf.constant(1), tf.constant(200.), tf.constant(200.))
        self.assertEqual(int(tf.reduce_sum(weights)), 0)

    def test_selective_search_backend_is_installed(self):
        self.assertTrue(selective_search_available())

    def test_inference_keeps_foreground_score_when_background_wins(self):
        class FakeModel:
            def __call__(self, inputs, training=False):
                count = tf.shape(inputs["rois"])[1]
                logits = tf.tile([[[2.0, 1.0, -2.0]]], [1, count, 1])
                boxes = tf.zeros((1, count, 4), tf.float32)
                return {"class_logits": logits, "bbox_regression": boxes}

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "image.jpg")
            cv2.imwrite(path, np.zeros((100, 100, 3), np.uint8))
            detections, _ = infer_image(
                FakeModel(), path, ["background", "object", "other"],
                proposals=np.asarray([[10, 10, 80, 80]], np.float32))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_id"], 1)


if __name__ == "__main__":
    unittest.main()
