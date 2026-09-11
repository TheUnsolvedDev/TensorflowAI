"""Focused regression tests for explicit Fast R-CNN run controls."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train import configure_run, reset_artifacts


class DummyModel:
    def __init__(self, load_error=None):
        self.load_error = load_error
        self.loads = []

    def load_weights(self, path, **kwargs):
        if self.load_error:
            raise self.load_error
        self.loads.append((path, kwargs))


class RunControlTests(unittest.TestCase):
    def test_default_missing_checkpoint_starts_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            result = configure_run(DummyModel(), os.path.join(directory, "model.weights.h5"), os.path.join(directory, "state.json"))
        self.assertEqual(result, (0, "default-fresh"))

    def test_default_loads_compatible_checkpoint_at_epoch_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "model.weights.h5")
            open(checkpoint, "wb").close()
            model = DummyModel()
            result = configure_run(model, checkpoint, os.path.join(directory, "state.json"))
        self.assertEqual(result, (0, "default-load"))
        self.assertEqual(model.loads, [(checkpoint, {})])

    def test_continue_restores_epoch_and_requires_state(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "model.weights.h5")
            state = os.path.join(directory, "state.json")
            open(checkpoint, "wb").close()
            with self.assertRaisesRegex(RuntimeError, "saved epoch state"):
                configure_run(DummyModel(), checkpoint, state, continue_run=True)
            with open(state, "w", encoding="utf-8") as file:
                json.dump({"epoch": 4}, file)
            self.assertEqual(configure_run(DummyModel(), checkpoint, state, continue_run=True), (4, "continue"))

    def test_reset_starts_fresh_and_clears_selected_log_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = os.path.join(directory, "old.json")
            open(artifact, "w", encoding="utf-8").close()
            reset_artifacts(directory)
            self.assertTrue(os.path.isdir(directory))
            self.assertFalse(os.path.exists(artifact))
            self.assertEqual(
                configure_run(DummyModel(), os.path.join(directory, "model.weights.h5"), os.path.join(directory, "state.json"), reset=True),
                (0, "reset"),
            )

    def test_incompatible_task_checkpoint_fails_without_partial_load(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "model.weights.h5")
            open(checkpoint, "wb").close()
            with self.assertRaisesRegex(RuntimeError, "incompatible checkpoint"):
                configure_run(DummyModel(ValueError("mismatch")), checkpoint, os.path.join(directory, "state.json"))


if __name__ == "__main__":
    unittest.main()
