"""Offline checks for shared gradient accumulation and compact checkpoints."""

import copy
import logging
import tempfile
import unittest
from pathlib import Path

import torch
from recbole.data.interaction import Interaction
from torch import nn
from torch.nn.utils import clip_grad_norm_

from trainer import CORETrainer


class ToyAdapterModel(nn.Module):
    """A trainable input projection followed by a frozen backbone."""

    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(1, 1, bias=False)
        self.backbone = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.input_projection.weight.fill_(0.4)
            self.backbone.weight.fill_(2.0)
        self.backbone.requires_grad_(False)

    def calculate_loss(self, interaction):
        prediction = self.backbone(self.input_projection(interaction["x"]))
        return nn.functional.mse_loss(prediction, interaction["y"])

    def checkpoint_state_dict(self):
        return {
            "input_projection.weight": self.input_projection.weight.detach().clone()
        }

    def other_parameter(self):
        return {"provenance": "offline-test"}


def make_trainer(model, accumulation_steps=2, clipping=None):
    """Skip logging/evaluator initialization to isolate the training algorithm."""
    trainer = object.__new__(CORETrainer)
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.enable_scaler = False
    trainer.enable_amp = False
    trainer.clip_grad_norm = clipping
    trainer.gradient_accumulation_steps = accumulation_steps
    trainer.epoch_records = []
    trainer.config = {"reg_weight": None}
    trainer.learner = "adamw"
    trainer.learning_rate = 1e-3
    trainer.weight_decay = 0.01
    trainer.logger = logging.getLogger("core-offline-test")
    trainer.cur_step = 1
    trainer.best_valid_score = 0.2
    trainer.optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    return trainer


class CORETrainerTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        self.y = torch.tensor([[1.0], [-1.0], [2.0], [0.0], [1.0]])

    def batch(self, start, stop):
        return Interaction({"x": self.x[start:stop], "y": self.y[start:stop]})

    def check_accumulation_against_full_batches(self, clipping):
        model = ToyAdapterModel()
        reference = copy.deepcopy(model)
        trainer = make_trainer(model, clipping=clipping)
        reference_optimizer = torch.optim.SGD(
            [p for p in reference.parameters() if p.requires_grad], lr=0.1
        )
        weighted_loss = 0.0
        # The first window contains unequal micro-batches (2 + 1), and the
        # final window contains only one batch instead of the configured two.
        for start, stop in [(0, 3), (3, 5)]:
            interaction = self.batch(start, stop)
            reference_optimizer.zero_grad()
            loss = reference.calculate_loss(interaction)
            weighted_loss += loss.item() * len(interaction)
            loss.backward()
            if clipping:
                clip_grad_norm_(
                    [p for p in reference.parameters() if p.requires_grad],
                    **clipping,
                )
            reference_optimizer.step()

        returned_loss = trainer._train_epoch(
            [self.batch(0, 2), self.batch(2, 3), self.batch(3, 5)], epoch_idx=3
        )
        torch.testing.assert_close(
            model.input_projection.weight, reference.input_projection.weight
        )
        self.assertAlmostEqual(returned_loss, weighted_loss / 5, places=6)
        self.assertIsNone(model.backbone.weight.grad)
        torch.testing.assert_close(model.backbone.weight, torch.tensor([[2.0]]))
        self.assertFalse(
            torch.equal(model.input_projection.weight, torch.tensor([[0.4]]))
        )
        self.assertEqual(len(trainer.epoch_records), 1)
        record = trainer.epoch_records[0]
        self.assertEqual(record["epoch"], 3)
        self.assertEqual(record["examples"], 5)
        self.assertEqual(record["optimizer_steps"], 2)
        self.assertAlmostEqual(record["train_loss"], returned_loss)
        self.assertEqual(record["cuda_peak_memory_mb"], 0.0)
        self.assertGreaterEqual(record["seconds"], 0.0)

    def test_sample_weighted_accumulation_and_partial_window(self):
        # No clipping here: clipping can conceal an incorrect gradient scale.
        self.check_accumulation_against_full_batches(clipping=None)

    def test_clipping_matches_full_batch_after_normalization(self):
        self.check_accumulation_against_full_batches(clipping={"max_norm": 0.5})

    def test_gradients_reach_input_projection_through_frozen_backbone(self):
        model = ToyAdapterModel()
        initial_weight = model.input_projection.weight.detach().clone()
        trainer = make_trainer(model, accumulation_steps=4)
        trainer._train_epoch([self.batch(0, 2)], epoch_idx=0)
        self.assertFalse(torch.equal(model.input_projection.weight, initial_weight))
        self.assertIsNone(model.backbone.weight.grad)
        self.assertEqual(trainer.epoch_records[0]["optimizer_steps"], 1)

    def test_optimizer_filters_frozen_parameters(self):
        model = ToyAdapterModel()
        trainer = make_trainer(model)
        for learner, optimizer_type in [("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW)]:
            with self.subTest(learner=learner):
                optimizer = trainer._build_optimizer(learner=learner)
                self.assertIsInstance(optimizer, optimizer_type)
                optimized = [p for group in optimizer.param_groups for p in group["params"]]
                self.assertEqual(len(optimized), 1)
                self.assertIs(optimized[0], model.input_projection.weight)
                self.assertEqual(optimizer.param_groups[0]["lr"], trainer.learning_rate)
                self.assertEqual(optimizer.param_groups[0]["weight_decay"], trainer.weight_decay)

    def test_compact_checkpoint_preserves_recbole_fields(self):
        model = ToyAdapterModel()
        trainer = make_trainer(model)
        with tempfile.TemporaryDirectory(prefix="core-trainer-") as directory:
            trainer.saved_model_file = str(Path(directory) / "checkpoint.pth")
            trainer._save_checkpoint(epoch=3, verbose=False)
            checkpoint = torch.load(trainer.saved_model_file, weights_only=False)
        self.assertEqual(
            set(checkpoint),
            {"config", "epoch", "cur_step", "best_valid_score", "state_dict", "other_parameter", "optimizer"},
        )
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["cur_step"], 1)
        self.assertEqual(checkpoint["best_valid_score"], 0.2)
        self.assertEqual(set(checkpoint["state_dict"]), {"input_projection.weight"})
        torch.testing.assert_close(
            checkpoint["state_dict"]["input_projection.weight"], model.input_projection.weight
        )
        self.assertEqual(checkpoint["other_parameter"], {"provenance": "offline-test"})
        self.assertEqual(checkpoint["optimizer"], trainer.optimizer.state_dict())


if __name__ == "__main__":
    unittest.main()
