"""Offline integration checks for grouped and standard RecBole evaluation."""

import copy
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from torch import nn

import recbole.evaluator.metrics as recbole_metrics
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.data.interaction import Interaction
from recbole.evaluator import Collector, Evaluator
from recbole.utils import EvaluatorType

from trainer import CORETrainer


class TinyFullSortLoader(FullSortEvalDataLoader):
    """Already-collated sequential batches, with the real full-sort type."""

    def __init__(self, batches, item_num):
        self.batches = batches
        self._dataset = SimpleNamespace(
            item_num=item_num,
            get_item_feature=lambda: Interaction(
                {"item_id": torch.arange(item_num)}
            ),
        )

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


class TinySampledLoader:
    def __init__(self, batches, item_num):
        self.batches = batches
        self._dataset = SimpleNamespace(item_num=item_num)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


class ToyScoringModel(nn.Module):
    ITEM_SEQ = "item_id_list"
    POS_ITEM_ID = "item_id"

    def __init__(self, scores):
        super().__init__()
        self.register_buffer("scores", scores)
        self.forward_calls = 0
        self.outputs = []
        self.fail_on_example = None

    def full_sort_predict(self, interaction):
        self.forward_calls += 1
        if self.training or torch.is_grad_enabled():
            raise AssertionError("The parent evaluator must disable training and autograd")
        examples = interaction["example_id"]
        if self.fail_on_example is not None and bool(
            examples.eq(self.fail_on_example).any()
        ):
            raise RuntimeError("injected evaluation failure")
        scores = self.scores[examples].clone()
        self.outputs.append(scores)
        return scores


def make_trainer(model):
    """Keep the real evaluation stack without filesystem/optimizer setup."""
    config = {
        "topk": [1, 3],
        "metrics": ["Recall", "MRR"],
        "metric_decimal_place": 4,
        "eval_args": {"mode": {"valid": "full", "test": "full"}},
        "eval_type": EvaluatorType.RANKING,
        "device": torch.device("cpu"),
        "single_spec": True,
    }
    trainer = object.__new__(CORETrainer)
    trainer.config = config
    trainer.model = model
    trainer.device = config["device"]
    trainer.gpu_available = False
    trainer.item_tensor = None
    trainer.tot_item_num = None
    trainer.eval_collector = Collector(config)
    trainer.evaluator = Evaluator(config)
    trainer.wandblogger = SimpleNamespace(log_eval_metrics=Mock())
    trainer.logger = logging.getLogger("core-grouped-offline-test")
    trainer.valid_metric = "mrr@3"
    trainer.epoch_records = []
    trainer.last_review_explore_result = None
    trainer._review_explore_accumulator = None
    return trainer


class GroupedTrainerTests(unittest.TestCase):
    def setUp(self):
        orders = [
            [2, 1, 3, 4, 5, 6],  # review target 2, rank 1
            [6, 1, 4, 2, 3, 5],  # review target 4, rank 3
            [1, 6, 2, 3, 4, 5],  # explore target 6, rank 2
            [1, 2, 3, 4, 5, 6],  # review target 5, rank 5
            [3, 2, 1, 4, 5, 6],  # explore target 3, rank 1
        ]
        self.scores = torch.zeros(5, 7)
        # Padding would win every row if the parent failed to exclude it.
        self.scores[:, 0] = 100.0
        for row, order in enumerate(orders):
            self.scores[row, order] = torch.arange(6, 0, -1).float()
        self.interaction = Interaction(
            {
                "example_id": torch.arange(5),
                "item_id_list": torch.tensor(
                    [[2, 2, 0], [1, 4, 0], [1, 3, 0], [5, 1, 0], [1, 2, 0]]
                ),
                "item_id": torch.tensor([2, 4, 6, 5, 3]),
            }
        )
        self.model = ToyScoringModel(self.scores.clone())
        self.trainer = make_trainer(self.model)

    def loader(self, slices=((0, 2), (2, 5))):
        batches = []
        for start, stop in slices:
            interaction = self.interaction[start:stop]
            batches.append(
                (interaction, None, torch.arange(len(interaction)), interaction["item_id"])
            )
        return TinyFullSortLoader(batches, self.scores.shape[1])

    def assert_metric_parity(self, result, groups):
        for key, value in result.items():
            self.assertEqual(
                value, round(groups["overall"][key], self.trainer.config["metric_decimal_place"])
            )

    def test_real_parent_evaluate_preserves_metrics_and_one_forward_per_batch(self):
        result = self.trainer.evaluate(self.loader(), load_best_model=False)
        groups = self.trainer.last_review_explore_result
        self.assertEqual(
            dict(result),
            {"recall@1": 0.4, "recall@3": 0.8, "mrr@1": 0.4, "mrr@3": 0.5667},
        )
        self.assert_metric_parity(result, groups)
        self.assertEqual(self.model.forward_calls, 2)
        self.assertEqual(groups["overall"]["count"], 5)
        self.assertEqual(groups["review"]["count"], 3)
        self.assertEqual(groups["explore"]["count"], 2)
        self.assertIsNone(self.trainer._review_explore_accumulator)
        self.trainer.wandblogger.log_eval_metrics.assert_called_once_with(result, head="eval")

    def test_true_target_groups_keep_history_and_full_catalog_candidates(self):
        self.trainer.evaluate(self.loader(), load_best_model=False)
        groups = self.trainer.last_review_explore_result
        self.assertAlmostEqual(groups["review"]["fraction"], 3 / 5)
        self.assertAlmostEqual(groups["review"]["recall@1"], 1 / 3)
        self.assertAlmostEqual(groups["review"]["recall@3"], 2 / 3)
        self.assertAlmostEqual(groups["review"]["mrr@3"], 4 / 9)
        self.assertAlmostEqual(groups["explore"]["recall@1"], 1 / 2)
        self.assertAlmostEqual(groups["explore"]["mrr@3"], 3 / 4)
        # Review rank 3 has unseen item 6 above it; explore rank 2 has history
        # item 1 above it. Any group-specific candidate restriction changes MRR.
        combined = torch.cat(self.model.outputs)
        self.assertTrue(bool(torch.isneginf(combined[:, 0]).all()))
        torch.testing.assert_close(combined[:, 1:], self.scores[:, 1:])
        torch.testing.assert_close(self.model.scores, self.scores)

    def test_repeated_evaluations_start_fresh_and_do_not_mutate_previous_report(self):
        self.trainer.evaluate(self.loader(), load_best_model=False)
        previous = self.trainer.last_review_explore_result
        snapshot = copy.deepcopy(previous)
        result = self.trainer.evaluate(self.loader(((4, 5),)), load_best_model=False)
        groups = self.trainer.last_review_explore_result
        self.assertEqual(groups["overall"]["count"], 1)
        self.assertEqual(groups["review"]["count"], 0)
        self.assertIsNone(groups["review"]["mrr@3"])
        self.assertEqual(groups["explore"]["count"], 1)
        self.assertEqual(previous, snapshot)
        self.assertIsNot(previous, groups)
        self.assert_metric_parity(result, groups)
        self.assertEqual(self.model.forward_calls, 3)

    def test_empty_evaluation_clears_previous_groups(self):
        self.trainer.evaluate(self.loader(), load_best_model=False)
        result = self.trainer.evaluate(self.loader(()), load_best_model=False)
        self.assertIsNone(result)
        self.assertIsNone(self.trainer.last_review_explore_result)
        self.assertIsNone(self.trainer._review_explore_accumulator)
        self.assertEqual(self.model.forward_calls, 2)

    def test_partial_evaluation_failure_clears_last_report_and_accumulator(self):
        self.trainer.evaluate(self.loader(), load_best_model=False)
        self.model.fail_on_example = 2
        with self.assertRaisesRegex(RuntimeError, "injected evaluation failure"):
            self.trainer.evaluate(self.loader(), load_best_model=False)
        self.assertIsNone(self.trainer.last_review_explore_result)
        self.assertIsNone(self.trainer._review_explore_accumulator)

    def test_recovery_after_partial_failure_has_no_stale_standard_metrics(self):
        self.model.fail_on_example = 2
        with self.assertRaises(RuntimeError):
            self.trainer.evaluate(self.loader(), load_best_model=False)
        self.model.fail_on_example = None
        result = self.trainer.evaluate(self.loader(), load_best_model=False)
        groups = self.trainer.last_review_explore_result
        self.assertEqual(groups["overall"]["count"], 5)
        self.assert_metric_parity(result, groups)

    def test_numpy_compatibility_does_not_change_global_aliases_or_module_binding(self):
        missing = object()
        aliases = {name: np.__dict__.get(name, missing) for name in ("float", "bool")}
        original_numpy = recbole_metrics.np
        self.trainer.evaluate(self.loader(), load_best_model=False)
        self.assertIs(recbole_metrics.np, original_numpy)
        for name, original in aliases.items():
            self.assertIs(np.__dict__.get(name, missing), original)
        self.model.fail_on_example = 2
        with self.assertRaises(RuntimeError):
            self.trainer.evaluate(self.loader(), load_best_model=False)
        self.assertIs(recbole_metrics.np, original_numpy)
        for name, original in aliases.items():
            self.assertIs(np.__dict__.get(name, missing), original)

    def test_invalid_positive_alignment_fails_and_clears_groups(self):
        for change in ("rows", "targets"):
            with self.subTest(change=change):
                trainer = make_trainer(ToyScoringModel(self.scores.clone()))
                loader = self.loader(((0, 2),))
                interaction, history, positive_u, positive_i = loader.batches[0]
                if change == "rows":
                    positive_u = positive_u.flip(0)
                else:
                    positive_i = positive_i.flip(0)
                loader.batches[0] = (interaction, history, positive_u, positive_i)
                with self.assertRaisesRegex(ValueError, "aligned target"):
                    trainer.evaluate(loader, load_best_model=False)
                self.assertIsNone(trainer.last_review_explore_result)
                self.assertIsNone(trainer._review_explore_accumulator)

    def test_valid_epoch_attaches_current_epoch_without_changing_older_records(self):
        older = {
            "epoch": 2,
            "valid_result": {"mrr@3": 0.1234},
            "valid_review_explore": {"saved": "previous-validation"},
        }
        self.trainer.epoch_records = [copy.deepcopy(older), {"epoch": 7, "train_loss": 1.0}]
        score, result = self.trainer._valid_epoch(self.loader())
        record = self.trainer.epoch_records[-1]
        self.assertEqual(score, result["mrr@3"])
        self.assertEqual(record["epoch"], 7)
        self.assertEqual(record["valid_result"], dict(result))
        self.assertEqual(record["valid_review_explore"], self.trainer.last_review_explore_result)
        self.assertEqual(self.trainer.epoch_records[0], older)
        saved_groups = copy.deepcopy(record["valid_review_explore"])
        self.trainer.evaluate(self.loader(((4, 5),)), load_best_model=False)
        self.assertEqual(record["valid_review_explore"], saved_groups)

    def test_non_full_sort_loader_is_rejected_without_extra_forward(self):
        loader = TinySampledLoader(self.loader().batches, self.scores.shape[1])
        with self.assertRaisesRegex(ValueError, "full-sort data loader"):
            self.trainer.evaluate(loader, load_best_model=False)
        self.assertEqual(self.model.forward_calls, 0)
        self.assertIsNone(self.trainer.last_review_explore_result)
        self.assertIsNone(self.trainer._review_explore_accumulator)


if __name__ == "__main__":
    unittest.main()
