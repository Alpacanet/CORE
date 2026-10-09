"""Offline ranking and group-aggregation checks; no models or data downloads."""

import unittest

import torch

from evaluator import ReviewExploreAccumulator


class ReviewExploreAccumulatorTests(unittest.TestCase):
    def setUp(self):
        # Real items 1..5 have exact ranks 1..5. Padding scores must be ignored.
        self.scores = torch.tensor([[999.0, 5.0, 4.0, 3.0, 2.0, 1.0]]).repeat(5, 1)
        self.targets = torch.tensor([1, 2, 3, 4, 5])
        self.history = torch.tensor([
            [1, 1, 0],  # Review, including duplicate historical clicks.
            [1, 0, 0],  # Explore, but the higher-scoring seen item stays ranked.
            [3, 0, 3],
            [4, 2, 0],
            [1, 2, 0],
        ])

    def test_known_ranks_padding_duplicates_and_full_catalog(self):
        accumulator = ReviewExploreAccumulator([1, 3, 10])
        accumulator.update(self.scores, self.history, self.targets)
        groups = accumulator.compute()
        self.assertEqual(groups['overall']['count'], 5)
        self.assertEqual(groups['review']['count'], 3)
        self.assertEqual(groups['explore']['count'], 2)
        self.assertEqual(groups['overall']['fraction'], 1.0)
        self.assertEqual(groups['review']['fraction'], 3 / 5)
        self.assertEqual(groups['explore']['fraction'], 2 / 5)
        self.assertEqual(groups['overall']['recall@1'], 1 / 5)
        self.assertEqual(groups['overall']['recall@3'], 3 / 5)
        self.assertAlmostEqual(groups['overall']['mrr@3'], (1 + 1 / 2 + 1 / 3) / 5)
        self.assertEqual(groups['review']['recall@1'], 1 / 3)
        self.assertEqual(groups['review']['recall@3'], 2 / 3)
        self.assertAlmostEqual(groups['review']['mrr@3'], (1 + 1 / 3) / 3)
        self.assertEqual(groups['explore']['recall@1'], 0.0)
        self.assertEqual(groups['explore']['recall@3'], 1 / 2)
        self.assertEqual(groups['explore']['mrr@3'], 1 / 4)
        for group in groups.values():
            self.assertEqual(group['recall@10'], 1.0)
        self.assertAlmostEqual(groups['overall']['mrr@10'], (1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5) / 5)
        self.assertAlmostEqual(groups['explore']['mrr@10'], (1 / 2 + 1 / 5) / 2)
        self.assertNotIn('recall@5', groups['overall'])

    def test_uneven_batches_match_one_pass_and_weighted_recomposition(self):
        one_pass = ReviewExploreAccumulator([1, 3, 10])
        one_pass.update(self.scores, self.history, self.targets)
        uneven = ReviewExploreAccumulator([1, 3, 10])
        for start, end in [(0, 2), (2, 3), (3, 5)]:
            uneven.update(self.scores[start:end], self.history[start:end], self.targets[start:end])
        result = uneven.compute()
        self.assertEqual(result, one_pass.compute())
        for key in ['recall@1', 'mrr@1', 'recall@3', 'mrr@3', 'recall@10', 'mrr@10']:
            recomposed = sum(result[group]['fraction'] * result[group][key]
                             for group in ['review', 'explore'])
            self.assertAlmostEqual(recomposed, result['overall'][key], places=15)
        self.assertEqual(uneven.compute(), result)

    def test_empty_group_metrics_are_none(self):
        for missing_group, history in [
            ('review', torch.zeros((5, 2), dtype=torch.long)),
            ('explore', self.targets.unsqueeze(1)),
        ]:
            with self.subTest(missing_group=missing_group):
                accumulator = ReviewExploreAccumulator(3)
                accumulator.update(self.scores, history, self.targets)
                group = accumulator.compute()[missing_group]
                self.assertEqual(group['count'], 0)
                self.assertEqual(group['fraction'], 0.0)
                self.assertIsNone(group['recall@3'])
                self.assertIsNone(group['mrr@3'])

    def test_no_examples_and_zero_width_history(self):
        accumulator = ReviewExploreAccumulator(1)
        for group in accumulator.compute().values():
            self.assertEqual(group, {'count': 0, 'fraction': 0.0,
                                     'recall@1': None, 'mrr@1': None})
        accumulator.update(self.scores[:0], self.history[:0], self.targets[:0])
        self.assertEqual(accumulator.compute()['overall']['count'], 0)
        accumulator.update(self.scores, torch.empty((5, 0), dtype=torch.long), self.targets)
        self.assertEqual(accumulator.compute()['review']['count'], 0)
        self.assertEqual(accumulator.compute()['explore']['count'], 5)

    def test_scores_are_not_mutated_and_gradients_are_not_retained(self):
        scores = self.scores.clone().requires_grad_(True)
        original = scores.detach().clone()
        accumulator = ReviewExploreAccumulator([1, 3])
        accumulator.update(scores, self.history, self.targets)
        torch.testing.assert_close(scores.detach(), original)
        self.assertTrue(scores.requires_grad)
        self.assertIsNone(scores.grad)
        self.assertIsInstance(accumulator.compute()['overall']['mrr@3'], float)

    def test_ties_follow_full_width_torch_topk(self):
        for width in [6, 7, 12]:
            with self.subTest(width=width):
                scores = torch.ones((1, width))
                scores[0, 0] = 999.0
                reference = scores.clone()
                reference[:, 0] = -torch.inf
                target = torch.topk(reference, 3, dim=1).indices[:, 0]
                accumulator = ReviewExploreAccumulator([1, 3])
                accumulator.update(scores, target.unsqueeze(1), target)
                self.assertEqual(accumulator.compute()['review']['recall@1'], 1.0)
                self.assertEqual(accumulator.compute()['review']['mrr@1'], 1.0)

    def test_invalid_topk(self):
        for value in [[], [0], [-1], [1, 1]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReviewExploreAccumulator(value)
        for value in [True, 1.5, '3', [1, False], [1.0]]:
            with self.subTest(value=value), self.assertRaises(TypeError):
                ReviewExploreAccumulator(value)

    def test_invalid_shapes_and_dtypes(self):
        invalid = [
            (self.scores[0], self.history, self.targets, ValueError),
            (self.scores.unsqueeze(0), self.history, self.targets, ValueError),
            (self.scores[:, :1], self.history, self.targets, ValueError),
            (self.scores, self.history[0], self.targets, ValueError),
            (self.scores, self.history, self.targets.unsqueeze(1), ValueError),
            (self.scores, self.history[:2], self.targets, ValueError),
            (self.scores, self.history, self.targets[:2], ValueError),
            (self.scores.long(), self.history, self.targets, TypeError),
            (self.scores, self.history.float(), self.targets, TypeError),
            (self.scores, self.history, self.targets.float(), TypeError),
            (self.scores.tolist(), self.history, self.targets, TypeError),
        ]
        for index, (scores, history, targets, error) in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(error):
                ReviewExploreAccumulator(3).update(scores, history, targets)

    def test_invalid_item_ids_and_catalog_changes(self):
        for target in [0, -1, 6]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                ReviewExploreAccumulator(3).update(self.scores[:1], self.history[:1], torch.tensor([target]))
        for item in [-1, 6]:
            with self.subTest(item=item), self.assertRaises(ValueError):
                ReviewExploreAccumulator(3).update(self.scores[:1], torch.tensor([[item]]), self.targets[:1])
        accumulator = ReviewExploreAccumulator(3)
        accumulator.update(self.scores, self.history, self.targets)
        original = accumulator.compute()
        with self.assertRaisesRegex(ValueError, 'catalog size'):
            accumulator.update(torch.ones((5, 7)), self.history, self.targets)
        self.assertEqual(accumulator.compute(), original)

    def test_small_integer_ids_are_validated_against_large_catalog_without_overflow(self):
        scores = torch.zeros((1, 37367))
        scores[:, 1] = 1.0
        for dtype in [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]:
            with self.subTest(dtype=dtype):
                accumulator = ReviewExploreAccumulator(1)
                target = torch.tensor([1], dtype=dtype)
                accumulator.update(scores, target.unsqueeze(1), target)
                self.assertEqual(accumulator.compute()['review']['recall@1'], 1.0)

    def test_nonfinite_candidates_are_rejected_but_padding_is_ignored(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            scores = self.scores[:1].clone()
            scores[:, 1] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite'):
                ReviewExploreAccumulator(3).update(scores, self.history[:1], self.targets[:1])
            scores = self.scores[:1].clone()
            scores[:, 0] = value
            accumulator = ReviewExploreAccumulator(3)
            accumulator.update(scores, self.history[:1], self.targets[:1])
            self.assertEqual(accumulator.compute()['overall']['recall@3'], 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_scores_accept_cpu_history_and_targets(self):
        accumulator = ReviewExploreAccumulator([1, 3, 10])
        accumulator.update(self.scores.cuda(), self.history, self.targets)
        reference = ReviewExploreAccumulator([1, 3, 10])
        reference.update(self.scores, self.history, self.targets)
        self.assertEqual(accumulator.compute(), reference.compute())


if __name__ == '__main__':
    unittest.main()
