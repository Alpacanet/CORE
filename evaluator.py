"""Review/explore metrics from existing full-catalog recommendation scores."""

from collections import Counter
from math import fsum
from numbers import Integral

import torch


class ReviewExploreAccumulator:
    """Aggregate single-target Recall/MRR without another model forward pass.

    A target present in the nonpadding, model-visible history is ``review``;
    every other target is ``explore``. Both groups use the same full-catalog
    ranking, with only padding ID 0 excluded. Ties follow ``torch.topk`` on the
    full-width score tensor, matching RecBole's ranking convention.
    """

    GROUPS = ('overall', 'review', 'explore')
    INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

    def __init__(self, topk):
        if isinstance(topk, Integral) and not isinstance(topk, bool):
            values = [topk]
        else:
            if isinstance(topk, (str, bytes, bool)):
                raise TypeError('topk must be an integer or an iterable of integers')
            try:
                values = list(topk)
            except TypeError as error:
                raise TypeError('topk must be an integer or an iterable of integers') from error
        if not values:
            raise ValueError('topk must contain at least one cutoff')
        if any(isinstance(k, bool) or not isinstance(k, Integral) for k in values):
            raise TypeError('Every topk cutoff must be an integer')
        if any(k < 1 for k in values):
            raise ValueError('Every topk cutoff must be positive')
        if len(set(values)) != len(values):
            raise ValueError('topk cutoffs must be unique')
        self.topk = tuple(sorted(int(k) for k in values))
        self._catalog_size = None
        self._counts = {group: 0 for group in self.GROUPS}
        # Integer histograms avoid batch-size-dependent floating-point sums.
        self._rank_counts = {group: Counter() for group in self.GROUPS}

    @torch.no_grad()
    def update(self, scores, item_seq, targets):
        """Consume scores [B,V], visible history [B,S], and one target [B].

        Histories and targets may be on another device; they are moved to the
        score device. The supplied tensors are never modified. Real-candidate
        scores must be finite so padding cannot enter a tie at minus infinity.
        """
        if not all(isinstance(value, torch.Tensor) for value in (scores, item_seq, targets)):
            raise TypeError('scores, item_seq, and targets must be torch tensors')
        if scores.ndim != 2 or scores.shape[1] < 2:
            raise ValueError('scores must have shape [B,V] with padding and at least one real item')
        if item_seq.ndim != 2:
            raise ValueError('item_seq must have shape [B,S]')
        if targets.ndim != 1:
            raise ValueError('targets must have shape [B], with exactly one target per row')
        batch_size, catalog_size = scores.shape
        if item_seq.shape[0] != batch_size or targets.shape[0] != batch_size:
            raise ValueError('scores, item_seq, and targets must have the same batch size')
        if not scores.is_floating_point():
            raise TypeError('scores must be floating-point values')
        if item_seq.dtype not in self.INTEGER_DTYPES or targets.dtype not in self.INTEGER_DTYPES:
            raise TypeError('item_seq and targets must contain integer item IDs')
        if self._catalog_size is not None and catalog_size != self._catalog_size:
            raise ValueError('The item catalog size must remain constant across updates')
        # Promote before comparisons: a large catalog bound otherwise wraps
        # when PyTorch converts scalar bounds to an int8/uint8 tensor's dtype.
        item_seq = item_seq.to(dtype=torch.int64)
        targets = targets.to(dtype=torch.int64)
        if bool(((targets <= 0) | (targets >= catalog_size)).any().item()):
            raise ValueError('targets must be real item IDs in [1,V-1], never padding')
        if bool(((item_seq < 0) | (item_seq >= catalog_size)).any().item()):
            raise ValueError('item_seq IDs must be in [0,V-1]')
        if not bool(torch.isfinite(scores[:, 1:]).all().item()):
            raise ValueError('Real-candidate scores must be finite')
        self._catalog_size = catalog_size
        if batch_size == 0:
            return

        item_seq = item_seq.to(device=scores.device)
        targets = targets.to(device=scores.device)
        review = ((item_seq != 0) & item_seq.eq(targets.unsqueeze(1))).any(dim=1)
        ranked_scores = scores.detach().clone()
        ranked_scores[:, 0] = -torch.inf
        ranked_width = min(self.topk[-1], catalog_size - 1)
        ranked_ids = torch.topk(ranked_scores, ranked_width, dim=1).indices
        matches = ranked_ids.eq(targets.unsqueeze(1))
        found = matches.any(dim=1)
        ranks = matches.to(torch.int64).argmax(dim=1) + 1
        # Misses remain outside every configured (possibly clamped) cutoff.
        ranks = torch.where(found, ranks, ranked_width + 1)
        membership = {
            'overall': torch.ones(batch_size, dtype=torch.bool, device=scores.device),
            'review': review,
            'explore': ~review,
        }
        for group, mask in membership.items():
            batch_ranks = ranks[mask].cpu().tolist()
            self._counts[group] += len(batch_ranks)
            self._rank_counts[group].update(batch_ranks)

    def compute(self):
        """Return counts, global fractions, and unrounded metric means.

        Metric keys retain the configured cutoff even when it exceeds the
        catalog. Empty groups have ``None`` metrics; an empty accumulator has
        zero counts and fractions. Calling this method does not reset state.
        """
        total = self._counts['overall']
        result = {}
        for group in self.GROUPS:
            count = self._counts[group]
            metrics = {'count': count, 'fraction': count / total if total else 0.0}
            for k in self.topk:
                if count == 0:
                    metrics[f'recall@{k}'] = None
                    metrics[f'mrr@{k}'] = None
                    continue
                cutoff = min(k, self._catalog_size - 1)
                hits = [(rank, number) for rank, number in self._rank_counts[group].items()
                        if rank <= cutoff]
                metrics[f'recall@{k}'] = sum(number for _, number in hits) / count
                metrics[f'mrr@{k}'] = fsum(number / rank for rank, number in hits) / count
            result[group] = metrics
        return result
