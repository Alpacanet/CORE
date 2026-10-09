"""Evaluate a trusted existing CORE checkpoint, including review/explore groups.

No training or model selection. Defaults to validation; test requires an explicit
--split test after the experimental scheme has been fixed. Original run files
are never overwritten.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from recbole.data.utils import get_dataloader
from recbole.utils import init_logger

from train import ROOT, _file_hash, _review_explore_report, _write_json, load_experiment
from trainer import CORETrainer


def evaluate_experiment(run_directory, split='valid', device=None,
                        eval_batch_size=None, limit=None):
    if split not in {'valid', 'test'}:
        raise ValueError('split must be valid or test')
    for name, value in [('eval_batch_size', eval_batch_size), ('limit', limit)]:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f'{name} must be a positive integer')
    source_directory = Path(run_directory).resolve()
    model, config, splits = load_experiment(source_directory, device=device)
    checkpoint_path = source_directory / 'best.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    best_epoch = int(checkpoint['epoch'])
    del checkpoint
    original_loader = splits[1 if split == 'valid' else 2]
    original_count = len(original_loader.dataset)
    selected_count = min(limit, original_count) if limit is not None else original_count
    if eval_batch_size is not None:
        config['eval_batch_size'] = eval_batch_size
    # Rebuild to honor optional evaluation batch size / subset. The vocabulary
    # and full candidate table are retained by Dataset.copy.
    dataset = original_loader.dataset
    if limit is not None:
        dataset = dataset.copy(dataset.inter_feat[:selected_count])
    loader = get_dataloader(config, split)(config, dataset, None, shuffle=False)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_directory = ROOT / 'results' / 'core' / f'evaluation-{stamp}-{config["dataset"]}-{uuid4().hex[:6]}'
    output_directory.mkdir(parents=True)
    config['checkpoint_dir'] = str(output_directory)
    init_logger(config)
    summary = {
        'status': 'running', 'run_kind': 'checkpoint_evaluation',
        'source_run_directory': str(source_directory),
        'checkpoint': str(checkpoint_path), 'checkpoint_sha256': _file_hash(checkpoint_path),
        'best_epoch': best_epoch, 'model': config['model'],
        'split': split, 'device': str(config['device']),
        'eval_batch_size': config['eval_batch_size'],
        'scope': 'prefix_subset' if limit is not None else 'full',
        'limit': limit, 'original_split_count': original_count,
        'evaluated_count': selected_count, 'result_directory': str(output_directory),
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_directory / 'metrics.json', summary)
    print(f'CORE_EVALUATION_DIR={output_directory}', flush=True)
    trainer = None
    try:
        trainer = CORETrainer(config, model)
        metrics = trainer.evaluate(loader, load_best_model=False, show_progress=False)
        groups = trainer.last_review_explore_result
        report = _review_explore_report(
            config['topk'], **{split: groups}, best_epoch=best_epoch, scope=summary['scope'],
        )
        report.update(source_run_directory=str(source_directory), split=split,
                      checkpoint_sha256=summary['checkpoint_sha256'],
                      original_split_count=original_count, evaluated_count=selected_count)
        _write_json(output_directory / 'review_explore.json', report)
        summary.update(status='complete', result=dict(metrics), review_explore=groups,
                       test_evaluated=split == 'test',
                       finished_at_utc=datetime.now(timezone.utc).isoformat())
        _write_json(output_directory / 'metrics.json', summary)
        print('REVIEW_EXPLORE=' + str(groups), flush=True)
        print(f'CORE_EVALUATION_COMPLETE={output_directory}', flush=True)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       error=str(error), finished_at_utc=datetime.now(timezone.utc).isoformat())
        _write_json(output_directory / 'metrics.json', summary)
        raise
    finally:
        if trainer is not None:
            trainer.tensorboard.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', required=True, help='Trusted local run containing best.pth and manifest.json')
    parser.add_argument('--split', choices=['valid', 'test'], default='valid')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None)
    parser.add_argument('--eval-batch-size', type=int, default=None)
    parser.add_argument('--limit', type=int, default=None, help='First N sessions ONLY, for engineering checks; not full-split results')
    args = parser.parse_args()
    evaluate_experiment(**vars(args))


if __name__ == '__main__':
    main()
