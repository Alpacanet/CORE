"""Unified ave, trm, and LLM training. Run from an IDE or use the CLI."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import sys
import traceback
from uuid import uuid4

# RecBole reads complete LOCAL checkpoints created by this runner. Never pass
# untrusted downloaded .pth files to this process.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.utils import get_dataloader
from recbole.utils import init_logger, init_seed

from core_ave import COREave
from core_trm import COREtrm
from trainer import CORETrainer


ROOT = Path(__file__).resolve().parent
VARIANTS = {
    'pretrained_lora': ('pretrained', 'lora'),
    'pretrained_frozen': ('pretrained', 'frozen'),
    'random_lora': ('random', 'lora'),
    'random_full': ('random', 'full'),
    # Offline engineering test ONLY: not a scientific comparator to Qwen.
    'tiny_random': ('tiny_random', 'lora'),
}


@dataclass
class ExperimentOptions:
    model: str = 'llm'
    variant: str = 'pretrained_lora'
    dataset: str = 'tmall'
    seed: int = 2020
    epochs: int = 30
    micro_batch_size: int = 16
    accumulation_steps: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 0.001
    core_branch: str = 'ave'
    fusion: str = 'gate'
    device: str = 'cuda'
    smoke: bool = False
    test: bool = False
    local_files_only: bool = False


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8')


def _review_explore_report(topk, *, valid=None, test=None, best_epoch=None, scope='full'):
    """Explicit analysis semantics shared by training and checkpoint evaluation."""
    return {
        'schema_version': 1,
        'definition': {
            'review': 'Ground-truth next item occurs in the nonpadding input session.',
            'explore': 'Ground-truth next item does not occur in the nonpadding input session.',
        },
        'session_scope': 'Model-visible input history after configured length truncation.',
        'ranking_policy': 'Full catalog; padding ID 0 excluded; history items remain eligible.',
        'topk': list(topk),
        'metric_precision': 'Unrounded sample means; RecBole summary metrics are rounded separately.',
        'scope': scope,
        'best_epoch': best_epoch,
        'valid': valid,
        'test': test,
    }


def _package_versions():
    """Record installed packages without requiring optional LLM dependencies."""
    packages = {}
    for name in ['torch', 'recbole', 'numpy', 'transformers', 'peft', 'accelerate']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return packages


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _subset_loader(config, loader, phase, count, seed):
    dataset = loader.dataset
    indices = np.random.default_rng(seed).choice(len(dataset), min(count, len(dataset)), replace=False)
    indices.sort()
    subset = dataset.copy(dataset.inter_feat[indices])
    result = get_dataloader(config, phase)(config, subset, None, shuffle=phase == 'train')
    return result, indices.tolist()


def _sample_predictions(model, dataset, device, count=5):
    model.eval()
    interaction = dataset.inter_feat[:min(count, len(dataset))].to(device)
    with torch.no_grad():
        scores = model.full_sort_predict(interaction)
        scores[:, 0] = -torch.inf
        best_scores, ids = scores.topk(min(20, model.n_items - 1), dim=-1)
    tokens = dataset.field2id_token[model.ITEM_ID]
    records = []
    for row in range(len(interaction)):
        history = interaction[model.ITEM_SEQ][row].tolist()
        target = int(interaction[model.POS_ITEM_ID][row])
        records.append({
            'history_item_ids': [str(tokens[x]) for x in history if x != 0],
            'target_item_id': str(tokens[target]),
            'target_group': 'review' if target in history and target != 0 else 'explore',
            'top20_item_ids': [str(tokens[x]) for x in ids[row].tolist()],
            'top20_scores': best_scores[row].float().tolist(),
        })
    return records


def load_experiment(run_directory, device=None):
    """Reload a TRUSTED local run without retraining.

    Returns (model, config, (train_loader, valid_loader, test_loader)). Uses the
    full original data splits, even when the saved training run was a smoke.
    Never use this pickle-based loader with an untrusted checkpoint.
    """
    run_directory = Path(run_directory).resolve()
    checkpoint = torch.load(run_directory / 'best.pth', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    selected_device = torch.device(device if device is not None else config['device'])
    if selected_device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; pass device="cpu" to load_experiment.')
    config['device'] = selected_device
    config['use_gpu'] = selected_device.type == 'cuda'
    config['data_path'] = str(ROOT / 'dataset' / config['dataset'])
    config['llm_cache_dir'] = str(ROOT / 'models' / 'hf')
    config['llm_local_files_only'] = True
    config['save_dataset'] = False
    config['save_dataloaders'] = False
    manifest = json.loads((run_directory / 'manifest.json').read_text(encoding='utf-8'))
    for split, expected in manifest['input_file_sha256'].items():
        source = ROOT / 'dataset' / config['dataset'] / f'{config["dataset"]}.{split}.inter'
        if _file_hash(source) != expected:
            raise ValueError(f'Dataset file changed since training: {source}')
    if config['model'] == 'COREllm':
        from core_llm import COREllm
        model_class = COREllm
    elif config['model'] in {'COREave', 'COREtrm'}:
        model_class = COREave if config['model'] == 'COREave' else COREtrm
    else:
        raise ValueError('Checkpoint is not from a supported CORE model')
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    splits = data_preparation(config, dataset)
    actual_tokens = [str(x) for x in dataset.field2id_token[config['ITEM_ID_FIELD']]]
    saved_tokens = json.loads((run_directory / 'item_tokens.json').read_text(encoding='utf-8'))
    if actual_tokens != saved_tokens:
        raise ValueError('Item vocabulary / internal ID mapping changed since training')
    model = model_class(config, splits[0].dataset).to(selected_device)
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    model.eval()
    return model, config, splits


def run_experiment(options=None, **overrides):
    """Callable entry point, e.g. run_experiment(smoke=True).

    All artifacts are saved in a fresh results/core/... directory. Full runs
    evaluate validation only by default; test=True is for the final selection.
    """
    if options is None:
        options = ExperimentOptions(**overrides)
    elif overrides:
        raise ValueError('Pass options OR keyword overrides, not both.')
    if options.model not in {'ave', 'trm', 'llm'} or options.variant not in VARIANTS:
        raise ValueError('Unknown model or LLM variant')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', options.dataset):
        raise ValueError('dataset must be a simple dataset name, not a path')
    if min(options.epochs, options.micro_batch_size, options.accumulation_steps,
           options.eval_batch_size) < 1 or options.learning_rate <= 0:
        raise ValueError('Epochs, batch sizes, accumulation, and learning rate must be positive')
    if options.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use device="cpu" only for tiny_random / CORE smoke tests.')
    if options.variant == 'random_full' and options.model == 'llm' and options.device == 'cuda':
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if memory_gb < 12:
            raise ValueError('random_full needs substantially more memory; do not run it on this <12GB GPU. '
                             'Use random_lora for the parameter-matched control.')

    label = options.model if options.model != 'llm' else options.variant
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = ROOT / 'results' / 'core' / f'{stamp}-{options.dataset}-{label}-{uuid4().hex[:6]}'
    run_dir.mkdir(parents=True)
    summary = {'status': 'running', 'run_kind': 'smoke' if options.smoke else 'full',
               'started_at_utc': datetime.now(timezone.utc).isoformat(),
               'options': asdict(options), 'result_directory': str(run_dir)}
    _write_json(run_dir / 'metrics.json', summary)
    print(f'CORE_RESULT_DIR={run_dir}', flush=True)
    trainer = None
    try:
        if options.model == 'llm':
            from core_llm import COREllm
            model_class = COREllm
        else:
            model_class = COREave if options.model == 'ave' else COREtrm
        llm_init, llm_tuning = VARIANTS[options.variant]
        settings = {
            'data_path': str(ROOT / 'dataset'), 'seed': options.seed,
            'llm_seed': options.seed, 'epochs': 1 if options.smoke else options.epochs,
            'train_batch_size': options.micro_batch_size,
            'gradient_accumulation_steps': options.accumulation_steps,
            'eval_batch_size': options.eval_batch_size,
            'learning_rate': options.learning_rate, 'learner': 'adam',
            'use_gpu': options.device == 'cuda', 'gpu_id': '0',
            'llm_dtype': 'bfloat16' if options.device == 'cuda' else 'float32',
            'llm_init': llm_init, 'llm_tuning': llm_tuning,
            'llm_core_branch': options.core_branch, 'llm_fusion': options.fusion,
            'llm_cache_dir': str(ROOT / 'models' / 'hf'),
            'llm_local_files_only': options.local_files_only,
            'checkpoint_dir': str(run_dir), 'show_progress': False,
            'save_dataset': False, 'save_dataloaders': False,
            'log_wandb': False, 'enable_amp': False, 'enable_scaler': False,
            'MAX_ITEM_LIST_LENGTH': 50, 'eval_step': 1,
            'clip_grad_norm': {'max_norm': 5.0},
        }
        # CLI options belong to this runner; do not let RecBole reinterpret
        # --model/--dataset/--epochs and silently override the recorded options.
        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0]]
            config = Config(model=model_class, dataset=options.dataset,
                            config_file_list=[str(ROOT / 'configs' / 'common.yaml'),
                                              str(ROOT / 'configs' / f'core_{options.model}.yaml')],
                            config_dict=settings)
        finally:
            sys.argv = original_argv
        # RecBole 1.2.1 derives device from gpu_id, ignoring use_gpu=False.
        # Honor the runner's explicit CPU choice even on a CUDA-capable host.
        config['device'] = torch.device(options.device)
        init_seed(config['seed'], config['reproducibility'])
        init_logger(config)
        _write_json(run_dir / 'config.json', config.final_config_dict)
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        original_counts = {name: len(loader.dataset) for name, loader in
                           [('train', train_data), ('valid', valid_data), ('test', test_data)]}
        selected = None
        if options.smoke:
            train_data, train_indices = _subset_loader(config, train_data, 'train', 256, options.seed)
            valid_data, valid_indices = _subset_loader(config, valid_data, 'valid', 128, options.seed + 1)
            test_data, test_indices = _subset_loader(config, test_data, 'test', 128, options.seed + 2)
            selected = {'train': train_indices, 'valid': valid_indices, 'test': test_indices}
            _write_json(run_dir / 'smoke_indices.json', selected)
        # Reject empty/gapped/left-padded sessions before they can produce NaNs
        # or change the last-state convention. Repeated items remain untouched.
        for loader in (train_data, valid_data, test_data):
            sequence = loader.dataset.inter_feat[config['ITEM_ID_FIELD'] + config['LIST_SUFFIX']]
            mask = sequence.ne(0)
            if bool((mask.sum(dim=1) == 0).any()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
                raise ValueError('CORE training expects nonempty, right-padded item sequences')
        init_seed(config['seed'], config['reproducibility'])
        model = model_class(config, train_data.dataset).to(config['device'])
        total_parameters = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tokens = [str(x) for x in dataset.field2id_token[config['ITEM_ID_FIELD']]]
        _write_json(run_dir / 'item_tokens.json', tokens)
        manifest = {
            'python': platform.python_version(),
            'packages': _package_versions(),
            'device': str(config['device']),
            'gpu': torch.cuda.get_device_name(0) if options.device == 'cuda' else None,
            'cuda': torch.version.cuda, 'total_parameters': total_parameters,
            'trainable_parameters': trainable_parameters,
            'nominal_effective_batch_size': options.micro_batch_size * options.accumulation_steps,
            'original_split_counts': original_counts,
            'used_split_counts': {name: len(loader.dataset) for name, loader in
                                 [('train', train_data), ('valid', valid_data), ('test', test_data)]},
            'n_items_including_padding': model.n_items,
            'item_vocab_sha256': hashlib.sha256(json.dumps(tokens, ensure_ascii=False,
                                                         separators=(',', ':')).encode()).hexdigest(),
            'input_file_sha256': {
                split: _file_hash(ROOT / 'dataset' / options.dataset / f'{options.dataset}.{split}.inter')
                for split in ('train', 'valid', 'test')},
            'model_signature': model.get_extra_state() if options.model == 'llm' else None,
            'notes': [
                'Explicit original benchmark train/valid/test files; no resplitting.',
                'Full-catalog scoring, padding ID excluded, previously clicked items allowed.',
                'No metadata, text, tokenizer, or language generation.',
                'CORE ID embeddings trained from scratch; no warm start from older checkpoints.',
                'Frozen/random backbone checkpoint reconstruction needs pinned model/config and seed.',
                'Accumulation matches sample weighting, not a monolithic batch dropout realization.',
                'random_lora controls frozen pretraining, not a fully trained random Transformer.',
                'Hybrid projection/fusion is not the original strict CORE RCE.',
                'Smoke results are engineering checks, not evidence of improvement.',
                'Review/explore groups use ground-truth target membership in the visible input session.',
            ],
        }
        _write_json(run_dir / 'manifest.json', manifest)
        print(f'Parameters: total={total_parameters:,}, trainable={trainable_parameters:,}; '
              f'effective batch={manifest["nominal_effective_batch_size"]}', flush=True)
        trainer = CORETrainer(config, model)
        trainer.saved_model_file = str(run_dir / 'best.pth')
        best_score, best_valid = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
        checkpoint = torch.load(trainer.saved_model_file, map_location=config['device'], weights_only=False)
        best_epoch = int(checkpoint['epoch'])
        best_valid_groups = next(
            record['valid_review_explore'] for record in trainer.epoch_records
            if record['epoch'] == best_epoch and 'valid_review_explore' in record
        )
        evaluate_test = options.test or options.smoke
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False) if evaluate_test else None
        test_groups = trainer.last_review_explore_result if evaluate_test else None
        if not evaluate_test:
            model.load_state_dict(checkpoint['state_dict'])
            model.load_other_parameter(checkpoint.get('other_parameter'))
        _write_json(run_dir / 'review_explore.json', _review_explore_report(
            config['topk'], valid=best_valid_groups, test=test_groups,
            best_epoch=best_epoch, scope='smoke_subset' if options.smoke else 'full',
        ))
        _write_json(run_dir / 'predictions.json', _sample_predictions(
            model, (test_data if evaluate_test else valid_data).dataset, config['device']))
        _write_json(run_dir / 'epochs.json', trainer.epoch_records)
        summary.update(status='complete', best_valid_score=float(best_score),
                       best_valid_result=dict(best_valid),
                       best_valid_review_explore=best_valid_groups,
                       test_result=dict(test_result) if test_result is not None else None,
                       test_review_explore=test_groups,
                       best_epoch=best_epoch,
                       prediction_split='test' if evaluate_test else 'valid',
                       checkpoint=str(run_dir / 'best.pth'),
                       epochs_completed=len(trainer.epoch_records),
                       actual_epochs_limit=config['epochs'],
                       used_split_counts=manifest['used_split_counts'],
                       nominal_effective_batch_size=manifest['nominal_effective_batch_size'],
                       test_evaluated=evaluate_test,
                       finished_at_utc=datetime.now(timezone.utc).isoformat())
        _write_json(run_dir / 'metrics.json', summary)
        print('BEST_VALID=' + json.dumps(best_valid), flush=True)
        print('BEST_VALID_REVIEW_EXPLORE=' + json.dumps(best_valid_groups), flush=True)
        print('TEST=' + json.dumps(test_result) if evaluate_test else 'TEST=not evaluated (use --test for final run)', flush=True)
        print(f'CORE_COMPLETE={run_dir}', flush=True)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       error=str(error), traceback=traceback.format_exc(),
                       finished_at_utc=datetime.now(timezone.utc).isoformat())
        _write_json(run_dir / 'metrics.json', summary)
        if trainer is not None:
            _write_json(run_dir / 'epochs.json', trainer.epoch_records)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['ave', 'trm', 'llm'], default='llm')
    parser.add_argument('--variant', choices=list(VARIANTS), default='pretrained_lora')
    parser.add_argument('--dataset', default='tmall')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--micro-batch-size', type=int, default=16)
    parser.add_argument('--accumulation-steps', type=int, default=16)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--core-branch', choices=['ave', 'trm'], default='ave')
    parser.add_argument('--fusion', choices=['gate', 'llm_only'], default='gate')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--smoke', action='store_true', help='1 epoch / train256 valid128 test128; retains full item catalog')
    parser.add_argument('--test', action='store_true', help='Evaluate test after model selection (full runs only at final stage)')
    parser.add_argument('--local-files-only', action='store_true')
    run_experiment(ExperimentOptions(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
