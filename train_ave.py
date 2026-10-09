"""Open this file in your IDE and click Run to train CORE-ave."""

from train import run_experiment


if __name__ == '__main__':
    run_experiment(
        model='ave',
        dataset='tmall',
        seed=2020,
        epochs=30,
        micro_batch_size=16,
        accumulation_steps=16,
        smoke=False,  # True checks one epoch on a small subset.
        test=False,   # True evaluates test after final model selection.
    )
