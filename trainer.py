"""Shared RecBole trainer for ave, trm, and continuous-input LLM models."""

from time import perf_counter

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from recbole.trainer import Trainer
from recbole.utils import get_gpu_usage, set_color


class CORETrainer(Trainer):
    """Accumulate example-weighted gradients without changing RecBole evaluation.

    ``calculate_loss`` must return a mean over the examples in a micro-batch,
    just as CORE's cross-entropy does. Unlike the standard RecBole trainer,
    the reported epoch loss is a sample-weighted mean, not a sum of batch
    means. The frozen backbone remains inside autograd so input adapters can
    receive gradients through it.
    """

    def __init__(self, config, model):
        if not config["single_spec"]:
            raise ValueError("CORETrainer supports one CPU or one GPU, not DDP.")
        accumulation_steps = config["gradient_accumulation_steps"]
        if accumulation_steps is None:
            accumulation_steps = 1
        if (
            isinstance(accumulation_steps, bool)
            or not isinstance(accumulation_steps, int)
            or accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be a positive integer.")
        self.gradient_accumulation_steps = accumulation_steps
        self.epoch_records = []
        super().__init__(config, model)

    def _build_optimizer(self, **kwargs):
        """Keep frozen weights out of the optimizer and its checkpoint state."""
        supplied_params = kwargs.pop("params", self.model.parameters())
        supplied_params = list(supplied_params)
        if supplied_params and isinstance(supplied_params[0], dict):
            trainable_params = []
            for group in supplied_params:
                filtered = [p for p in group["params"] if p.requires_grad]
                if filtered:
                    trainable_params.append({**group, "params": filtered})
        else:
            trainable_params = [p for p in supplied_params if p.requires_grad]
        if not trainable_params:
            raise ValueError("CORETrainer needs at least one trainable parameter.")
        return super()._build_optimizer(params=trainable_params, **kwargs)

    def _step_accumulated_gradients(self, scaler, examples):
        """Turn accumulated batch sums into a window mean, then clip and step."""
        scaler.unscale_(self.optimizer)
        parameters = [
            p for group in self.optimizer.param_groups for p in group["params"]
        ]
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(examples)
        if self.clip_grad_norm:
            clip_grad_norm_(parameters, **self.clip_grad_norm)

        previous_scale = scaler.get_scale()
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        # GradScaler lowers its scale when it skips an update due to overflow.
        return int(scaler.get_scale() >= previous_scale)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        cuda_device = self.device.type == "cuda"
        if cuda_device:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = perf_counter()
        scaler = torch.amp.GradScaler(
            "cuda", enabled=self.enable_scaler and cuda_device
        )
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        self.optimizer.zero_grad(set_to_none=True)
        component_sums = None
        tuple_loss = None
        total_examples = 0
        window_examples = 0
        window_batches = 0
        optimizer_steps = 0
        for interaction in iter_data:
            interaction = interaction.to(self.device)
            examples = len(interaction)
            if examples < 1:
                continue
            with torch.autocast(
                device_type=self.device.type, enabled=self.enable_amp
            ):
                losses = loss_func(interaction)
                is_tuple = isinstance(losses, tuple)
                components = losses if is_tuple else (losses,)
                loss = sum(components)
            if not components or loss.ndim != 0:
                raise ValueError("calculate_loss must return scalar mean loss(es).")
            if not torch.isfinite(loss).item():
                raise ValueError("Training loss is not finite.")
            if component_sums is None:
                component_sums = [0.0] * len(components)
                tuple_loss = is_tuple
            elif tuple_loss != is_tuple or len(component_sums) != len(components):
                raise ValueError("calculate_loss changed its return structure.")
            for index, component in enumerate(components):
                component_sums[index] += component.detach().item() * examples

            # Keep autograd through the frozen LM to the trainable input adapter.
            scaler.scale(loss * examples).backward()
            total_examples += examples
            window_examples += examples
            window_batches += 1
            if window_batches == self.gradient_accumulation_steps:
                optimizer_steps += self._step_accumulated_gradients(
                    scaler, window_examples
                )
                window_examples = 0
                window_batches = 0
            if cuda_device and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )

        # An incomplete final window uses its actual sample count, not the
        # configured full-window size, so its gradient is not underscaled.
        if window_examples:
            optimizer_steps += self._step_accumulated_gradients(
                scaler, window_examples
            )
        if total_examples == 0:
            raise ValueError("Training data contains no examples.")
        if cuda_device:
            torch.cuda.synchronize(self.device)
        mean_components = tuple(value / total_examples for value in component_sums)
        mean_loss = sum(mean_components)
        self.epoch_records.append(
            {
                "epoch": epoch_idx,
                "train_loss": mean_loss,
                "examples": total_examples,
                "optimizer_steps": optimizer_steps,
                "seconds": perf_counter() - started,
                "cuda_peak_memory_mb": (
                    torch.cuda.max_memory_allocated(self.device) / (1024**2)
                    if cuda_device
                    else 0.0
                ),
            }
        )
        return mean_components if tuple_loss else mean_components[0]

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        saved_model_file = kwargs.pop("saved_model_file", self.saved_model_file)
        checkpoint_state = getattr(self.model, "checkpoint_state_dict", None)
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "state_dict": (
                checkpoint_state() if checkpoint_state is not None else self.model.state_dict()
            ),
            "other_parameter": self.model.other_parameter(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file, pickle_protocol=4)
        if verbose:
            self.logger.info(
                set_color("Saving current", "blue") + f": {saved_model_file}"
            )
