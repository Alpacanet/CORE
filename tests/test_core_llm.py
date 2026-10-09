"""Offline contract tests for continuous item-ID adaptation to a causal LM.

These deliberately use a tiny local Qwen architecture; running the tests never
downloads weights or reads the Tmall dataset.
"""

import copy
import io
import unittest

import torch
from transformers import Qwen3Config, Qwen3Model

from core_llm import COREllm


class TinyDataset:
    def __init__(self, tokens=None):
        self.field2id_token = {
            "item_id": tokens or ["[PAD]", "101", "205", "309", "410", "511", "612"]
        }
        self.field2seqlen = {"item_id_list": 8}

    def num(self, field):
        return len(self.field2id_token[field])


def model_config(tuning="frozen"):
    return {
        "USER_ID_FIELD": "session_id",
        "ITEM_ID_FIELD": "item_id",
        "LIST_SUFFIX": "_list",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "MAX_ITEM_LIST_LENGTH": 8,
        "NEG_PREFIX": "neg_",
        "device": torch.device("cpu"),
        "embedding_size": 8,
        "loss_type": "CE",
        "sess_dropout": 0.0,
        "item_dropout": 0.0,
        "temperature": 0.07,
        "llm_init": "random",
        "llm_tuning": tuning,
        "llm_dtype": "float32",
        "llm_seed": 2020,
        "llm_name": "test-qwen",
        "llm_revision": "test",
        "llm_gradient_checkpointing": False,
        "llm_core_branch": "ave",
        "llm_fusion": "gate",
        "lora_rank": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "gate_bias": 1.0,
    }


def make_model(tuning="frozen", dataset=None, init_mode="random", config_overrides=None):
    config = model_config(tuning)
    config["llm_init"] = init_mode
    if config_overrides:
        config.update(config_overrides)
    # Compact random-backbone checkpoints rely on the same initialization seed.
    # Set it before creating the injected backbone, not only before COREllm.
    torch.manual_seed(config["llm_seed"])
    backbone = Qwen3Model(
        Qwen3Config(
            vocab_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
    )
    return COREllm(config, dataset or TinyDataset(), backbone=backbone)


def interaction():
    return {
        "item_id_list": torch.tensor([[1, 2, 3, 0], [4, 2, 0, 0]]),
        "item_length": torch.tensor([3, 2]),
        "item_id": torch.tensor([4, 3]),
    }


class COREllmTests(unittest.TestCase):
    def test_full_sort_and_cross_entropy_contract(self):
        for tuning in ("frozen", "lora", "full"):
            with self.subTest(tuning=tuning):
                model = make_model(tuning)
                model.eval()
                scores = model.full_sort_predict(interaction())
                self.assertEqual(tuple(scores.shape), (2, 7))
                self.assertTrue(torch.isfinite(scores).all().item())
                loss = model.calculate_loss(interaction())
                self.assertEqual(loss.ndim, 0)
                self.assertTrue(torch.isfinite(loss).item())
                self.assertGreater(loss.item(), 0.0)

    def test_right_padding_does_not_change_recommendations(self):
        model = make_model("lora").eval()
        with torch.no_grad():
            short = model.full_sort_predict(
                {"item_id_list": torch.tensor([[1, 2, 3]])}
            )
            padded = model.full_sort_predict(
                {"item_id_list": torch.tensor([[1, 2, 3, 0, 0, 0]])}
            )
            mixed_batch = model.full_sort_predict(
                {"item_id_list": torch.tensor([[1, 2, 3, 0], [4, 5, 6, 1]])}
            )
        torch.testing.assert_close(short, padded, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(short[0], mixed_batch[0], rtol=1e-5, atol=1e-5)

    def test_backbone_cannot_see_future_item_ids(self):
        model = make_model("lora").eval()
        outputs = []

        def collect_hidden(_module, _args, output):
            outputs.append(output.last_hidden_state.detach().clone())

        # Inspect states from the actual recommendation forward path so an
        # accidental bidirectional mask in that path is detected.
        handle = model.backbone.register_forward_hook(collect_hidden)
        try:
            with torch.no_grad():
                model.forward(torch.tensor([[1, 2, 3]]))
                model.forward(torch.tensor([[1, 5, 6]]))
        finally:
            handle.remove()
        self.assertEqual(len(outputs), 2)
        torch.testing.assert_close(outputs[0][:, 0], outputs[1][:, 0])
        self.assertFalse(torch.allclose(outputs[0][:, -1], outputs[1][:, -1]))

    def test_frozen_backbone_keeps_input_adapter_gradient_path(self):
        model = make_model("frozen").train()
        model.calculate_loss(interaction()).backward()
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        projection_gradients = [p.grad for p in model.input_projection.parameters()]
        self.assertTrue(all(g is not None for g in projection_gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in projection_gradients), 0)
        self.assertGreater(model.item_embedding.weight.grad[1:].abs().sum().item(), 0)

    def test_lora_only_trains_adapters_in_backbone(self):
        model = make_model("lora").train()
        model.calculate_loss(interaction()).backward()
        adapters = [(name, p) for name, p in model.backbone.named_parameters()
                    if "lora_" in name]
        base = [(name, p) for name, p in model.backbone.named_parameters()
                if "lora_" not in name]
        self.assertTrue(adapters)
        self.assertTrue(base)
        self.assertTrue(all(p.requires_grad for _, p in adapters))
        self.assertTrue(all(not p.requires_grad and p.grad is None for _, p in base))
        self.assertTrue(all(p.grad is not None for _, p in adapters))
        # LoRA initializes one factor to zero, so some factor gradients may be
        # zero on step one. At least one adapter must receive a nonzero signal.
        self.assertGreater(sum(p.grad.abs().sum().item() for _, p in adapters), 0)

    def test_compact_checkpoint_restores_predictions_after_training(self):
        model = make_model("lora").train()
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=0.01
        )
        model.calculate_loss(interaction()).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model.full_sort_predict(interaction())
        compact = model.checkpoint_state_dict()
        compact_size = sum(v.numel() for v in compact.values() if torch.is_tensor(v))
        full_size = sum(v.numel() for v in model.state_dict().values() if torch.is_tensor(v))
        self.assertLess(compact_size, full_size / 2)

        # Exercise serialization too, not just an in-memory copy of tensors.
        buffer = io.BytesIO()
        torch.save(compact, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=False)
        restored = make_model("lora")
        restored.load_state_dict(checkpoint)
        restored.eval()
        with torch.no_grad():
            actual = restored.full_sort_predict(interaction())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_checkpoint_rejects_same_size_different_item_mapping(self):
        model = make_model("lora")
        checkpoint = copy.deepcopy(model.checkpoint_state_dict())
        tokens = list(TinyDataset().field2id_token["item_id"])
        tokens[1], tokens[2] = tokens[2], tokens[1]
        restored = make_model("lora", TinyDataset(tokens))
        with self.assertRaises((ValueError, RuntimeError)):
            restored.load_state_dict(checkpoint)

    def test_compact_checkpoint_rejects_missing_trainable_parameter(self):
        model = make_model("lora")
        checkpoint = copy.deepcopy(model.checkpoint_state_dict())
        del checkpoint["input_projection.weight"]
        restored = make_model("lora")
        with self.assertRaisesRegex(RuntimeError, "input_projection.weight"):
            restored.load_state_dict(checkpoint)

    def test_checkpoint_rejects_changed_scoring_temperature_before_loading(self):
        model = make_model("lora")
        checkpoint = copy.deepcopy(model.checkpoint_state_dict())
        # Make a checkpoint weight observably different to verify semantic
        # validation occurs before any parameters are copied into the model.
        checkpoint["item_embedding.weight"].add_(1.0)
        restored = make_model("lora", config_overrides={"temperature": 0.5})
        before = restored.item_embedding.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "temperature"):
            restored.load_state_dict(checkpoint)
        torch.testing.assert_close(restored.item_embedding.weight, before)

    def test_pretrained_and_random_controls_start_with_identical_adapters(self):
        # Inject identical local backbones to isolate experiment initialization
        # from pretrained loading. The init-mode label must not change items,
        # projections, gates, LoRA parameters, or the resulting predictions.
        random_control = make_model("lora", init_mode="random").eval()
        pretrained_control = make_model("lora", init_mode="pretrained").eval()
        random_trainable = {name: p for name, p in random_control.named_parameters()
                            if p.requires_grad}
        pretrained_trainable = {
            name: p for name, p in pretrained_control.named_parameters() if p.requires_grad
        }
        self.assertEqual(random_trainable.keys(), pretrained_trainable.keys())
        for name in random_trainable:
            with self.subTest(parameter=name):
                torch.testing.assert_close(random_trainable[name], pretrained_trainable[name])
        with torch.no_grad():
            torch.testing.assert_close(
                random_control.full_sort_predict(interaction()),
                pretrained_control.full_sort_predict(interaction()),
            )


if __name__ == "__main__":
    unittest.main()
