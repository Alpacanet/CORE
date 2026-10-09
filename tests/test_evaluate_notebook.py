"""Headless notebook checks without notebook packages or real model loading."""

import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "evaluate.ipynb"


def notebook_cells():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return notebook, {cell["id"]: cell for cell in notebook["cells"]}


def cell_source(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def execute_cell(cell, namespace):
    code = compile(ast.parse(cell_source(cell)), f"evaluate.ipynb::{cell['id']}", "exec")
    with redirect_stdout(io.StringIO()):
        exec(code, namespace)


class TinyDataset:
    """Dataset.copy retains the item catalog when interactions are sliced."""

    def __init__(self, interactions, tokens=None):
        self.inter_feat = list(interactions)
        self.item_tokens = tokens if tokens is not None else tuple(str(i) for i in range(7))
        self.item_num = len(self.item_tokens)
        self.copy_calls = []

    def __len__(self):
        return len(self.inter_feat)

    def copy(self, interactions):
        self.copy_calls.append(list(interactions))
        return TinyDataset(interactions, self.item_tokens)


def grouped_metrics(count):
    review_count = count // 2
    groups = {}
    for group, size in (("overall", count), ("review", review_count), ("explore", count - review_count)):
        groups[group] = {"count": size, "fraction": size / count if count else 0.0}
        for k in (10, 20):
            groups[group][f"recall@{k}"] = 0.5 if size else None
            groups[group][f"mrr@{k}"] = 0.25 if size else None
    return groups


class NotebookStructureTests(unittest.TestCase):
    def test_v45_schema_ids_and_unexecuted_cells(self):
        notebook, cells = notebook_cells()
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["nbformat_minor"], 5)
        self.assertEqual(len(notebook["cells"]), 13)
        self.assertEqual(len(cells), len(notebook["cells"]))
        self.assertIsInstance(notebook["metadata"], dict)
        self.assertEqual(notebook["metadata"]["kernelspec"]["language"], "python")
        for cell in cells.values():
            self.assertRegex(cell["id"], r"^[A-Za-z0-9_-]{1,64}$")
            self.assertIn(cell["cell_type"], {"markdown", "code"})
            self.assertIsInstance(cell["metadata"], dict)
            self.assertIsInstance(cell_source(cell), str)
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_all_six_code_cells_compile_without_notebook_dependencies(self):
        notebook, _ = notebook_cells()
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 6)
        for cell in code_cells:
            with self.subTest(cell=cell["id"]):
                compile(ast.parse(cell_source(cell)), f"evaluate.ipynb::{cell['id']}", "exec")

    def test_bootstrap_parameters_and_definition_do_not_load_or_evaluate_models(self):
        _, cells = notebook_cells()
        forbidden = Mock(side_effect=AssertionError("Bootstrap must not start model work"))
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=Mock(return_value=False))
        fake_torch.load = forbidden
        fake_utils = ModuleType("recbole.data.utils")
        fake_utils.get_dataloader = forbidden
        fake_recbole_utils = ModuleType("recbole.utils")
        fake_recbole_utils.init_logger = forbidden
        fake_train = ModuleType("train")
        fake_train.ROOT = PROJECT_ROOT
        for name in ("_file_hash", "_review_explore_report", "_write_json", "load_experiment", "run_experiment"):
            setattr(fake_train, name, forbidden)
        fake_trainer = ModuleType("trainer")
        fake_trainer.CORETrainer = forbidden
        modules = {
            "pandas": ModuleType("pandas"), "torch": fake_torch,
            "recbole.data.utils": fake_utils, "recbole.utils": fake_recbole_utils,
            "train": fake_train, "trainer": fake_trainer,
        }
        namespace = {"__name__": "core_notebook_bootstrap_test"}
        with patch.dict(sys.modules, modules), patch.object(sys, "path", list(sys.path)), patch.object(
            Path, "cwd", return_value=PROJECT_ROOT / "tests"
        ):
            for cell_id in ("imports", "parameters", "evaluation-function"):
                execute_cell(cells[cell_id], namespace)
        forbidden.assert_not_called()
        self.assertEqual(namespace["PROJECT_ROOT"], PROJECT_ROOT)
        self.assertEqual(namespace["SPLIT"], "valid")
        self.assertIsNone(namespace["LIMIT"])
        self.assertEqual(namespace["DEVICE"], "cpu")
        self.assertEqual(namespace["EVAL_BATCH_SIZE"], 32)
        mapping = namespace["RUN_DIRECTORIES"]
        self.assertEqual(set(mapping), {"ave", "trm", "llm"})
        self.assertEqual(namespace["RUN_DIRECTORY"], mapping["ave"])
        for key, label in (("ave", "ave"), ("trm", "trm"), ("llm", "pretrained_lora")):
            self.assertEqual(mapping[key].parent, PROJECT_ROOT / "results" / "phase1")
            self.assertIn(f"-tmall-{label}-", mapping[key].name)
        self.assertTrue(callable(namespace["evaluate_experiment"]))

    def test_function_retains_original_public_signature(self):
        _, cells = notebook_cells()
        namespace = {}
        execute_cell(cells["evaluation-function"], namespace)
        parameters = list(inspect.signature(namespace["evaluate_experiment"]).parameters.values())
        self.assertEqual(
            [(parameter.name, parameter.default) for parameter in parameters],
            [("run_directory", inspect.Parameter.empty), ("split", "valid"),
             ("device", None), ("eval_batch_size", None), ("limit", None)],
        )
        self.assertTrue(all(parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters))

    def test_run_cell_clears_old_result_before_a_failed_call(self):
        _, cells = notebook_cells()
        evaluate = Mock(side_effect=RuntimeError("injected run failure"))
        namespace = {
            "result": {"status": "complete"}, "evaluate_experiment": evaluate,
            "RUN_DIRECTORY": PROJECT_ROOT / "saved-run", "SPLIT": "valid",
            "DEVICE": "cpu", "EVAL_BATCH_SIZE": 32, "LIMIT": None,
        }
        with self.assertRaisesRegex(RuntimeError, "injected run failure"):
            execute_cell(cells["run-evaluation"], namespace)
        self.assertIsNone(namespace["result"])
        evaluate.assert_called_once_with(
            run_directory=namespace["RUN_DIRECTORY"], split="valid", device="cpu",
            eval_batch_size=32, limit=None,
        )


class NotebookEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary_parent = PROJECT_ROOT / "results" / "core"
        temporary_parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="notebook-tests-", dir=temporary_parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "original-run"
        self.source.mkdir()
        (self.source / "best.pth").write_bytes(b"trusted-test-fixture-not-a-real-checkpoint")
        (self.source / "metrics.json").write_text('{"status":"original-complete"}', encoding="utf-8")
        (self.source / "manifest.json").write_text('{"original":true}', encoding="utf-8")
        self.original_files = self.source_snapshot()
        self.checkpoint_hash = hashlib.sha256((self.source / "best.pth").read_bytes()).hexdigest()
        self.config = {"dataset": "tmall", "model": "COREave", "device": "cpu", "topk": [10, 20], "eval_batch_size": 32}
        self.splits = tuple(SimpleNamespace(dataset=TinyDataset(range(size))) for size in (3, 5, 6))
        self.model = Mock()
        self.trainer = Mock()
        self.trainer.evaluate.return_value = {"recall@10": 0.5, "mrr@10": 0.25, "recall@20": 0.5, "mrr@20": 0.25}
        self.trainer.last_review_explore_result = grouped_metrics(5)
        self.loader_factory = Mock(side_effect=lambda config, dataset, sampler, shuffle: SimpleNamespace(dataset=dataset))

        def report(topk, *, valid=None, test=None, best_epoch=None, scope="full"):
            return {"topk": list(topk), "valid": valid, "test": test, "best_epoch": best_epoch, "scope": scope}

        def write_json(path, value):
            Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

        self.namespace = {
            "Path": Path, "datetime": datetime, "timezone": timezone, "uuid4": uuid4,
            "ROOT": self.root, "torch": SimpleNamespace(load=Mock(return_value={"epoch": 6})),
            "load_experiment": Mock(return_value=(self.model, self.config, self.splits)),
            "get_dataloader": Mock(return_value=self.loader_factory), "init_logger": Mock(),
            "CORETrainer": Mock(return_value=self.trainer),
            "_file_hash": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "_review_explore_report": Mock(side_effect=report), "_write_json": write_json,
        }
        _, cells = notebook_cells()
        execute_cell(cells["evaluation-function"], self.namespace)
        self.evaluate = self.namespace["evaluate_experiment"]

    def source_snapshot(self):
        return {path.name: path.read_bytes() for path in self.source.iterdir() if path.is_file()}

    def call(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return self.evaluate(self.source, **kwargs)

    def assert_source_untouched_and_no_training(self):
        self.assertEqual(self.source_snapshot(), self.original_files)
        self.trainer.fit.assert_not_called()
        self.model.calculate_loss.assert_not_called()

    def assert_saved_success(self, summary, split, count, scope):
        output = Path(summary["result_directory"])
        self.assertEqual(output.parent, self.root / "results" / "core")
        self.assertNotEqual(output, self.source)
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        report = json.loads((output / "review_explore.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics, summary)
        for saved in (metrics, report):
            self.assertEqual(saved["checkpoint_sha256"], self.checkpoint_hash)
            self.assertEqual(saved["best_epoch"], 6)
            self.assertEqual(saved["source_run_directory"], str(self.source.resolve()))
            self.assertEqual(saved["split"], split)
            self.assertEqual(saved["scope"], scope)
            self.assertEqual(saved["evaluated_count"], count)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["test_evaluated"], split == "test")
        self.assertEqual(report[split], self.trainer.last_review_explore_result)
        self.assertIsNone(report["test" if split == "valid" else "valid"])
        self.assert_source_untouched_and_no_training()
        self.trainer.tensorboard.close.assert_called_once_with()
        self.namespace["torch"].load.assert_called_once_with(
            self.source.resolve() / "best.pth", map_location="cpu", weights_only=False
        )

    def test_full_default_validation_saves_metadata_without_subset_copy(self):
        summary = self.call()
        self.assert_saved_success(summary, "valid", 5, "full")
        self.assertIsNone(summary["limit"])
        self.assertEqual(summary["original_split_count"], 5)
        self.assertEqual(summary["eval_batch_size"], 32)
        self.namespace["load_experiment"].assert_called_once_with(self.source.resolve(), device=None)
        self.namespace["get_dataloader"].assert_called_once_with(self.config, "valid")
        self.assertEqual(self.splits[1].dataset.copy_calls, [])
        loader = self.trainer.evaluate.call_args.args[0]
        self.assertIs(loader.dataset, self.splits[1].dataset)
        self.assertEqual(self.trainer.evaluate.call_args.kwargs, {"load_best_model": False, "show_progress": False})

    def test_prefix_validation_preserves_catalog_and_marks_subset(self):
        self.trainer.last_review_explore_result = grouped_metrics(2)
        summary = self.call(device="cpu", eval_batch_size=4, limit=2)
        self.assert_saved_success(summary, "valid", 2, "prefix_subset")
        self.assertEqual(summary["original_split_count"], 5)
        self.assertEqual(summary["eval_batch_size"], 4)
        loader = self.trainer.evaluate.call_args.args[0]
        self.assertEqual(loader.dataset.inter_feat, [0, 1])
        self.assertIs(loader.dataset.item_tokens, self.splits[1].dataset.item_tokens)
        self.assertEqual(loader.dataset.item_num, self.splits[1].dataset.item_num)
        self.assertEqual(self.splits[1].dataset.copy_calls, [[0, 1]])
        self.loader_factory.assert_called_once_with(self.config, loader.dataset, None, shuffle=False)

    def test_explicit_test_split_uses_test_data_and_report_key(self):
        self.trainer.last_review_explore_result = grouped_metrics(3)
        summary = self.call(split="test", limit=3)
        self.assert_saved_success(summary, "test", 3, "prefix_subset")
        self.assertEqual(summary["original_split_count"], 6)
        self.namespace["get_dataloader"].assert_called_once_with(self.config, "test")
        self.assertEqual(self.splits[1].dataset.copy_calls, [])
        self.assertEqual(self.splits[2].dataset.copy_calls, [[0, 1, 2]])

    def test_failures_and_interruptions_record_status_and_close_tensorboard(self):
        for error, status in ((RuntimeError("mock evaluation failure"), "failed"), (KeyboardInterrupt("mock stop"), "interrupted")):
            with self.subTest(status=status):
                self.trainer.reset_mock()
                self.trainer.evaluate.side_effect = error
                with self.assertRaises(type(error)):
                    self.call()
                files = sorted((self.root / "results" / "core").glob("evaluation-*/metrics.json"))
                matching = [json.loads(path.read_text(encoding="utf-8")) for path in files]
                recorded = next(value for value in matching if value["status"] == status)
                self.assertEqual(recorded["error"], str(error))
                self.assertEqual(recorded["best_epoch"], 6)
                self.assertEqual(recorded["checkpoint_sha256"], self.checkpoint_hash)
                self.assertIn("finished_at_utc", recorded)
                self.assertFalse((Path(recorded["result_directory"]) / "review_explore.json").exists())
                self.trainer.tensorboard.close.assert_called_once_with()
                self.assert_source_untouched_and_no_training()

    def test_invalid_parameters_fail_before_model_loading(self):
        for parameter in ("eval_batch_size", "limit"):
            for value in (True, False, 0, -1, 1.5, "2"):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaisesRegex(ValueError, f"{parameter} must be a positive integer"):
                        self.call(**{parameter: value})
        with self.assertRaisesRegex(ValueError, "split must be valid or test"):
            self.call(split="train")
        self.namespace["load_experiment"].assert_not_called()
        self.namespace["torch"].load.assert_not_called()
        self.namespace["CORETrainer"].assert_not_called()
        self.assert_source_untouched_and_no_training()


if __name__ == "__main__":
    unittest.main()
