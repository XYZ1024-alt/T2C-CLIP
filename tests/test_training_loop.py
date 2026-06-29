from pathlib import Path
import tempfile
import unittest

import torch

from t2c_clip.evaluation import ReIDMetrics
from t2c_clip.loops import TrainingLoopConfig, run_training_loop


class ProgressRecorder:
    def __init__(self):
        self.descriptions: list[str] = []
        self.items: list[int] = []

    def __call__(self, iterable, **kwargs):
        self.descriptions.append(kwargs["desc"])
        self.items = list(iterable)
        return self.items


class TqdmLikeProgressRecorder:
    def __init__(self):
        self.items: list[int] = []
        self.messages: list[str] = []
        self.postfixes: list[dict[str, str]] = []

    def __call__(self, iterable, **kwargs):
        self.items = list(iterable)
        return self

    def __iter__(self):
        return iter(self.items)

    def set_postfix(self, values):
        self.postfixes.append(values)

    def write(self, message):
        self.messages.append(message)


class ProgressBarRecorder:
    def __init__(self, iterable, **kwargs):
        self.items = list(iterable)
        self.kwargs = kwargs
        self.messages: list[str] = []
        self.postfixes: list[dict[str, str]] = []

    def __iter__(self):
        return iter(self.items)

    def set_postfix(self, values):
        self.postfixes.append(values)

    def write(self, message):
        self.messages.append(message)


class MultiProgressRecorder:
    def __init__(self):
        self.bars: list[ProgressBarRecorder] = []

    def __call__(self, iterable, **kwargs):
        bar = ProgressBarRecorder(iterable, **kwargs)
        self.bars.append(bar)
        return bar


class TrainingLoopTest(unittest.TestCase):
    def test_default_interval_validates_every_five_epochs_and_saves_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            progress = ProgressRecorder()
            trained_epochs: list[int] = []
            validated_epochs: list[int] = []

            result = run_training_loop(
                model=model,
                optimizer=optimizer,
                config=TrainingLoopConfig(total_epochs=10, checkpoint_dir=Path(tmp)),
                train_one_epoch=lambda epoch, reporter: trained_epochs.append(epoch),
                validate=lambda epoch: _metric(epoch, validated_epochs, {5: 0.3, 10: 0.2}),
                progress_factory=progress,
            )
            best_payload = _load_checkpoint(Path(tmp) / "best.pth")
            last_payload = _load_checkpoint(Path(tmp) / "last.pth")

        self.assertEqual(trained_epochs, list(range(1, 11)))
        self.assertEqual(validated_epochs, [5, 10])
        self.assertEqual(result.best_map, 0.3)
        self.assertEqual(best_payload["epoch"], 5)
        self.assertEqual(last_payload["epoch"], 10)
        self.assertEqual(last_payload["best_map"], 0.3)

    def test_custom_interval_updates_best_only_when_map_improves(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            result = run_training_loop(
                model=model,
                optimizer=None,
                config=TrainingLoopConfig(total_epochs=6, validation_interval=2, checkpoint_dir=Path(tmp)),
                train_one_epoch=lambda epoch, reporter: None,
                validate=lambda epoch: ReIDMetrics(map={2: 0.2, 4: 0.5, 6: 0.4}[epoch], cmc={1: 0.0}),
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            best_payload = _load_checkpoint(Path(tmp) / "best.pth")
            last_payload = _load_checkpoint(Path(tmp) / "last.pth")

        self.assertEqual([row.is_best for row in result.history if row.metrics is not None], [True, True, False])
        self.assertEqual(result.best_map, 0.5)
        self.assertEqual(best_payload["epoch"], 4)
        self.assertEqual(last_payload["epoch"], 6)
        self.assertEqual(last_payload["metrics"]["mAP"], 0.4)

    def test_validation_interval_must_be_positive(self):
        with self.assertRaises(ValueError):
            TrainingLoopConfig(total_epochs=1, validation_interval=0)

    def test_validation_metrics_are_reported_to_progress_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = TqdmLikeProgressRecorder()
            run_training_loop(
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                config=TrainingLoopConfig(total_epochs=1, validation_interval=1, checkpoint_dir=Path(tmp)),
                train_one_epoch=_tracked_empty_train_epoch,
                validate=lambda epoch: ReIDMetrics(map=0.25, cmc={1: 0.5, 5: 0.75, 10: 0.9}),
                progress_factory=progress,
            )

        self.assertEqual(
            progress.postfixes,
            [{"mAP": "0.2500", "best_mAP": "0.2500", "rank1": "0.5000", "rank5": "0.7500", "rank10": "0.9000"}],
        )
        self.assertEqual(
            progress.messages,
            ["epoch=1 mAP=0.2500 rank1=0.5000 rank5=0.7500 rank10=0.9000 best_mAP=0.2500 best=True"],
        )

    def test_each_epoch_is_reported_even_without_validation_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = TqdmLikeProgressRecorder()
            run_training_loop(
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                config=TrainingLoopConfig(total_epochs=3, validation_interval=2, checkpoint_dir=Path(tmp)),
                train_one_epoch=_tracked_empty_train_epoch,
                validate=lambda epoch: ReIDMetrics(map=0.25, cmc={1: 0.5}),
                progress_factory=progress,
            )

        self.assertEqual(
            progress.messages,
            [
                "epoch=1 done",
                "epoch=2 mAP=0.2500 rank1=0.5000 best_mAP=0.2500 best=True",
                "epoch=3 done",
            ],
        )

    def test_training_metrics_are_reported_each_epoch(self):
        logged: list[tuple[int, dict[str, float]]] = []
        with tempfile.TemporaryDirectory() as tmp:
            progress = TqdmLikeProgressRecorder()
            run_training_loop(
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                config=TrainingLoopConfig(total_epochs=2, validation_interval=2, checkpoint_dir=Path(tmp)),
                train_one_epoch=_tracked_epoch_metric_train,
                validate=lambda epoch: ReIDMetrics(map=0.25, cmc={1: 0.5}),
                progress_factory=progress,
                train_metric_logger=lambda epoch, metrics: logged.append((epoch, dict(metrics))),
            )

        self.assertEqual(
            progress.postfixes,
            [
                {"loss": "1.0000", "lr": "0.0100"},
                {"loss": "2.0000", "lr": "0.0100"},
                {
                    "loss": "2.0000",
                    "lr": "0.0100",
                    "mAP": "0.2500",
                    "best_mAP": "0.2500",
                    "rank1": "0.5000",
                },
            ],
        )
        self.assertEqual(
            progress.messages,
            [
                "epoch=1 loss=1.0000 lr=0.0100",
                "epoch=2 loss=2.0000 lr=0.0100 mAP=0.2500 rank1=0.5000 best_mAP=0.2500 best=True",
            ],
        )
        self.assertEqual(logged, [(1, {"loss": 1.0, "lr": 0.01}), (2, {"loss": 2.0, "lr": 0.01})])

    def test_batch_metrics_are_reported_with_train_step_and_epoch_progress(self):
        logged: list[tuple[int, dict[str, float]]] = []
        with tempfile.TemporaryDirectory() as tmp:
            progress = MultiProgressRecorder()
            run_training_loop(
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                config=TrainingLoopConfig(total_epochs=2, validation_interval=3, checkpoint_dir=Path(tmp)),
                train_one_epoch=_batch_reporting_train_epoch,
                validate=lambda epoch: ReIDMetrics(map=0.25, cmc={1: 0.5}),
                progress_factory=progress,
                train_step_metric_logger=lambda step, metrics: logged.append((step, dict(metrics))),
            )

        self.assertEqual([bar.kwargs["desc"] for bar in progress.bars], ["epoch 1/2", "epoch 2/2"])
        self.assertEqual(
            progress.bars[0].postfixes,
            [
                {"loss": "1.1000", "lr": "0.0100"},
                {"loss": "1.2000", "lr": "0.0100"},
                {"loss": "1.1500", "lr": "0.0100"},
            ],
        )
        self.assertEqual(
            progress.bars[1].postfixes,
            [
                {"loss": "2.1000", "lr": "0.0100"},
                {"loss": "2.2000", "lr": "0.0100"},
                {"loss": "2.1500", "lr": "0.0100"},
            ],
        )
        self.assertEqual(
            logged,
            [
                (1, {"loss": 1.1, "lr": 0.01}),
                (2, {"loss": 1.2, "lr": 0.01}),
                (3, {"loss": 2.1, "lr": 0.01}),
                (4, {"loss": 2.2, "lr": 0.01}),
            ],
        )

    def test_validation_metrics_are_sent_to_metric_logger(self):
        logged: list[tuple[int, ReIDMetrics, float | None, bool]] = []
        with tempfile.TemporaryDirectory() as tmp:
            run_training_loop(
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                config=TrainingLoopConfig(total_epochs=1, validation_interval=1, checkpoint_dir=Path(tmp)),
                train_one_epoch=lambda epoch, reporter: None,
                validate=lambda epoch: ReIDMetrics(map=0.25, cmc={1: 0.5}),
                progress_factory=lambda iterable, **kwargs: iterable,
                metric_logger=lambda epoch, metrics, best_map, is_best: logged.append(
                    (epoch, metrics, best_map, is_best)
                ),
            )

        self.assertEqual(logged, [(1, ReIDMetrics(map=0.25, cmc={1: 0.5}), 0.25, True)])

    def test_sanity_gate_off_when_offset_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_training_loop(
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                config=TrainingLoopConfig(
                    total_epochs=10, validation_interval=5, checkpoint_dir=Path(tmp),
                    sanity_check_offset=0,
                ),
                train_one_epoch=lambda epoch, reporter: None,
                validate=lambda epoch: ReIDMetrics(map=0.0123, cmc={1: 0.0123}),
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            # Tolerating a flat random-init floor for the whole run is the OFF-default behaviour.
            self.assertEqual(result.best_map, 0.0123)

    def test_sanity_gate_raises_when_best_map_stays_at_random_floor(self):
        from t2c_clip.loops import SanityCheckFailed
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SanityCheckFailed) as context:
                run_training_loop(
                    model=torch.nn.Linear(2, 2),
                    optimizer=None,
                    config=TrainingLoopConfig(
                        total_epochs=20, validation_interval=5, checkpoint_dir=Path(tmp),
                        sanity_check_offset=11,
                        sanity_improvement_factor=1.5,
                    ),
                    train_one_epoch=lambda epoch, reporter: None,
                    validate=lambda epoch: ReIDMetrics(map=0.0123, cmc={1: 0.0123}),
                    progress_factory=lambda iterable, **kwargs: iterable,
                )
        # First validation batch happens at epoch 5 (best=0.0123, first=0.0123);
        # at epoch 15 (second validation past the offset) the gate fires since
        # best (0.0123) < first (0.0123) * 1.5.
        self.assertEqual(context.exception.first_map, 0.0123)
        self.assertEqual(context.exception.best_map, 0.0123)
        self.assertEqual(context.exception.epoch, 15)

    def test_sanity_gate_passes_when_best_map_clears_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_training_loop(
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                config=TrainingLoopConfig(
                    total_epochs=20, validation_interval=5, checkpoint_dir=Path(tmp),
                    sanity_check_offset=10,
                    sanity_improvement_factor=1.5,
                ),
                train_one_epoch=lambda epoch, reporter: None,
                # epoch 5 (first): 0.01; epoch 10 (second, past offset): 0.03 -> above 1.5x floor.
                validate=lambda epoch: ReIDMetrics(
                    map={5: 0.01, 10: 0.03, 15: 0.05, 20: 0.07}[epoch],
                    cmc={1: 0.0},
                ),
                progress_factory=lambda iterable, **kwargs: iterable,
            )
        self.assertEqual(result.best_map, 0.07)

    def test_sanity_gate_does_not_fire_with_only_one_validation_observation(self):
        # The first validation event alone cannot express improvement, so the gate must wait.
        with tempfile.TemporaryDirectory() as tmp:
            run_training_loop(
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                config=TrainingLoopConfig(
                    total_epochs=1, validation_interval=1, checkpoint_dir=Path(tmp),
                    sanity_check_offset=1,
                    sanity_improvement_factor=1.5,
                ),
                train_one_epoch=lambda epoch, reporter: None,
                validate=lambda epoch: ReIDMetrics(map=0.0123, cmc={1: 0.0123}),
                progress_factory=lambda iterable, **kwargs: iterable,
            )

    def test_sanity_check_offset_must_be_non_negative(self):
        with self.assertRaises(ValueError):
            TrainingLoopConfig(total_epochs=1, checkpoint_dir=Path("/tmp"), sanity_check_offset=-1)

    def test_sanity_improvement_factor_must_be_positive(self):
        with self.assertRaises(ValueError):
            TrainingLoopConfig(total_epochs=1, checkpoint_dir=Path("/tmp"), sanity_improvement_factor=0.0)


def _metric(epoch: int, validated_epochs: list[int], values: dict[int, float]) -> ReIDMetrics:
    validated_epochs.append(epoch)
    return ReIDMetrics(map=values[epoch], cmc={1: values[epoch]})


def _tracked_empty_train_epoch(epoch, reporter) -> None:
    for _ in reporter.batches([0]):
        pass


def _tracked_epoch_metric_train(epoch, reporter) -> dict[str, float]:
    for _ in reporter.batches([0]):
        pass
    return {"loss": float(epoch), "lr": 0.01}


def _batch_reporting_train_epoch(epoch, reporter) -> dict[str, float]:
    losses: list[float] = []
    for batch_number in reporter.batches([1, 2]):
        loss = epoch + batch_number / 10.0
        reporter.report_batch({"loss": loss, "lr": 0.01})
        losses.append(loss)
    return {"loss": sum(losses) / len(losses), "lr": 0.01}


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)
